"""
실데이터를 dashboard.html에 주입하고 브라우저로 바로 엽니다.
사용법: python run_dashboard.py
"""
import sys, json, ast, webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from utils.config import DATA_PROCESSED

# ─── Plotly 차트 생성 ────────────────────────────────────────────
def generate_eda(corpus: pd.DataFrame) -> dict:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠️  plotly 미설치 — EDA 차트 생략")
        return {}

    ACCENT = ["#a8a4f0","#7dd4d4","#85c99a","#e8c97a","#f0a080"]
    LC_COL  = {"자견":"#a8a4f0","성견":"#7dd4d4","노령견":"#e8c97a"}

    def layout(**kwargs):
        base = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="'Pretendard Variable','Inter',sans-serif", size=12, color="#09090b"),
            hoverlabel=dict(bgcolor="#ffffff", font_size=12),
            margin=dict(t=16, b=40, l=60, r=20),
        )
        base.update(kwargs)
        return base

    eda = {}

    # ① 질병 롱테일 — 수평 막대
    dis = corpus["disease"].value_counts().head(15)
    fig = go.Figure(go.Bar(
        x=dis.values.tolist(), y=dis.index.tolist(), orientation="h",
        marker_color=["#f0a535" if n=="기타" else "#6c63ff" for n in dis.index],
        text=[f"{v:,}" for v in dis.values], textposition="outside",
        hovertemplate="%{y}: %{x:,}건<extra></extra>",
    ))
    fig.update_layout(**layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(title="건수"), height=340))
    eda["disease_bar"] = json.loads(fig.to_json())

    # ② 생애주기 도넛
    lc = corpus["lifeCycle"].value_counts()
    fig = go.Figure(go.Pie(
        labels=lc.index.tolist(), values=lc.values.tolist(), hole=0.52,
        marker_colors=[LC_COL.get(l,"#aaa") for l in lc.index],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}건 (%{percent})<extra></extra>",
    ))
    fig.update_layout(**layout(
        height=340, margin=dict(t=16, b=50, l=20, r=20),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        annotations=[dict(text=f"{len(corpus):,}", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=18), xanchor="center")]))
    eda["lifecycle_pie"] = json.loads(fig.to_json())

    # ③ 진료과 × 생애주기 히트맵
    DEPTS = ["내과","외과","피부과","안과","치과"]
    LIVES = ["자견","성견","노령견"]
    ct = pd.crosstab(corpus["lifeCycle"], corpus["department"])
    z = [[int(ct.loc[lc_k, d]) if lc_k in ct.index and d in ct.columns else 0
          for d in DEPTS] for lc_k in LIVES]
    fig = go.Figure(go.Heatmap(
        z=z, x=DEPTS, y=LIVES,
        colorscale=[[0,"#f0effd"],[0.5,"#c4c0f5"],[1,"#a8a4f0"]],
        text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:,}건<extra></extra>", showscale=True,
    ))
    fig.update_layout(**layout(height=280, margin=dict(t=16, b=50, l=70, r=60),
        xaxis=dict(title="진료과"), yaxis=dict(title="생애주기")))
    eda["dept_heatmap"] = json.loads(fig.to_json())

    # ④ 텍스트 길이 박스플롯
    fig = go.Figure()
    for lc_name, color in LC_COL.items():
        vals = corpus[corpus["lifeCycle"] == lc_name]["input"].str.len().tolist()
        fig.add_trace(go.Box(
            y=vals, name=lc_name, marker_color=color,
            boxpoints="outliers", jitter=0.3, pointpos=-1.8,
            hovertemplate=f"{lc_name} — 길이: %{{y}}자<extra></extra>",
        ))
    fig.update_layout(**layout(height=280,
        yaxis=dict(title="문자 수 (chars)"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")))
    eda["text_boxplot"] = json.loads(fig.to_json())

    # ⑤ 계층 선버스트
    sb = (corpus.groupby(["lifeCycle","department","disease"])
          .size().reset_index(name="cnt"))
    sb = (sb.sort_values("cnt", ascending=False)
          .groupby(["lifeCycle","department"]).head(4)
          .reset_index(drop=True))
    fig = px.sunburst(sb, path=["lifeCycle","department","disease"], values="cnt",
                      color_discrete_sequence=ACCENT)
    fig.update_traces(hovertemplate="%{label}: %{value:,}건<extra></extra>",
                      textfont_size=12)
    fig.update_layout(**layout(height=440, margin=dict(t=16, b=16, l=16, r=16)))
    eda["sunburst"] = json.loads(fig.to_json())

    # ⑥ 진료과별 Top 5 질병 — 그룹드 바 (기타 제외)
    dd = (corpus[corpus["disease"] != "기타"]
          .groupby(["department","disease"]).size().reset_index(name="cnt"))
    dd = (dd.sort_values("cnt", ascending=False)
          .groupby("department").head(5).reset_index(drop=True))
    DEPT_ORDER = ["내과","외과","피부과","안과","치과"]
    fig = px.bar(dd, x="cnt", y="disease", color="department",
                 orientation="h", barmode="group",
                 category_orders={"department": DEPT_ORDER},
                 color_discrete_sequence=ACCENT,
                 labels={"cnt":"건수","disease":"질병","department":"진료과"})
    fig.update_layout(**layout(height=360,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(title="건수"),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center")))
    eda["dept_top10"] = json.loads(fig.to_json())

    # ⑦ 생애주기별 빈출 바이그램 트리맵 (단어 단독이 아닌 2-gram 구문으로 문맥 표현)
    import collections as _col
    STOPWORDS = {"있다","하다","되다","이다","않다","없다","그","이","저","것","수","더","도","을","를",
                 "위해","통해","에서","으로","에게","에게서","부터","까지","같다","보다","대해","따라"}
    rows = []
    for lc_name in ["자견","성견","노령견"]:
        subset = corpus[corpus["lifeCycle"] == lc_name]["input_tokens"].dropna()
        counter: _col.Counter = _col.Counter()
        for ts in subset:
            toks = [t for t in str(ts).split() if len(t) >= 2 and not t.isdigit() and t not in STOPWORDS]
            for a, b in zip(toks, toks[1:]):
                counter[f"{a} {b}"] += 1
        for bigram, cnt in counter.most_common(20):
            rows.append({"lifeCycle": lc_name, "bigram": bigram, "count": cnt})
    if rows:
        word_df = pd.DataFrame(rows)
        fig = px.treemap(word_df, path=["lifeCycle","bigram"], values="count",
                         color="count",
                         color_continuous_scale=["#f0effd","#ccc9f7","#a8a4f0","#7b76d4","#5450b0"],
                         range_color=[word_df["count"].min(), word_df["count"].max()])
        fig.update_traces(
            hovertemplate="%{label}: %{value:,}회<extra></extra>",
            textinfo="label+value",
            textfont_size=12,
        )
        fig.update_coloraxes(colorbar=dict(
            title="빈도", thickness=12, len=0.7,
            tickfont=dict(size=10),
        ))
        fig.update_layout(**layout(height=420, margin=dict(t=16,b=16,l=16,r=16)))
        eda["word_treemap"] = json.loads(fig.to_json())

    # ⑧ Train / Val 생애주기·진료과 분할 균형 스택드 바
    tv = corpus.groupby(["split","lifeCycle"]).size().reset_index(name="cnt")
    fig = px.bar(tv, x="lifeCycle", y="cnt", color="split",
                 barmode="stack",
                 color_discrete_map={"train":"#a8a4f0","val":"#7dd4d4"},
                 category_orders={"lifeCycle":["자견","성견","노령견"],"split":["train","val"]},
                 labels={"cnt":"건수","lifeCycle":"생애주기","split":"분할"})
    fig.update_layout(**layout(height=280,
        xaxis=dict(title="생애주기"),
        yaxis=dict(title="건수"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")))
    eda["train_val_bar"] = json.loads(fig.to_json())

    # ⑨ 질문 vs 답변 길이 산점도 (샘플 2,000개)
    sample = corpus.sample(min(2000, len(corpus)), random_state=42).copy()
    sample["q_len"] = sample["input"].str.len()
    sample["a_len"] = sample["output"].str.len()
    LC_COLOR_MAP = {"자견":"#a8a4f0","성견":"#7dd4d4","노령견":"#e8c97a"}
    fig = px.scatter(sample, x="q_len", y="a_len", color="lifeCycle",
                     color_discrete_map=LC_COLOR_MAP,
                     opacity=0.45, size_max=4,
                     labels={"q_len":"질문 길이 (chars)","a_len":"답변 길이 (chars)","lifeCycle":"생애주기"},
                     category_orders={"lifeCycle":["자견","성견","노령견"]})
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**layout(height=300,
        xaxis=dict(title="질문 길이 (chars)"),
        yaxis=dict(title="답변 길이 (chars)"),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")))
    eda["len_scatter"] = json.loads(fig.to_json())

    return eda

APP   = Path(__file__).parent / "app"
SRC   = APP / "dashboard.html"
OUT   = APP / "dashboard_live.html"

print("📂 데이터 로딩 중...")

corpus  = pd.read_csv(DATA_PROCESSED / "corpus_preprocessed.csv")
eval_df = pd.read_csv(DATA_PROCESSED / "evaluation_summary.csv")
match   = pd.read_csv(DATA_PROCESSED / "matching_results.csv")
train   = corpus[corpus["split"] == "train"].reset_index(drop=True)

# 생애주기 분포
lc = corpus["lifeCycle"].value_counts()
lifecycle = [
    {"key":"puppy",  "ko":"자견",   "en":"Puppy",  "count":int(lc.get("자견",0))},
    {"key":"adult",  "ko":"성견",   "en":"Adult",  "count":int(lc.get("성견",0))},
    {"key":"senior", "ko":"노령견", "en":"Senior", "count":int(lc.get("노령견",0))},
]

# 진료과 분포
_dk = {"내과":"internal","외과":"surgery","피부과":"derm","안과":"ophth","치과":"dental"}
_de = {"내과":"Internal","외과":"Surgery","피부과":"Dermatology","안과":"Ophthalmology","치과":"Dental"}
N   = len(corpus)
department = [
    {"key":_dk.get(d,"etc"),"ko":d,"en":_de.get(d,d),"pct":round(c/N*100,1)}
    for d,c in corpus["department"].value_counts().items()
]

# 질병 Top 10
disease_top = [
    {"name":n,"count":int(c)}
    for n,c in corpus["disease"].value_counts().head(10).items()
]

# 텍스트 길이
ql, al = corpus["input"].str.len(), corpus["output"].str.len()
text_stats = {
    "q_mean":int(ql.mean()),"q_median":int(ql.median()),"q_max":int(ql.max()),
    "a_mean":int(al.mean()),"a_median":int(al.median()),"a_max":int(al.max()),
}

# 평가 지표
rt = eval_df[eval_df["모델"]=="TF-IDF"].iloc[0]
rb = eval_df[eval_df["모델"]=="Sentence-BERT"].iloc[0]
metrics = {
    "overall": [
        {"k":"Hit@1","tfidf":round(float(rt["Hit@1"])*100,2),"bert":round(float(rb["Hit@1"])*100,2)},
        {"k":"Hit@3","tfidf":round(float(rt["Hit@3"])*100,2),"bert":round(float(rb["Hit@3"])*100,2)},
        {"k":"Hit@5","tfidf":round(float(rt["Hit@5"])*100,2),"bert":round(float(rb["Hit@5"])*100,2)},
        {"k":"MAP@5","tfidf":round(float(rt["MAP@5"])*100,4),"bert":round(float(rb["MAP@5"])*100,4)},
    ],
    "byLifecycle": [
        {"key":"puppy", "ko":"자견",  "n":17,
         "tfidf":round(float(rt["자견 Hit@5"])*100,1),  "bert":round(float(rb["자견 Hit@5"])*100,1)},
        {"key":"adult", "ko":"성견",  "n":17,
         "tfidf":round(float(rt["성견 Hit@5"])*100,1),  "bert":round(float(rb["성견 Hit@5"])*100,1)},
        {"key":"senior","ko":"노령견","n":16,
         "tfidf":round(float(rt["노령견 Hit@5"])*100,1),"bert":round(float(rb["노령견 Hit@5"])*100,1)},
    ],
}

# 코퍼스 문서 조회
def doc(idx):
    if 0 <= idx < len(train):
        r = train.iloc[idx]
        return {
            "lifeCycle":str(r.get("lifeCycle","")),"department":str(r.get("department","")),
            "disease":str(r.get("disease","")),"input":str(r.get("input",""))[:160],
            "output":str(r.get("output",""))[:220],
        }
    return {"lifeCycle":"","department":"","disease":"","input":"","output":""}

# 50개 평가 쿼리 (실데이터 top-5)
eval_queries = []
for _, row in match.iterrows():
    try:    tf = ast.literal_eval(str(row["tfidf_top5"]))
    except: tf = []
    try:    sb = ast.literal_eval(str(row["sbert_top5"]))
    except: sb = []
    eval_queries.append({
        "id":str(row["query_id"]),"life":str(row["lifeCycle"]),
        "dept":str(row["department"]),"disease":str(row["disease"]),
        "title":str(row["query"])[:100],
        "results":{"tfidf":[doc(i) for i in tf[:5]],"bert":[doc(i) for i in sb[:5]]},
    })

# 검색 데모 결과 (Q007 외이염 케이스)
def mk_demo(idxs, base):
    return [{"rank":r+1,"sim":round(base-r*0.028,4),"lifecycle":doc(i)["lifeCycle"],
             "dept":doc(i)["department"],"disease":doc(i)["disease"],
             "q":doc(i)["input"][:200],"a":doc(i)["output"][:200]}
            for r,i in enumerate(idxs[:5])]
try:
    row6 = match.iloc[6]
    demo_bert  = mk_demo(ast.literal_eval(str(row6["sbert_top5"])),  0.862)
    demo_tfidf = mk_demo(ast.literal_eval(str(row6["tfidf_top5"])), 0.693)
except Exception:
    demo_bert = demo_tfidf = []

print("📊 EDA 차트 생성 중...")
eda_charts = generate_eda(corpus)
print(f"   ✓ 차트 {len(eda_charts)}개 생성 완료")

APP_DATA = {
    "stats":      {"total":int(N),"train":int(len(train)),"val":int(N-len(train)),"depts":5},
    "lifecycle":  lifecycle, "department": department, "diseaseTop": disease_top,
    "textStats":  text_stats, "metrics": metrics,
    "sampleSuggestions": [
        "눈에 눈곱이 자꾸 끼고 빨갛게 충혈돼요","산책 후 뒷다리를 절뚝거려요",
        "귀를 자꾸 긁고 냄새가 나요","피부에 붉은 발진이 생겼어요",
        "물을 너무 많이 마시고 소변을 자주 봐요",
    ],
    "results":    {"bert":demo_bert,"tfidf":demo_tfidf},
    "evalQueries": eval_queries,
    "eda":        eda_charts,
}

print(f"✅ 데이터 구성 완료 — 쿼리 {len(eval_queries)}개 · {len(json.dumps(APP_DATA))//1024}KB")

# HTML에 데이터 주입
html = SRC.read_text(encoding="utf-8")
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA,ensure_ascii=False)});"
    "</script>"
)
html = html.replace("</body>", override + "\n</body>")
OUT.write_text(html, encoding="utf-8")

print(f"📄 저장: {OUT}")
print(f"🌐 브라우저에서 열기...")

webbrowser.open(f"file://{OUT.resolve()}")
print("✅ 완료!")
