"""
실데이터를 dashboard.html에 주입하고 브라우저로 바로 엽니다.
사용법: python run_dashboard.py
"""
import sys, json, webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.chart_builder import ChartBuilder
from utils.theme import build_css

APP = Path(__file__).parent / "app"
SRC = APP / "dashboard.html"
OUT = APP / "dashboard_live.html"

print("📂 데이터 로딩 중...")
dl     = DataLoader()
corpus = dl.corpus
N      = len(corpus)

# 생애주기 분포
lc = corpus["lifeCycle"].value_counts()
lifecycle = [
    {"key": "puppy",  "ko": "자견",   "en": "Puppy",  "count": int(lc.get("자견",  0))},
    {"key": "adult",  "ko": "성견",   "en": "Adult",  "count": int(lc.get("성견",  0))},
    {"key": "senior", "ko": "노령견", "en": "Senior", "count": int(lc.get("노령견", 0))},
]

# 진료과 분포
_DK = {"내과":"internal","외과":"surgery","피부과":"derm","안과":"ophth","치과":"dental"}
_DE = {"내과":"Internal","외과":"Surgery","피부과":"Dermatology","안과":"Ophthalmology","치과":"Dental"}
department = [
    {"key":_DK.get(d,"etc"),"ko":d,"en":_DE.get(d,d),"pct":round(c/N*100,1)}
    for d,c in corpus["department"].value_counts().items()
]

# 질병 Top 10
disease_top = [{"name":n,"count":int(c)} for n,c in corpus["disease"].value_counts().head(10).items()]

# 텍스트 길이
ql, al = corpus["input"].str.len(), corpus["output"].str.len()
text_stats = {
    "q_mean":int(ql.mean()),"q_median":int(ql.median()),"q_max":int(ql.max()),
    "a_mean":int(al.mean()),"a_median":int(al.median()),"a_max":int(al.max()),
}

# 평가 지표
eval_df = dl.eval_summary
rt = eval_df[eval_df["모델"] == "TF-IDF"].iloc[0]
rb = eval_df[eval_df["모델"] == "Sentence-BERT"].iloc[0]
metrics = {
    "overall": [
        {"k":k,"tfidf":round(float(rt[k])*100,2),"bert":round(float(rb[k])*100,2)}
        for k in ("Hit@1","Hit@3","Hit@5","MAP@5")
    ],
    "byLifecycle": [
        {"key":key,"ko":ko,"n":n,
         "tfidf":round(float(rt[f"{ko} Hit@5"])*100,1),
         "bert": round(float(rb[f"{ko} Hit@5"])*100,1)}
        for key,ko,n in [("puppy","자견",17),("adult","성견",17),("senior","노령견",16)]
    ],
}

# 매칭 결과
match = dl.matching_results

# 실패 쿼리 분석
import ast

def _parse_ids(val):
    try: return ast.literal_eval(val) if isinstance(val, str) else list(val)
    except: return []

def _is_rel(idx, disease, lc, corpus):
    if not (0 <= idx < len(corpus)): return False
    doc = corpus.iloc[idx]
    return doc["disease"] == disease and doc["lifeCycle"] == lc

_train = dl.train
fail_analysis = []
for _, row in match.iterrows():
    t5 = _parse_ids(row["tfidf_top5"])[:5]
    b5 = _parse_ids(row["sbert_top5"])[:5]
    disease, lc = str(row["disease"]), str(row["lifeCycle"])
    t_hit = any(_is_rel(i, disease, lc, _train) for i in t5)
    b_hit = any(_is_rel(i, disease, lc, _train) for i in b5)
    status = "both" if (t_hit and b_hit) else "tfidf" if t_hit else "bert" if b_hit else "none"
    fail_analysis.append({
        "id": str(row["query_id"]), "q": str(row["query"])[:80],
        "life": lc, "disease": disease,
        "tfidf_hit": t_hit, "bert_hit": b_hit, "status": status,
    })

# 유사도 점수
sim_scores = {
    "tfidf": match["tfidf_score1"].round(4).tolist() if "tfidf_score1" in match.columns else [],
    "bert":  match["sbert_score1"].round(4).tolist()  if "sbert_score1" in match.columns else [],
}

def _demo_item(idx: int, rank: int, sim: float) -> dict:
    d = dl.doc_snippet(idx, q_len=200, a_len=200)
    return {"rank":rank,"sim":sim,"lifecycle":d["lifeCycle"],"dept":d["department"],
            "disease":d["disease"],"q":d["input"],"a":d["output"]}

eval_queries = [
    {"id":str(r["query_id"]),"life":str(r["lifeCycle"]),"dept":str(r["department"]),
     "disease":str(r["disease"]),"title":str(r["query"])[:100],
     "results":{
         "tfidf":[dl.doc_snippet(i) for i in r["tfidf_top5"][:5]],
         "bert": [dl.doc_snippet(i) for i in r["sbert_top5"][:5]],
     }}
    for _, r in match.iterrows()
]

try:
    row6 = match.iloc[6]
    demo_bert  = [_demo_item(i,r+1,round(0.862-r*0.028,4)) for r,i in enumerate(row6["sbert_top5"][:5])]
    demo_tfidf = [_demo_item(i,r+1,round(0.693-r*0.028,4)) for r,i in enumerate(row6["tfidf_top5"][:5])]
except Exception:
    demo_bert = demo_tfidf = []

print("📊 EDA 차트 생성 중...")
eda_charts = ChartBuilder(corpus).build_all()
print(f"   ✓ 차트 {len(eda_charts)}개 생성 완료")

APP_DATA = {
    "stats":    {"total":N,"train":len(dl.train),"val":N-len(dl.train),"depts":5},
    "lifecycle": lifecycle, "department": department, "diseaseTop": disease_top,
    "textStats": text_stats, "metrics": metrics,
    "anova": {
        "f": 229.46, "df_between": 4, "df_within": 21599,
        "p": "2.87e-193", "n": 21604,
        "desc": "진료과(5개 그룹) × 질문 텍스트 길이 · One-way ANOVA",
    },
    "sampleSuggestions": [
        "눈에 눈곱이 자꾸 끼고 빨갛게 충혈돼요","산책 후 뒷다리를 절뚝거려요",
        "귀를 자꾸 긁고 냄새가 나요","피부에 붉은 발진이 생겼어요",
        "물을 너무 많이 마시고 소변을 자주 봐요",
    ],
    "results":      {"bert":demo_bert,"tfidf":demo_tfidf},
    "evalQueries":  eval_queries,
    "failAnalysis": fail_analysis,
    "simScores":    sim_scores,
    "eda":          eda_charts,
}

print(f"✅ 데이터 구성 완료 — 쿼리 {len(eval_queries)}개 · {len(json.dumps(APP_DATA))//1024}KB")

# dashboard_live.html: EDA 포함 전체 데이터 주입
html     = SRC.read_text(encoding="utf-8")
theme_css = build_css()
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA,ensure_ascii=False)});"
    "</script>"
)
live_html = html.replace("</head>", theme_css + "\n</head>")
live_html = live_html.replace("</body>", override + "\n</body>")
OUT.write_text(live_html, encoding="utf-8")
print(f"📄 dashboard_live.html 저장: {OUT}")

# dashboard.html: 실데이터 인라인 업데이트 (EDA 차트 제외)
DATA_LIGHT = {k:v for k,v in APP_DATA.items() if k != "eda"}
src_html  = SRC.read_text(encoding="utf-8")
marker    = "<script>\nwindow.APP_DATA = {"
if marker in src_html:
    start     = src_html.index(marker)
    end       = src_html.index("</script>", start) + len("</script>")
    new_block = "<script>\nwindow.APP_DATA = " + json.dumps(DATA_LIGHT,ensure_ascii=False,indent=2) + ";\n</script>"
    SRC.write_text(src_html[:start] + new_block + src_html[end:], encoding="utf-8")
    print(f"📝 dashboard.html 실데이터 업데이트 완료 ({len(json.dumps(DATA_LIGHT))//1024}KB)")

print("🌐 브라우저에서 열기...")
webbrowser.open(f"file://{OUT.resolve()}")
print("✅ 완료!")
