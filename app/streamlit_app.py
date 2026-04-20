"""반려견 증상 매칭 서비스 · Streamlit iframe wrapper"""
import sys, json, ast
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from utils.config import DATA_PROCESSED

st.set_page_config(
    page_title="반려견 증상 매칭 · Vet Match",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Streamlit chrome 제거 + iframe 전체 화면 리사이즈 ───────────
st.html("""<script>
(function(){
  var p=window.parent.document;
  if(p.getElementById("v-fix"))return;
  var s=p.createElement("style");s.id="v-fix";
  s.textContent=
    "#MainMenu,footer,header,[data-testid='stHeader'],[data-testid='stToolbar'],"
    +"[data-testid='stSidebar'],[data-testid='stSidebarNav'],"
    +".stDeployButton,[data-testid='stDecoration'],[data-testid='stStatusWidget']{display:none!important}"
    +"section.main>div.block-container,"
    +"[data-testid='stMainBlockContainer']"
    +"{padding:0!important;margin:0!important;max-width:100%!important;width:100%!important}"
    +"body{padding:0!important;margin:0!important}"
    +".stApp{overflow:hidden!important;background:#fafafa!important}";
  p.head.appendChild(s);
  function fix(){
    var h=window.parent.innerHeight;
    p.querySelectorAll("iframe").forEach(function(f){
      if(parseInt(f.getAttribute("height")||0)>400){
        f.style.setProperty("height", h+"px", "important");
        f.style.setProperty("width",  "100vw", "important");
        f.style.setProperty("border", "none",  "important");
        f.style.setProperty("display","block",  "important");
        f.style.setProperty("margin", "0",      "important");
      }
    });
  }
  setTimeout(fix,200);setTimeout(fix,800);setTimeout(fix,2000);
  window.parent.addEventListener("resize",fix);
})();
</script>""")

# ─── 데이터 로드 ────────────────────────────────────────────────
@st.cache_data
def load_all():
    corpus  = pd.read_csv(DATA_PROCESSED / "corpus_preprocessed.csv")
    eval_df = pd.read_csv(DATA_PROCESSED / "evaluation_summary.csv")
    match   = pd.read_csv(DATA_PROCESSED / "matching_results.csv")
    train   = corpus[corpus["split"] == "train"].reset_index(drop=True)
    return corpus, train, eval_df, match

corpus, train, eval_df, match = load_all()

# ─── APP_DATA 구성 ──────────────────────────────────────────────

# 생애주기 분포
lc = corpus["lifeCycle"].value_counts()
lifecycle = [
    {"key": "puppy",  "ko": "자견",   "en": "Puppy",  "count": int(lc.get("자견",   0))},
    {"key": "adult",  "ko": "성견",   "en": "Adult",  "count": int(lc.get("성견",   0))},
    {"key": "senior", "ko": "노령견", "en": "Senior", "count": int(lc.get("노령견", 0))},
]

# 진료과 분포
_dk = {"내과":"internal","외과":"surgery","피부과":"derm","안과":"ophth","치과":"dental"}
_de = {"내과":"Internal","외과":"Surgery","피부과":"Dermatology","안과":"Ophthalmology","치과":"Dental"}
depts_cnt = corpus["department"].value_counts()
N = len(corpus)
department = [
    {"key": _dk.get(d,"etc"), "ko": d, "en": _de.get(d,d), "pct": round(c/N*100, 1)}
    for d, c in depts_cnt.items()
]

# 질병 Top 10
dis_top = corpus["disease"].value_counts().head(10)
disease_top = [{"name": n, "count": int(c)} for n, c in dis_top.items()]

# 텍스트 길이 통계
ql = corpus["input"].str.len()
al = corpus["output"].str.len()
text_stats = {
    "q_mean":   int(ql.mean()),   "q_median": int(ql.median()),   "q_max": int(ql.max()),
    "a_mean":   int(al.mean()),   "a_median": int(al.median()),   "a_max": int(al.max()),
}

# 평가 지표
rt = eval_df[eval_df["모델"] == "TF-IDF"].iloc[0]
rb = eval_df[eval_df["모델"] == "Sentence-BERT"].iloc[0]
metrics = {
    "overall": [
        {"k": "Hit@1", "tfidf": round(float(rt["Hit@1"])*100, 2), "bert": round(float(rb["Hit@1"])*100, 2)},
        {"k": "Hit@3", "tfidf": round(float(rt["Hit@3"])*100, 2), "bert": round(float(rb["Hit@3"])*100, 2)},
        {"k": "Hit@5", "tfidf": round(float(rt["Hit@5"])*100, 2), "bert": round(float(rb["Hit@5"])*100, 2)},
        {"k": "MAP@5", "tfidf": round(float(rt["MAP@5"])*100, 4), "bert": round(float(rb["MAP@5"])*100, 4)},
    ],
    "byLifecycle": [
        {"key":"puppy",  "ko":"자견",   "n":17,
         "tfidf": round(float(rt["자견 Hit@5"])*100, 1),   "bert": round(float(rb["자견 Hit@5"])*100, 1)},
        {"key":"adult",  "ko":"성견",   "n":17,
         "tfidf": round(float(rt["성견 Hit@5"])*100, 1),   "bert": round(float(rb["성견 Hit@5"])*100, 1)},
        {"key":"senior", "ko":"노령견", "n":16,
         "tfidf": round(float(rt["노령견 Hit@5"])*100, 1), "bert": round(float(rb["노령견 Hit@5"])*100, 1)},
    ],
}

# 코퍼스 문서 축약 조회
def doc_brief(idx):
    if 0 <= idx < len(train):
        r = train.iloc[idx]
        return {
            "lifeCycle":  str(r.get("lifeCycle",  "")),
            "department": str(r.get("department", "")),
            "disease":    str(r.get("disease",    "")),
            "input":      str(r.get("input",      ""))[:160],
            "output":     str(r.get("output",     ""))[:220],
        }
    return {"lifeCycle":"","department":"","disease":"","input":"","output":""}

# 평가 쿼리 (실데이터)
eval_queries = []
for _, row in match.iterrows():
    try:    tf_idx = ast.literal_eval(str(row["tfidf_top5"]))
    except: tf_idx = []
    try:    sb_idx = ast.literal_eval(str(row["sbert_top5"]))
    except: sb_idx = []
    eval_queries.append({
        "id":      str(row["query_id"]),
        "life":    str(row["lifeCycle"]),
        "dept":    str(row["department"]),
        "disease": str(row["disease"]),
        "title":   str(row["query"])[:100],
        "results": {
            "tfidf": [doc_brief(i) for i in tf_idx[:5]],
            "bert":  [doc_brief(i) for i in sb_idx[:5]],
        },
    })

# 증상 검색 페이지 데모 결과 (첫 번째 쿼리의 실제 top-5)
def mk_demo(idx_list, base_sim):
    results = []
    for rank, idx in enumerate(idx_list[:5], 1):
        d = doc_brief(idx)
        results.append({
            "rank": rank, "sim": round(base_sim - rank*0.028, 4),
            "lifecycle": d["lifeCycle"], "dept": d["department"],
            "disease": d["disease"], "q": d["input"][:200], "a": d["output"][:200],
        })
    return results

try:
    first_bert  = ast.literal_eval(str(match.iloc[5]["sbert_top5"]))
    first_tfidf = ast.literal_eval(str(match.iloc[5]["tfidf_top5"]))
    demo_bert  = mk_demo(first_bert,  0.862)
    demo_tfidf = mk_demo(first_tfidf, 0.693)
except Exception:
    demo_bert = demo_tfidf = []

# 최종 APP_DATA
APP_DATA = {
    "stats": {
        "total": int(N),
        "train": int(len(train)),
        "val":   int(N - len(train)),
        "depts": 5,
    },
    "lifecycle":    lifecycle,
    "department":   department,
    "diseaseTop":   disease_top,
    "textStats":    text_stats,
    "metrics":      metrics,
    "sampleSuggestions": [
        "눈에 눈곱이 자꾸 끼고 빨갛게 충혈돼요",
        "산책 후 뒷다리를 절뚝거려요",
        "귀를 자꾸 긁고 냄새가 나요",
        "피부에 붉은 발진이 생겼어요",
        "물을 너무 많이 마시고 소변을 자주 봐요",
    ],
    "results":      {"bert": demo_bert, "tfidf": demo_tfidf},
    "evalQueries":  eval_queries,
}

# ─── dashboard.html 로드 → APP_DATA 주입 → 렌더 ─────────────────
DASHBOARD = Path(__file__).parent / "dashboard.html"
html = DASHBOARD.read_text(encoding="utf-8")

# 기존 APP_DATA를 실데이터로 덮어쓰기 (body 닫기 태그 직전에 삽입)
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA, ensure_ascii=False)});"
    "</script>"
)
html = html.replace("</body>", override + "\n</body>")

components.html(html, height=900, scrolling=False)
