"""
대시보드 3개 영역 데이터 생성 → app/dashboard_live.html (BERT 불필요, pandas+plotly만).
  1) naver        — 네이버 보호자 질문 (건강 상담 탭)
  2) eda          — AI Hub 코퍼스 차트 (EDA 탭, Plotly JSON)
  3) fullMatching — 매칭 뷰어 전체 2,399건

이미 dashboard.html에 임베딩된 APP_DATA(stats·evalQueries 등) 위에 이 3개 키만
override 주입한다. 실행:
    .venv-dash/bin/python scripts/build_dashboard_data.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.chart_builder import ChartBuilder
from utils.theme import build_css

DP  = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
APP = ROOT / "app"

print("📂 corpus 로드 (utf-8-sig — BOM 제거)...")
corpus = pd.read_csv(DP / "corpus_preprocessed.csv", encoding="utf-8-sig")

# ── 1) naver (네이버 보호자 질문) ──────────────────────────────
print("🗂  naver...")
ndf = pd.read_csv(EXT / "naver_questions.csv", encoding="utf-8-sig")
ko = {"symptom": "증상", "emergency": "응급", "treatment": "처치"}
counts = ndf["intent"].value_counts().to_dict()
samples = {}
for it in ("symptom", "emergency", "treatment"):
    sub = ndf.loc[ndf["intent"] == it, "query"].dropna()
    sub = sub[sub.str.len().between(15, 90)]
    picked = sub.sample(min(12, len(sub)), random_state=42)
    samples[ko[it]] = [str(s)[:88] for s in picked.tolist()]
naver = {
    "total": int(len(ndf)),
    "counts": {ko[k]: int(counts.get(k, 0)) for k in ("symptom", "emergency", "treatment")},
    "samples": samples,
}

# ── 2) eda (AI Hub 코퍼스 차트) ────────────────────────────────
print("📊 eda charts...")
eda = ChartBuilder(corpus).build_all()

# ── 3) fullMatching (매칭 뷰어 전체 2,399건) ──────────────────
print("🔎 fullMatching...")
fm = pd.read_csv(DP / "full_matching_results.csv", encoding="utf-8-sig").reset_index(drop=True)
val = corpus[corpus["split"] == "validation"].reset_index(drop=True)
fm["query"] = val["input"].astype(str).str.slice(0, 110).values
cols = ["query", "lifeCycle", "department", "disease", "tfidf_hit1", "bert_hit1"]
full_matching = fm[cols].to_dict(orient="records")

extra = {"naver": naver, "eda": eda, "fullMatching": full_matching}
print("── 크기(KB) ──")
for k, v in extra.items():
    print(f"  {k}: {len(json.dumps(v, ensure_ascii=False)) // 1024} KB")

# ── dashboard_live.html 생성 (3개 키 override + theme 주입) ────
src = (APP / "dashboard.html").read_text(encoding="utf-8")
override = (
    "<script>window.APP_DATA=Object.assign(window.APP_DATA||{},"
    + json.dumps(extra, ensure_ascii=False)
    + ");</script>"
)
live = src.replace("</head>", build_css() + "\n</head>", 1)
live = live.replace("</body>", override + "\n</body>", 1)
out = APP / "dashboard_live.html"
out.write_text(live, encoding="utf-8")
print(f"✅ {out.relative_to(ROOT)}: {out.stat().st_size // 1024} KB")
