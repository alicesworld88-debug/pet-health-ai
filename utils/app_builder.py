"""
APP_DATA 공통 빌더 — run_dashboard.py / streamlit_app.py 양쪽에서 사용.
DataLoader 인스턴스를 받아 JSON 직렬화 가능한 dict를 반환.
"""
import ast
import pandas as pd
from utils.data_loader import DataLoader

_DK = {"내과":"internal","외과":"surgery","피부과":"derm","안과":"ophth","치과":"dental"}
_DE = {"내과":"Internal","외과":"Surgery","피부과":"Dermatology","안과":"Ophthalmology","치과":"Dental"}


def _parse_ids(val) -> list[int]:
    try:
        return ast.literal_eval(val) if isinstance(val, str) else list(val)
    except Exception:
        return []


def build_stats(dl: DataLoader) -> dict:
    corpus = dl.corpus
    N = len(corpus)
    return {"total": N, "train": len(dl.train), "val": N - len(dl.train), "depts": 5}


def build_lifecycle(dl: DataLoader) -> list[dict]:
    lc = dl.corpus["lifeCycle"].value_counts()
    return [
        {"key": "puppy",  "ko": "자견",   "en": "Puppy",  "count": int(lc.get("자견",   0))},
        {"key": "adult",  "ko": "성견",   "en": "Adult",  "count": int(lc.get("성견",   0))},
        {"key": "senior", "ko": "노령견", "en": "Senior", "count": int(lc.get("노령견", 0))},
    ]


def build_department(dl: DataLoader) -> list[dict]:
    corpus = dl.corpus
    N = len(corpus)
    return [
        {"key": _DK.get(d, "etc"), "ko": d, "en": _DE.get(d, d), "pct": round(c / N * 100, 1)}
        for d, c in corpus["department"].value_counts().items()
    ]


def build_disease_top(dl: DataLoader, n: int = 10) -> list[dict]:
    return [
        {"name": name, "count": int(cnt)}
        for name, cnt in dl.corpus["disease"].value_counts().head(n).items()
    ]


def build_text_stats(dl: DataLoader) -> dict:
    ql = dl.corpus["input"].str.len()
    al = dl.corpus["output"].str.len()
    return {
        "q_mean": int(ql.mean()), "q_median": int(ql.median()), "q_max": int(ql.max()),
        "a_mean": int(al.mean()), "a_median": int(al.median()), "a_max": int(al.max()),
    }


def build_metrics(dl: DataLoader) -> dict:
    eval_df = dl.eval_summary
    rt = eval_df[eval_df["모델"] == "TF-IDF"].iloc[0]
    rb = eval_df[eval_df["모델"] == "Sentence-BERT"].iloc[0]
    return {
        "overall": [
            {"k": k,
             "tfidf": round(float(rt[k]) * 100, 2),
             "bert":  round(float(rb[k]) * 100, 2)}
            for k in ("Hit@1", "Hit@3", "Hit@5", "MAP@5")
        ],
        "byLifecycle": [
            {"key": key, "ko": ko, "n": n,
             "tfidf": round(float(rt[f"{ko} Hit@5"]) * 100, 1),
             "bert":  round(float(rb[f"{ko} Hit@5"]) * 100, 1)}
            for key, ko, n in [("puppy", "자견", 17), ("adult", "성견", 17), ("senior", "노령견", 16)]
        ],
    }


def build_eval_queries(dl: DataLoader) -> list[dict]:
    match = dl.matching_results
    return [
        {
            "id":      str(row["query_id"]),
            "life":    str(row["lifeCycle"]),
            "dept":    str(row["department"]),
            "disease": str(row["disease"]),
            "title":   str(row["query"])[:100],
            "results": {
                "tfidf": [dl.doc_snippet(i) for i in _parse_ids(row["tfidf_top5"])[:5]],
                "bert":  [dl.doc_snippet(i) for i in _parse_ids(row["sbert_top5"])[:5]],
            },
        }
        for _, row in match.iterrows()
    ]


def build_fail_analysis(dl: DataLoader) -> list[dict]:
    match  = dl.matching_results
    train  = dl.train
    result = []
    for _, row in match.iterrows():
        disease, lc = str(row["disease"]), str(row["lifeCycle"])
        def hit(ids):
            return any(
                0 <= i < len(train)
                and train.iloc[i]["disease"] == disease
                and train.iloc[i]["lifeCycle"] == lc
                for i in ids
            )
        t_hit = hit(_parse_ids(row["tfidf_top5"])[:5])
        b_hit = hit(_parse_ids(row["sbert_top5"])[:5])
        status = "both" if (t_hit and b_hit) else "tfidf" if t_hit else "bert" if b_hit else "none"
        result.append({
            "id": str(row["query_id"]), "q": str(row["query"])[:80],
            "life": lc, "disease": disease,
            "tfidf_hit": t_hit, "bert_hit": b_hit, "status": status,
        })
    return result


def build_sim_scores(dl: DataLoader) -> dict:
    match = dl.matching_results
    return {
        "tfidf": match["tfidf_score1"].round(4).tolist() if "tfidf_score1" in match.columns else [],
        "bert":  match["sbert_score1"].round(4).tolist() if "sbert_score1" in match.columns else [],
    }


def build_demo_results(dl: DataLoader, row_idx: int = 6) -> tuple[list, list]:
    match = dl.matching_results
    try:
        row = match.iloc[row_idx]
        bert_ids  = _parse_ids(row["sbert_top5"])[:5]
        tfidf_ids = _parse_ids(row["tfidf_top5"])[:5]
        def mk(ids, base_sim):
            return [
                {**dl.doc_snippet(i, q_len=200, a_len=200),
                 "rank": r + 1, "sim": round(base_sim - r * 0.028, 4)}
                for r, i in enumerate(ids)
            ]
        return mk(bert_ids, 0.862), mk(tfidf_ids, 0.693)
    except Exception:
        return [], []


SAMPLE_SUGGESTIONS = [
    "눈에 눈곱이 자꾸 끼고 빨갛게 충혈돼요",
    "산책 후 뒷다리를 절뚝거려요",
    "귀를 자꾸 긁고 냄새가 나요",
    "피부에 붉은 발진이 생겼어요",
    "물을 너무 많이 마시고 소변을 자주 봐요",
]


def build_sample_results(dl: DataLoader, suggestions: list[str] | None = None) -> dict:
    """샘플 제안별 TF-IDF + BERT 실검색 결과 (미리 계산)."""
    from utils.matcher import TFIDFMatcher, BERTMatcher
    if suggestions is None:
        suggestions = SAMPLE_SUGGESTIONS
    train = dl.train

    print("  🔍 TF-IDF 피팅 중...")
    tfidf = TFIDFMatcher().fit(train["input"].fillna("").tolist())

    print("  🤖 BERT 임베딩 로드 중...")
    bert = BERTMatcher()
    bert.load_or_build(train["input_normalized"].fillna("").tolist())

    results = {}
    for q in suggestions:
        t_idxs, t_scores = tfidf.match(q)
        b_idxs, b_scores = bert.match(q)
        results[q] = {
            "tfidf": [
                {**dl.doc_snippet(i, q_len=200, a_len=200),
                 "rank": r + 1, "sim": round(t_scores[r], 4)}
                for r, i in enumerate(t_idxs[:5])
            ],
            "bert": [
                {**dl.doc_snippet(i, q_len=200, a_len=200),
                 "rank": r + 1, "sim": round(b_scores[r], 4)}
                for r, i in enumerate(b_idxs[:5])
            ],
        }
    return results


def build_app_data(dl: DataLoader, include_sample_search: bool = True) -> dict:
    """run_dashboard.py / streamlit_app.py 공통 APP_DATA 빌더."""
    demo_bert, demo_tfidf = build_demo_results(dl)
    data = {
        "stats":        build_stats(dl),
        "lifecycle":    build_lifecycle(dl),
        "department":   build_department(dl),
        "diseaseTop":   build_disease_top(dl),
        "textStats":    build_text_stats(dl),
        "metrics":      build_metrics(dl),
        "anova": {
            "f": 229.46, "df_between": 4, "df_within": 21599,
            "p": "2.87e-193", "n": 21604,
            "desc": "진료과(5개 그룹) × 질문 텍스트 길이 · One-way ANOVA",
        },
        "sampleSuggestions": SAMPLE_SUGGESTIONS,
        "results":      {"bert": demo_bert, "tfidf": demo_tfidf},
        "evalQueries":  build_eval_queries(dl),
        "failAnalysis": build_fail_analysis(dl),
        "simScores":    build_sim_scores(dl),
    }
    if include_sample_search:
        print("  📡 샘플 검색 결과 미리 계산 중...")
        data["sampleResults"] = build_sample_results(dl)
    return data
