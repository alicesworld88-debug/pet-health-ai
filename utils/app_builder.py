"""
APP_DATA 빌더 — run_dashboard.py에서 사용.
DataLoader 인스턴스를 받아 JSON 직렬화 가능한 dict를 반환.
"""
import ast
import time
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


def build_eval_queries(dl: DataLoader, tfidf=None, bert=None) -> list[dict]:
    match = dl.matching_results
    queries = []
    for _, row in match.iterrows():
        q_text = str(row["query"])
        if tfidf is not None and bert is not None:
            t_idxs, t_scores = tfidf.match(q_text, top_k=5)
            b_idxs, b_scores = bert.match(q_text, top_k=5)
            tfidf_docs = [{**dl.doc_snippet(i), "sim": round(float(t_scores[r]), 4)}
                          for r, i in enumerate(t_idxs[:5])]
            bert_docs  = [{**dl.doc_snippet(i), "sim": round(float(b_scores[r]), 4)}
                          for r, i in enumerate(b_idxs[:5])]
        else:
            s_t = float(row.get("tfidf_score1", 0))
            s_b = float(row.get("sbert_score1", 0))
            tfidf_docs = [{**dl.doc_snippet(i), "sim": round(s_t, 4) if r == 0 else None}
                          for r, i in enumerate(_parse_ids(row["tfidf_top5"])[:5])]
            bert_docs  = [{**dl.doc_snippet(i), "sim": round(s_b, 4) if r == 0 else None}
                          for r, i in enumerate(_parse_ids(row["sbert_top5"])[:5])]
        queries.append({
            "id":      str(row["query_id"]),
            "life":    str(row["lifeCycle"]),
            "dept":    str(row["department"]),
            "disease": str(row["disease"]),
            "title":   q_text[:100],
            "results": {"tfidf": tfidf_docs, "bert": bert_docs},
        })
    return queries


def build_fail_analysis(dl: DataLoader) -> list[dict]:
    # 소프트 매치: top-k 결과 중 쿼리와 동일한 disease+lifeCycle 조합 문서가
    # 하나라도 있으면 hit으로 판정. 의미적 유사 문서 검색이 목적이므로
    # 정확한 문서 재현(exact match)보다 완화된 기준 적용.
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


def build_demo_results(dl: DataLoader, tfidf=None, bert=None, row_idx: int = 6) -> tuple[list, list]:
    match = dl.matching_results
    try:
        row = match.iloc[row_idx]
        q_text = str(row["query"])
        if tfidf is not None and bert is not None:
            t_idxs, t_scores = tfidf.match(q_text, top_k=5)
            b_idxs, b_scores = bert.match(q_text, top_k=5)
            bert_docs  = [{**dl.doc_snippet(i, q_len=200, a_len=200),
                           "rank": r + 1, "sim": round(float(b_scores[r]), 4)}
                          for r, i in enumerate(b_idxs[:5])]
            tfidf_docs = [{**dl.doc_snippet(i, q_len=200, a_len=200),
                           "rank": r + 1, "sim": round(float(t_scores[r]), 4)}
                          for r, i in enumerate(t_idxs[:5])]
        else:
            bert_ids  = _parse_ids(row["sbert_top5"])[:5]
            tfidf_ids = _parse_ids(row["tfidf_top5"])[:5]
            s_b = float(row.get("sbert_score1", 0))
            s_t = float(row.get("tfidf_score1", 0))
            bert_docs  = [{**dl.doc_snippet(i, q_len=200, a_len=200),
                           "rank": r + 1, "sim": round(s_b, 4) if r == 0 else None}
                          for r, i in enumerate(bert_ids)]
            tfidf_docs = [{**dl.doc_snippet(i, q_len=200, a_len=200),
                           "rank": r + 1, "sim": round(s_t, 4) if r == 0 else None}
                          for r, i in enumerate(tfidf_ids)]
        return bert_docs, tfidf_docs
    except Exception:
        return [], []


SAMPLE_SUGGESTIONS = [
    # 내과
    "밥을 안먹어요",
    "구토를 자꾸 해요",
    "설사가 며칠째 계속돼요",
    "기침을 자주 하고 콧물이 나요",
    "물을 너무 많이 마시고 소변을 자주 봐요",
    "숨을 헐떡거리고 호흡이 빨라요",
    # 외과
    "산책 후 뒷다리를 절뚝거려요",
    "배가 빵빵하게 부풀어올랐어요",
    "목이나 몸에 혹이 만져져요",
    "앞다리를 들고 잘 걷지 못해요",
    # 피부과
    "피부에 붉은 발진이 생겼어요",
    "온몸을 자꾸 긁고 핥아요",
    "털이 많이 빠지고 피부가 보여요",
    "피부에 딱지가 생기고 각질이 일어나요",
    # 안과
    "눈에 눈곱이 자꾸 끼고 빨갛게 충혈돼요",
    "눈을 자꾸 비비고 눈물이 많이 나요",
    # 치과
    "입냄새가 너무 심하고 잇몸이 빨개요",
    "이빨이 흔들리고 밥을 씹기 힘들어해요",
    # 귀
    "귀를 자꾸 긁고 냄새가 나요",
    "귀를 계속 흔들고 머리를 기울여요",
]


def _load_matchers(dl: DataLoader):
    """TF-IDF + BERT 매처를 한 번만 로드해서 반환."""
    from utils.matcher import TFIDFMatcher, BERTMatcher
    train = dl.train
    print("  🔍 TF-IDF 피팅 중...")
    tfidf = TFIDFMatcher().fit(train["input"].fillna("").tolist())
    print("  🤖 BERT 임베딩 로드 중...")
    bert = BERTMatcher()
    bert.load_or_build(train["input_normalized"].fillna("").tolist())
    return tfidf, bert


def build_sample_results(dl: DataLoader, suggestions: list[str] | None = None,
                         matchers=None) -> dict:
    """샘플 제안별 TF-IDF + BERT 실검색 결과 (미리 계산)."""
    if suggestions is None:
        suggestions = SAMPLE_SUGGESTIONS
    tfidf, bert = matchers if matchers else _load_matchers(dl)
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
    """APP_DATA 전체 빌드 — run_dashboard.py 에서 호출."""
    print("  📡 매처 로드 중 (TF-IDF + BERT)...")
    tfidf, bert = _load_matchers(dl)

    print("  ⏱️  추론 시간 측정 중...")
    _sample_q = SAMPLE_SUGGESTIONS[0]
    tfidf.match(_sample_q, top_k=5)   # TF-IDF 워밍업
    bert.match(_sample_q, top_k=5)    # BERT 워밍업
    _t0 = time.perf_counter(); tfidf.match(_sample_q, top_k=5); tfidf_ms = round((time.perf_counter() - _t0) * 1000, 1)
    _t0 = time.perf_counter(); bert.match(_sample_q, top_k=5);  bert_ms  = round((time.perf_counter() - _t0) * 1000, 1)

    demo_bert, demo_tfidf = build_demo_results(dl, tfidf=tfidf, bert=bert)
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
        "inferTime": {"tfidf": tfidf_ms, "bert": bert_ms},
        "sampleSuggestions": SAMPLE_SUGGESTIONS,
        "results":      {"bert": demo_bert, "tfidf": demo_tfidf},
        "evalQueries":  build_eval_queries(dl, tfidf=tfidf, bert=bert),
        "failAnalysis": build_fail_analysis(dl),
        "simScores":    build_sim_scores(dl),
    }
    if include_sample_search:
        print("  📡 샘플 검색 결과 미리 계산 중...")
        data["sampleResults"] = build_sample_results(dl, matchers=(tfidf, bert))
    return data
