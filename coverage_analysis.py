"""
지식iN 커버리지 갭 분석
==========================================
지식iN 실제 보호자 질문 114,439건을 수의사 코퍼스(21,604건)에 검색해
top-1 코사인 유사도 분포로 "우리 데이터가 못 다루는 질문 비율"을 정량화한다.

- Gemini 호출 0회 (로컬 BERT 임베딩 + 행렬곱만) → 전체 11만 건 처리 가능
- 결과: 커버리지 등급별 비율, intent별 커버리지, 갭 질문 예시
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED

NAVER     = Path("data/external/naver_questions.csv")
CORPUS    = DATA_PROCESSED / "corpus_preprocessed.csv"
EMB_PATH  = DATA_PROCESSED / "embeddings" / "full_embeddings.npy"
OUT_MD    = Path("docs/coverage_analysis.md")

THR_HIGH = 0.70   # 이상 → 잘 커버됨
THR_MID  = 0.50   # 이상 → 부분 커버 / 미만 → 갭(미커버)


def main():
    # 1. 코퍼스 임베딩 로드 + 정규화
    print("코퍼스 임베딩 로드...")
    corpus_emb = np.load(EMB_PATH).astype("float32")
    corpus_emb /= (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)
    corpus_df = pd.read_csv(CORPUS)
    print(f"  코퍼스 {corpus_emb.shape[0]:,}건 × {corpus_emb.shape[1]}차원")

    # 2. 지식iN 질문 로드 (대표성 있는 2만 건 무작위 샘플 — CPU 인코딩 시간 단축)
    df = pd.read_csv(NAVER).dropna(subset=["query", "intent"]).reset_index(drop=True)
    _total = len(df)
    SAMPLE_N = 80000
    if _total > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
        print(f"{SAMPLE_N:,}건 무작위 샘플링 (전체 {_total:,}건 중, seed=42)")
    queries = df["query"].tolist()
    print(f"지식iN 질문 {len(queries):,}건 인코딩 중...")

    # 3. 질문 임베딩 (배치)
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    q_emb = model.encode(
        queries, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    # 4. 코퍼스와 top-1 유사도 (메모리 절약 배치 행렬곱)
    print("코퍼스 대비 top-1 유사도 계산...")
    max_sim = np.empty(len(q_emb), dtype="float32")
    top_idx = np.empty(len(q_emb), dtype="int32")
    B = 2000
    for i in range(0, len(q_emb), B):
        s = q_emb[i:i + B] @ corpus_emb.T          # (B, 21604)
        max_sim[i:i + B] = s.max(axis=1)
        top_idx[i:i + B] = s.argmax(axis=1)
    df["max_sim"] = max_sim
    df["top_idx"] = top_idx

    # 5. 커버리지 등급
    def grade(s: float) -> str:
        if s >= THR_HIGH: return "covered"
        if s >= THR_MID:  return "partial"
        return "gap"
    df["coverage"] = df["max_sim"].map(grade)

    n = len(df)
    cov = (df["coverage"] == "covered").sum()
    par = (df["coverage"] == "partial").sum()
    gap = (df["coverage"] == "gap").sum()

    print("\n" + "=" * 55)
    print(f"전체 {n:,}건 커버리지")
    print("=" * 55)
    print(f"  ✅ 잘 커버 (≥{THR_HIGH}): {cov:,} ({cov/n:.1%})")
    print(f"  🟡 부분 ({THR_MID}~{THR_HIGH}): {par:,} ({par/n:.1%})")
    print(f"  ❌ 갭 (<{THR_MID}):       {gap:,} ({gap/n:.1%})")
    print(f"  유사도 평균={df['max_sim'].mean():.3f} 중앙값={df['max_sim'].median():.3f}")

    # 6. intent별 커버리지
    print("\nintent별 커버 비율 (≥0.70):")
    by_intent = df.groupby("intent")["max_sim"].agg(
        평균="mean",
        커버율=lambda s: (s >= THR_HIGH).mean(),
        갭율=lambda s: (s < THR_MID).mean(),
    )
    print(by_intent.round(3).to_string())

    # 7. 갭 질문 예시 (유사도 최하위)
    gap_examples = df.nsmallest(8, "max_sim")[["query", "intent", "max_sim"]]

    # ── Markdown ──
    md = ["# 지식iN 커버리지 갭 분석\n",
          f"\n> 지식iN 실제 보호자 질문 **{n:,}건**을 수의사 코퍼스(21,604건)에 검색해 "
          "top-1 유사도로 커버리지를 정량화. Gemini 호출 0회 (로컬 BERT 임베딩).\n",
          "\n## 전체 커버리지\n",
          f"\n| 등급 | 기준 | 건수 | 비율 |",
          f"\n|------|------|----:|----:|",
          f"\n| ✅ 잘 커버 | 유사도 ≥ {THR_HIGH} | {cov:,} | **{cov/n:.1%}** |",
          f"\n| 🟡 부분 커버 | {THR_MID} ~ {THR_HIGH} | {par:,} | {par/n:.1%} |",
          f"\n| ❌ 갭 (미커버) | < {THR_MID} | {gap:,} | **{gap/n:.1%}** |",
          f"\n\n- top-1 유사도 평균 **{df['max_sim'].mean():.3f}**, 중앙값 {df['max_sim'].median():.3f}\n",
          "\n## intent별 커버리지\n",
          "\n| intent | 평균 유사도 | 커버율(≥0.70) | 갭율(<0.50) |",
          "\n|--------|:--------:|:----------:|:---------:|"]
    for it, r in by_intent.iterrows():
        md.append(f"\n| {it} | {r['평균']:.3f} | {r['커버율']:.1%} | {r['갭율']:.1%} |")
    md.append("\n\n## 갭 질문 예시 (코퍼스와 가장 동떨어진 질문)\n")
    for _, r in gap_examples.iterrows():
        md.append(f"\n- *(유사도 {r['max_sim']:.2f}, {r['intent']})* {r['query'][:70]}…")
    md.append("\n\n## 해석\n")
    md.append(f"\n- 실제 보호자 질문의 **{cov/n:.0%}**는 우리 수의사 코퍼스로 답변 가능, "
              f"**{gap/n:.0%}**는 코퍼스에 유사 사례가 없어 답변 신뢰도가 낮음(데이터 갭).")
    md.append("\n- 갭 영역을 보강하면 챗봇 커버리지를 직접 끌어올릴 수 있음 → 후속 데이터 수집 우선순위.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")
    df[["query", "intent", "max_sim", "coverage"]].to_csv(
        "data/external/naver_coverage.csv", index=False)
    print(f"\n✓ 보고서: {OUT_MD}")
    print(f"✓ 상세 CSV: data/external/naver_coverage.csv")


if __name__ == "__main__":
    main()
