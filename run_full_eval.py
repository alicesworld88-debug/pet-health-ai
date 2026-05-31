"""
전체 validation 평가 실행 스크립트
conda run: /opt/anaconda3/envs/pet-health/bin/python run_full_eval.py
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
DATA = ROOT / 'data' / 'processed'

print("=== 데이터 로드 ===")
corpus = pd.read_csv(DATA / 'corpus_preprocessed.csv')
train_df = corpus[corpus['split'] == 'train'].reset_index(drop=True)
val_df   = corpus[corpus['split'] == 'validation'].reset_index(drop=True)
print(f"Train: {len(train_df):,}건 / Validation: {len(val_df):,}건")

# ── TF-IDF ──────────────────────────────────────────────────
print("\n=== TF-IDF 인덱스 구축 ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(train_df['input_tokens'].fillna('').tolist())
val_tfidf   = tfidf.transform(val_df['input_tokens'].fillna('').tolist())
print(f"행렬 크기: {train_tfidf.shape}, 어휘: {len(tfidf.vocabulary_):,}")

# ── BERT 임베딩 ────────────────────────────────────────────
print("\n=== BERT 임베딩 로드/계산 ===")
train_bert = np.load(DATA / 'embeddings' / 'db_embeddings.npy')
val_emb_path = DATA / 'embeddings' / 'val_embeddings.npy'

if val_emb_path.exists():
    val_bert = np.load(val_emb_path)
    print(f"Validation 임베딩 로드: {val_bert.shape}")
else:
    from sentence_transformers import SentenceTransformer
    print("Validation 임베딩 계산 중... (3-5분 소요)")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    val_bert = model.encode(
        val_df['input_normalized'].fillna('').tolist(),
        batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    np.save(val_emb_path, val_bert)
    print(f"저장 완료: {val_emb_path}")

# L2 정규화
def norm(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)

train_bert_n = norm(train_bert)
val_bert_n   = norm(val_bert)

# ── 소프트 매치 / 메트릭 ────────────────────────────────────
TOP_K = 5
BATCH = 200

def eval_model(sim_fn, label):
    hits_list, hit1_bin = [], []
    for start in range(0, len(val_df), BATCH):
        end = min(start + BATCH, len(val_df))
        sims = sim_fn(start, end)                          # (batch, 19205)
        top_idx = np.argsort(-sims, axis=1)[:, :TOP_K]

        for i, row in enumerate(range(start, end)):
            q_dept = val_df.iloc[row]['department']
            q_lc   = val_df.iloc[row]['lifeCycle']
            hits = [
                train_df.iloc[j]['department'] == q_dept and
                train_df.iloc[j]['lifeCycle']  == q_lc
                for j in top_idx[i]
            ]
            hits_list.append(hits)
            hit1_bin.append(1 if hits[0] else 0)

    n = len(hits_list)
    h1 = sum(1 for h in hits_list if h[0]) / n
    h3 = sum(1 for h in hits_list if any(h[:3])) / n
    h5 = sum(1 for h in hits_list if any(h[:5])) / n
    ap = sum(
        sum(s/k for k, s in enumerate(
            [sum(h[:i+1]) for i in range(TOP_K)], 1) if h[k-1])
        / max(sum(h), 1)
        for h in hits_list
    ) / n
    print(f"\n--- {label} (n={n:,}) ---")
    print(f"Hit@1: {h1:.4f} ({h1:.1%})")
    print(f"Hit@3: {h3:.4f} ({h3:.1%})")
    print(f"Hit@5: {h5:.4f} ({h5:.1%})")
    print(f"MAP@5: {ap:.4f} ({ap:.1%})")
    return np.array(hit1_bin)

print("\n=== 평가 시작 ===")

tfidf_hit1 = eval_model(
    lambda s, e: cosine_similarity(val_tfidf[s:e], train_tfidf).toarray()
                 if hasattr(cosine_similarity(val_tfidf[s:e], train_tfidf), 'toarray')
                 else cosine_similarity(val_tfidf[s:e], train_tfidf),
    "TF-IDF"
)

bert_hit1 = eval_model(
    lambda s, e: val_bert_n[s:e] @ train_bert_n.T,
    "BERT"
)

# ── McNemar Test ────────────────────────────────────────────
print("\n=== McNemar Test ===")
b = int(np.sum((tfidf_hit1 == 1) & (bert_hit1 == 0)))
c = int(np.sum((tfidf_hit1 == 0) & (bert_hit1 == 1)))
both = int(np.sum((tfidf_hit1 == 1) & (bert_hit1 == 1)))
miss = int(np.sum((tfidf_hit1 == 0) & (bert_hit1 == 0)))

print(f"2×2 분할표:")
print(f"  둘 다 맞춤:        {both}")
print(f"  TF-IDF만 맞춤 (b): {b}")
print(f"  BERT만 맞춤 (c):   {c}")
print(f"  둘 다 틀림:        {miss}")

if b + c > 0:
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = chi2_dist.sf(chi2_stat, df=1)
    print(f"\nχ² = {chi2_stat:.4f}, p = {p:.4f}")
    print("✅ 유의미 (p<0.05)" if p < 0.05 else "❌ 유의미하지 않음 (p≥0.05)")

# ── Bootstrap ────────────────────────────────────────────────
print("\n=== Paired Bootstrap (2,000회) ===")
np.random.seed(42)
n = len(tfidf_hit1)
diffs = [
    bert_hit1[idx := np.random.randint(0, n, n)].mean() - tfidf_hit1[idx].mean()
    for _ in range(2000)
]
ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
obs = bert_hit1.mean() - tfidf_hit1.mean()
print(f"관측 차이: {obs:+.4f} ({obs:+.1%})")
print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print("✅ BERT 우위 유의미" if ci_lo > 0 else ("TF-IDF 우위 유의미" if ci_hi < 0 else "유의미한 차이 없음"))

# ── 생애주기별 ──────────────────────────────────────────────
print("\n=== 생애주기별 Hit@1 ===")
val_df = val_df.copy()
val_df['tfidf_hit1'] = tfidf_hit1
val_df['bert_hit1']  = bert_hit1
for lc in ['자견', '성견', '노령견']:
    sub = val_df[val_df['lifeCycle'] == lc]
    t, b_ = sub['tfidf_hit1'].mean(), sub['bert_hit1'].mean()
    print(f"  {lc} (n={len(sub)}): TF-IDF {t:.1%} vs BERT {b_:.1%}  {'BERT↑' if b_>t else 'TF-IDF↑' if t>b_ else '동률'} ({b_-t:+.1%})")

# ── 저장 ────────────────────────────────────────────────────
val_df[['lifeCycle','department','disease','tfidf_hit1','bert_hit1']].to_csv(
    DATA / 'full_matching_results.csv', index=False
)
print("\n✅ full_matching_results.csv 저장 완료")
