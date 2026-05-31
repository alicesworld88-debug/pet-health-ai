"""
Q3: TF-IDF vs BERT 추론 비용 측정
- 레이턴시: 100회 반복, 평균/중앙값/p95
- 메모리: sparse matrix vs 임베딩 .npy 실제 크기
"""
import sys, time, tracemalloc, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'processed'

QUERIES = [
    "강아지가 밥을 안 먹어요",
    "슬개골 탈구 수술 후 회복이 얼마나 걸려요",
    "강아지 눈에서 눈물이 많이 나와요",
    "피부가 빨갛고 가려워해요",
    "강아지가 갑자기 기침을 많이 해요",
]
N_REPEAT = 100

print("데이터 로드...")
corpus = pd.read_csv(DATA / 'corpus_preprocessed.csv')
train = corpus[corpus['split'] == 'train'].reset_index(drop=True)

# ── TF-IDF 구축 ──────────────────────────────────────────────
print("TF-IDF 학습 중...")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(train['input_tokens'].fillna('').tolist())

# ── BERT 임베딩 로드 ─────────────────────────────────────────
print("BERT 임베딩 로드...")
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
train_emb = np.load(DATA / 'embeddings' / 'db_embeddings.npy')
train_emb_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)

print("\n" + "="*55)
print("  레이턴시 측정 (쿼리당 100회 반복)")
print("="*55)

# ── TF-IDF 레이턴시 ──────────────────────────────────────────
tfidf_times = []
for query in QUERIES:
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        vec = tfidf.transform([query])
        scores = cosine_similarity(vec, tfidf_matrix).flatten()
        _ = scores.argsort()[::-1][:5]
        times.append((time.perf_counter() - t0) * 1000)
    tfidf_times.extend(times)

tfidf_arr = np.array(tfidf_times)
print(f"\nTF-IDF (n={len(tfidf_arr)}회)")
print(f"  평균:   {tfidf_arr.mean():.2f} ms")
print(f"  중앙값: {np.median(tfidf_arr):.2f} ms")
print(f"  p95:    {np.percentile(tfidf_arr, 95):.2f} ms")
print(f"  최소:   {tfidf_arr.min():.2f} ms")
print(f"  최대:   {tfidf_arr.max():.2f} ms")

# ── BERT 레이턴시 (캐시 있음 — 임베딩 파일 로드 후) ─────────────
bert_times = []
for query in QUERIES:
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        q_emb = bert_model.encode([query], normalize_embeddings=True)[0]
        scores = (train_emb_norm @ q_emb).flatten()
        _ = scores.argsort()[::-1][:5]
        times.append((time.perf_counter() - t0) * 1000)
    bert_times.extend(times)

bert_arr = np.array(bert_times)
print(f"\nBERT — 쿼리 인코딩 포함 (캐시 히트, n={len(bert_arr)}회)")
print(f"  평균:   {bert_arr.mean():.2f} ms")
print(f"  중앙값: {np.median(bert_arr):.2f} ms")
print(f"  p95:    {np.percentile(bert_arr, 95):.2f} ms")
print(f"  최소:   {bert_arr.min():.2f} ms")
print(f"  최대:   {bert_arr.max():.2f} ms")

speed_ratio = bert_arr.mean() / tfidf_arr.mean()
print(f"\n  → BERT가 TF-IDF보다 {speed_ratio:.1f}배 느림 (평균 기준)")

# ── 메모리 측정 ──────────────────────────────────────────────
print("\n" + "="*55)
print("  메모리 사용량")
print("="*55)

# TF-IDF sparse matrix
import scipy.sparse as sp
tfidf_bytes = (tfidf_matrix.data.nbytes +
               tfidf_matrix.indices.nbytes +
               tfidf_matrix.indptr.nbytes)
print(f"\nTF-IDF sparse matrix")
print(f"  크기: {tfidf_matrix.shape[0]:,} × {tfidf_matrix.shape[1]:,}")
print(f"  비영 원소: {tfidf_matrix.nnz:,}")
print(f"  메모리: {tfidf_bytes / 1024**2:.1f} MB")

# BERT embeddings
bert_bytes = train_emb.nbytes
print(f"\nBERT embeddings (train {train_emb.shape[0]:,}건)")
print(f"  크기: {train_emb.shape[0]:,} × {train_emb.shape[1]}")
print(f"  dtype: {train_emb.dtype}")
print(f"  메모리: {bert_bytes / 1024**2:.1f} MB")

full_emb_path = DATA / 'embeddings' / 'full_embeddings.npy'
if full_emb_path.exists():
    full_emb = np.load(full_emb_path)
    print(f"\nBERT embeddings (전체 {full_emb.shape[0]:,}건)")
    print(f"  메모리: {full_emb.nbytes / 1024**2:.1f} MB")

# ── 초기화 시간 ──────────────────────────────────────────────
print("\n" + "="*55)
print("  초기화 시간 (모델/인덱스 로드)")
print("="*55)

t0 = time.perf_counter()
_v = TfidfVectorizer()
_m = _v.fit_transform(train['input_tokens'].fillna('').tolist())
tfidf_init = (time.perf_counter() - t0) * 1000
print(f"\nTF-IDF fit+transform: {tfidf_init:.0f} ms")

t0 = time.perf_counter()
_e = np.load(DATA / 'embeddings' / 'db_embeddings.npy')
bert_load = (time.perf_counter() - t0) * 1000
print(f"BERT 임베딩 로드(.npy): {bert_load:.0f} ms")

# BERT model load (cold start)
del bert_model; gc.collect()
t0 = time.perf_counter()
bert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
bert_model_load = (time.perf_counter() - t0) * 1000
print(f"BERT 모델 로드(cold start): {bert_model_load:.0f} ms")

# ── 요약표 ───────────────────────────────────────────────────
print("\n" + "="*55)
print("  최종 비교표")
print("="*55)
print(f"{'항목':<25} {'TF-IDF':>12} {'BERT':>12}")
print("-" * 50)
print(f"{'쿼리 레이턴시 평균':<25} {tfidf_arr.mean():>10.2f}ms {bert_arr.mean():>10.2f}ms")
print(f"{'쿼리 레이턴시 p95':<25} {np.percentile(tfidf_arr,95):>10.2f}ms {np.percentile(bert_arr,95):>10.2f}ms")
print(f"{'인덱스 메모리':<25} {tfidf_bytes/1024**2:>10.1f}MB {bert_bytes/1024**2:>10.1f}MB")
print(f"{'초기화 시간':<25} {tfidf_init:>10.0f}ms {bert_model_load:>10.0f}ms")
print("-" * 50)
print(f"\n결론: BERT는 TF-IDF보다 레이턴시 {speed_ratio:.1f}배 느리지만,")
print(f"      임베딩이 사전 계산돼 있어 실시간 추론 부담이 없음 (Batch Processing)")
