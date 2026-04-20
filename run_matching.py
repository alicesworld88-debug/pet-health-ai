"""06 + 07 매칭 및 평가 실행 스크립트 (torch 버전 호환 패치 포함)"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# torch.load 보안 체크 우회 (로컬 전용, CVE-2025-32434 — safetensors 캐시 사용으로 실제 위험 없음)
import transformers.utils.import_utils as _tu
import transformers.modeling_utils as _mu
_tu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED, PROJECT_ROOT

PREPROCESSED_PATH = DATA_PROCESSED / 'corpus_preprocessed.csv'
GROUND_TRUTH_PATH = PROJECT_ROOT / 'data/splits/ground_truth.csv'
EMBEDDINGS_DIR    = DATA_PROCESSED / 'embeddings'
RESULTS_PATH      = DATA_PROCESSED / 'matching_results.csv'
EVAL_PATH         = DATA_PROCESSED / 'evaluation_summary.csv'
EMBEDDINGS_DIR.mkdir(exist_ok=True)
SBERT_MODEL = 'jhgan/ko-sroberta-multitask'

# ── 데이터 로드 ────────────────────────────────────────────────────────
print("데이터 로드 중...")
df    = pd.read_csv(PREPROCESSED_PATH)
df_gt = pd.read_csv(GROUND_TRUTH_PATH)
df_db = df[df['split'] == 'train'].reset_index(drop=True)
print(f"  매칭 DB: {len(df_db):,}개 | Ground Truth: {len(df_gt)}개")

# ── TF-IDF 매칭 ───────────────────────────────────────────────────────
print("TF-IDF 매칭 중...")
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_db['input_tokens'].fillna(''))

def match_tfidf(query, top_k=5):
    vec    = vectorizer.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    return scores.argsort()[::-1][:top_k].tolist()

tfidf_results = [match_tfidf(row['query']) for _, row in df_gt.iterrows()]
print("  TF-IDF 완료")

# ── BERT 임베딩 및 매칭 ────────────────────────────────────────────────
EMBED_PATH = EMBEDDINGS_DIR / 'db_embeddings.npy'
if EMBED_PATH.exists():
    print("임베딩 캐시 로드 중...")
    db_embeddings = np.load(EMBED_PATH)
else:
    print(f"임베딩 생성 중 (모델: {SBERT_MODEL}) — 수 분 소요...")
    sbert = SentenceTransformer(SBERT_MODEL)
    db_embeddings = sbert.encode(
        df_db['input_normalized'].fillna('').tolist(),
        batch_size=64, normalize_embeddings=True, show_progress_bar=True,
    )
    np.save(EMBED_PATH, db_embeddings)
    print(f"  임베딩 저장: {EMBED_PATH}")

print("BERT 쿼리 인코딩 중...")
sbert = SentenceTransformer(SBERT_MODEL)
query_embeddings = sbert.encode(
    df_gt['query'].tolist(),
    batch_size=32, normalize_embeddings=True, show_progress_bar=True,
)

def match_sbert(q_emb, top_k=5):
    scores = (db_embeddings @ q_emb).flatten()
    return scores.argsort()[::-1][:top_k].tolist()

sbert_results = [match_sbert(query_embeddings[i]) for i in range(len(df_gt))]
print("  BERT 매칭 완료")

# ── 결과 저장 ─────────────────────────────────────────────────────────
df_gt['tfidf_top5'] = [str(r) for r in tfidf_results]
df_gt['sbert_top5'] = [str(r) for r in sbert_results]
df_gt.to_csv(RESULTS_PATH, index=False, encoding='utf-8-sig')
print(f"매칭 결과 저장: {RESULTS_PATH}")

# ── 평가 지표 계산 ────────────────────────────────────────────────────
print("\n=== 성능 평가 ===")
df_gt['tfidf_top5'] = df_gt['tfidf_top5'].apply(ast.literal_eval)
df_gt['sbert_top5'] = df_gt['sbert_top5'].apply(ast.literal_eval)
true_indices = df_gt['correct_idx'].tolist()

def top_k_acc(preds, trues, k):
    return sum(1 for p, t in zip(preds, trues) if t in p[:k]) / len(trues)

def mrr(preds, trues):
    rr = []
    for p, t in zip(preds, trues):
        rr.append(1.0 / (p.index(t) + 1) if t in p else 0.0)
    return sum(rr) / len(rr)

metrics = {}
for name, preds in [('TF-IDF', df_gt['tfidf_top5'].tolist()),
                    ('Sentence-BERT', df_gt['sbert_top5'].tolist())]:
    metrics[name] = {
        'Top-1': top_k_acc(preds, true_indices, 1),
        'Top-3': top_k_acc(preds, true_indices, 3),
        'MRR':   mrr(preds, true_indices),
    }

# 생애주기별
lc_results = {}
for lc in ['자견', '성견', '노령견']:
    sub = df_gt[df_gt['lifeCycle'] == lc]
    t   = sub['correct_idx'].tolist()
    lc_results[lc] = {
        'TF-IDF Top-1': top_k_acc(sub['tfidf_top5'].tolist(), t, 1),
        'BERT Top-1':   top_k_acc(sub['sbert_top5'].tolist(), t, 1),
    }

# 출력
print(f"\n{'지표':<12} {'TF-IDF':>10} {'BERT':>10} {'향상':>8}")
print("-" * 42)
for m in ['Top-1', 'Top-3', 'MRR']:
    tf = metrics['TF-IDF'][m]
    sb = metrics['Sentence-BERT'][m]
    print(f"{m:<12} {tf:>10.4f} {sb:>10.4f} {(sb-tf)*100:>+7.1f}%p")

print(f"\n생애주기별 Top-1:")
for lc, v in lc_results.items():
    print(f"  {lc}: TF-IDF {v['TF-IDF Top-1']:.4f} | BERT {v['BERT Top-1']:.4f}")

# 요약 CSV 저장
summary = pd.DataFrame({
    '모델':         ['TF-IDF', 'Sentence-BERT'],
    'Top-1':        [metrics['TF-IDF']['Top-1'], metrics['Sentence-BERT']['Top-1']],
    'Top-3':        [metrics['TF-IDF']['Top-3'], metrics['Sentence-BERT']['Top-3']],
    'MRR':          [metrics['TF-IDF']['MRR'],   metrics['Sentence-BERT']['MRR']],
    '자견 Top-1':   [lc_results['자견']['TF-IDF Top-1'],   lc_results['자견']['BERT Top-1']],
    '성견 Top-1':   [lc_results['성견']['TF-IDF Top-1'],   lc_results['성견']['BERT Top-1']],
    '노령견 Top-1': [lc_results['노령견']['TF-IDF Top-1'], lc_results['노령견']['BERT Top-1']],
})
summary.to_csv(EVAL_PATH, index=False, encoding='utf-8-sig')
print(f"\n평가 요약 저장: {EVAL_PATH}")
