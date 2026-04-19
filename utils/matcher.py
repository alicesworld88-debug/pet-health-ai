"""
TF-IDF / Sentence-BERT 매칭 공통 모듈
노트북(06_matching.ipynb)과 앱(streamlit_app.py)에서 공유 사용.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED

SBERT_MODEL = "jhgan/ko-sroberta-multitask"
EMBED_PATH  = DATA_PROCESSED / "embeddings" / "db_embeddings.npy"


# ── TF-IDF ────────────────────────────────────────────────────────────

def build_tfidf(corpus: list[str]) -> tuple[TfidfVectorizer, object]:
    """형태소 토큰 리스트로 TF-IDF 행렬 생성."""
    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def match_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = 5,
) -> tuple[list[int], list[float]]:
    """쿼리를 TF-IDF 벡터로 변환 후 코사인 유사도 기준 상위 top_k 반환."""
    vec    = vectorizer.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx.tolist(), scores[idx].tolist()


# ── Sentence-BERT ─────────────────────────────────────────────────────

def load_or_build_embeddings(
    corpus: list[str],
    model: SentenceTransformer,
    embed_path: Path = EMBED_PATH,
    batch_size: int = 64,
) -> np.ndarray:
    """사전 계산된 임베딩이 있으면 로드, 없으면 생성 후 저장."""
    if embed_path.exists():
        return np.load(embed_path)

    embeddings = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embed_path, embeddings)
    return embeddings


def match_sbert(
    query: str,
    model: SentenceTransformer,
    db_embeddings: np.ndarray,
    top_k: int = 5,
) -> tuple[list[int], list[float]]:
    """쿼리 임베딩 코사인 유사도 기준 상위 top_k 반환."""
    q_emb  = model.encode([query], normalize_embeddings=True)[0]
    scores = (db_embeddings @ q_emb).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx.tolist(), scores[idx].tolist()


# ── 평가 지표 ─────────────────────────────────────────────────────────

def top_k_accuracy(
    pred_list: list[list[int]],
    true_list: list[int],
    k: int,
) -> float:
    """상위 k개 안에 정답이 포함된 비율."""
    correct = sum(
        1 for pred, true in zip(pred_list, true_list)
        if true in pred[:k]
    )
    return correct / len(true_list)


def mean_reciprocal_rank(
    pred_list: list[list[int]],
    true_list: list[int],
) -> float:
    """정답 순위 역수의 평균 (MRR)."""
    rr_list = []
    for pred, true in zip(pred_list, true_list):
        if true in pred:
            rr_list.append(1.0 / (pred.index(true) + 1))
        else:
            rr_list.append(0.0)
    return sum(rr_list) / len(rr_list)
