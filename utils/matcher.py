"""
TF-IDF / Sentence-BERT 매칭 모듈.

클래스 API (run_*.py):
    TFIDFMatcher, BERTMatcher

함수 API (노트북 하위 호환):
    build_tfidf, match_tfidf, load_or_build_embeddings, match_sbert,
    top_k_accuracy, mean_reciprocal_rank
"""
import numpy as np
from pathlib import Path
from functools import cached_property
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED

SBERT_MODEL = "jhgan/ko-sroberta-multitask"
EMBED_PATH  = DATA_PROCESSED / "embeddings" / "db_embeddings.npy"


# ── 클래스 API ────────────────────────────────────────────────────────

class TFIDFMatcher:
    """TF-IDF 벡터 공간 기반 유사 문서 검색기."""

    def __init__(self):
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def fit(self, tokens: list[str]) -> "TFIDFMatcher":
        """형태소 토큰 리스트로 TF-IDF 행렬 빌드."""
        self._vectorizer = TfidfVectorizer()
        self._matrix     = self._vectorizer.fit_transform(tokens)
        return self

    def match(self, query: str, top_k: int = 5) -> tuple[list[int], list[float]]:
        """쿼리 → 코사인 유사도 기준 상위 top_k (인덱스, 점수) 반환."""
        if self._vectorizer is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        vec    = self._vectorizer.transform([query])
        scores = cosine_similarity(vec, self._matrix).flatten()
        idx    = scores.argsort()[::-1][:top_k]
        return idx.tolist(), scores[idx].tolist()


class BERTMatcher:
    """Sentence-BERT 임베딩 기반 유사 문서 검색기 (임베딩 캐싱 지원)."""

    def __init__(
        self,
        model_name: str = SBERT_MODEL,
        embed_path: Path = EMBED_PATH,
    ):
        self.model_name  = model_name
        self.embed_path  = embed_path
        self._embeddings: np.ndarray | None = None
        self.__model: SentenceTransformer | None = None

    @cached_property
    def _model(self) -> SentenceTransformer:
        print(f"  BERT 모델 로딩: {self.model_name}")
        return SentenceTransformer(self.model_name)

    def load_or_build(self, corpus: list[str], batch_size: int = 64) -> "BERTMatcher":
        """임베딩 캐시가 있으면 로드, 없으면 생성 후 저장."""
        if self.embed_path.exists():
            print(f"  임베딩 캐시 로드: {self.embed_path}")
            self._embeddings = np.load(self.embed_path)
        else:
            print("  임베딩 생성 중 (GPU/CPU)...")
            self._embeddings = self._model.encode(
                corpus, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=True,
            )
            self.embed_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embed_path, self._embeddings)
            print(f"  임베딩 저장: {self.embed_path}")
        return self

    def match(self, query: str, top_k: int = 5) -> tuple[list[int], list[float]]:
        """쿼리 → 코사인 유사도 기준 상위 top_k (인덱스, 점수) 반환."""
        if self._embeddings is None:
            raise RuntimeError("load_or_build()를 먼저 호출하세요.")
        q_emb  = self._model.encode([query], normalize_embeddings=True)[0]
        scores = (self._embeddings @ q_emb).flatten()
        idx    = scores.argsort()[::-1][:top_k]
        return idx.tolist(), scores[idx].tolist()


# ── 함수 API (노트북 하위 호환) ───────────────────────────────────────

def build_tfidf(corpus: list[str]) -> tuple[TfidfVectorizer, object]:
    m = TFIDFMatcher().fit(corpus)
    return m._vectorizer, m._matrix


def match_tfidf(query, vectorizer, tfidf_matrix, top_k=5):
    vec    = vectorizer.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx.tolist(), scores[idx].tolist()


def load_or_build_embeddings(corpus, model, embed_path=EMBED_PATH, batch_size=64):
    if embed_path.exists():
        return np.load(embed_path)
    embeddings = model.encode(corpus, batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=True)
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embed_path, embeddings)
    return embeddings


def match_sbert(query, model, db_embeddings, top_k=5):
    q_emb  = model.encode([query], normalize_embeddings=True)[0]
    scores = (db_embeddings @ q_emb).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx.tolist(), scores[idx].tolist()


def top_k_accuracy(pred_list, true_list, k):
    return sum(1 for p, t in zip(pred_list, true_list) if t in p[:k]) / len(true_list)


def mean_reciprocal_rank(pred_list, true_list):
    rr = []
    for p, t in zip(pred_list, true_list):
        rr.append(1.0 / (p.index(t) + 1) if t in p else 0.0)
    return sum(rr) / len(rr)
