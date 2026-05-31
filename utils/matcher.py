"""
TF-IDF / Sentence-BERT 매칭 모듈.

클래스 API (run_*.py):
    TFIDFMatcher, BERTMatcher

함수 API (노트북 하위 호환):
    build_tfidf, match_tfidf, load_or_build_embeddings, match_sbert,
    top_k_accuracy, mean_reciprocal_rank
"""
import re
import numpy as np
from pathlib import Path
from functools import cached_property
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.config import DATA_PROCESSED

# 한국어 조사·어미 근사 제거 (corpus의 형태소 토큰과 어휘 일치율 향상)
_JOSA = re.compile(r'(이|가|을|를|은|는|에|의|와|과|로|으로|에서|도|만|까지|부터|이나|이라고|이라|야|아|여|이여)$')

def _normalize_query(text: str) -> str:
    """쿼리 어절에서 주요 조사·어미를 제거해 TF-IDF 어휘와 매칭률을 높임."""
    tokens = text.strip().split()
    stripped = [_JOSA.sub('', t) for t in tokens]
    return ' '.join(stripped)

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
        """쿼리 → 코사인 유사도 기준 상위 top_k (인덱스, 점수) 반환.
        corpus는 형태소 토큰으로 구축됐으므로 쿼리도 같은 방식으로 전처리.
        """
        if self._vectorizer is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        # 조사·어미 제거 후 어절 분리 (KoNLPy 없이 근사 처리)
        normalized = _normalize_query(query)
        vec    = self._vectorizer.transform([normalized])
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
