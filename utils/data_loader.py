"""공통 데이터 로딩 모듈 — 지연 로딩 + 인스턴스 캐싱."""
import ast
import pandas as pd
from functools import cached_property
from pathlib import Path

from utils.config import DATA_PROCESSED, DATA_SPLITS


class DataLoader:
    """전처리·평가 CSV를 한 번만 읽고 캐싱하는 데이터 접근 객체."""

    def __init__(self, processed_dir: Path = DATA_PROCESSED, splits_dir: Path = DATA_SPLITS):
        self._processed = processed_dir
        self._splits    = splits_dir

    @cached_property
    def corpus(self) -> pd.DataFrame:
        return pd.read_csv(self._processed / "corpus_preprocessed.csv")

    @cached_property
    def train(self) -> pd.DataFrame:
        return self.corpus[self.corpus["split"] == "train"].reset_index(drop=True)

    @cached_property
    def val(self) -> pd.DataFrame:
        return self.corpus[self.corpus["split"] == "validation"].reset_index(drop=True)

    @cached_property
    def ground_truth(self) -> pd.DataFrame:
        return pd.read_csv(self._splits / "ground_truth.csv")

    @cached_property
    def matching_results(self) -> pd.DataFrame:
        df = pd.read_csv(self._processed / "matching_results.csv")
        for col in ("tfidf_top5", "sbert_top5"):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)))
        return df

    @cached_property
    def eval_summary(self) -> pd.DataFrame:
        return pd.read_csv(self._processed / "evaluation_summary.csv")

    def doc_snippet(self, idx: int, q_len: int = 160, a_len: int = 220) -> dict:
        """train 코퍼스에서 idx 행의 주요 필드만 반환."""
        if not (0 <= idx < len(self.train)):
            return {"lifecycle": "", "dept": "", "disease": "", "q": "", "a": ""}
        r = self.train.iloc[idx]
        return {
            "lifecycle": str(r.get("lifeCycle", "")),
            "dept":      str(r.get("department", "")),
            "disease":   str(r.get("disease", "")),
            "q":         str(r.get("input", ""))[:q_len],
            "a":         str(r.get("output", ""))[:a_len],
        }
