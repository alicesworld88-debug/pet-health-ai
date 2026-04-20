"""평가 지표 계산 모듈."""
import pandas as pd


class Evaluator:
    """Ground Truth 기반 Hit@k / MAP@k 평가기.

    정답 판정: 검색 결과 문서의 disease + lifeCycle 이 쿼리와 동일한 경우.
    """

    def __init__(self, ground_truth: pd.DataFrame, train_corpus: pd.DataFrame):
        self.gt = ground_truth.reset_index(drop=True)
        self.db = train_corpus.reset_index(drop=True)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _is_relevant(self, pred_idx: int, query_row: pd.Series) -> bool:
        if not (0 <= pred_idx < len(self.db)):
            return False
        doc = self.db.iloc[pred_idx]
        return (doc["disease"] == query_row["disease"] and
                doc["lifeCycle"] == query_row["lifeCycle"])

    def _relevance_list(self, pred_idxs: list[int], query_row: pd.Series) -> list[bool]:
        return [self._is_relevant(i, query_row) for i in pred_idxs]

    # ── 공개 API ─────────────────────────────────────────────────────────

    def hit_at_k(self, pred_list: list[list[int]], k: int) -> float:
        """상위 k 안에 정답 문서가 하나라도 포함된 비율."""
        hits = sum(
            any(self._is_relevant(idx, self.gt.iloc[i]) for idx in preds[:k])
            for i, preds in enumerate(pred_list)
        )
        return hits / len(pred_list)

    def map_at_k(self, pred_list: list[list[int]], k: int = 5) -> float:
        """Mean Average Precision @k."""
        ap_list = []
        for i, preds in enumerate(pred_list):
            row = self.gt.iloc[i]
            rel = self._relevance_list(preds[:k], row)
            if not any(rel):
                ap_list.append(0.0)
                continue
            precision_sum = sum(
                sum(rel[:j+1]) / (j + 1)
                for j, r in enumerate(rel) if r
            )
            ap_list.append(precision_sum / k)
        return sum(ap_list) / len(ap_list)

    def summary(
        self,
        tfidf_preds: list[list[int]],
        bert_preds:  list[list[int]],
        ks: tuple[int, ...] = (1, 3, 5),
    ) -> pd.DataFrame:
        """모델별 전체 지표 DataFrame."""
        rows = []
        for model, preds in [("TF-IDF", tfidf_preds), ("Sentence-BERT", bert_preds)]:
            row: dict = {"모델": model}
            for k in ks:
                row[f"Hit@{k}"] = round(self.hit_at_k(preds, k), 4)
            row["MAP@5"] = round(self.map_at_k(preds, 5), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def by_lifecycle(
        self,
        tfidf_preds: list[list[int]],
        bert_preds:  list[list[int]],
        k: int = 5,
    ) -> dict[str, dict[str, float]]:
        """생애주기별 Hit@k 딕셔너리."""
        result: dict[str, dict[str, float]] = {}
        for lc in self.gt["lifeCycle"].unique():
            mask  = self.gt["lifeCycle"] == lc
            idxs  = self.gt.index[mask].tolist()
            tf_sub = [tfidf_preds[i] for i in idxs]
            bt_sub = [bert_preds[i]  for i in idxs]
            result[lc] = {
                "n":     len(idxs),
                "tfidf": round(self.hit_at_k(tf_sub, k), 4),
                "bert":  round(self.hit_at_k(bt_sub, k), 4),
            }
        return result
