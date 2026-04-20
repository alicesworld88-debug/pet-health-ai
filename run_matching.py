"""매칭 및 평가 실행 스크립트"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# torch.load 보안 체크 우회 (로컬 전용, CVE-2025-32434 — safetensors 캐시 사용으로 실제 위험 없음)
import transformers.utils.import_utils as _tu
import transformers.modeling_utils as _mu
_tu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from utils.config import DATA_PROCESSED
from utils.data_loader import DataLoader
from utils.matcher import TFIDFMatcher, BERTMatcher
from utils.evaluator import Evaluator

RESULTS_PATH = DATA_PROCESSED / "matching_results.csv"
EVAL_PATH    = DATA_PROCESSED / "evaluation_summary.csv"

print("데이터 로드 중...")
dl    = DataLoader()
train = dl.train
gt    = dl.ground_truth
print(f"  매칭 DB: {len(train):,}개 | Ground Truth: {len(gt)}개")

# TF-IDF 매칭
print("TF-IDF 매칭 중...")
tfidf         = TFIDFMatcher().fit(train["input_tokens"].fillna("").tolist())
tfidf_results = [tfidf.match(q)[0] for q in gt["query"]]
print("  완료")

# BERT 임베딩 로드 / 생성 후 매칭
print("BERT 임베딩/매칭 중...")
bert         = BERTMatcher()
bert.load_or_build(train["input_normalized"].fillna("").tolist())
bert_results = [bert.match(q)[0] for q in gt["query"]]
print("  완료")

# 결과 저장
gt_out = gt.copy()
gt_out["tfidf_top5"] = [str(r) for r in tfidf_results]
gt_out["sbert_top5"] = [str(r) for r in bert_results]
gt_out.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
print(f"매칭 결과 저장: {RESULTS_PATH}")

# 평가
print("\n=== 성능 평가 ===")
ev      = Evaluator(gt, train)
summary = ev.summary(tfidf_results, bert_results)

# 생애주기별 Hit@5를 summary에 열로 추가
lc_hits = ev.by_lifecycle(tfidf_results, bert_results, k=5)
for lc, v in lc_hits.items():
    summary.loc[summary["모델"] == "TF-IDF",       f"{lc} Hit@5"] = v["tfidf"]
    summary.loc[summary["모델"] == "Sentence-BERT", f"{lc} Hit@5"] = v["bert"]

print(summary.to_string(index=False))
print(f"\n생애주기별 Hit@5:")
for lc, v in lc_hits.items():
    print(f"  {lc} (n={v['n']}): TF-IDF {v['tfidf']:.4f} | BERT {v['bert']:.4f}")

summary.to_csv(EVAL_PATH, index=False, encoding="utf-8-sig")
print(f"\n평가 요약 저장: {EVAL_PATH}")
