"""
TF-IDF 사전 학습 피클 생성 — Lambda 콜드스타트 단축용.

corpus_preprocessed.csv의 input_tokens로 TfidfVectorizer를 fit하고
(vectorizer, matrix)를 data/processed/tfidf_prefit.pkl로 저장한다.
런타임(chat.build_pipeline)은 이 피클이 있으면 21,604문서 벡터화를 건너뛴다.

사용: python scripts/build_tfidf_prefit.py
이후 scripts/upload_data.py로 S3 업로드.
"""
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import DATA_PROCESSED
from utils.matcher import TFIDFMatcher

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main() -> None:
    corpus_path = DATA_PROCESSED / "corpus_preprocessed.csv"
    out_path = DATA_PROCESSED / "tfidf_prefit.pkl"

    if not corpus_path.exists():
        print(f"❌ corpus 없음: {corpus_path} (노트북으로 생성 필요)")
        sys.exit(1)

    print(f"📖 corpus 로드: {corpus_path}")
    with open(corpus_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    tokens = [r.get("input_tokens", "") for r in rows]
    print(f"   {len(tokens):,}건")

    print("🔧 TF-IDF fit 중...")
    m = TFIDFMatcher().fit(tokens)

    with open(out_path, "wb") as f:
        pickle.dump((m._vectorizer, m._matrix), f)
    size_kb = out_path.stat().st_size // 1024
    print(f"✅ 저장: {out_path} ({size_kb}KB)")


if __name__ == "__main__":
    main()
