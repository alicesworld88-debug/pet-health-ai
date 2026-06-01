"""
BERT 임베딩(full_embeddings.npy) 로컬 생성 — SageMaker 불필요.

corpus_preprocessed.csv의 input_normalized를 ko-sroberta로 인코딩해
data/processed/embeddings/full_embeddings.npy 로 저장한다.
이후 scripts/upload_data.py 로 S3 업로드하면 BERT Lambda가 콜드스타트에 로드한다.

사용: python scripts/build_embeddings.py
참고: 21,604건 CPU 인코딩은 수 분 소요. GPU가 있으면 자동 사용.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import DATA_PROCESSED

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main() -> None:
    corpus_path = DATA_PROCESSED / "corpus_preprocessed.csv"
    out_path = DATA_PROCESSED / "embeddings" / "full_embeddings.npy"

    if not corpus_path.exists():
        print(f"❌ corpus 없음: {corpus_path} (노트북으로 생성 필요)")
        sys.exit(1)

    print(f"📖 corpus 로드: {corpus_path}")
    with open(corpus_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [r.get("input_normalized", "") for r in rows]
    print(f"   {len(texts):,}건")

    # 무거운 import는 여기서 (torch/sentence-transformers)
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("🔧 모델 로딩: jhgan/ko-sroberta-multitask")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    print("🧮 임베딩 생성 중... (수 분 소요)")
    emb = model.encode(
        texts, batch_size=64,
        normalize_embeddings=True, show_progress_bar=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"✅ 저장: {out_path}  shape={emb.shape} ({size_mb:.1f}MB)")
    print("   → python scripts/upload_data.py 로 S3 업로드")


if __name__ == "__main__":
    main()
