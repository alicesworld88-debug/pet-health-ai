"""
런타임 데이터 파일을 S3에 업로드 — Lambda가 콜드스타트에 내려받는다.

업로드 대상 (data/processed/ → s3://<bucket>/pet-health-ai/data/processed/):
  - corpus_preprocessed.csv        (필수)
  - intent_classifier.pkl          (필수)
  - tfidf_prefit.pkl               (선택, TF-IDF 콜드스타트 단축)
  - embeddings/full_embeddings.npy (BERT 함수 필수)

사용: python scripts/upload_data.py [--bucket 버킷명]
사전: aws configure / boto3, 노트북으로 data/processed 생성
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import DATA_PROCESSED, S3_BUCKET, S3_PREFIX, AWS_REGION

# (로컬 상대경로, S3 키 접미사, 필수여부)
FILES = [
    ("corpus_preprocessed.csv",        "corpus_preprocessed.csv",        True),
    ("intent_classifier.pkl",          "intent_classifier.pkl",          True),
    ("tfidf_prefit.pkl",               "tfidf_prefit.pkl",               False),
    ("embeddings/full_embeddings.npy", "embeddings/full_embeddings.npy", False),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=S3_BUCKET)
    p.add_argument("--region", default=AWS_REGION)
    args = p.parse_args()

    import boto3
    s3 = boto3.client("s3", region_name=args.region)
    base_key = f"{S3_PREFIX}/data/processed"

    uploaded = 0
    for rel, key_suffix, required in FILES:
        local = DATA_PROCESSED / rel
        key = f"{base_key}/{key_suffix}"
        if not local.exists():
            msg = "❌ 필수 파일 없음" if required else "⏭️  선택 파일 없음 → skip"
            print(f"{msg}: {local}")
            if required:
                sys.exit(1)
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        print(f"📤 {local.name} → s3://{args.bucket}/{key} ({size_mb:.1f}MB)")
        s3.upload_file(str(local), args.bucket, key)
        uploaded += 1

    print(f"\n✅ 업로드 완료: {uploaded}개")


if __name__ == "__main__":
    main()
