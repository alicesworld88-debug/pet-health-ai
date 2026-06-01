"""
런타임 데이터 경로 해소 — 로컬 / Lambda(S3→/tmp) 공용.

Lambda는 패키지가 read-only이고 `/tmp`만 쓰기 가능하다.
DATA_SOURCE=s3 이면 콜드스타트에 S3에서 `/tmp/data/processed/...`로 내려받고,
로컬 모드면 기존 `utils.config.DATA_PROCESSED`를 그대로 쓴다.

`/tmp`는 웜 호출 사이에 유지되므로 ensure_s3_file()은 멱등(존재하면 skip)하게 동작한다.
"""
from pathlib import Path

from utils.config import DATA_SOURCE, DATA_PROCESSED, S3_BUCKET, S3_PREFIX, AWS_REGION

# Lambda /tmp 미러 경로 (로컬 data/processed 레이아웃과 동일)
_TMP_PROCESSED = Path("/tmp/data/processed")

# ── S3 키 (utils/config의 버킷·프리픽스 재사용) ─────────────────────────
_BASE = f"{S3_PREFIX}/data/processed"
CORPUS_KEY = f"{_BASE}/corpus_preprocessed.csv"
INTENT_KEY = f"{_BASE}/intent_classifier.pkl"
EMB_KEY    = f"{_BASE}/embeddings/full_embeddings.npy"
PREFIT_KEY = f"{_BASE}/tfidf_prefit.pkl"


def is_s3() -> bool:
    return DATA_SOURCE == "s3"


def data_dir() -> Path:
    """런타임 데이터 디렉터리. s3 모드면 /tmp 미러, 아니면 로컬 DATA_PROCESSED."""
    return _TMP_PROCESSED if is_s3() else DATA_PROCESSED


def ensure_s3_file(key: str, local: Path, *, optional: bool = False) -> bool:
    """
    s3 모드일 때 local이 없으면 S3에서 내려받는다 (멱등).
    로컬 모드면 no-op. 다운로드/존재 시 True, optional 파일이 없어 실패하면 False.
    """
    local = Path(local)
    if local.exists():
        return True
    if not is_s3():
        return local.exists()

    local.parent.mkdir(parents=True, exist_ok=True)
    import boto3  # Lambda 런타임 기본 제공
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.download_file(S3_BUCKET, key, str(local))
        return True
    except Exception as e:
        if optional:
            print(f"[runtime_paths] optional 파일 없음 → skip: s3://{S3_BUCKET}/{key} ({e})")
            return False
        raise
