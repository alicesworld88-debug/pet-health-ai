import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# -------------------------------------------------------------------
# 실행 환경 설정
# DATA_SOURCE = "local"  → 로컬 Desktop 경로 사용 (현재)
# DATA_SOURCE = "s3"     → S3 경로 사용 (AWS 마이그레이션 후)
# -------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "local")
AWS_REGION   = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET    = os.getenv("S3_BUCKET_NAME", "alices-project-storage")
S3_PREFIX    = "pet-health-ai"

# 로컬 경로 — .env의 LOCAL_DATA_ROOT 또는 환경변수로 지정
# 예: LOCAL_DATA_ROOT=/path/to/59.반려견.../3.개방데이터/1.데이터
_LOCAL_ROOT      = Path(os.getenv("LOCAL_DATA_ROOT", "./data/raw"))
LOCAL_TRAIN_PATH = _LOCAL_ROOT / "Training/02.라벨링데이터"
LOCAL_VAL_PATH   = _LOCAL_ROOT / "Validation/02.라벨링데이터"

# S3 경로 (마이그레이션 후 사용)
S3_TRAIN_PATH = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/raw/training/"
S3_VAL_PATH   = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/raw/validation/"

# 프로젝트 내부 경로
PROJECT_ROOT   = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
DATA_SPLITS    = PROJECT_ROOT / "data/splits"


def get_train_path() -> str:
    if DATA_SOURCE == "s3":
        return S3_TRAIN_PATH
    return str(LOCAL_TRAIN_PATH)


def get_val_path() -> str:
    if DATA_SOURCE == "s3":
        return S3_VAL_PATH
    return str(LOCAL_VAL_PATH)


def is_s3() -> bool:
    return DATA_SOURCE == "s3"
