# AWS 마이그레이션 가이드

현재 로컬 환경에서 개발 후 AWS(Pattern C)로 전환하는 단계별 가이드.

---

## 아키텍처 (Pattern C)

```
로컬 개발 단계                  AWS 전환 후
─────────────────────          ──────────────────────────
로컬 JSON 파일                 S3 (데이터 저장)
    ↓                               ↓
Jupyter (로컬)          →      SageMaker Notebook (BERT 임베딩)
    ↓                               ↓
Streamlit (로컬)        →      EC2 t2.micro (Streamlit 데모)
네이버 API (수동)       →      Lambda (주기적 수집)
```

---

## Step 1. AWS 계정 셋업

```bash
# IAM 사용자 생성 후 키 발급
# 권한: S3FullAccess, SageMakerFullAccess, LambdaFullAccess

# .env 파일 설정 (절대 git에 올리지 말 것)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=pet-health-ai-data
DATA_SOURCE=local  # S3 전환 후 → s3 로 변경
```

---

## Step 2. S3 버킷 생성 및 데이터 업로드

```bash
# 버킷 생성
aws s3 mb s3://pet-health-ai-data --region ap-northeast-2

# 로컬 데이터 → S3 업로드 (1회)
aws s3 sync \
  "$LOCAL_DATA_ROOT/Training/02.라벨링데이터/" \
  s3://pet-health-ai-data/raw/training/

aws s3 sync \
  "$LOCAL_DATA_ROOT/Validation/02.라벨링데이터/" \
  s3://pet-health-ai-data/raw/validation/

# 처리된 CSV도 업로드
aws s3 cp data/processed/corpus_raw.csv s3://pet-health-ai-data/processed/corpus_raw.csv
```

---

## Step 3. 노트북 S3 전환

`.env`에서 `DATA_SOURCE=s3` 변경하면 `utils/config.py`가 자동으로 S3 경로를 반환.

각 노트북의 주석 처리된 S3 코드 블록 활성화:

```python
# 01_data_collection.ipynb — load_json_parallel() 내 S3 블록 활성화
import s3fs
fs = s3fs.S3FileSystem()
files = [Path(f) for f in fs.glob(data_path + '*.json')]
```

추가 패키지 설치:
```bash
pip install s3fs boto3
```

---

## Step 4. SageMaker — Sentence-BERT 임베딩 생성

> t2.micro(1GB RAM)에서는 BERT 모델 로딩 불가 → SageMaker 사용 필수

```python
# SageMaker Notebook Instance에서 실행
# 인스턴스 타입: ml.t3.medium (무료 티어) 또는 ml.g4dn.xlarge (GPU)

from sentence_transformers import SentenceTransformer
import numpy as np
import boto3

model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 한국어 모델

# 임베딩 생성
embeddings = model.encode(df['input'].tolist(), batch_size=64, show_progress_bar=True)

# S3에 저장
np.save('/tmp/embeddings.npy', embeddings)
s3 = boto3.client('s3')
s3.upload_file('/tmp/embeddings.npy', 'pet-health-ai-data', 'embeddings/train_embeddings.npy')
```

---

## Step 5. EC2 — Streamlit 배포

```bash
# EC2 t2.micro 인스턴스 생성 (Amazon Linux 2)
# 포트 8501 인바운드 규칙 추가

# 서버 접속 후
pip install streamlit boto3 numpy pandas scikit-learn

# Streamlit은 S3에서 임베딩 로드만 (생성 안 함)
# → 메모리 문제 없음
streamlit run app/streamlit_app.py --server.port 8501
```

---

## Step 6. Lambda — 네이버 지식iN 주기 수집

```python
# Lambda 함수 (Python 3.11)
# 트리거: EventBridge (주 1회 등)
# 환경변수: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, S3_BUCKET_NAME

import json, os, boto3, urllib.request

def lambda_handler(event, context):
    query = "강아지 질병"
    url = f"https://openapi.naver.com/v1/search/kin.json?query={query}&display=100"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", os.environ["NAVER_CLIENT_ID"])
    req.add_header("X-Naver-Client-Secret", os.environ["NAVER_CLIENT_SECRET"])

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())

    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=os.environ["S3_BUCKET_NAME"],
        Key=f"naver/kin_{event.get('date', 'latest')}.json",
        Body=json.dumps(data, ensure_ascii=False)
    )
    return {"statusCode": 200, "body": f"수집 완료: {len(data['items'])}건"}
```

---

## 비기능 요구사항

| 항목 | 목표 |
|------|------|
| Streamlit 응답 시간 | 쿼리 입력 후 3초 이내 결과 반환 |
| Lambda 수집 주기 | 주 1회 (EventBridge 트리거) |
| S3 데이터 보존 기간 | 프로젝트 종료 후 90일 |
| 월 비용 한도 | $5 이하 (Budget 알림 설정) |
| EC2 가용성 | 데모 발표 기간 중 상시 운영 |

---

## 장애 대응

### S3 접근 실패
```python
# utils/config.py — S3 접근 불가 시 로컬로 자동 fallback
def get_train_path() -> str:
    if DATA_SOURCE == "s3":
        try:
            import boto3
            boto3.client('s3').head_bucket(Bucket=S3_BUCKET)
            return S3_TRAIN_PATH
        except Exception:
            print("[경고] S3 접근 실패 → 로컬 경로로 fallback")
            return str(LOCAL_TRAIN_PATH)
    return str(LOCAL_TRAIN_PATH)
```

### Lambda 장애
| 상황 | 대응 |
|------|------|
| 타임아웃 | 제한 시간 30초 설정, 재시도 2회 |
| 네이버 API 한도 초과 | 하루 25,000건 제한 → 쿼리 분산 수집 |
| S3 업로드 실패 | Lambda Dead Letter Queue(DLQ) 설정 |

### SageMaker 중단
- 임베딩 생성은 1회성 작업 → 결과를 S3에 저장
- 재실행 필요 시 S3에서 `.npy` 로드로 대체
```python
# 임베딩 재생성 방지 — S3에 있으면 로드, 없으면 생성
if s3_file_exists('embeddings/train_embeddings.npy'):
    embeddings = np.load(download_from_s3('embeddings/train_embeddings.npy'))
else:
    embeddings = model.encode(...)
    upload_to_s3(embeddings, 'embeddings/train_embeddings.npy')
```

### EC2 다운
- Streamlit 프로세스 자동 재시작: `systemd` 서비스 등록
```bash
# /etc/systemd/system/streamlit.service
[Unit]
Description=Streamlit Pet Health AI

[Service]
ExecStart=/usr/bin/streamlit run /home/ec2-user/app/streamlit_app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

### 네이버 API 키 노출
- `.env` 파일은 절대 git push 금지 (`.gitignore` 확인)
- AWS Lambda 환경변수로 관리 (코드에 하드코딩 금지)
- 노출 시: 즉시 네이버 개발자센터에서 키 재발급

---

## 체크리스트

- [ ] AWS 계정 생성 + Budget $5 알림 설정
- [ ] IAM 사용자 생성 + 키 발급
- [ ] `.env` 파일 설정 (gitignore 확인)
- [ ] S3 버킷 생성
- [ ] 로컬 데이터 S3 업로드 (`aws s3 sync`)
- [ ] `DATA_SOURCE=s3`로 전환 후 노트북 테스트
- [ ] SageMaker에서 BERT 임베딩 생성 + S3 저장
- [ ] EC2 인스턴스 생성 + Streamlit 배포
- [ ] Lambda 함수 배포 + EventBridge 트리거 설정
