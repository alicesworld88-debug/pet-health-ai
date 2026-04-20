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
run_dashboard.py (로컬) →      S3 정적 호스팅 (React 대시보드) ★ 데모 배포
Streamlit (로컬)        →      EC2 t2.micro (실시간 검색 필요 시)
네이버 API (수동)       →      Lambda (주기적 수집)
```

---

## Step 0-A. React 대시보드 → S3 정적 호스팅 ★ 권장 데모 방식

> `dashboard_live.html`은 단일 HTML 파일 — EC2/서버 불필요.  
> 원클릭 배포: `python deploy_aws.py`

### 수동 배포 방법

```bash
# 1) 실데이터 주입된 HTML 생성
python run_dashboard.py --no-browser

# 2) S3 버킷 생성 (퍼블릭 정적 호스팅)
BUCKET=pet-health-ai-demo
aws s3 mb s3://$BUCKET --region ap-northeast-2
aws s3 website s3://$BUCKET --index-document dashboard_live.html

# 3) 퍼블릭 읽기 허용 (Block Public Access OFF 먼저 확인)
aws s3api put-public-access-block \
  --bucket $BUCKET \
  --public-access-block-configuration \
  "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

aws s3api put-bucket-policy --bucket $BUCKET --policy "{
  \"Version\":\"2012-10-17\",
  \"Statement\":[{\"Effect\":\"Allow\",\"Principal\":\"*\",
  \"Action\":\"s3:GetObject\",
  \"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"

# 4) HTML 업로드
aws s3 cp app/dashboard_live.html s3://$BUCKET/dashboard_live.html \
  --content-type "text/html; charset=utf-8"

# 접속 URL:
# http://pet-health-ai-demo.s3-website.ap-northeast-2.amazonaws.com/dashboard_live.html
```

---

## Step 0. 빌링 알림 설정 (필수 — 먼저 할 것)

> 먼저 설정하지 않으면 예상치 못한 요금 발생 가능

```
AWS 콘솔 → Billing → Budgets → Create Budget
- Budget type: Cost budget
- Amount: $5
- 알림 조건: 실제 비용 80% 초과 시 이메일 발송
```

---

## Step 1. IAM 사용자 + CLI 설정

```bash
# 1) AWS 콘솔 → IAM → 사용자 → 사용자 생성
#    권한: AmazonS3FullAccess, AmazonSageMakerFullAccess,
#          AmazonEC2FullAccess, AmazonSSMFullAccess
#    액세스 키 발급 후 저장

# 2) CLI 설정 (터미널)
aws configure
# AWS Access Key ID:     발급받은 키 입력
# AWS Secret Access Key: 발급받은 시크릿 입력
# Default region:        ap-northeast-2
# Output format:         json

# 3) 설정 확인
aws sts get-caller-identity
```

> **키 관리**: 실제 키는 로컬 `.env` 파일에만 보관 (git 절대 업로드 금지)  
> Lambda/EC2 키는 AWS Systems Manager Parameter Store(무료)로 관리

---

## Step 2. 키 관리 — Parameter Store (무료)

> Secrets Manager($0.40/건/월) 대신 Parameter Store Standard(무료) 사용

```bash
# 네이버 API 키 저장
aws ssm put-parameter \
  --name "/pet-health-ai/naver-client-id" \
  --value "실제키값" \
  --type SecureString \
  --region ap-northeast-2

aws ssm put-parameter \
  --name "/pet-health-ai/naver-client-secret" \
  --value "실제시크릿값" \
  --type SecureString \
  --region ap-northeast-2
```

```python
# Lambda / EC2 코드에서 읽기
import boto3

def get_parameter(name: str) -> str:
    ssm = boto3.client('ssm', region_name='ap-northeast-2')
    return ssm.get_parameter(Name=name, WithDecryption=True)['Parameter']['Value']

naver_id     = get_parameter('/pet-health-ai/naver-client-id')
naver_secret = get_parameter('/pet-health-ai/naver-client-secret')
```

---

## Step 3. S3 버킷 생성 및 데이터 업로드

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

# 전처리된 CSV 업로드
aws s3 cp data/processed/corpus_preprocessed.csv \
  s3://pet-health-ai-data/processed/corpus_preprocessed.csv
```

---

## Step 4. 노트북 S3 전환

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

## Step 5. SageMaker — Sentence-BERT 임베딩 생성

> t2.micro(1GB RAM)에서는 BERT 모델 로딩 불가 → SageMaker 사용 필수

```python
# SageMaker Notebook Instance에서 실행
# 인스턴스 타입: ml.t3.medium (무료 티어) 또는 ml.g4dn.xlarge (GPU)

from sentence_transformers import SentenceTransformer
import numpy as np
import boto3

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

embeddings = model.encode(df['input'].tolist(), batch_size=64, show_progress_bar=True)

# S3에 저장
np.save('/tmp/embeddings.npy', embeddings)
s3 = boto3.client('s3')
s3.upload_file('/tmp/embeddings.npy', 'pet-health-ai-data', 'embeddings/db_embeddings.npy')
```

---

## Step 6. EC2 — Streamlit 배포

```bash
# EC2 t2.micro 인스턴스 생성 (Amazon Linux 2)
# 보안 그룹 인바운드: 포트 8501 (Streamlit), 22 (SSH) 허용

# SSH 접속 후 환경 구성
sudo yum update -y
pip install streamlit boto3 numpy pandas scikit-learn sentence-transformers

# 코드 배포
git clone <repo-url>
cd pet-health-ai
cp .env.example .env
# .env에 LOCAL_DATA_ROOT, AWS 설정 입력

# systemd 서비스 등록 (자동 재시작)
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit Pet Health AI

[Service]
ExecStart=/usr/local/bin/streamlit run /home/ec2-user/pet-health-ai/app/streamlit_app.py --server.port 8501
Restart=always
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable streamlit
sudo systemctl start streamlit
```

---

## Step 7. Lambda — 네이버 지식iN 수집 (추후 구현)

```python
# Lambda 함수 (Python 3.11)
# 트리거: EventBridge (주 1회)
# 키는 Parameter Store에서 읽음 (하드코딩 금지)

import json, os, boto3, urllib.request

def get_parameter(name):
    ssm = boto3.client('ssm', region_name='ap-northeast-2')
    return ssm.get_parameter(Name=name, WithDecryption=True)['Parameter']['Value']

def lambda_handler(event, context):
    client_id     = get_parameter('/pet-health-ai/naver-client-id')
    client_secret = get_parameter('/pet-health-ai/naver-client-secret')

    url = "https://openapi.naver.com/v1/search/kin.json?query=강아지+질병&display=100"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())

    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='pet-health-ai-data',
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
| S3 업로드 실패 | Dead Letter Queue(DLQ) 설정 |

### SageMaker 중단
- 임베딩 생성은 1회성 작업 → 결과를 S3에 저장
- 재실행 필요 시 S3에서 `.npy` 로드로 대체

### EC2 다운
- systemd `Restart=always` 로 프로세스 자동 재시작

### 키 노출 시
- Parameter Store에서 해당 파라미터 즉시 삭제 후 재생성
- 네이버 키 노출 시: 네이버 개발자센터에서 즉시 재발급

---

## 체크리스트

- [ ] AWS Budget $5 알림 설정
- [ ] IAM 사용자 생성 + 키 발급 + CLI 설정
- [ ] Parameter Store에 네이버 키 저장
- [ ] S3 버킷 생성 + 데이터 업로드
- [ ] `DATA_SOURCE=s3`로 전환 후 노트북 테스트
- [ ] SageMaker에서 BERT 임베딩 생성 + S3 저장
- [ ] EC2 인스턴스 생성 + Streamlit 배포 + systemd 등록
- [ ] Lambda 함수 배포 + EventBridge 트리거 설정 (추후)
