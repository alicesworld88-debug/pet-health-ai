# Lambda 배포 가이드 (AWS SAM)

`pet-health-ai`를 AWS Lambda + API Gateway 서비스로 배포한다.

- **TfidfFunction / BertFunction** — 단일 컨테이너 이미지(`Dockerfile`)를 공유, `RETRIEVER_TYPE` env로만 분기
  - Tfidf 1024MB / Bert 2048MB. TF-IDF는 torch를 lazy import라 런타임에 로드하지 않음
- 단일 HTTP API에서 경로 프리픽스로 분기: `/tfidf/*`, `/bert/*`
- 데이터(corpus/intent/embeddings)는 S3에서 콜드스타트에 `/tmp`로 로드

```
[S3 대시보드] ─POST {API}/tfidf/chat─► [API Gateway] ─► [Tfidf Lambda] ┐ (공유 이미지)
                                                 └────► [Bert Lambda]  ┘  ──► Gemini
[S3 data] ◄─ 콜드스타트 다운로드 ─┘
```

---

## 0. 사전 준비물
- AWS 계정 + `aws configure` (자격증명, region `ap-northeast-2`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Docker (공유 컨테이너 이미지 빌드용)
- `data/processed/` 생성물 (노트북 실행): `corpus_preprocessed.csv`, `intent_classifier.pkl`
- BERT 배포 시: `embeddings/full_embeddings.npy` — 로컬에서 `python scripts/build_embeddings.py`로 생성

---

## 1. Vertex API 키 → SSM Parameter Store (1회)
키는 코드/템플릿에 넣지 않고 SSM SecureString으로 관리한다. 런타임에 `generator._get_api_key()`가 `VERTEX_API_KEY_PARAM` 경로에서 읽는다.

```bash
aws ssm put-parameter \
  --name "/pet-health-ai/VERTEX_API_KEY" \
  --value "실제_API_키" \
  --type SecureString \
  --region ap-northeast-2
```

## 2. 검색기 산출물 생성 (로컬)
```bash
# (선택) TF-IDF 사전학습 피클 — 콜드스타트 단축
python scripts/build_tfidf_prefit.py     # data/processed/tfidf_prefit.pkl

# (BERT 쓸 때) 임베딩 생성 — SageMaker 불필요, 로컬에서 수 분
python scripts/build_embeddings.py       # data/processed/embeddings/full_embeddings.npy
```
> 로컬 RAM/시간이 부족하면 Google Colab(무료 GPU)에서 같은 스크립트 실행 후 `.npy`만 받아 업로드해도 된다.

## 3. 데이터 S3 업로드
> ⚠️ 이 **데이터 버킷은 비공개로 유지**한다 (Lambda는 IAM으로 읽음). 절대 공개 설정하지 말 것.

```bash
# 데이터 버킷이 없으면 먼저 생성 (이미 있으면 생략 — upload_data.py는 버킷을 만들지 않음)
aws s3 mb s3://alices-project-storage --region ap-northeast-2

python scripts/upload_data.py            # corpus/intent (+prefit, embeddings) 업로드
# 버킷 기본값: utils/config.py의 S3_BUCKET (alices-project-storage)
```
S3 레이아웃:
```
s3://<bucket>/pet-health-ai/data/processed/corpus_preprocessed.csv
                                          /intent_classifier.pkl
                                          /tfidf_prefit.pkl                # 선택
                                          /embeddings/full_embeddings.npy  # BERT
```

## 4. 빌드 & 배포
```bash
sam build                        # 공유 컨테이너 이미지 빌드 (docker 필요)
sam deploy --guided             # 최초 1회 (이후 sam deploy)
#   - Stack name: pet-health-ai
#   - Region: ap-northeast-2
#   - S3BucketName: <데이터 버킷>  (기본 alices-project-storage)
#   - 이미지 ECR 리포: resolve_image_repos=true 로 자동 생성
```
배포 후 출력(Outputs)에서 `ApiBaseUrl`, `TfidfChatUrl`, `BertChatUrl` 확인.

## 5. 대시보드(S3 정적 호스팅) 연결
> ⚠️ `deploy_aws.py`는 대상 버킷을 **전체 공개**로 설정한다. 따라서 **데이터 버킷과 다른 별도 공개 버킷**을 써야 한다
> (데이터 버킷을 그대로 쓰면 corpus·임베딩·분류기까지 공개됨). `--bucket`으로 데모 전용 버킷을 지정한다.

```bash
# API_BASE = ApiBaseUrl + /tfidf (또는 /bert)
python deploy_aws.py \
  --bucket pet-health-ai-demo \
  --api-base "https://xxxx.execute-api.ap-northeast-2.amazonaws.com/tfidf"
```
이 스크립트가 `dashboard_live.html`에 `window.API_BASE`를 주입하고 (공개) 데모 버킷에 업로드한다.

---

## 6. 검증

### 로컬
```bash
# 코어 import가 torch를 로드하지 않는지 확인 (lazy import 검증)
python -c "import sys, chat; assert 'torch' not in sys.modules; print('ok')"

# SAM 로컬 호출 (실제 S3 사용 → AWS 자격증명 필요)
sam build
sam local invoke TfidfFunction -e events/chat.json \
  --env-vars '{"TfidfFunction":{"VERTEX_API_KEY":"<키>"}}'
```

### 배포 후
```bash
API=https://xxxx.execute-api.ap-northeast-2.amazonaws.com

curl $API/tfidf/health          # {"status":"ok","pipeline":"tfidf"}
curl $API/bert/health           # {"status":"ok","pipeline":"bert"}

curl -X POST $API/tfidf/chat -H 'content-type: application/json' \
  -d '{"query":"강아지가 기침을 해요"}'
# → {"intent":..., "answer":..., "sources":[...≤3], "clarify_question":...}

# CORS preflight
curl -i -X OPTIONS $API/tfidf/chat \
  -H 'Origin: http://example.com' -H 'Access-Control-Request-Method: POST'
```

### 대시보드 E2E
S3 사이트 접속 → AI 채팅 탭에서 질문 전송 → 네트워크 호출이 `{API}/tfidf/chat`로 가는지, 응답(intent/sources)이 렌더되는지 확인.

---

## 참고 / 트러블슈팅
- **CORS**: FastAPI `CORSMiddleware`(app/api.py) 한 곳에서만 처리한다. API Gateway에 CorsConfiguration을 추가하면 `Access-Control-Allow-Origin`이 중복되어 브라우저가 차단하니 넣지 말 것. (OPTIONS preflight도 Lambda의 Starlette가 처리)
- **쓰로틀**: HTTP API `DefaultRouteSettings`로 burst 10 / rate 5 req/s 제한(비용 남용 방지). 트래픽이 늘면 `template.yaml`에서 상향.
- **타임아웃**: API Gateway HTTP API 통합 한도 ~30초. Gemini 호출은 기본 22초 제한(env `GEMINI_TIMEOUT`로 조정). Lambda Timeout 29초.
- **BERT 콜드스타트가 29초 초과 위험**: 첫 요청에 모델 로드(~8s)+임베딩 다운로드(63MB)+Gemini(최대 22s)가 한 요청에 몰림. 대응: ① BERT 함수에 `GEMINI_TIMEOUT=18` 환경변수로 여유 확보, ② MemorySize 상향(3008MB)으로 로드 가속, ③ 상시 빠른 응답 필요 시 Provisioned Concurrency(2GB ≈ 월 $22, 데모엔 비추천). 데모는 TF-IDF 위주 사용 권장.
- **pickle 버전 정합성**: `intent_classifier.pkl`/`tfidf_prefit.pkl`은 로컬 sklearn으로 생성된다. 이미지의 `scikit-learn` 버전이 크게 다르면 unpickle 경고/실패 가능 → 가급적 동일 버전으로 맞추거나(requirements-serving.txt 핀 고정), 피클을 이미지 안에서 생성.
- **HF 오프라인 로드**: 모델은 이미지 `/opt/hf`에 베이크되고 `HF_HUB_OFFLINE=1`로 로드된다. 만약 락 파일 쓰기 오류가 나면 콜드스타트에 `/opt/hf`를 `/tmp`로 복사 후 `HF_HOME=/tmp/hf`로 로드하도록 변경.
- **SSM 키 복호화**: 기본 `aws/ssm` 키면 `ssm:GetParameter`만으로 충분. 고객관리형 키(CMK)로 바꾸면 IAM에 `kms:Decrypt` 추가 필요.
- **공유 이미지**: 두 함수가 동일 `Dockerfile`(`DockerTag: pet-health-serving`)을 사용. TF-IDF 함수도 이미지에 torch가 있지만 lazy import라 런타임 미로드 → 실행 성능 영향 없음. ECR 보관료 ~$0.3–0.6/월.
