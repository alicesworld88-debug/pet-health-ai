# 배포 사전 점검 체크리스트

`sam deploy` 전에 한 번에 점검한다. 자동 검사는 한 줄로:

```bash
bash scripts/preflight.sh                 # 빠른 정적 검증 (배포/변경 없음)
bash scripts/preflight.sh --build         # + sam build (Docker 이미지, 수 분)
bash scripts/preflight.sh --local-invoke  # + sam local invoke (AWS 자격증명 필요)
```

- **FAIL 0개** → 진행. WARN은 내용 확인 후 진행.
- 스크립트는 **읽기 전용**(리소스 생성·변경 없음). SSM 키 **값은 절대 출력하지 않음**(이름만 확인).

---

## A. 자동 검사 항목 (scripts/preflight.sh)

| 단계 | 검사 | 통과 기준 |
|------|------|-----------|
| 1 도구 | aws / sam / docker / python | 설치·실행 확인 |
| 2 코드 | `py_compile` 전 파일 | 구문 오류 0 |
| 2 코드 | **import 격리** | `import chat` 시 torch/sentence_transformers/pandas 미로드 |
| 2 코드 | events/chat.json | JSON 유효 |
| 3 템플릿 | **`sam validate`** | 템플릿 유효 |
| 3 템플릿 | **`sam validate --lint`** | cfn-lint 통과 |
| 4 데이터 | corpus / intent (필수) | 로컬 존재 |
| 4 데이터 | tfidf_prefit / embeddings | 선택/ BERT용 |
| 5 AWS | sts / S3 버킷 / S3 객체 / SSM 키 | 자격증명 있을 때 |

---

## B. 수동 점검 항목 (스크립트로 자동화 불가)

### B-1. 시크릿
- [ ] SSM에 Vertex 키 등록: `aws ssm put-parameter --name /pet-health-ai/VERTEX_API_KEY --type SecureString --value <키> --region ap-northeast-2`
- [ ] 키가 git에 커밋되지 않았는지 (`.env`만, `git status` 확인)
- [ ] SSM을 **CMK**로 바꿨다면 IAM에 `kms:Decrypt` 추가했는지

### B-2. 데이터 업로드
- [ ] `python scripts/build_tfidf_prefit.py` (선택, 콜드스타트 단축)
- [ ] `python scripts/upload_data.py` 로 S3 업로드 완료
- [ ] BERT 쓸 거면 `full_embeddings.npy` 생성·업로드했는지 (`python scripts/build_embeddings.py` — 로컬)
- [ ] **pickle 버전 정합성**: prefit/intent pkl을 만든 로컬 `scikit-learn` 버전이 이미지(`requirements-serving.txt`)와 호환되는지

### B-3. 비용 가드 (배포 전 권장)
- [ ] AWS Budgets $5 알림 설정 (`docs/aws_migration.md` Step 0)
- [ ] HTTP API 쓰로틀 확인: `template.yaml`의 `DefaultRouteSettings` (burst 10 / rate 5) — 트래픽 예상에 맞게 조정
- [ ] BERT 함수 사용 시 콜드스타트 비용·지연 인지 (Provisioned Concurrency는 상시 과금 ≈ 월 $22)

### B-4. 템플릿 리뷰 (눈으로 확인)
- [ ] 두 함수 모두 `PackageType: Image` (Globals 아님)
- [ ] `CorsConfiguration` 없음 (CORS는 FastAPI 한 곳) — 중복 헤더 방지
- [ ] IAM이 `s3:GetObject`(데이터 경로)·`ssm:GetParameter`(키 경로)만 허용하는지 (최소권한)
- [ ] `S3BucketName` 파라미터가 실제 데이터 버킷과 일치

---

## C. 배포 & 사후 검증

```bash
sam build
sam deploy --guided          # 최초 1회 (이후 sam deploy)

# Outputs에서 ApiBaseUrl 확보 후
API=https://xxxx.execute-api.ap-northeast-2.amazonaws.com
curl $API/tfidf/health       # {"status":"ok","pipeline":"tfidf"}
curl $API/bert/health        # {"status":"ok","pipeline":"bert"}
curl -X POST $API/tfidf/chat -H 'content-type: application/json' -d '{"query":"강아지가 기침을 해요"}'

# CORS: 응답 헤더에 Access-Control-Allow-Origin이 "정확히 1개"인지 (중복 아님) 확인
curl -i -X POST $API/tfidf/chat -H 'Origin: http://example.com' \
  -H 'content-type: application/json' -d '{"query":"기침"}' | grep -i access-control-allow-origin

# 대시보드 연결
python deploy_aws.py --api-base "$API/tfidf"
```

사후 체크:
- [ ] `/tfidf/health`·`/bert/health` 200
- [ ] `/tfidf/chat` 정상 응답 (intent/answer/sources)
- [ ] **`Access-Control-Allow-Origin` 헤더가 1개만** (중복이면 CORS 설정 점검)
- [ ] BERT 첫 호출(콜드) 29초 내 응답 — 초과하면 `GEMINI_TIMEOUT=18` 또는 메모리 상향
- [ ] 대시보드에서 채팅이 `{API}/tfidf/chat` 호출·렌더

> 상세 절차·트러블슈팅은 [lambda_deploy.md](lambda_deploy.md) 참고.
