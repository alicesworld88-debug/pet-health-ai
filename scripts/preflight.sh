#!/usr/bin/env bash
# 배포 사전 점검 (읽기 전용 — 배포/리소스 변경 없음)
#
# 사용:
#   bash scripts/preflight.sh                 # 빠른 정적 검증 (기본)
#   bash scripts/preflight.sh --build         # + sam build (Docker 이미지, 수 분 소요)
#   bash scripts/preflight.sh --local-invoke  # + sam local invoke (AWS 자격증명 필요)
#
# 환경변수 오버라이드: AWS_REGION, S3_BUCKET_NAME, VERTEX_API_KEY_PARAM

cd "$(dirname "$0")/.." || exit 1

REGION="${AWS_REGION:-ap-northeast-2}"
BUCKET="${S3_BUCKET_NAME:-alices-project-storage}"
SSM_PARAM="${VERTEX_API_KEY_PARAM:-/pet-health-ai/VERTEX_API_KEY}"
PREFIX="pet-health-ai/data/processed"

DO_BUILD=0; DO_INVOKE=0
for a in "$@"; do
  case "$a" in
    --build) DO_BUILD=1 ;;
    --local-invoke) DO_INVOKE=1 ;;
  esac
done

PASS=0; WARN=0; FAIL=0
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; PASS=$((PASS+1)); }
warn() { printf "  \033[33m▲ WARN\033[0m %s\n" "$1"; WARN=$((WARN+1)); }
bad()  { printf "  \033[31m✗ FAIL\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
hdr()  { printf "\n\033[1m%s\033[0m\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

PY=python3; have python3 || PY=python

# ── 1. 필수 도구 ───────────────────────────────────────────────
hdr "1. 도구"
have aws    && ok "aws CLI: $(aws --version 2>&1 | head -1)"              || bad "aws CLI 미설치"
have sam    && ok "SAM CLI: $(sam --version 2>&1)"                        || bad "SAM CLI 미설치 (sam validate/build 불가)"
have docker && (docker info >/dev/null 2>&1 && ok "Docker 실행 중" || warn "Docker 설치됐으나 데몬 미실행 (build 시 필요)") || warn "Docker 미설치 (이미지 빌드 불가)"
have "$PY"  && ok "Python: $($PY --version 2>&1)"                         || bad "python 미설치"

# ── 2. 코드 정적 검증 ──────────────────────────────────────────
hdr "2. 코드"
if $PY -m py_compile chat.py app/api.py app/lambda_handler.py utils/matcher.py \
      utils/generator.py utils/runtime_paths.py scripts/upload_data.py \
      scripts/build_tfidf_prefit.py deploy_aws.py 2>/tmp/pf_pyc.log; then
  ok "py_compile 통과 (전 서빙 파일)"
else
  bad "py_compile 실패 — $(cat /tmp/pf_pyc.log | head -1)"
fi

# torch lazy import 격리 (서빙 코어가 torch를 끌어오지 않아야 함)
$PY - <<'PYEOF' 2>/tmp/pf_iso.log
import sys, types
m = types.ModuleType("dotenv"); m.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = m
import chat
heavy = [x for x in ("torch","sentence_transformers","pandas") if x in sys.modules]
sys.exit(1 if heavy else 0)
PYEOF
case $? in
  0) ok "import 격리: chat 임포트 시 torch/sentence_transformers/pandas 미로드" ;;
  1) bad "import 격리 실패: 서빙 코어가 무거운 모듈을 import함 (matcher lazy import 확인)" ;;
  *) warn "import 격리 미검증: $(head -1 /tmp/pf_iso.log) (의존성 미설치 가능 — 빌드 환경에서 재확인)" ;;
esac

# events/chat.json JSON 유효성
$PY -c "import json; json.load(open('events/chat.json'))" 2>/dev/null \
  && ok "events/chat.json 유효" || bad "events/chat.json 파싱 실패"

# ── 3. SAM 템플릿 검증 ─────────────────────────────────────────
hdr "3. 템플릿 (sam validate)"
if have sam; then
  if sam validate --region "$REGION" >/tmp/pf_val.log 2>&1; then
    ok "sam validate 통과"
  else
    bad "sam validate 실패 — $(tail -3 /tmp/pf_val.log | tr '\n' ' ')"
  fi
  if sam validate --lint >/tmp/pf_lint.log 2>&1; then
    ok "sam validate --lint (cfn-lint) 통과"
  else
    warn "cfn-lint 경고/실패 — $(tail -3 /tmp/pf_lint.log | tr '\n' ' ')"
  fi
else
  warn "SAM CLI 없어 템플릿 검증 건너뜀"
fi

# ── 4. 로컬 데이터 산출물 ──────────────────────────────────────
hdr "4. 로컬 데이터 (data/processed)"
chk_file(){ [ -f "$1" ] && ok "$(basename "$1") 존재 ($(du -h "$1" | cut -f1))" || { [ "$2" = req ] && bad "$1 없음 (노트북으로 생성 필요)" || warn "$1 없음 ($2)"; }; }
chk_file data/processed/corpus_preprocessed.csv req
chk_file data/processed/intent_classifier.pkl req
chk_file data/processed/tfidf_prefit.pkl "선택 — TF-IDF 콜드스타트 단축"
chk_file data/processed/embeddings/full_embeddings.npy "BERT 함수 사용 시 필수"

# ── 5. AWS 컨텍스트 (자격증명 있을 때만) ──────────────────────
hdr "5. AWS 컨텍스트 (선택)"
if have aws && aws sts get-caller-identity >/tmp/pf_sts.log 2>&1; then
  ok "자격증명 유효: $(aws sts get-caller-identity --query Account --output text 2>/dev/null) / region=$REGION"
  # S3 버킷
  aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 \
    && ok "S3 버킷 접근 가능: $BUCKET" || warn "S3 버킷 미존재/권한없음: $BUCKET (upload_data.py가 생성)"
  # S3 데이터 업로드 여부
  for f in corpus_preprocessed.csv intent_classifier.pkl; do
    aws s3 ls "s3://$BUCKET/$PREFIX/$f" >/dev/null 2>&1 \
      && ok "S3 업로드됨: $f" || warn "S3 미업로드: $f (scripts/upload_data.py 실행)"
  done
  aws s3 ls "s3://$BUCKET/$PREFIX/embeddings/full_embeddings.npy" >/dev/null 2>&1 \
    && ok "S3 업로드됨: full_embeddings.npy (BERT)" || warn "S3 미업로드: full_embeddings.npy (BERT 함수만 필요)"
  # SSM 파라미터 (값은 절대 출력하지 않음 — 이름만)
  aws ssm get-parameter --name "$SSM_PARAM" --query Parameter.Name --output text --region "$REGION" >/dev/null 2>&1 \
    && ok "SSM 키 등록됨: $SSM_PARAM" || bad "SSM 키 없음: $SSM_PARAM (aws ssm put-parameter 필요)"
else
  warn "AWS 자격증명 없음 → S3/SSM 점검 건너뜀 (aws configure 후 재실행)"
fi

# ── 6. (선택) sam build ────────────────────────────────────────
if [ "$DO_BUILD" = 1 ]; then
  hdr "6. sam build (Docker 이미지)"
  if sam build >/tmp/pf_build.log 2>&1; then
    ok "sam build 성공"
  else
    bad "sam build 실패 — $(tail -3 /tmp/pf_build.log | tr '\n' ' ')"
  fi
fi

# ── 7. (선택) sam local invoke ────────────────────────────────
if [ "$DO_INVOKE" = 1 ]; then
  hdr "7. sam local invoke (TfidfFunction)"
  if sam local invoke TfidfFunction -e events/chat.json >/tmp/pf_invoke.log 2>&1; then
    grep -q '"intent"' /tmp/pf_invoke.log \
      && ok "로컬 호출 응답 OK (intent 포함)" \
      || warn "호출됐으나 응답에 intent 없음 — $(tail -2 /tmp/pf_invoke.log | tr '\n' ' ')"
  else
    bad "sam local invoke 실패 — $(tail -3 /tmp/pf_invoke.log | tr '\n' ' ')"
  fi
fi

# ── 요약 ───────────────────────────────────────────────────────
hdr "결과"
printf "  PASS=%d  WARN=%d  FAIL=%d\n" "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf "\033[31m▶ FAIL 항목 해결 후 배포하세요.\033[0m\n"; exit 1
elif [ "$WARN" -gt 0 ]; then
  printf "\033[33m▶ WARN 항목 확인 후 진행 가능. (sam build && sam deploy)\033[0m\n"; exit 0
else
  printf "\033[32m▶ 사전 점검 통과 — sam build && sam deploy 진행 가능.\033[0m\n"; exit 0
fi
