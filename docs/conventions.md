# 프로젝트 컨벤션

## 커밋 메시지

형식: `타입: 한글 설명`

| 타입 | 용도 |
|------|------|
| `feat` | 새 기능 / 노트북 추가 |
| `fix` | 오류 수정 |
| `data` | 데이터 수집 / 처리 |
| `docs` | 문서 작성 / 수정 |
| `chore` | 환경설정 / 패키지 |

**규칙**
- 타입 소문자 + 콜론 + 한 칸 띄우고 설명
- 설명은 동사로 시작 (추가, 수정, 삭제, 작성, 구현)
- 한 줄, 50자 이내

**예시**
```
feat: 데이터 수집 노트북 추가
feat: KoNLPy 전처리 파이프라인 구현
fix: JSON 파싱 오류 처리 추가
data: AI Hub corpus CSV 저장
docs: AWS 마이그레이션 가이드 작성
chore: requirements.txt 패키지 추가
```

---

## 브랜치 전략

```
main           ← 항상 동작하는 상태 유지 (제출 기준)
└── dev        ← 개발 작업 브랜치
    ├── feat/data-collection
    ├── feat/preprocessing
    ├── feat/eda
    └── feat/matching
```

**규칙**
- `main`에 직접 push 금지 — 반드시 `dev`에서 작업 후 merge
- 노트북 1개 완성 → `dev` 커밋 → `main` merge
- 브랜치명: `feat/작업명` (소문자, 하이픈)

---

## 파일 / 노트북 네이밍

```
notebooks/
├── 01_data_collection.ipynb     # 번호_작업명 (소문자, 언더스코어)
├── 02_data_validation.ipynb
├── 03_preprocessing.ipynb
├── 04_eda.ipynb
├── 05_ground_truth.ipynb
├── 06_matching.ipynb
└── 07_evaluation.ipynb

utils/
├── config.py        # 환경 설정
└── (추가 모듈).py   # 소문자, 언더스코어

docs/
├── conventions.md       # 이 파일
├── aws_migration.md
└── progress.md
```

---

## 코드 스타일

- 변수명 / 함수명: 영어 스네이크케이스 (`load_json_files`, `df_train`)
- 주석 / docstring / 노트북 설명 셀: 한글
- 들여쓰기: 스페이스 4칸
- 함수 하나의 역할은 하나 (단일 책임)

---

## 환경변수 관리

```bash
# .env 파일 (로컬 전용 — git push 절대 금지)
DATA_SOURCE=local
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=pet-health-ai-data
NAVER_CLIENT_ID=your_naver_id
NAVER_CLIENT_SECRET=your_naver_secret
```

- `.env.example`에 키 이름만 기록 (값 없이) → git에 포함
- 코드에 키 하드코딩 금지 → 항상 `os.getenv()` 사용
- AWS Lambda는 콘솔 환경변수로 관리

---

## 데이터 버전 관리

```
data/
├── raw/              # 원본 데이터 (수정 금지)
│   ├── training/
│   └── validation/
├── processed/        # 전처리 결과
│   ├── corpus_raw.csv          # 01_data_collection 결과
│   ├── corpus_validated.csv    # 02_data_validation 결과
│   └── corpus_preprocessed.csv # 03_preprocessing 결과
└── splits/           # 학습/검증/테스트 분리
    ├── train.csv
    ├── val.csv
    └── ground_truth.csv        # 05_ground_truth 수동 레이블
```

**규칙**
- `data/` 전체 gitignore — 원본 데이터 절대 git push 금지
- 단계별 처리 결과는 별도 파일로 저장 (덮어쓰기 금지)
- S3 업로드 후 로컬 `data/processed/` 와 S3 내용 동기화 유지
