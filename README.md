# 반려견 증상 매칭 AI 시스템

> AI Hub 반려견 Q&A 말뭉치 기반 생애주기 건강 분석 및 의미 매칭 시스템

[![Status](https://img.shields.io/badge/status-complete-brightgreen)]()
[![Course](https://img.shields.io/badge/course-Data_Mining-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

---

## 프로젝트 개요

반려견 보호자가 자연어로 증상을 입력하면, 수의사 Q&A 말뭉치에서 의미적으로 유사한 답변을 추천하는 시스템입니다.  
**TF-IDF**와 **Sentence-BERT** 두 방법의 성능을 정량 지표(Hit@1/Hit@3/Hit@5/MAP@5)로 비교하여 의미 기반 검색의 실질적 우위를 검증합니다.

**🌐 라이브 대시보드**: [S3 배포 링크](https://alices-project-storage.s3.ap-northeast-2.amazonaws.com/pet-health-ai/dashboard/index.html)

---

## 연구 질문

1. 생애주기(자견 / 성견 / 노령견)별 주요 질병 빈도 분포와 특징은 무엇인가?
2. Sentence-BERT 임베딩 기반 매칭은 TF-IDF 코사인 유사도보다 의미적 유사도를 더 잘 반영하는가?
3. 사용자 입력 쿼리와 기존 Q&A 데이터 간 유사도 매칭으로 보호자에게 실질적으로 유의미한 정보를 제공할 수 있는가?

---

## 분석 파이프라인

| # | 노트북 | 내용 | 출력 |
|---|--------|------|------|
| 01 | `01_data_collection.ipynb` | AI Hub JSON 병렬 로드, Train/Val 통합 | `corpus_raw.csv` |
| 02 | `02_data_validation.ipynb` | 결측치·중복·이상치·클래스 불균형 처리 | `corpus_validated.csv` |
| 03 | `03_preprocessing.ipynb` | 구어체 정규화, KoNLPy 형태소 분석, 불용어 제거 | `corpus_preprocessed.csv` |
| 04 | `04_eda.ipynb` | 생애주기별 질병 분포, 워드클라우드, Sunburst, Long-tail | `eda_figures/` |
| 05 | `05_ground_truth.ipynb` | 평가용 쿼리 50개 반자동 구축 | `ground_truth.csv` |
| 06 | `06_matching.ipynb` | TF-IDF / Sentence-BERT 매칭 실행 | `matching_results.csv` |
| 07 | `07_evaluation.ipynb` | Hit@1/3/5, MAP@5 성능 비교 시각화 | `evaluation_summary.csv` |

---

## 데이터

- **출처**: [AI Hub — 반려견 성장 및 질병 관련 말뭉치](https://aihub.or.kr)
- **구성**: 내과 / 안과 / 외과 / 치과 / 피부과 5개 진료과 Q&A
- **규모**: Training 19,418개 + Validation 2,427개 = 총 21,845개 JSON
- **라벨**: `lifeCycle` (자견/성견/노령견), `disease`, `department`

> 원본 데이터는 AI Hub 이용 약관에 따라 레포에 포함되지 않습니다.

---

## 폴더 구조

```
pet-health-ai/
├── notebooks/                     # 분석 노트북 (실행 순서대로)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_ground_truth.ipynb
│   ├── 06_matching.ipynb
│   └── 07_evaluation.ipynb
├── app/
│   └── dashboard.html             # 인터랙티브 대시보드 (S3 배포용)
├── utils/
│   ├── config.py                  # 경로·환경 설정 (local ↔ S3 전환)
│   ├── matcher.py                 # TF-IDF / BERT 공통 매칭 함수
│   ├── app_builder.py             # APP_DATA 빌더
│   ├── chart_builder.py           # Plotly 차트 생성
│   └── theme.py                   # 디자인 시스템 (색상 팔레트)
├── data/
│   ├── processed/                 # 전처리 결과 CSV, 임베딩 .npy
│   └── splits/                    # ground_truth.csv
├── docs/
│   ├── concept_model.md           # 도메인 개념 모델 정의
│   ├── evaluation_tables.md       # 보고서용 평가 수치
│   ├── aws_migration.md           # AWS 마이그레이션 가이드
│   └── conventions.md             # 커밋·코드 컨벤션
├── run_dashboard.py               # 대시보드 로컬 실행
├── deploy_aws.py                  # S3 배포 스크립트
├── requirements.txt
└── README.md
```

---

## 설치 및 실행

```bash
# 1. 레포 클론
git clone https://github.com/alicesworld88-debug/pet-health-ai.git
cd pet-health-ai

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 노트북 순서대로 실행 (01 → 07)
jupyter notebook

# 5. 대시보드 로컬 실행
python run_dashboard.py
```

> **데이터 경로**: `utils/config.py`의 `_LOCAL_ROOT`를 AI Hub 데이터 압축 해제 경로로 수정하세요.

---

## 핵심 기술

| 구분 | 내용 |
|------|------|
| 형태소 분석 | KoNLPy Okt — 명사/동사/형용사 추출, 불용어 제거 |
| 구어체 정규화 | 구어체 → 표준 의학 용어 변환 사전 (`COLLOQUIAL_MAP`) |
| TF-IDF 매칭 | `TfidfVectorizer` + 코사인 유사도 |
| BERT 매칭 | `jhgan/ko-sroberta-multitask`, 임베딩 사전 계산 후 `.npy` 캐싱 |
| 평가 지표 | Hit@1/3/5, MAP@5 (50개 Ground Truth, 소프트 매치 기준) |
| 시각화 | Plotly + React (Babel standalone) 인터랙티브 대시보드 |

---

## AWS 아키텍처

```
S3 (alices-project-storage)
└── pet-health-ai/
    ├── dashboard/index.html   ← 정적 대시보드 (현재 배포)
    └── data/
        ├── processed/         ← 전처리 CSV
        ├── embeddings/        ← BERT 임베딩 .npy
        └── splits/            ← Ground Truth
```

> 최종 발표 시 동적 검색(사용자 실시간 입력) 구현 예정: Lambda + S3 + BERT 추론

---

## 성능 요약

| 지표 | TF-IDF | Sentence-BERT | 향상 |
|------|--------|--------------|------|
| Hit@1 | 18.0% | **24.0%** | +6.0 %p |
| Hit@3 | 48.0% | **52.0%** | +4.0 %p |
| Hit@5 | 62.0% | 62.0% | ±0 %p |
| MAP@5 | 9.74% | **12.87%** | +3.13 %p |

> 평가 기준: 검증셋 50 queries (자견 17 · 성견 17 · 노령견 16), 소프트 매치

---

## 작성자

성균관대학교 데이터마이닝 과제 (2025)
