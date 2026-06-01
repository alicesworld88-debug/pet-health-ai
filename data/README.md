# 데이터 명세 (Data Specification)

본 문서는 프로젝트에서 사용·생성된 모든 데이터의 **출처, 라이선스, 규모, 스키마, 용도**를 정리한다.
원천 데이터의 재배포 라이선스 제한과 GitHub 파일 용량 제한(100MB)에 따라, 데이터는 **GitHub 저장소**와 **AWS S3(객체 스토리지)**에 분산 배포한다.

---

## 1. 데이터 출처 및 라이선스

| 출처 | 데이터 | 라이선스 | 배포 |
|------|--------|---------|------|
| **AI Hub** (반려동물 질병 데이터) | 수의사 Q&A 코퍼스 (원천·전처리), BERT 임베딩 | AI Hub 이용약관 — **재배포 라이선스 제한** | S3 (출처 표기) |
| **네이버 지식iN** (Open API) | 보호자 질문 114,439건 | 비영리·연구 목적 수집 | GitHub |
| **본 연구 산출물** | 학습 모델·평가 지표·시각화·Ground Truth | 프로젝트 자체 생성 | GitHub |

> ⚠️ AI Hub 원천 데이터는 라이선스상 재배포가 제한되므로 GitHub에 포함하지 않으며, S3에 별도 보관하고 원 출처 이용약관 준수를 전제로 제공한다.

---

## 2. 디렉토리 구조

```
data/
├── external/
│   ├── naver_questions.csv      # [GitHub] 네이버 지식iN 수집 질문 (114,439건)
│   └── naver_coverage.csv       # [GitHub] 지식iN ↔ 코퍼스 커버리지 분석 결과
├── processed/
│   ├── corpus_raw.csv           # [S3] AI Hub 원천 Q&A 코퍼스
│   ├── corpus_validated.csv     # [S3] 결측·중복·이상치 처리 후
│   ├── corpus_preprocessed.csv  # [S3] 형태소 분석·구어체 정규화 완료 (117MB)
│   ├── intent_classifier.pkl    # [GitHub] 학습된 intent 분류기 (TF-IDF+LR)
│   ├── evaluation_summary.csv   # [GitHub] GT 50건 평가 지표
│   ├── full_evaluation_summary.csv  # [GitHub] Validation 2,399건 평가 지표
│   ├── matching_results.csv     # [GitHub] GT 매칭 결과
│   ├── full_matching_results.csv    # [GitHub] 전체 매칭 결과
│   ├── anova_result.csv         # [GitHub] 진료과×질문길이 ANOVA
│   ├── class_distribution.png   # [GitHub] 클래스 분포
│   ├── eda_figures/             # [GitHub] EDA 시각화 산출물 (png·html)
│   └── embeddings/              # [S3] 사전 계산 BERT 임베딩 (.npy)
│       ├── db_embeddings.npy        #   학습 코퍼스 임베딩 (56MB)
│       ├── full_embeddings.npy      #   전체 코퍼스 임베딩 (63MB)
│       └── val_embeddings.npy       #   Validation 임베딩 (7MB)
└── splits/
    └── ground_truth.csv         # [GitHub] 수동 평가용 쿼리 50건
```

---

## 3. GitHub 저장소 포함 데이터

### 3.1 네이버 지식iN 수집 데이터 — `external/naver_questions.csv`
- **규모**: 114,439행 (symptom 60,301 / emergency 27,408 / treatment 26,730)
- **스키마**: `query`(보호자 질문 텍스트) · `intent`(증상/처치/응급 라벨) · `source`(수집 출처)
- **용도**: intent 분류기 학습, RAG 평가 입력, 커버리지 분석
- **수집 방법**: 네이버 지식iN Open API (`search/kin.json`), 반려견 키워드 필터링

### 3.2 모델 산출물 — `processed/intent_classifier.pkl`
- **구성**: `TfidfVectorizer(ngram 1-2, max_features 5000)` + `LogisticRegression(class_weight=balanced)`
- **성능**: 정확도 84.2%, F1(macro) 0.83
- **용도**: 채팅 파이프라인의 의도 분류 (`chat.py`)

### 3.3 평가 지표 결과
| 파일 | 내용 |
|------|------|
| `evaluation_summary.csv` | Ground Truth 50건 기준 Hit@1/3/5·MAP@5 (TF-IDF vs BERT) |
| `full_evaluation_summary.csv` | Validation 2,399건 전체 평가 + McNemar 검정 |
| `matching_results.csv` / `full_matching_results.csv` | 쿼리별 top-k 매칭 결과 (성공/실패) |
| `anova_result.csv` | 진료과(5군) × 질문 길이 One-way ANOVA |

### 3.4 평가용 Ground Truth — `splits/ground_truth.csv`
- **규모**: 수동 구축 쿼리 50건
- **스키마**: 쿼리 텍스트 + 정답 disease·lifeCycle 라벨
- **용도**: Hit@k·MAP@5 평가 기준

### 3.5 EDA 시각화 산출물 — `processed/eda_figures/`
- 생애주기·진료과·질병 분포, 텍스트 길이 박스플롯, 워드클라우드, 선버스트/히트맵(html)

---

## 4. S3 (객체 스토리지) 배포 데이터

> 버킷: `s3://alices-project-storage/pet-health-ai/data/`
> AI Hub 원천 데이터 및 대용량 임베딩. **AI Hub 이용약관 준수를 전제로 제공.**

| 파일 | 규모 | 스키마 / 내용 | 용도 |
|------|------|--------------|------|
| `corpus_raw.csv` | 37MB | `lifeCycle·department·disease·input·output` | AI Hub 원천 Q&A |
| `corpus_validated.csv` | 37MB | 위 + 정제 | 결측·중복·이상치 처리 후 |
| `corpus_preprocessed.csv` | 117MB | 위 + `input_clean·output_clean·input_tokens·input_normalized` | 형태소·구어체 정규화 완료 (검색 코퍼스 21,604건) |
| `embeddings/db_embeddings.npy` | 56MB | `(N, 768)` float32 | 학습 코퍼스 BERT 임베딩 |
| `embeddings/full_embeddings.npy` | 63MB | `(21604, 768)` float32 | 전체 코퍼스 BERT 임베딩 |
| `embeddings/val_embeddings.npy` | 7MB | `(2399, 768)` float32 | Validation BERT 임베딩 |

**다운로드 (AWS CLI)**
```bash
aws s3 sync s3://alices-project-storage/pet-health-ai/data/ ./data/
```

---

## 5. 데이터 처리 파이프라인

```
AI Hub 원천 JSON
   │  01_data_collection
   ▼
corpus_raw.csv  ──02_validation──▶  corpus_validated.csv
   │  03_preprocessing (형태소·구어체 정규화 COLLOQUIAL_MAP 34개)
   ▼
corpus_preprocessed.csv  ──BERT 인코딩──▶  embeddings/*.npy
   │
   ▼
06_matching · 07~08_evaluation  ──▶  *_evaluation_summary.csv

네이버 지식iN Open API  ──▶  naver_questions.csv  ──▶  intent_classifier.pkl
                                          └──▶  커버리지 분석  ──▶  naver_coverage.csv
```

재현 절차는 루트 [`README.md`](../README.md)의 분석 파이프라인(노트북 01~10) 참조.
