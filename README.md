# pet-health-ai

> AI Hub 반려견 Q&A 말뭉치 기반 생애주기 건강 분석 및 의미 매칭 시스템

[![Status](https://img.shields.io/badge/status-in_progress-yellow)]()
[![Course](https://img.shields.io/badge/course-Data_Mining-blue)]()

---

## 프로젝트 개요

반려견 보호자가 자연어로 증상을 입력하면, 수의사 Q&A 말뭉치에서 의미적으로 유사한 답변을 추천하는 시스템을 구축합니다. **TF-IDF**와 **Sentence-BERT** 두 가지 매칭 방식의 성능을 비교하여, 의미 기반 검색의 실질적 효용을 검증합니다.

---

## 연구 질문 (RQ)

1. 생애주기(자견/성견/노령견)별 주요 질병 빈도 분포와 특징은 무엇인가?
2. Sentence-BERT 임베딩 기반 매칭은 TF-IDF 코사인 유사도보다 의미적 유사도를 더 잘 반영하는가?
3. 사용자 입력 질문과 기존 데이터 간 유사도 매칭을 통해 보호자에게 실질적으로 유의미한 정보를 제공할 수 있는가?

---

## 접근 방법

| 단계 | 설명 | 도구 |
|------|------|------|
| 1. 데이터 수집 | AI Hub 5개 진료과 Q&A 통합 | Python, pandas |
| 2. 전처리 | 결측치 처리, 형태소 분석, 불용어 제거 | KoNLPy (Okt) |
| 3. EDA | 진료과별/생애주기별 분포 분석 | matplotlib, plotly |
| 4. 매칭 비교 | TF-IDF vs Sentence-BERT | scikit-learn, sentence-transformers |
| 5. 시각화 | 워드클라우드, 매칭 결과 비교 차트 | wordcloud, plotly |
| 6. 배포 (예정) | 인터랙티브 검색 데모 | Streamlit |

---

## 데이터

- **출처**: AI Hub - 반려견 성장 및 질병관련 말뭉치
- **구성**: 내과, 안과, 외과, 치과, 피부과 5개 진료과 Q&A
- **라벨**: lifeCycle, disease, department

> 데이터는 AI Hub 라이선스에 따라 본 레포에 포함되지 않습니다.

---

## 폴더 구조

```
pet-health-ai/
├── notebooks/          # Jupyter 분석 노트북
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_eda.ipynb
├── utils/              # 재사용 함수 모듈
├── data/               # 원본 데이터 (gitignore)
├── reports/            # 중간/최종 보고서 PDF
├── .env.example        # API 키 템플릿
├── requirements.txt
└── README.md
```

---

## 설치 및 실행

```bash
git clone https://github.com/alicesworld88-debug/pet-health-ai.git
cd pet-health-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## 진행 상황

### 완료
- [x] **W02~W05** - W02 워크시트 작성
- [x] **W06** - 교수님 피드백 수령 (Sentence-BERT 비교 권장)
- [x] **W07** - 피드백 반영 및 GitHub 셋업

### 진행 중 (W08 중간고사 대비)
- [ ] 데이터 수집 노트북 (01_data_collection.ipynb)
- [ ] 전처리 파이프라인 (02_preprocessing.ipynb)
- [ ] EDA 및 시각화 (03_eda.ipynb)
- [ ] 중간 보고서 PDF

### 예정 (W09 이후)
- [ ] TF-IDF vs Sentence-BERT 매칭 실험
- [ ] 온톨로지 기반 지식 그래프 설계 (확장)
- [ ] 네이버 지식iN API 보조 데이터 수집
- [ ] Streamlit 데모 개발 및 최종 발표

---

## 작성자

**정은영** (2025720370)  
성균관대학교 빅데이터 대학원
