# TODO — 데이터마이닝 최종 보고서 준비

**최종 업데이트**: 2026-05-31  
**과목**: 데이터마이닝 (성균관대 대학원 빅데이터학과)  
**현재 상태**: 중간보고서 제출 완료 → 교수님 질문 4건 + 동료 피드백 반영 + RAG 파이프라인 개발 진행 중

---

## 1단계: 즉시 할 것

### 환경 설정

- [ ] **conda 환경 의존성 충돌 해결** — `pet-health` 환경에서 PyTorch/NumPy 버전 충돌 발생. `requirements.txt`가 `numpy>=2.3.3`을 요구하는데 sentence-transformers 구버전이 NumPy 1.x를 요구할 수 있음
  - 해결 방법: `sentence-transformers>=3.0.0` 고정 + `torch>=2.4` + `numpy>=2.0` 조합으로 `environment.yml` 작성
  - `pip install --upgrade sentence-transformers torch numpy`로 최신 버전 통일 시도
- [ ] **`.env` 파일 설정** — `utils/generator.py`가 `VERTEX_API_KEY`, `VERTEX_MODEL` 환경변수 필요. `.env` 파일에 Gemini API 키 추가
- [ ] **`chat.py` + `utils/generator.py` 커밋** — 두 파일이 아직 미커밋 상태. GitHub에 push (public repo: https://github.com/alicesworld88-debug/pet-health-ai)

### Q1: Hit@1 통계적 유의성 검정 (McNemar test) — 교수 질문 1

- [ ] **McNemar test 구현** — TF-IDF vs BERT의 Hit@1 (18% vs 24%)이 통계적으로 유의한지 검정. Ground Truth 50개 쿼리 기준으로 각 모델의 맞춤/틀림 이진 벡터 생성 후 McNemar 검정 실행
  - 입력: `data/splits/ground_truth.csv`, `data/processed/matching_results.csv`
  - `scipy.stats.mcnemar()` 사용, 2×2 분할표 구성 (둘 다 맞춤 / TF-IDF만 맞춤 / BERT만 맞춤 / 둘 다 틀림)
- [ ] **Paired Bootstrap 구현** (McNemar 보완) — 50개 샘플에서 1000회 부트스트랩 재샘플링으로 BERT 우위의 95% 신뢰구간 계산
  - n=50은 소표본이므로 McNemar + Bootstrap 병행 권장
- [ ] **결과를 `08_statistical_test.ipynb`에 정리** — 검정 통계량, p-value, 신뢰구간, 해석 포함

---

## 2단계: 단기 과제 (교수님 Q2/Q3/Q4 + 동료 피드백)

### Q2: 성견 TF-IDF 우위 가설 정량 검증 — 교수 질문 2

- [ ] **성견 구간 명사 비율 분석** — `corpus_preprocessed.csv`에서 생애주기별 질문 텍스트의 명사/전체 토큰 비율 계산. 성견이 다른 생애주기보다 명사 비율이 높다면 TF-IDF 우위 설명 가능
  - KoNLPy Okt 품사 태깅 활용, 생애주기별 boxplot 시각화
- [ ] **동의어 빈도 분석** — COLLOQUIAL_MAP에 등재된 구어체 표현이 생애주기별 질문에 얼마나 자주 등장하는지 카운트. 성견 구간에서 구어체 빈도가 낮다면 가설 지지
- [ ] **구어체 표현 빈도 비교** — 생애주기별 `input` 텍스트에서 구어체 지표어(ㅠㅠ, ~해요, ~같아요 등) 빈도 분포 비교
- [ ] **결과를 `09_adult_dog_analysis.ipynb`에 정리** — 표 + 시각화 + 해석 포함

### Q3: BERT 추론 비용 측정 — 교수 질문 3

- [ ] **레이턴시(Latency) 측정** — 동일 쿼리 100회 반복 기준 TF-IDF vs BERT 평균/중앙값/p95 응답시간 측정
  - `time.perf_counter()` 또는 `timeit` 사용
  - BERT는 임베딩 캐시 있을 때와 없을 때 모두 측정 (캐시 히트 vs 콜드 스타트)
- [ ] **메모리 사용량 측정** — `tracemalloc` 또는 `memory_profiler`로 TF-IDF 행렬 vs BERT 임베딩(.npy) 메모리 점유 비교
  - TF-IDF 행렬: sparse matrix 크기 (21,604 × vocabulary)
  - BERT 임베딩: 21,604 × 768 float32 = 약 63MB 예상
- [ ] **결과를 `10_cost_analysis.ipynb`에 정리** — 표 형식(TF-IDF vs BERT: 응답시간, 메모리, 초기화 시간) + 언제 어떤 모델을 쓸지 권고

### Q4: Query Intent Classifier 구현 — 교수 질문 4

- [ ] **2-stage 시스템 설계 문서 작성** — Query → Intent Classifier → (증상/처치/응급) → 각 경로별 Retriever 전략
  - `chat.py`의 `classify_intent()`는 규칙 기반으로 이미 뼈대 구현됨. 이를 ML 분류기로 고도화
- [ ] **Naver 지식iN 데이터 수집** — 실제 보호자 질문 30~50건 수집, 증상/처치/응급으로 수동 레이블링
  - 수집 방법: Naver 지식iN 검색 (반려견 증상, 반려견 수술, 반려견 응급) → 복붙 or requests 크롤링
  - 파일 위치: `data/external/naver_questions.csv` (컬럼: query, intent)
- [ ] **Intent Classifier 노트북 작성** (`11_intent_classifier.ipynb`)
  - 방법 A (규칙 기반, 이미 구현): `chat.py`의 `_EMERGENCY_KEYWORDS`, `_TREATMENT_KEYWORDS` 확장
  - 방법 B (ML 기반): TF-IDF + Logistic Regression 또는 BERT fine-tuning (30~50건으로 소규모 fine-tuning)
  - 평가: 수집한 30~50건에 대해 accuracy, confusion matrix 출력

### 동료 피드백 반영

- [ ] **COLLOQUIAL_MAP 근거 문서화** (안명현, 이용균, 장진환 질문) — 현재 16개 규칙의 출처·선정 기준을 `docs/colloquial_map_rationale.md`에 정리
  - 각 규칙이 어떤 실제 데이터 패턴에서 나왔는지 예시 포함
- [ ] **COLLOQUIAL_MAP 확장** — 16개 → 30개 이상으로 확장. `notebooks/03_preprocessing.ipynb` 에서 미매핑 구어체 표현 목록 추출 후 추가
- [ ] **"언제 어떤 모델이 더 나은가" 결론 명확화** (손훤아 피드백) — Q3 비용 측정 결과를 바탕으로 명확한 권고 작성
  - 예: "실시간 응답 필요 + 자견 쿼리 → BERT / 배치 처리 + 성견 쿼리 → TF-IDF"
- [ ] **성능 수치 맥락 설명 추가** (장진환, 양세미 피드백) — Hit@1 18% vs 24%가 낮아 보일 수 있으므로 태스크 난이도 설명 추가 (21,604건 중 정확한 1건 매칭, 소프트 매치 기준)
  - 보고서 3절 또는 결론에 베이스라인 비교 또는 태스크 특성 설명 1~2문장 추가

---

## 3단계: 개발 (RAG 파이프라인 + 온톨로지 + 채팅 UI)

### 온톨로지 + BERT 융합

- [ ] **온톨로지 구조 정의** — 도메인 개념 모델(`docs/concept_model.md`) 기반으로 질병/증상/생애주기 간 관계를 JSON 또는 dict 형태로 코드화
  - 파일 위치: `utils/ontology.py`
- [ ] **Query Expansion 구현** — 입력 쿼리에 COLLOQUIAL_MAP 정규화 + 온톨로지 동의어 확장 적용 후 검색
  - `chat.py`의 `retrieve()` 전처리 단계에 삽입
- [ ] **Ontology Re-ranking 구현** — BERT Top-K 결과를 온톨로지 관계(disease 카테고리, lifeCycle 일치)로 재정렬
  - `ChatPipeline.retrieve()` 반환값에 re-rank 로직 추가

### RAG 파이프라인 완성

- [ ] **`utils/generator.py` API 엔드포인트 검증** — Gemini 2.5 Flash Lite API URL 형식 확인 (현재 `aiplatform.googleapis.com` 엔드포인트 사용 중, Vertex AI vs Google AI Studio 구분 필요)
  - Vertex AI: 프로젝트 ID + 리전 필요 → `VERTEX_PROJECT_ID`, `VERTEX_LOCATION` 환경변수 추가 필요
  - Google AI Studio (더 간단): `generativelanguage.googleapis.com` 엔드포인트 + API 키만 필요
- [ ] **`chat.py` 엔드-투-엔드 테스트** — 실제 쿼리 5건으로 전체 파이프라인 동작 확인 (query → intent → retrieve → generate → ChatResponse)
- [ ] **응급 intent 처리 강화** — `emergency` intent일 때 답변 첫 줄에 "즉시 동물병원 방문을 권장합니다" 문구 강제 삽입 (안전성 강화)
- [ ] **`app/streamlit_app.py` 채팅 UI 연동** — 기존 대시보드에 채팅 탭 추가 또는 별도 Streamlit 채팅 페이지 구성
  - `ChatPipeline.chat()` 호출 → `ChatResponse.answer` 표시 + `ChatResponse.sources` 참고 출처 표시

### GitHub 정리

- [ ] **`chat.py`, `utils/generator.py` 커밋 및 push** — 커밋 메시지 한글, 코드/변수명 영어 (conventions.md 준수)
- [ ] **`requirements.txt` 업데이트** — `requests`, `python-dotenv`는 이미 있음. `memory_profiler` (Q3 측정용) 추가 여부 검토
- [ ] **README.md 아키텍처 섹션 업데이트** — 현재 파이프라인 다이어그램에 RAG 흐름 (BERT → Gemini) 반영

---

## 참고: 파일 위치 요약

| 작업 | 파일 |
|------|------|
| 통계 검정 (Q1) | `notebooks/08_statistical_test.ipynb` (신규 생성) |
| 성견 분석 (Q2) | `notebooks/09_adult_dog_analysis.ipynb` (신규 생성) |
| 비용 측정 (Q3) | `notebooks/10_cost_analysis.ipynb` (신규 생성) |
| Intent 분류기 (Q4) | `notebooks/11_intent_classifier.ipynb` (신규 생성) |
| 지식iN 데이터 | `data/external/naver_questions.csv` (신규 생성) |
| COLLOQUIAL_MAP 근거 | `docs/colloquial_map_rationale.md` (신규 생성) |
| 온톨로지 코드 | `utils/ontology.py` (신규 생성) |
| 채팅 파이프라인 | `chat.py` (기존, 미커밋) |
| 생성기 | `utils/generator.py` (기존, 미커밋) |
