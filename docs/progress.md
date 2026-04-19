# 프로젝트 진행 현황

**과제**: 데이터마이닝 최종 보고서  
**주제**: AI Hub 반려견 질병 말뭉치 기반 생애주기별 분석 및 유사 질문 매칭  
**최종 업데이트**: 2026-04-19

---

## 노트북 진행 현황

| 단계 | 노트북 | 상태 | 완료일 | 커밋 |
|------|--------|------|--------|------|
| 데이터 수집 | `01_data_collection.ipynb` | ✅ 완료 | 2026-04-19 | `85f2cc3` |
| 데이터 검증 | `02_data_validation.ipynb` | ✅ 완료 | 2026-04-19 | `12c931a` |
| 전처리 | `03_preprocessing.ipynb` | ✅ 완료 | 2026-04-19 | `5f1ab13` |
| EDA | `04_eda.ipynb` | ✅ 완료 | 2026-04-19 | `e200175`, `cc3a13e` |
| Ground Truth 구축 | `05_ground_truth.ipynb` | ✅ 완료 | 2026-04-19 | `746a015` |
| 매칭 실험 | `06_matching.ipynb` | ✅ 완료 | 2026-04-19 | `6dd132d` |
| 성능 평가 | `07_evaluation.ipynb` | ✅ 완료 | 2026-04-19 | `1ffa28b` |
| 서비스 앱 | `app/streamlit_app.py` | ✅ 완료 | 2026-04-19 | `86033bf` |

---

## 교수 피드백 반영 현황

| 피드백 항목 | 반영 여부 | 구현 위치 |
|------------|----------|----------|
| AI Hub 데이터 확보 | ✅ 완료 | 01_data_collection.ipynb |
| TF-IDF 한계 인식 및 기술 | ✅ 반영 | 07_evaluation.ipynb 섹션 9 |
| Sentence-BERT 비교 실험 | ✅ 완료 | 06_matching.ipynb, 07_evaluation.ipynb |
| Ground Truth 평가셋 구축 | ✅ 완료 | 05_ground_truth.ipynb (50개 쿼리) |
| 생애주기별 분포 분석 | ✅ 완료 | 04_eda.ipynb |
| 데이터 검증 및 전처리 문서화 | ✅ 완료 | 02, 03 노트북 |

---

## AWS 마이그레이션 현황 (Pattern C)

| 항목 | 상태 | 비고 |
|------|------|------|
| IAM + S3 버킷 생성 | ⬜ 예정 | `pet-health-ai-data` |
| 원본 JSON → S3 업로드 | ⬜ 예정 | `raw/training/`, `raw/validation/` |
| 전처리 CSV → S3 업로드 | ⬜ 예정 | `processed/` |
| SageMaker: BERT 임베딩 생성 | ⬜ 예정 | `embeddings/db_embeddings.npy` |
| EC2 t2.micro: Streamlit 배포 | ⬜ 예정 | 포트 8501, systemd |
| Lambda: 스케줄 트리거 | ⬜ 예정 | 주간 실행 |

> 마이그레이션 상세 가이드: [aws_migration.md](aws_migration.md)

---

## 데이터 현황

| 항목 | 값 |
|------|-----|
| Training | 19,418개 JSON |
| Validation | 2,427개 JSON |
| 전체 | 21,845개 |
| 진료과 | 내과 / 안과 / 외과 / 치과 / 피부과 |
| lifeCycle | 자견 / 성견 / 노령견 |
| Ground Truth 평가셋 | 50개 쿼리 (생애주기별 균등 분포) |

---

## 생성된 산출물

| 파일 | 설명 |
|------|------|
| `data/processed/corpus_raw.csv` | 원본 통합 데이터 |
| `data/processed/corpus_validated.csv` | 검증 완료 데이터 |
| `data/processed/corpus_preprocessed.csv` | 전처리 완료 데이터 |
| `data/processed/matching_results.csv` | TF-IDF / BERT 매칭 결과 |
| `data/processed/evaluation_summary.csv` | 성능 지표 요약 |
| `data/processed/embeddings/db_embeddings.npy` | BERT 임베딩 캐시 |
| `data/processed/eda_figures/` | EDA 시각화 이미지 / HTML |
| `data/processed/eval_figures/` | 성능 비교 시각화 |
| `data/splits/ground_truth.csv` | 평가셋 쿼리-정답 쌍 |
