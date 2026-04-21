# 연구 도메인 개념 모델

보고서 ① 프로젝트 개요 — 1.4절 "연구 도메인 개념 모델" 참고용

---

## 클래스 정의

| 클래스 | 설명 | 예시 |
|--------|------|------|
| `Dog` | 반려견 개체 | 개별 환자 |
| `LifeCycle` | 생애주기 단계 | 자견 / 성견 / 노령견 |
| `Disease` | 질병명 | 파보바이러스, 슬개골 탈구 |
| `Symptom` | 증상 키워드 | 구토, 설사, 파행, 식욕부진 |
| `Department` | 진료과 | 내과 / 외과 / 피부과 / 안과 / 치과 |
| `Question` | 보호자 질문 (Q) | input 필드 |
| `Answer` | 수의사 답변 (A) | output 필드 |

---

## 관계 정의

| 관계 | 도메인 → 범위 | 설명 |
|------|-------------|------|
| `hasLifeCycle` | Dog → LifeCycle | 반려견은 특정 생애주기에 속함 |
| `hasDepartment` | Disease → Department | 질병은 특정 진료과에서 다뤄짐 |
| `hasSymptom` | Disease → Symptom | 질병은 증상을 가짐 |
| `commonIn` | Disease → LifeCycle | 질병은 특정 생애주기에 많이 발생 |
| `mentions` | Question → Symptom | 질문은 증상 키워드를 언급 |
| `diagnoses` | Answer → Disease | 답변은 질병을 진단/설명 |
| `relatedTo` | Question → Answer | 질문-답변 쌍 연결 |

---

## 다이어그램 구조 (draw.io 작성 참고)

```
         Dog
          |
    hasLifeCycle
          |
       LifeCycle ←── commonIn ──── Disease
                                      |
                              hasDepartment  hasSymptom
                                      |          |
                                 Department   Symptom
                                                 |
                                              mentions
                                                 |
                              Question ──── relatedTo ──── Answer
                                                               |
                                                           diagnoses
                                                               |
                                                           Disease
```

---

## draw.io 작성 가이드

1. draw.io 접속 → 새 다이어그램
2. 클래스: 사각형(Rectangle) — 클래스명 입력
3. 관계: 화살표(Arrow) — 관계명 레이블 추가
4. 색상 구분:
   - **파란색**: 핵심 클래스 (Dog, Disease, Symptom)
   - **초록색**: 메타 클래스 (LifeCycle, Department)
   - **주황색**: 데이터 클래스 (Question, Answer)

---

## 보고서 ① 개요 서술 예시

> 본 연구에서 다루는 반려견 질병 도메인은 위 개념 모델로 정의된다.  
> 핵심 개념은 `Dog`, `LifeCycle`, `Disease`, `Symptom` 네 클래스이며,  
> `hasSymptom` 관계를 통해 질병과 증상이 연결된다.  
> TF-IDF가 포착하지 못하는 동의어(`구토` ↔ `토해요`)와 상위 개념(`파행` ↔ `절뚝`)은  
> 모두 개념 모델의 `Symptom` 클래스 내 인스턴스 관계로 설명 가능하며,  
> 이것이 Sentence-BERT 도입의 개념적 근거이다.

---

## 보고서 ⑤ 인사이트 TF-IDF 한계 서술 예시

| 실패 유형 | TF-IDF 매칭 예시 | 개념 모델 관점 해석 |
|----------|----------------|-----------------|
| Symptom 동의어 | "밥 안 먹어요" ↔ "사료 거부" 미매칭 | 동일 Symptom 인스턴스, 표현만 다름 |
| 상위 개념 | "관절 아파요" ↔ "절뚝거려요" 미매칭 | hasSymptom 관계로 연결된 동일 Disease |
| Disease-Symptom | "구토+설사+혈변" ↔ "파보바이러스" 미매칭 | mentions → diagnoses 관계 무시 |
