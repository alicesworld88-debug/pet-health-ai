"""
네이버 지식iN 수집 데이터로 intent classifier 학습
저장된 분류기: data/processed/intent_classifier.pkl
"""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_PROCESSED = Path("data/processed")
NAVER_DATA = Path("data/external/naver_questions.csv")
MODEL_PATH = DATA_PROCESSED / "intent_classifier.pkl"

# 1. 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv(NAVER_DATA)
print(f"총 {len(df):,}건 로드됨")
print(df["intent"].value_counts())

# 2. 결측치 + 중복 제거 (데이터 품질 — 완전중복·query중복 제거)
df = df.dropna(subset=["query", "intent"])
_before = len(df)
df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
print(f"중복 제거: {_before:,} → {len(df):,} (-{_before-len(df):,})")
X = df["query"].tolist()
y = df["intent"].tolist()

# 3. 파이프라인 구축: TF-IDF + LogisticRegression
print("\n분류기 학습 중...")
clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )),
    ("lr", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
    )),
])

clf.fit(X, y)

# 4. 성능 평가 (전체 데이터 기준)
train_pred = clf.predict(X)
accuracy = (train_pred == y).sum() / len(y)
print(f"\n정확도: {accuracy:.1%}")
print("\n분류 리포트:")
print(classification_report(y, train_pred, target_names=['emergency', 'symptom', 'treatment']))

# 5. 저장
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)
print(f"\n✓ 저장: {MODEL_PATH}")

# 6. 테스트
print("\n테스트 예제:")
test_queries = [
    "강아지가 초콜릿을 먹었어요",
    "우리 개 기침이 심해요",
    "슬개골 수술 비용이 얼마예요",
]
for q in test_queries:
    pred = clf.predict([q])[0]
    print(f"  '{q}' → {pred}")
