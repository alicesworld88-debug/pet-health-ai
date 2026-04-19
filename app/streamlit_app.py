"""
반려견 증상 매칭 서비스 — Streamlit 앱
로컬: streamlit run app/streamlit_app.py
EC2:  systemd로 8501 포트 서비스 (docs/aws_migration.md Step 5 참고)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import DATA_PROCESSED

# ── 경로 설정 ─────────────────────────────────────────────────────────
PREPROCESSED_PATH = DATA_PROCESSED / "corpus_preprocessed.csv"
EMBED_PATH        = DATA_PROCESSED / "embeddings" / "db_embeddings.npy"
SBERT_MODEL       = "jhgan/ko-sroberta-multitask"


# ── 리소스 로드 (캐싱) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 및 데이터 로딩 중...")
def load_resources():
    df_db = pd.read_csv(PREPROCESSED_PATH)
    df_db = df_db[df_db["split"] == "train"].reset_index(drop=True)

    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_db["input_tokens"].fillna(""))

    if EMBED_PATH.exists():
        db_embeddings = np.load(EMBED_PATH)
    else:
        model         = SentenceTransformer(SBERT_MODEL)
        db_embeddings = model.encode(
            df_db["input_normalized"].fillna("").tolist(),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBED_PATH, db_embeddings)

    sbert_model = SentenceTransformer(SBERT_MODEL)
    return df_db, vectorizer, tfidf_matrix, db_embeddings, sbert_model


# ── 매칭 함수 ─────────────────────────────────────────────────────────
def match_tfidf(query, vectorizer, tfidf_matrix, top_k=5):
    vec    = vectorizer.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx, scores[idx]


def match_sbert(query, sbert_model, db_embeddings, top_k=5):
    q_emb  = sbert_model.encode([query], normalize_embeddings=True)[0]
    scores = (db_embeddings @ q_emb).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return idx, scores[idx]


def show_results(indices, scores, df, label, color):
    st.markdown(f"#### {label}")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = df.iloc[idx]
        with st.expander(
            f"**{rank}위** | {row['lifeCycle']} · {row.get('department','')} · {row.get('disease','')}  |  유사도 {score:.3f}",
            expanded=(rank == 1),
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**질문**")
                st.write(row["input"])
            with c2:
                st.markdown("**수의사 답변**")
                st.write(row["output"])


# ── UI ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="반려견 증상 매칭 서비스",
    page_icon="🐾",
    layout="wide",
)

st.title("🐾 반려견 증상 매칭 서비스")
st.caption("증상을 입력하면 유사한 수의사 Q&A를 찾아드립니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio(
        "검색 모드",
        ["🔀 TF-IDF vs BERT 비교", "Sentence-BERT 단독", "TF-IDF 단독"],
        index=0,
    )
    top_k = st.slider("결과 개수", min_value=1, max_value=5, value=3)
    lc_filter = st.selectbox(
        "생애주기 필터 (선택)",
        ["전체", "자견", "성견", "노령견"],
        index=0,
    )
    st.divider()
    st.caption("모델: jhgan/ko-sroberta-multitask")
    st.caption("데이터: AI Hub 반려견 질병 말뭉치")

df_db, vectorizer, tfidf_matrix, db_embeddings, sbert_model = load_resources()

if lc_filter != "전체":
    mask         = df_db["lifeCycle"] == lc_filter
    df_search    = df_db[mask].reset_index(drop=True)
    search_embed = db_embeddings[df_db[mask].index.values]
    search_tfidf = tfidf_matrix[df_db[mask].index.values]
else:
    df_search    = df_db
    search_embed = db_embeddings
    search_tfidf = tfidf_matrix

query = st.text_area(
    "증상을 입력하세요",
    placeholder="예: 강아지가 밥을 안 먹고 자꾸 토해요. 3일째 기운이 없어요.",
    height=100,
)

search_btn = st.button("🔍 검색", type="primary")

# ── 검색 실행 ─────────────────────────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("검색 중..."):
        tf_idx, tf_scores  = match_tfidf(query, vectorizer, search_tfidf, top_k)
        sb_idx, sb_scores  = match_sbert(query, sbert_model, search_embed, top_k)

    if mode == "🔀 TF-IDF vs BERT 비교":
        st.subheader("📊 TF-IDF vs Sentence-BERT 비교")
        st.caption("같은 질문에 대해 두 방법이 어떻게 다른 결과를 내는지 확인하세요.")
        col_tf, col_sb = st.columns(2)
        with col_tf:
            show_results(tf_idx, tf_scores, df_search, "🔵 TF-IDF", "blue")
        with col_sb:
            show_results(sb_idx, sb_scores, df_search, "🟠 Sentence-BERT", "orange")

    elif mode == "Sentence-BERT 단독":
        show_results(sb_idx, sb_scores, df_search, "🟠 Sentence-BERT 결과", "orange")

    else:
        show_results(tf_idx, tf_scores, df_search, "🔵 TF-IDF 결과", "blue")

elif search_btn:
    st.warning("증상을 입력해주세요.")

st.divider()
st.caption("데이터마이닝 과제 · 2025")
