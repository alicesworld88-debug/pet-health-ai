"""
채팅 API 백엔드 — FastAPI
실행: python run_api.py  (또는 uvicorn app.api:app --reload)

보안:
  - API 키는 .env에만 존재, 응답에 절대 포함되지 않음
  - 브라우저는 /chat 엔드포인트만 호출 (키 미노출)
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chat import build_pipeline

# 검색기 종류 — Lambda는 함수별 env(RETRIEVER_TYPE)로 tfidf/bert 선택
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "tfidf")

app = FastAPI(title="반려견 증상 매칭 AI", docs_url=None, redoc_url=None)

# CORS — S3 대시보드에서도 호출 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 메인 파이프라인 — Lambda는 env(RETRIEVER_TYPE)로 선택, 지연 초기화 + 전역 캐싱
# (콜드스타트에 1회 빌드 → 웜 호출에서 재사용)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"파이프라인 초기화 중... ({RETRIEVER_TYPE})")
        _pipeline = build_pipeline(retriever_type=RETRIEVER_TYPE)
        print("초기화 완료")
    return _pipeline


# /chat/compare(4방향 비교 — 로컬 대시보드)용: tfidf·bert 둘 다 지연 초기화
# (Lambda 단일 함수에서는 호출되지 않으므로 무겁게 미리 만들지 않는다)
_pipe_tfidf = None
_pipe_bert = None


def _get_compare_pipelines():
    global _pipe_tfidf, _pipe_bert
    if _pipe_tfidf is None:
        _pipe_tfidf = build_pipeline(retriever_type="tfidf")
    if _pipe_bert is None:
        _pipe_bert = build_pipeline(retriever_type="bert")
    return _pipe_tfidf, _pipe_bert


# 비교용 지표 임베딩 모델 (lazy load — 첫 비교 요청 시 1회 로드)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("비교 지표용 BERT 임베딩 모델 로드 중...")
        _embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _embedder


# ── 요청/응답 스키마 ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str

class SourceDoc(BaseModel):
    input: str
    output: str
    score: float
    lifecycle: str = ""
    disease: str = ""

class ChatResponseBody(BaseModel):
    intent: str
    answer: str
    sources: list[SourceDoc]
    clarify_question: str | None = None   # VeNom 분산 증상 되묻기


class AnswerMetric(BaseModel):
    """답변 1건의 정량 지표."""
    relevance: float       # 질문 관련성 (질문↔답변 코사인)
    grounded: float        # 수의학 근거성 (답변↔참고답변 max 코사인)

class CompareResponseBody(BaseModel):
    """4방향 비교 — A.Gemini 단독 / B.TF-IDF 검색 / C.BERT 검색 / D.RAG."""
    intent: str
    gemini_only: str        # A. 검색 없이 LLM만
    tfidf_retrieval: str    # B. TF-IDF 검색 top-1 원문
    bert_retrieval: str     # C. BERT 검색 top-1 원문
    rag: str                # D. BERT 검색 + Gemini 생성 (메인 시스템)
    sources: list[SourceDoc]
    metrics: dict[str, AnswerMetric]   # key: gemini_only / tfidf_retrieval / bert_retrieval / rag


# ── 엔드포인트 ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pipeline": RETRIEVER_TYPE}


@app.post("/chat", response_model=ChatResponseBody)
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="query가 너무 깁니다 (최대 500자).")

    resp = get_pipeline().chat(query)

    # 응답에 API 키 등 민감 정보 절대 포함하지 않음
    return ChatResponseBody(
        intent=resp.intent,
        answer=resp.answer,
        clarify_question=resp.clarify_question,
        sources=[
            SourceDoc(input=d.input, output=d.output, score=d.score,
                      lifecycle=getattr(d, "lifecycle", ""), disease=getattr(d, "disease", ""))
            for d in resp.sources[:5]
        ],
    )


@app.post("/chat/compare", response_model=CompareResponseBody)
def chat_compare(req: ChatRequest):
    """
    같은 질문에 대해 3방향 답변을 한 번에 반환 — 검색(RAG)의 효과를 직접 비교.
        A. gemini_only   : 검색 없이 LLM만 (대조군)
        B. retrieval_only: BERT/TF-IDF 검색 top-1 수의사 답변 원문
        C. rag           : 검색 + 생성 (현재 챗봇)
    """
    from utils.generator import generate_answer_solo

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="query가 너무 깁니다 (최대 500자).")

    pipe_tfidf, pipe_bert = _get_compare_pipelines()
    # D. RAG (BERT 검색 + Gemini 생성) — 메인 시스템
    resp = pipe_bert.chat(query)
    bert_sources = resp.sources
    bert_retrieval = bert_sources[0].output if bert_sources else "(검색 결과 없음)"
    # B. TF-IDF 검색 top-1 원본 (corpus는 list[dict])
    t_ret    = pipe_tfidf.agents["symptom"].retriever
    t_corpus = pipe_tfidf.agents["symptom"].corpus
    t_idx, _ = t_ret.match(query, top_k=1)
    tfidf_retrieval = str(t_corpus[t_idx[0]]["output"]) if len(t_idx) else "(검색 결과 없음)"
    # A. Gemini 단독 (검색 없음)
    gemini_only = generate_answer_solo(query)

    # ── 정량 지표 (참고풀 = BERT 검색 결과, BERT 임베딩 코사인) ──
    from sentence_transformers import util
    emb = _get_embedder()
    q_emb = emb.encode(query, convert_to_tensor=True)
    ref_texts = [d.output for d in bert_sources]
    ref_embs = emb.encode(ref_texts, convert_to_tensor=True) if ref_texts else None

    def _metric(answer: str, exclude_top1: bool = False) -> AnswerMetric:
        a_emb = emb.encode(answer, convert_to_tensor=True)
        relevance = float(util.cos_sim(q_emb, a_emb).item())
        grounded = 0.0
        if ref_embs is not None:
            sims = util.cos_sim(a_emb, ref_embs)[0]
            # C(BERT 검색 원본)는 자기자신(top-1)이 참고풀에 포함 → 제외해 공정 비교
            if exclude_top1 and len(sims) > 1:
                grounded = float(sims[1:].max())
            else:
                grounded = float(sims.max())
        return AnswerMetric(relevance=round(relevance, 3), grounded=round(grounded, 3))

    metrics = {
        "gemini_only":     _metric(gemini_only),
        "tfidf_retrieval": _metric(tfidf_retrieval),
        "bert_retrieval":  _metric(bert_retrieval, exclude_top1=True),
        "rag":             _metric(resp.answer),
    }

    return CompareResponseBody(
        intent=resp.intent,
        gemini_only=gemini_only,
        tfidf_retrieval=tfidf_retrieval,
        bert_retrieval=bert_retrieval,
        rag=resp.answer,
        sources=[
            SourceDoc(input=d.input, output=d.output, score=d.score)
            for d in bert_sources[:3]
        ],
        metrics=metrics,
    )


# ── 대시보드 HTML 서빙 ────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """
    dashboard_live.html을 서빙합니다.
    run_api.py가 APP_DATA를 주입한 파일을 생성합니다.
    """
    live = ROOT / "app" / "dashboard_live.html"
    base = ROOT / "app" / "dashboard.html"
    target = live if live.exists() else base
    if not target.exists():
        return HTMLResponse(
            content="<h1>반려견 증상 매칭 AI API</h1><p>대시보드 파일이 없습니다. API는 /chat, /health 사용.</p>",
        )
    html = target.read_text(encoding="utf-8")
    # 같은 Lambda가 대시보드를 서빙하므로, 대시보드의 루트경로 호출(/chat, /chat/compare)에
    # 함수 prefix(/tfidf · /bert)를 자동으로 붙여 같은 함수의 API로 가도록 fetch를 보정한다.
    # (CDN 등 https 절대경로는 영향 없음 — '/'로 시작하는 경로만 보정)
    base_path = os.getenv("API_BASE_PATH", "").rstrip("/")
    if base_path:
        # 기본 prefix(base_path)를 루트 호출에 붙이되, 이미 /tfidf/ · /bert/ 로
        # 시작하는 명시 경로(검색 패널의 모델별 호출)는 그대로 둔다 → 이중 prefix 방지.
        patch = (
            "<script>(function(){var _f=window.fetch;window.fetch=function(u,o){"
            'if(typeof u==="string"&&u.charAt(0)==="/"&&!/^\\/(tfidf|bert)\\//.test(u))'
            f'u="{base_path}"+u;'
            "return _f(u,o);};})();</script>"
        )
        html = html.replace("</head>", patch + "\n</head>", 1)
    # 브라우저가 옛 대시보드를 캐싱하지 않도록 (갱신 즉시 반영)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache, must-revalidate"})
