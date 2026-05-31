"""
채팅 API 백엔드 — FastAPI
실행: python run_api.py  (또는 uvicorn app.api:app --reload)

보안:
  - API 키는 .env에만 존재, 응답에 절대 포함되지 않음
  - 브라우저는 /chat 엔드포인트만 호출 (키 미노출)
"""
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from chat import build_pipeline

app = FastAPI(title="반려견 증상 매칭 AI", docs_url=None, redoc_url=None)

# CORS — S3 대시보드에서도 호출 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 파이프라인 초기화 (서버 시작 시 1회)
print("파이프라인 초기화 중... (TF-IDF)")
_pipeline = build_pipeline(retriever_type="tfidf")
print("초기화 완료")


# ── 요청/응답 스키마 ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str

class SourceDoc(BaseModel):
    input: str
    output: str
    score: float

class ChatResponseBody(BaseModel):
    intent: str
    answer: str
    sources: list[SourceDoc]
    clarify_question: str | None = None   # VeNom 분산 증상 되묻기


# ── 엔드포인트 ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "tfidf"}


@app.post("/chat", response_model=ChatResponseBody)
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")
    if len(query) > 500:
        raise HTTPException(status_code=400, detail="query가 너무 깁니다 (최대 500자).")

    resp = _pipeline.chat(query)

    # 응답에 API 키 등 민감 정보 절대 포함하지 않음
    return ChatResponseBody(
        intent=resp.intent,
        answer=resp.answer,
        clarify_question=resp.clarify_question,
        sources=[
            SourceDoc(input=d.input, output=d.output, score=d.score)
            for d in resp.sources[:3]
        ],
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
    return HTMLResponse(content=target.read_text(encoding="utf-8"))
