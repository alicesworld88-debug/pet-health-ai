"""
AWS Lambda 진입점 — Mangum으로 FastAPI(app.api:app)를 ASGI 어댑트.

두 함수(Tfidf/Bert)가 동일 핸들러를 공유하며, 차이는 env로만:
  - RETRIEVER_TYPE: tfidf | bert  (app.api가 읽음)
  - API_BASE_PATH:  /tfidf | /bert (HTTP API 프리픽스 제거용)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mangum import Mangum

from app.api import app

# API Gateway HTTP API에서 /tfidf · /bert 프리픽스를 떼어내 FastAPI는 /chat·/health만 본다.
handler = Mangum(app, api_gateway_base_path=os.getenv("API_BASE_PATH", "/"))
