"""
채팅 포함 대시보드 서버 실행
사용법: python run_api.py

실행 후 브라우저에서 http://localhost:8000 접속
  - 기존 대시보드 탭 (데이터, EDA, 평가 등) + AI 채팅 탭
"""
import sys, json, webbrowser, time, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.chart_builder import ChartBuilder
from utils.app_builder import build_app_data
from utils.theme import build_css

APP  = Path(__file__).parent / "app"
SRC  = APP / "dashboard.html"
OUT  = APP / "dashboard_live.html"

print("데이터 로딩 중...")
dl = DataLoader()

print("APP_DATA 구성 중...")
APP_DATA = build_app_data(dl)

print("EDA 차트 생성 중...")
eda_charts = ChartBuilder(dl.corpus).build_all()
APP_DATA["eda"] = eda_charts
print(f"완료 — EDA 차트 {len(eda_charts)}개 · {len(json.dumps(APP_DATA)) // 1024}KB")

# dashboard_live.html 생성 (EDA 데이터 + 테마 CSS 포함)
html = SRC.read_text(encoding="utf-8")
theme_css = build_css()
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA, ensure_ascii=False)});"
    "</script>"
)
live_html = html.replace("</head>", theme_css + "\n</head>")
live_html = live_html.replace("</body>", override + "\n</body>")
OUT.write_text(live_html, encoding="utf-8")
print(f"dashboard_live.html 생성 완료")

# 브라우저 자동 열기 (서버 시작 후 1.5초 대기)
def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

threading.Thread(target=open_browser, daemon=True).start()

# FastAPI 서버 시작
print("서버 시작: http://localhost:8000")
import uvicorn
uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
