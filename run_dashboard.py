"""
실데이터를 dashboard.html에 주입하고 브라우저로 바로 엽니다.
사용법: python run_dashboard.py
"""
import sys, json, webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.chart_builder import ChartBuilder
from utils.app_builder import build_app_data
from utils.theme import build_css

APP = Path(__file__).parent / "app"
SRC = APP / "dashboard.html"
OUT = APP / "dashboard_live.html"

print("📂 데이터 로딩 중...")
dl = DataLoader()

print("📊 APP_DATA 구성 중...")
APP_DATA = build_app_data(dl)

print("📊 EDA 차트 생성 중...")
eda_charts = ChartBuilder(dl.corpus).build_all()
print(f"   ✓ 차트 {len(eda_charts)}개 생성 완료")
APP_DATA["eda"] = eda_charts

print(f"✅ 데이터 구성 완료 — 쿼리 {len(APP_DATA['evalQueries'])}개 · {len(json.dumps(APP_DATA))//1024}KB")

# dashboard_live.html: EDA 포함 전체 데이터 주입
html = SRC.read_text(encoding="utf-8")
theme_css = build_css()
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA,ensure_ascii=False)});"
    "</script>"
)
live_html = html.replace("</head>", theme_css + "\n</head>")
live_html = live_html.replace("</body>", override + "\n</body>")
OUT.write_text(live_html, encoding="utf-8")
print(f"📄 dashboard_live.html 저장: {OUT}")

# dashboard.html: 실데이터 인라인 업데이트 (EDA 차트 제외)
DATA_LIGHT = {k: v for k, v in APP_DATA.items() if k != "eda"}
src_html = SRC.read_text(encoding="utf-8")
marker = "<script>\nwindow.APP_DATA = {"
if marker in src_html:
    start = src_html.index(marker)
    end = src_html.index("</script>", start) + len("</script>")
    new_block = "<script>\nwindow.APP_DATA = " + json.dumps(DATA_LIGHT, ensure_ascii=False, indent=2) + ";\n</script>"
    SRC.write_text(src_html[:start] + new_block + src_html[end:], encoding="utf-8")
    print(f"📝 dashboard.html 실데이터 업데이트 완료 ({len(json.dumps(DATA_LIGHT))//1024}KB)")

print("🌐 브라우저에서 열기...")
webbrowser.open(f"file://{OUT.resolve()}")
print("✅ 완료!")
