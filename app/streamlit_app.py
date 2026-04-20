"""반려견 증상 매칭 서비스 · Streamlit iframe wrapper"""
import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from utils.data_loader import DataLoader
from utils.app_builder import build_app_data
from utils.theme import build_css

st.set_page_config(
    page_title="반려견 증상 매칭 · Vet Match",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Streamlit chrome 제거 + iframe 전체 화면 리사이즈 ───────────────
st.html("""<script>
(function(){
  var p=window.parent.document;
  if(p.getElementById("v-fix"))return;
  var s=p.createElement("style");s.id="v-fix";
  s.textContent=
    "#MainMenu,footer,header,[data-testid='stHeader'],[data-testid='stToolbar'],"
    +"[data-testid='stSidebar'],[data-testid='stSidebarNav'],"
    +".stDeployButton,[data-testid='stDecoration'],[data-testid='stStatusWidget']{display:none!important}"
    +"section.main>div.block-container,"
    +"[data-testid='stMainBlockContainer']"
    +"{padding:0!important;margin:0!important;max-width:100%!important;width:100%!important}"
    +"body{padding:0!important;margin:0!important}"
    +".stApp{overflow:hidden!important;background:#fafafa!important}";
  p.head.appendChild(s);
  function fix(){
    var h=window.parent.innerHeight;
    p.querySelectorAll("iframe").forEach(function(f){
      if(parseInt(f.getAttribute("height")||0)>400){
        f.style.setProperty("height", h+"px", "important");
        f.style.setProperty("width",  "100vw", "important");
        f.style.setProperty("border", "none",  "important");
        f.style.setProperty("display","block",  "important");
        f.style.setProperty("margin", "0",      "important");
      }
    });
  }
  setTimeout(fix,200);setTimeout(fix,800);setTimeout(fix,2000);
  window.parent.addEventListener("resize",fix);
})();
</script>""")

# ─── 데이터 로드 및 APP_DATA 구성 ───────────────────────────────────
@st.cache_data(show_spinner="데이터 로딩 중...")
def load_app_data():
    dl = DataLoader()
    return build_app_data(dl)

APP_DATA = load_app_data()

# ─── dashboard.html 로드 → theme CSS + APP_DATA 주입 → 렌더 ─────────
DASHBOARD = Path(__file__).parent / "dashboard.html"
html = DASHBOARD.read_text(encoding="utf-8")

theme_css = build_css()
override = (
    "<script>"
    f"window.APP_DATA=Object.assign(window.APP_DATA||{{}},{json.dumps(APP_DATA, ensure_ascii=False)});"
    "</script>"
)
html = html.replace("</head>", theme_css + "\n</head>")
html = html.replace("</body>", override + "\n</body>")

components.html(html, height=900, scrolling=False)
