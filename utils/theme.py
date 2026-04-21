"""
디자인 시스템 — 단일 색상 소스.
chart_builder.py, run_dashboard.py 양쪽에서 import.
"""

# ── 청록 계열 팔레트 ──────────────────────────────────────────────────
TEAL_50  = "#F0FDFA"
TEAL_100 = "#CCFBF1"
TEAL_300 = "#86EFCA"
TEAL_500 = "#3A9188"
TEAL_600 = "#0F766E"
TEAL_900 = "#134E4A"

# ── 시맨틱 컬러 ──────────────────────────────────────────────────────
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER  = "#EF4444"
NEUTRAL = "#6B7280"

# ── UI 공통 ──────────────────────────────────────────────────────────
TEXT_COLOR  = "#1F2937"   # 차트 제목·수치 — 거의 검정
AXIS_COLOR  = "#4B5563"   # 축 라벨·범례 — 진회색
HOVER_BG    = "#FFFFFF"
GRID_COLOR  = "#E5E0D8"

# ── 5-범주 팔레트 (진료과) ────────────────────────────────────────────
PALETTE = [
    "#3A9188",  # 내과  — deep teal
    "#E07856",  # 외과  — deep coral
    "#F4A261",  # 피부과 — apricot
    "#6B708F",  # 안과  — lavender grey
    "#F59E0B",  # 치과  — amber
]

DEPT_COLORS = dict(zip(["내과", "외과", "피부과", "안과", "치과"], PALETTE))

# ── 생애주기 ─────────────────────────────────────────────────────────
LC_COLOR = {
    "자견":   "#E89A54",  # 진한 살구
    "성견":   "#3A9188",  # 진한 틸
    "노령견": "#6B708F",  # 라벤더 그레이
}

# ── 모델 비교 ────────────────────────────────────────────────────────
COLOR_BERT       = "#3A9188"  # deep teal — BERT
COLOR_TFIDF      = "#E07856"  # deep coral — TF-IDF
COLOR_VALIDATION = TEAL_300   # 연청록 — val split

# ── 기타 ─────────────────────────────────────────────────────────────
COLOR_OTHER   = NEUTRAL
COLOR_PRIMARY = TEAL_500

# ── 그라디언트 스케일 ─────────────────────────────────────────────────
HEATMAP_SCALE = [[0, TEAL_100], [0.5, TEAL_500], [1, TEAL_900]]
# 이산형 5단계 — 각 구간을 동일 색으로 묶어 범례를 읽기 쉽게
HEATMAP_SCALE_DISCRETE = [
    [0.00, TEAL_100], [0.20, TEAL_100],
    [0.20, "#A8E6D9"], [0.40, "#A8E6D9"],
    [0.40, TEAL_300],  [0.60, TEAL_300],
    [0.60, TEAL_500],  [0.80, TEAL_500],
    [0.80, TEAL_900],  [1.00, TEAL_900],
]
TREEMAP_SCALE = [TEAL_100, TEAL_300, TEAL_500, TEAL_600, TEAL_900]

# ── CSS 주입용 변수 문자열 생성 ────────────────────────────────────────
def build_css() -> str:
    lc = LC_COLOR
    return f"""<style id="theme-override">
/* ─ 전역 배경 (라이트모드) ─ */
:root {{ --bg-canvas-override: #FAFAF9; }}
:root:not([data-theme="dark"]) body, :root:not([data-theme="dark"]) .shell {{ background: #FAFAF9 !important; }}
:root[data-theme="dark"] body {{ background: var(--bg) !important; }}

/* ─ 모델 바 ─ */
.bar-fill.bert  {{ background: {COLOR_BERT}; }}
.bar-fill.tfidf {{ background: {COLOR_TFIDF}; }}
.bar-fill.solid {{ background: {COLOR_BERT}; }}
.sim-bar .fill  {{ background: {COLOR_BERT}; }}
.model-col.bert  .model-col-head {{ background: {COLOR_BERT}18; color: {TEAL_600}; border-color: transparent; }}
.model-col.tfidf .model-col-head {{ background: {COLOR_TFIDF}15; color: #9D4B30; border-color: transparent; }}
.model-col.bert  .model-col-head .swatch {{ background: {COLOR_BERT}; }}
.model-col.tfidf .model-col-head .swatch {{ background: {COLOR_TFIDF}; }}
.model-col.tfidf .sim-bar .fill {{ background: {COLOR_TFIDF}; }}

/* ─ 생애주기 태그 ─ */
.tag.life-puppy  {{ color: #9A3412; background: #FEF3C7; border-color: {lc['자견']}60; }}
.tag.life-adult  {{ color: {TEAL_600}; background: {lc['성견']}20; border-color: {lc['성견']}50; }}
.tag.life-senior {{ color: #5B5E72; background: {lc['노령견']}20; border-color: {lc['노령견']}60; }}

/* ─ 다크모드 생애주기 태그 ─ */
:root[data-theme="dark"] .tag.life-puppy  {{ color: #FCD34D; background: #92400E30; }}
:root[data-theme="dark"] .tag.life-adult  {{ color: #5EEAD4; background: {TEAL_600}30; }}
:root[data-theme="dark"] .tag.life-senior {{ color: #C0C3D8; background: #6B708F30; }}

/* ─ 선버스트 hover 팝아웃 ─ */
.sunburstlayer path {{
  transition: filter 0.18s ease;
  cursor: pointer;
}}
.sunburstlayer path:hover {{
  filter: brightness(1.15) drop-shadow(0 2px 6px rgba(0,0,0,0.25));
}}

/* ─ 액센트 변수 오버라이드 ─ */
:root {{
  --accent-h: 174;
  --color-bert:  {COLOR_BERT};
  --color-tfidf: {COLOR_TFIDF};
  --success: {SUCCESS};
}}
</style>"""
