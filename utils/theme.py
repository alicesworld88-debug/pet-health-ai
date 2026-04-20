"""
디자인 시스템 — 단일 색상 소스.
chart_builder.py, run_dashboard.py 양쪽에서 import.
"""

# ── 청록 계열 팔레트 ──────────────────────────────────────────────────
TEAL_50  = "#F0FDFA"
TEAL_100 = "#CCFBF1"
TEAL_300 = "#86EFCA"
TEAL_500 = "#2A9D8F"
TEAL_600 = "#0F766E"
TEAL_900 = "#134E4A"

# ── 시맨틱 컬러 ──────────────────────────────────────────────────────
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER  = "#EF4444"
NEUTRAL = "#94a3b8"

# ── UI 공통 ──────────────────────────────────────────────────────────
TEXT_COLOR  = "#111827"
HOVER_BG    = "#FFFFFF"
GRID_COLOR  = "#E5E7EB"

# ── 5-범주 팔레트 (진료과) ────────────────────────────────────────────
PALETTE = [
    "#2A9D8F",  # 내과  — teal
    "#E76F51",  # 외과  — terracotta
    "#F4A261",  # 피부과 — apricot
    "#264653",  # 안과  — deep teal
    "#F59E0B",  # 치과  — amber
]

DEPT_COLORS = dict(zip(["내과", "외과", "피부과", "안과", "치과"], PALETTE))

# ── 생애주기 ─────────────────────────────────────────────────────────
LC_COLOR = {
    "자견":   "#F4A261",  # apricot
    "성견":   "#2A9D8F",  # teal
    "노령견": "#264653",  # deep teal
}

# ── 모델 비교 ────────────────────────────────────────────────────────
COLOR_BERT       = "#2A9D8F"  # teal  — BERT
COLOR_TFIDF      = "#E76F51"  # terracotta — TF-IDF
COLOR_VALIDATION = TEAL_300   # 연청록 — val split

# ── 기타 ─────────────────────────────────────────────────────────────
COLOR_OTHER   = NEUTRAL
COLOR_PRIMARY = TEAL_500

# ── 그라디언트 스케일 ─────────────────────────────────────────────────
HEATMAP_SCALE = [[0, TEAL_100], [0.5, TEAL_500], [1, TEAL_900]]
TREEMAP_SCALE = [TEAL_100, TEAL_300, TEAL_500, TEAL_600, TEAL_900]

# ── CSS 주입용 변수 문자열 생성 ────────────────────────────────────────
def build_css() -> str:
    lc = LC_COLOR
    return f"""<style id="theme-override">
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
.tag.life-senior {{ color: {lc['노령견']}; background: {lc['노령견']}20; border-color: {lc['노령견']}60; }}

/* ─ 다크모드 생애주기 태그 ─ */
:root[data-theme="dark"] .tag.life-puppy  {{ color: #FCD34D; background: #92400E30; }}
:root[data-theme="dark"] .tag.life-adult  {{ color: #5EEAD4; background: {TEAL_600}30; }}
:root[data-theme="dark"] .tag.life-senior {{ color: #99F6E4; background: #26465355; }}

/* ─ 액센트 변수 오버라이드 ─ */
:root {{
  --accent-h: 174;
  --color-bert:  {COLOR_BERT};
  --color-tfidf: {COLOR_TFIDF};
  --success: {SUCCESS};
}}
</style>"""
