"""
디자인 시스템 — 단일 색상 소스.
chart_builder.py, run_dashboard.py 양쪽에서 import.
"""

# ── 블루 계열 팔레트 ──────────────────────────────────────────────────
BLUE_50  = "#eff6ff"
BLUE_100 = "#dbeafe"
BLUE_300 = "#93c5fd"
BLUE_500 = "#3b82f6"
BLUE_700 = "#1d4ed8"
BLUE_900 = "#1e3a8a"

# ── 시맨틱 컬러 ──────────────────────────────────────────────────────
SUCCESS = "#10b981"   # 초록 — 성공/긍정
WARNING = "#f59e0b"   # 앰버 — 경고/중간
DANGER  = "#ef4444"   # 빨강 — 위험/부정
NEUTRAL = "#94a3b8"   # 슬레이트 — 기본/기타

# ── 5-범주 팔레트 (진료과) ────────────────────────────────────────────
PALETTE = [
    "#3b82f6",  # 내과  — blue
    "#ef4444",  # 외과  — red
    "#10b981",  # 피부과 — green
    "#f59e0b",  # 안과  — amber
    "#8b5cf6",  # 치과  — purple
]

DEPT_COLORS = dict(zip(["내과", "외과", "피부과", "안과", "치과"], PALETTE))

# ── 생애주기 (파랑 순차 — 밝음→진함으로 성장 표현) ───────────────────
LC_COLOR = {
    "자견":   "#60a5fa",  # blue-400  — 밝고 어림
    "성견":   "#3b82f6",  # blue-500  — 표준
    "노령견": "#1e3a8a",  # blue-900  — 진하고 성숙
}

# ── 모델 비교 ────────────────────────────────────────────────────────
COLOR_BERT  = "#3b82f6"  # blue — BERT (더 나은 모델 → 강조)
COLOR_TFIDF = "#94a3b8"  # gray — TF-IDF (baseline)

# ── 기타 ─────────────────────────────────────────────────────────────
COLOR_OTHER   = NEUTRAL
COLOR_PRIMARY = BLUE_500

# ── 그라디언트 스케일 ─────────────────────────────────────────────────
HEATMAP_SCALE = [[0, BLUE_100], [0.5, BLUE_500], [1, BLUE_900]]
TREEMAP_SCALE = [BLUE_100, BLUE_300, BLUE_500, BLUE_700, BLUE_900]

# ── CSS 주입용 변수 문자열 생성 ────────────────────────────────────────
def build_css() -> str:
    lc = LC_COLOR
    return f"""<style id="theme-override">
/* ─ 모델 바 ─ */
.bar-fill.bert  {{ background: {COLOR_BERT}; }}
.bar-fill.tfidf {{ background: {COLOR_TFIDF}; }}
.bar-fill.solid {{ background: {COLOR_BERT}; }}
.sim-bar .fill  {{ background: {COLOR_BERT}; }}
.model-col.bert .model-col-head .swatch {{ background: {COLOR_BERT}; }}

/* ─ 생애주기 태그 ─ */
.tag.life-puppy  {{ color: {lc['자견']};   background: {lc['자견']}18;   border-color: {lc['자견']}40; }}
.tag.life-adult  {{ color: {lc['성견']};   background: {lc['성견']}18;   border-color: {lc['성견']}40; }}
.tag.life-senior {{ color: {lc['노령견']}; background: {lc['노령견']}18; border-color: {lc['노령견']}40; }}

/* ─ 다크모드 생애주기 태그 ─ */
:root[data-theme="dark"] .tag.life-puppy  {{ color: #93c5fd; background: #1e3a8a30; }}
:root[data-theme="dark"] .tag.life-adult  {{ color: #3b82f6; background: #1e3a8a40; }}
:root[data-theme="dark"] .tag.life-senior {{ color: #60a5fa; background: #1e3a8a55; }}

/* ─ 액센트 변수 오버라이드 ─ */
:root {{
  --accent-h: 217;
  --color-bert:  {COLOR_BERT};
  --color-tfidf: {COLOR_TFIDF};
  --success: {SUCCESS};
}}
</style>"""
