"""Plotly 차트 생성 모듈 — theme.py에서 색상 import."""
import json
import pandas as pd
import plotly.graph_objects as go

from utils.theme import (
    LC_COLOR, COLOR_PRIMARY,
    HEATMAP_SCALE_DISCRETE, WARNING,
    TEXT_COLOR, AXIS_COLOR, HOVER_BG, GRID_COLOR,
)

_DEPTS = ["내과", "외과", "피부과", "안과", "치과"]
_LIVES = ["자견", "성견", "노령견"]
_OPACITY = 0.85


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────

def _layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="'Pretendard Variable','Inter',sans-serif", size=13, color=TEXT_COLOR),
        hoverlabel=dict(bgcolor=HOVER_BG, font_size=13),
        margin=dict(t=16, b=40, l=60, r=20),
    )
    base.update(kwargs)
    return base


def _axis(title="", **kwargs) -> dict:
    return dict(
        title=title,
        tickfont=dict(size=12, color=AXIS_COLOR),
        title_font=dict(size=12, color=AXIS_COLOR),
        gridcolor=GRID_COLOR,
        **kwargs,
    )


def _to_json(fig) -> dict:
    return json.loads(fig.to_json())


def _heatmap_annotations(z, x_labels, y_labels, threshold=0.65, size=11) -> list:
    """셀 밝기에 따라 텍스트 색 자동 전환 (진한 배경→흰색, 연한 배경→검정)."""
    flat = [v for row in z for v in row]
    max_val = max(flat) if flat else 1
    anns = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            color = "#FFFFFF" if (val / max_val) > threshold else "#1F2937"
            anns.append(dict(
                x=x_labels[j], y=y_labels[i],
                text=f"{int(val):,}",
                showarrow=False,
                font=dict(size=size, color=color, weight=700,
                          family="'Pretendard Variable','Inter',sans-serif"),
            ))
    return anns


def _crosstab_heatmap(df, row_col, col_col, row_order, col_order,
                      x_title, height, margin, ann_size,
                      exclude_row_val=None, top_n=None) -> dict:
    """범용 crosstab 히트맵 — row/col 컬럼과 순서만 넘기면 재사용 가능."""
    if exclude_row_val:
        df = df[df[row_col] != exclude_row_val]
    if top_n:
        row_order = df[row_col].value_counts().head(top_n).index.tolist()

    ct = (pd.crosstab(df[row_col], df[col_col])
            .reindex(row_order).reindex(columns=col_order).fillna(0))
    z = ct.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=col_order, y=row_order,
        colorscale=HEATMAP_SCALE_DISCRETE, opacity=_OPACITY,
        xgap=3, ygap=3,
        hovertemplate="%{y} × %{x}: %{z:,}건<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(**_layout(
        height=height, margin=margin,
        xaxis=dict(showgrid=False, zeroline=False, **_axis(x_title)),
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, **_axis()),
        annotations=_heatmap_annotations(z, col_order, row_order, size=ann_size),
    ))
    return _to_json(fig)


def _lifecycle_boxplot(df, col, y_title, hover_label) -> go.Figure:
    """생애주기별 박스플롯 공통 생성."""
    fig = go.Figure()
    for lc, color in LC_COLOR.items():
        vals = df[df["lifeCycle"] == lc][col].str.len().tolist()
        fig.add_trace(go.Box(
            y=vals, name=lc, marker_color=color, opacity=_OPACITY,
            boxpoints="outliers", jitter=0.3, pointpos=-1.8,
            hovertemplate=f"{lc} — {hover_label}: %{{y}}자<extra></extra>",
        ))
    fig.update_layout(**_layout(
        height=300,
        yaxis=_axis(y_title),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                    font=dict(size=12, color=AXIS_COLOR)),
    ))
    return fig


# ── ChartBuilder ─────────────────────────────────────────────────────────────

class ChartBuilder:
    """Plotly 차트 팩토리 — EDA 6종."""

    def __init__(self, corpus: pd.DataFrame):
        self.df = corpus

    # ── 1: 생애주기 × 질병 Top 히트맵 ────────────────────────────────────
    def lifecycle_disease_heatmap(self, top_n: int = 12) -> dict:
        return _crosstab_heatmap(
            self.df, row_col="disease", col_col="lifeCycle",
            row_order=None, col_order=_LIVES, x_title="생애주기",
            height=420, margin=dict(t=16, b=40, l=120, r=60), ann_size=12,
            exclude_row_val="기타", top_n=top_n,
        )

    # ── 2: 질문 길이 박스플롯 ─────────────────────────────────────────────
    def text_boxplot(self) -> dict:
        return _to_json(_lifecycle_boxplot(self.df, "input", "질문 길이 (chars)", "질문 길이"))

    # ── 3: 질문 길이 히스토그램 ──────────────────────────────────────────
    def question_len_histogram(self) -> dict:
        fig = go.Figure()
        for lc, color in LC_COLOR.items():
            vals = self.df[self.df["lifeCycle"] == lc]["input"].str.len().tolist()
            fig.add_trace(go.Histogram(
                x=vals, name=lc, marker_color=color, opacity=0.65,
                nbinsx=60,
                hovertemplate=f"{lc}: %{{y}}건<extra></extra>",
            ))
        fig.update_layout(**_layout(
            height=280, barmode="overlay",
            xaxis=_axis("질문 길이 (chars)"),
            yaxis=_axis("빈도"),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                        font=dict(size=12, color=AXIS_COLOR)),
        ))
        return _to_json(fig)

    # ── 4: 진료과별 문서 수 + 평균 질문 길이 ────────────────────────────
    def dept_dual_axis(self) -> dict:
        stats = (self.df.groupby("department")
                 .agg(cnt=("input", "count"),
                      avg_q=("input", lambda x: round(x.str.len().mean(), 1)))
                 .reindex(_DEPTS))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats.index.tolist(), y=stats["cnt"].tolist(),
            name="문서 수", marker_color=COLOR_PRIMARY, opacity=_OPACITY,
            yaxis="y", hovertemplate="%{x}: %{y:,}건<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stats.index.tolist(), y=stats["avg_q"].tolist(),
            name="평균 질문 길이", mode="lines+markers",
            marker=dict(size=9, color=WARNING, symbol="circle"),
            line=dict(color=WARNING, width=2.5),
            opacity=_OPACITY,
            yaxis="y2", hovertemplate="%{x}: %{y:.1f}자<extra></extra>",
        ))
        fig.update_layout(**_layout(
            height=300,
            yaxis=_axis("문서 수"),
            yaxis2=dict(title="", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(size=12, color=AXIS_COLOR)),
            annotations=[dict(
                xref="paper", yref="paper", x=1.0, y=-0.15,
                text="우축: 평균 질문 길이 (chars)",
                showarrow=False, xanchor="right",
                font=dict(size=11, color=TEXT_COLOR),
            )],
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                        font=dict(size=12, color=AXIS_COLOR)),
            margin=dict(t=16, b=40, l=60, r=70),
        ))
        return _to_json(fig)

    # ── 5: 진료과 × 질병 Top 히트맵 ─────────────────────────────────────
    def dept_disease_heatmap(self, top_n: int = 10) -> dict:
        return _crosstab_heatmap(
            self.df, row_col="disease", col_col="department",
            row_order=None, col_order=_DEPTS, x_title="진료과",
            height=340, margin=dict(t=16, b=50, l=130, r=60), ann_size=11,
            exclude_row_val="기타", top_n=top_n,
        )

    # ── 6: 트리맵 계층 데이터 (React용 — Plotly JSON 아님) ───────────────
    def treemap_data(self) -> dict:
        # 진료과 타일 크기: 전체 레코드 기준 (기타 포함) → 올바른 비율
        dept_totals = (self.df.groupby(["lifeCycle", "department"])
                       .size().reset_index(name="total"))
        # 상위 질병: 기타 제외
        disease_g = (self.df[self.df["disease"] != "기타"]
                     .groupby(["lifeCycle", "department", "disease"])
                     .size().reset_index(name="cnt"))
        result = {}
        for lc, lc_df in dept_totals.groupby("lifeCycle"):
            result[lc] = {}
            for _, row in lc_df.iterrows():
                dept = row["department"]
                top4_df = disease_g[
                    (disease_g["lifeCycle"] == lc) &
                    (disease_g["department"] == dept)
                ]
                top4 = top4_df.nlargest(4, "cnt")[["disease", "cnt"]].values.tolist()
                result[lc][dept] = {
                    "total": int(row["total"]),
                    "top": [[str(d), int(c)] for d, c in top4],
                }
        return result

    # ── 전체 빌드 ─────────────────────────────────────────────────────────
    def build_all(self) -> dict:
        steps = [
            ("① 생애주기 × 질병 히트맵",     "lifecycle_disease_heatmap", self.lifecycle_disease_heatmap),
            ("② 질문 길이 박스플롯",           "text_boxplot",              self.text_boxplot),
            ("③ 질문 길이 히스토그램",         "question_len_histogram",    self.question_len_histogram),
            ("④ 진료과 문서 수 + 질문 길이",   "dept_dual_axis",            self.dept_dual_axis),
            ("⑤ 진료과 × 질병 히트맵",        "dept_disease_heatmap",      self.dept_disease_heatmap),
            ("⑥ 트리맵 데이터",               "treemap_data",              self.treemap_data),
        ]
        charts = {}
        for label, key, fn in steps:
            print(f"  {label}...")
            charts[key] = fn()
        return charts
