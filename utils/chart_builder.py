"""Plotly 차트 생성 모듈 — theme.py에서 색상 import."""
import json
import collections
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.theme import (
    PALETTE, LC_COLOR, COLOR_OTHER, COLOR_PRIMARY,
    HEATMAP_SCALE, TREEMAP_SCALE, SUCCESS, WARNING, DANGER,
    COLOR_BERT, COLOR_TFIDF, NEUTRAL,
)

_DEPTS = ["내과", "외과", "피부과", "안과", "치과"]
_LIVES = ["자견", "성견", "노령견"]


def _layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Pretendard Variable','Inter',sans-serif", size=12, color="#09090b"),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12),
        margin=dict(t=16, b=40, l=60, r=20),
    )
    base.update(kwargs)
    return base


def _to_json(fig) -> dict:
    return json.loads(fig.to_json())


class ChartBuilder:
    """재사용 가능한 Plotly 차트 팩토리."""

    def __init__(self, corpus: pd.DataFrame):
        self.df = corpus

    # ── 차트 1: 질병 롱테일 수평 막대 ────────────────────────────────────

    def disease_bar(self, top_n: int = 15) -> dict:
        dis = self.df["disease"].value_counts().head(top_n)
        # 1위는 강조색, 나머지는 단계적으로 흐려지는 blue
        colors = []
        for i, n in enumerate(dis.index):
            if n == "기타":
                colors.append(COLOR_OTHER)
            elif i == 0:
                colors.append(COLOR_PRIMARY)
            else:
                opacity = max(0.35, 1.0 - i * 0.05)
                colors.append(f"rgba(59,130,246,{opacity:.2f})")
        fig = go.Figure(go.Bar(
            x=dis.values.tolist(), y=dis.index.tolist(), orientation="h",
            marker_color=colors,
            text=[f"{v:,}" for v in dis.values], textposition="outside",
            hovertemplate="%{y}: %{x:,}건<extra></extra>",
        ))
        fig.update_layout(**_layout(
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis=dict(title="건수", gridcolor="#f1f5f9"), height=360,
        ))
        return _to_json(fig)

    # ── 차트 2: 생애주기 도넛 ────────────────────────────────────────────

    def lifecycle_pie(self) -> dict:
        lc = self.df["lifeCycle"].value_counts().reindex(_LIVES).dropna()
        fig = go.Figure(go.Pie(
            labels=lc.index.tolist(), values=lc.values.tolist(), hole=0.52,
            marker_colors=[LC_COLOR.get(l, "#aaa") for l in lc.index],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,}건 (%{percent})<extra></extra>",
        ))
        fig.update_layout(**_layout(
            height=340, margin=dict(t=16, b=50, l=20, r=20),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            annotations=[dict(text=f"{len(self.df):,}", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=18), xanchor="center")],
        ))
        return _to_json(fig)

    # ── 차트 3: 진료과 × 생애주기 히트맵 ────────────────────────────────

    def dept_heatmap(self) -> dict:
        ct = pd.crosstab(self.df["lifeCycle"], self.df["department"])
        z  = [[int(ct.loc[lc, d]) if lc in ct.index and d in ct.columns else 0
               for d in _DEPTS] for lc in _LIVES]
        fig = go.Figure(go.Heatmap(
            z=z, x=_DEPTS, y=_LIVES,
            colorscale=HEATMAP_SCALE,
            text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:,}건<extra></extra>", showscale=True,
        ))
        fig.update_layout(**_layout(
            height=280, margin=dict(t=16, b=50, l=70, r=60),
            xaxis=dict(title="진료과"), yaxis=dict(title="생애주기"),
        ))
        return _to_json(fig)

    # ── 차트 4: 텍스트 길이 박스플롯 ─────────────────────────────────────

    def text_boxplot(self) -> dict:
        fig = go.Figure()
        for lc, color in LC_COLOR.items():
            vals = self.df[self.df["lifeCycle"] == lc]["input"].str.len().tolist()
            fig.add_trace(go.Box(
                y=vals, name=lc, marker_color=color,
                boxpoints="outliers", jitter=0.3, pointpos=-1.8,
                hovertemplate=f"{lc} — 길이: %{{y}}자<extra></extra>",
            ))
        fig.update_layout(**_layout(
            height=280,
            yaxis=dict(title="문자 수 (chars)"),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        ))
        return _to_json(fig)

    # ── 차트 5: 계층 선버스트 ────────────────────────────────────────────

    def sunburst(self) -> dict:
        sb = (self.df.groupby(["lifeCycle", "department", "disease"])
              .size().reset_index(name="cnt"))
        sb = (sb.sort_values("cnt", ascending=False)
              .groupby(["lifeCycle", "department"]).head(4)
              .reset_index(drop=True))
        fig = px.sunburst(sb, path=["lifeCycle", "department", "disease"],
                          values="cnt", color_discrete_sequence=PALETTE)
        fig.update_traces(hovertemplate="%{label}: %{value:,}건<extra></extra>",
                          textfont_size=12)
        fig.update_layout(**_layout(height=440, margin=dict(t=16, b=16, l=16, r=16)))
        return _to_json(fig)

    # ── 차트 6: 진료과별 Top 5 질병 ─────────────────────────────────────

    def dept_top5(self) -> dict:
        dd = (self.df[self.df["disease"] != "기타"]
              .groupby(["department", "disease"]).size().reset_index(name="cnt"))
        dd = (dd.sort_values("cnt", ascending=False)
              .groupby("department").head(5).reset_index(drop=True))
        fig = px.bar(dd, x="cnt", y="disease", color="department",
                     orientation="h", barmode="group",
                     category_orders={"department": _DEPTS},
                     color_discrete_sequence=PALETTE,
                     labels={"cnt": "건수", "disease": "질병", "department": "진료과"})
        fig.update_layout(**_layout(
            height=400,
            margin=dict(t=16, b=60, l=60, r=20),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        ))
        return _to_json(fig)

    # ── 차트 7: 생애주기별 빈출 2-gram 트리맵 ────────────────────────────

    def word_treemap(self) -> dict:
        STOPWORDS = {
            "있다", "하다", "되다", "이다", "않다", "없다", "그", "이", "저", "것", "수",
            "더", "도", "을", "를", "위해", "통해", "에서", "으로", "에게", "부터",
            "까지", "같다", "보다", "대해", "따라",
        }
        rows = []
        for lc in _LIVES:
            subset  = self.df[self.df["lifeCycle"] == lc]["input_tokens"].dropna()
            counter: collections.Counter = collections.Counter()
            for ts in subset:
                toks = [t for t in str(ts).split()
                        if len(t) >= 2 and not t.isdigit() and t not in STOPWORDS]
                for a, b in zip(toks, toks[1:]):
                    counter[f"{a} {b}"] += 1
            for bigram, cnt in counter.most_common(20):
                rows.append({"lifeCycle": lc, "bigram": bigram, "count": cnt})
        if not rows:
            return {}
        word_df = pd.DataFrame(rows)
        fig = px.treemap(
            word_df, path=["lifeCycle", "bigram"], values="count",
            color="count",
            color_continuous_scale=TREEMAP_SCALE,
            range_color=[word_df["count"].min(), word_df["count"].max()],
        )
        fig.update_traces(
            hovertemplate="%{label}: %{value:,}회<extra></extra>",
            textinfo="label+value", textfont_size=12,
        )
        fig.update_coloraxes(colorbar=dict(title="빈도", thickness=12, len=0.7,
                                           tickfont=dict(size=10)))
        fig.update_layout(**_layout(height=420, margin=dict(t=16, b=16, l=16, r=16)))
        return _to_json(fig)

    # ── 차트 8: Train / Val 생애주기 분할 ────────────────────────────────

    def train_val_bar(self) -> dict:
        tv = self.df.groupby(["split", "lifeCycle"]).size().reset_index(name="cnt")
        fig = px.bar(
            tv, x="lifeCycle", y="cnt", color="split", barmode="stack",
            color_discrete_map={"train": "#a8a4f0", "val": "#7dd4d4"},
            category_orders={"lifeCycle": _LIVES, "split": ["train", "val"]},
            labels={"cnt": "건수", "lifeCycle": "생애주기", "split": "분할"},
        )
        fig.update_layout(**_layout(
            height=280,
            xaxis=dict(title="생애주기"), yaxis=dict(title="건수"),
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        ))
        return _to_json(fig)

    # ── 차트 9: 질문 vs 답변 길이 산점도 ─────────────────────────────────

    def len_scatter(self, sample_n: int = 2000) -> dict:
        sample = self.df.sample(min(sample_n, len(self.df)), random_state=42).copy()
        sample["q_len"] = sample["input"].str.len()
        sample["a_len"] = sample["output"].str.len()
        fig = px.scatter(
            sample, x="q_len", y="a_len", color="lifeCycle",
            color_discrete_map=LC_COLOR, opacity=0.45,
            labels={"q_len": "질문 길이 (chars)", "a_len": "답변 길이 (chars)",
                    "lifeCycle": "생애주기"},
            category_orders={"lifeCycle": _LIVES},
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**_layout(
            height=300,
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        ))
        return _to_json(fig)

    # ── 차트 10: 진료과별 문서 수 + 평균 질문 길이 (dual-axis) ────────────

    def dept_dual_axis(self) -> dict:
        stats = (self.df.groupby("department")
                 .agg(cnt=("input", "count"),
                      avg_q=("input", lambda x: round(x.str.len().mean(), 1)),
                      avg_a=("output", lambda x: round(x.str.len().mean(), 1)))
                 .reindex(_DEPTS))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats.index.tolist(), y=stats["cnt"].tolist(),
            name="문서 수", marker_color=COLOR_PRIMARY, opacity=0.85,
            yaxis="y", hovertemplate="%{x}: %{y:,}건<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stats.index.tolist(), y=stats["avg_q"].tolist(),
            name="평균 질문 길이", mode="lines+markers",
            marker=dict(size=9, color=WARNING, symbol="circle"),
            line=dict(color=WARNING, width=2.5),
            yaxis="y2", hovertemplate="%{x}: %{y:.1f}자<extra></extra>",
        ))
        fig.update_layout(**_layout(
            height=300,
            yaxis=dict(title="문서 수", gridcolor="#f1f5f9"),
            yaxis2=dict(title="평균 질문 길이 (chars)",
                        overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(t=16, b=40, l=60, r=70),
        ))
        return _to_json(fig)

    # ── 차트 11: 생애주기 × 진료과 100% 스택 바 ──────────────────────────

    def lifecycle_dept_stacked(self) -> dict:
        ct = pd.crosstab(self.df["lifeCycle"], self.df["department"],
                         normalize="index").reindex(_LIVES)[_DEPTS] * 100
        fig = go.Figure()
        for i, dept in enumerate(_DEPTS):
            vals = ct[dept].round(1).tolist()
            fig.add_trace(go.Bar(
                name=dept, y=_LIVES, x=vals, orientation="h",
                marker_color=PALETTE[i],
                text=[f"{v:.0f}%" for v in vals], textposition="inside",
                insidetextanchor="middle", textfont=dict(size=11, color="#fff"),
                hovertemplate=f"{dept}: %{{x:.1f}}%<extra></extra>",
            ))
        fig.update_layout(**_layout(
            barmode="stack", height=220,
            xaxis=dict(title="비율 (%)", range=[0, 100], gridcolor="#f1f5f9", ticksuffix="%"),
            yaxis=dict(title=""),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            margin=dict(t=16, b=50, l=70, r=20),
        ))
        return _to_json(fig)

    # ── 전체 차트 딕셔너리 반환 ──────────────────────────────────────────

    def build_all(self) -> dict:
        print("  ① 질병 롱테일 막대...")
        charts = {"disease_bar": self.disease_bar()}
        print("  ② 생애주기 도넛...")
        charts["lifecycle_pie"] = self.lifecycle_pie()
        print("  ③ 진료과×생애주기 히트맵...")
        charts["dept_heatmap"] = self.dept_heatmap()
        print("  ④ 텍스트 길이 박스플롯...")
        charts["text_boxplot"] = self.text_boxplot()
        print("  ⑤ 계층 선버스트...")
        charts["sunburst"] = self.sunburst()
        print("  ⑥ 진료과별 Top5 질병...")
        charts["dept_top10"] = self.dept_top5()
        print("  ⑦ 빈출 2-gram 트리맵...")
        charts["word_treemap"] = self.word_treemap()
        print("  ⑧ Train/Val 분할 막대...")
        charts["train_val_bar"] = self.train_val_bar()
        print("  ⑨ 질문·답변 길이 산점도...")
        charts["len_scatter"] = self.len_scatter()
        print("  ⑩ 진료과 dual-axis (문서 수 + 평균 질문 길이)...")
        charts["dept_dual_axis"] = self.dept_dual_axis()
        print("  ⑪ 생애주기 × 진료과 100% 스택 바...")
        charts["lifecycle_dept_stacked"] = self.lifecycle_dept_stacked()
        return charts
