"""차트.

색은 계측 트렌드 화면의 관습을 따랐다. 실측은 짙은 남색(기준선),
예측은 앰버(계측기 지시침), 잔차는 회색조. 이 세 가지 외에는 색을 늘리지 않는다.
웹폰트는 쓰지 않는다 — 폐쇄망에서 로딩되지 않으면 레이아웃이 깨진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INK = "#0E1620"
MUTED = "#66768A"
GRID = "#E7EBF0"
ACTUAL = "#0B4F8C"
PREDICTED = "#C77B02"
POSITIVE = "#1F6F5C"
NEGATIVE = "#A32015"
SERIES = ["#0B4F8C", "#C77B02", "#2E7D5B", "#7A4E9E", "#0E7C86",
          "#A8322D", "#64707F", "#B4632A"]

FONT_STACK = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", '
              '"Apple SD Gothic Neo", "Noto Sans KR", sans-serif')


def _px():
    import plotly.graph_objects as go
    return go


# ─────────────────────────────────────────────────────────────
# 큰 시계열 그리기
# ─────────────────────────────────────────────────────────────
MAX_POINTS = 4000      # 화면 가로 픽셀보다 촘촘히 그려도 눈에 보이지 않는다
GL_THRESHOLD = 1500    # 이보다 많으면 SVG 대신 WebGL 로 그린다


def thin(obj, max_points: int = MAX_POINTS):
    """점이 너무 많으면 일정 간격으로 솎아낸다.

    12,000행짜리 시계열을 SVG 로 그리면 브라우저가 점 하나마다 DOM 노드를 만든다.
    차트 몇 개가 한 화면에 겹치면 스크롤조차 버벅인다. 화면 가로 해상도보다 촘촘한
    점은 어차피 보이지 않으므로 줄여도 그림이 같다.

    **원본은 건드리지 않는다.** 통계·모델 계산에는 항상 전체 데이터를 쓰고,
    여기서 줄이는 것은 그리기 직전의 사본뿐이다.
    """
    n = len(obj)
    if n <= max_points:
        return obj
    step = int(np.ceil(n / max_points))
    return obj.iloc[::step]


def thin_pair(*series, max_points: int = MAX_POINTS):
    """여러 시계열을 같은 간격으로 솎아낸다. 서로 어긋나면 안 되기 때문이다."""
    n = max((len(s) for s in series), default=0)
    if n <= max_points:
        return series
    step = int(np.ceil(n / max_points))
    return tuple(s.iloc[::step] for s in series)


def _line(go, n: int):
    """점 수에 따라 SVG(Scatter) 와 WebGL(Scattergl) 을 고른다.

    Scattergl 은 GPU 로 그려서 수천 점도 부드럽다. 다만 점이 적을 때는 Scatter 쪽
    선이 더 곱게 나오므로 임계값을 두고 갈아탄다.
    """
    return go.Scattergl if n > GL_THRESHOLD else go.Scatter


def apply_layout(fig, title: str = "", height: int = 420, ylabel: str = "", xlabel: str = ""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        height=height,
        font=dict(family=FONT_STACK, size=12, color=INK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=56, r=24, t=48 if title else 20, b=44),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, zeroline=False, title=xlabel)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, zeroline=False, title=ylabel)
    return fig


def actual_vs_pred(
    actual: pd.Series,
    predicted: pd.Series,
    train_end=None,
    title: str = "실측 대비 예측",
    ylabel: str = "",
):
    """시계열 라인차트. 학습 구간과 홀드아웃 경계를 표시한다."""
    go = _px()
    a, p = thin_pair(actual, predicted)
    Line = _line(go, len(a))
    fig = go.Figure()
    fig.add_trace(Line(
        x=a.index, y=a.to_numpy(), name="실측",
        mode="lines", line=dict(color=ACTUAL, width=1.6)))
    fig.add_trace(Line(
        x=p.index, y=p.to_numpy(), name="예측",
        mode="lines", line=dict(color=PREDICTED, width=1.6, dash="solid")))

    if train_end is not None:
        fig.add_vline(x=train_end, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=train_end, yref="paper", y=1.02, text="홀드아웃 시작",
                           showarrow=False, font=dict(size=11, color=MUTED), xanchor="left")
    return apply_layout(fig, title, ylabel=ylabel)


def residual_series(actual: pd.Series, predicted: pd.Series, title: str = "잔차 (실측 − 예측)"):
    go = _px()
    common = actual.index.intersection(predicted.index)
    resid = thin(actual.loc[common] - predicted.loc[common])
    Line = _line(go, len(resid))
    fig = go.Figure()
    fig.add_trace(Line(x=resid.index, y=resid.to_numpy(), mode="lines",
                       line=dict(color=MUTED, width=1.2), name="잔차"))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    return apply_layout(fig, title, height=260)


def residual_band(
    stats: pd.DataFrame,
    outlier_points: pd.DataFrame | None = None,
    title: str = "잔차 추이 · rolling 평균과 ±1σ 밴드",
):
    """잔차 + rolling 평균 + ±1σ 밴드 + 이상점.

    밴드가 0 선을 벗어나 머무르는 구간이 계통 편향이 생긴 구간이다.
    """
    go = _px()
    fig = go.Figure()
    # 배경 곡선만 솎아낸다. 이상점은 원본 그대로 찍어야 위치가 어긋나지 않는다.
    s = thin(stats)
    idx = s.index
    up = (s["roll_mean"] + s["roll_std"]).to_numpy()
    dn = (s["roll_mean"] - s["roll_std"]).to_numpy()

    fig.add_trace(go.Scatter(x=idx, y=up, mode="lines", name="+1σ",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=dn, mode="lines", name="±1σ",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(102,118,138,0.16)", hoverinfo="skip"))
    Line = _line(go, len(s))
    fig.add_trace(Line(x=idx, y=s["residual"].to_numpy(), mode="lines",
                       name="잔차", line=dict(color=MUTED, width=0.8), opacity=0.55))
    fig.add_trace(Line(x=idx, y=s["roll_mean"].to_numpy(), mode="lines",
                       name="rolling 평균", line=dict(color=ACTUAL, width=1.8)))
    fig.add_hline(y=0, line=dict(color=INK, width=1))

    if outlier_points is not None and not outlier_points.empty:
        outlier_points = outlier_points.head(2000)
        fig.add_trace(go.Scatter(
            x=outlier_points.index, y=outlier_points["residual"].to_numpy(),
            mode="markers", name="이상점",
            marker=dict(size=7, color=NEGATIVE, symbol="circle-open", line=dict(width=1.6)),
            customdata=outlier_points[["robust_z", "방향"]].to_numpy(),
            hovertemplate="%{x}<br>잔차 %{y:.4g}<br>z %{customdata[0]:.2f} "
                          "(%{customdata[1]})<extra></extra>"))
    return apply_layout(fig, title, height=330)


def residual_drift(table: pd.DataFrame, title: str = "구간별 잔차 통계"):
    """구간을 잘라 MAE 와 표준편차가 이동하는지 본다."""
    go = _px()
    fig = go.Figure()
    labels = [f"{int(r['구간'])}구간" for _, r in table.iterrows()]
    fig.add_trace(go.Bar(x=labels, y=table["MAE"].to_numpy(), name="MAE",
                         marker_color=ACTUAL, opacity=0.85))
    fig.add_trace(go.Scatter(x=labels, y=table["std"].to_numpy(), name="표준편차",
                             mode="lines+markers", yaxis="y2",
                             line=dict(color=PREDICTED, width=2),
                             marker=dict(size=7)))
    fig = apply_layout(fig, title, height=300, ylabel="MAE")
    fig.update_layout(
        hovermode="x unified",
        # titlefont= 은 plotly 5 에서 제거됐다. title=dict(font=...) 가 현재 형식.
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    title=dict(text="표준편차", font=dict(color=PREDICTED)),
                    tickfont=dict(color=PREDICTED)))
    return fig


def backtest_series(table: pd.DataFrame, metric: str = "R2",
                    title: str = "구간별 성능 (rolling backtest)"):
    """시기마다 성능이 어떻게 달라지는지. 평균선에서 크게 벗어난 구간을 찾는다."""
    go = _px()
    ok = table[table["status"] == "ok"] if "status" in table.columns else table
    if ok.empty or metric not in ok.columns:
        return apply_layout(go.Figure(), title, height=300)

    v = ok[metric].astype(float)
    mean = float(v.mean())
    labels = [f"{i}구간<br>{pd.Timestamp(s):%m-%d}"
              for i, s in zip(ok["구간"], ok["평가시작"])]
    colors = [NEGATIVE if abs(x - mean) > v.std() else ACTUAL for x in v]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=v.to_numpy(), marker_color=colors, name=metric,
                         customdata=ok[["n_train", "n_test"]].to_numpy(),
                         hovertemplate=(f"{metric}=%{{y:.4f}}<br>"
                                        "학습 %{customdata[0]:,}행 · 평가 %{customdata[1]:,}행"
                                        "<extra></extra>")))
    fig.add_hline(y=mean, line=dict(color=PREDICTED, width=1.5, dash="dash"),
                  annotation_text=f"평균 {mean:.4f}", annotation_position="top left")
    fig = apply_layout(fig, title, height=320, ylabel=metric)
    fig.update_layout(hovermode="closest")
    return fig


def residual_acf(acf: pd.DataFrame, n: int, title: str = "잔차 자기상관"):
    """lag>=1 의 상관이 신뢰구간 밖에 남아 있으면 못 뽑아낸 시간 구조가 있다."""
    go = _px()
    ci = 1.96 / np.sqrt(max(n, 2))
    colors = [NEGATIVE if abs(v) > ci else MUTED for v in acf["acf"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=acf["lag"].to_numpy(), y=acf["acf"].to_numpy(),
                         marker_color=colors, name="ACF"))
    for s in (ci, -ci):
        fig.add_hline(y=s, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig = apply_layout(fig, title, height=280, xlabel="lag (행)", ylabel="자기상관")
    fig.update_layout(hovermode="closest")
    return fig


def scatter_actual_pred(actual: pd.Series, predicted: pd.Series, title: str = "실측 vs 예측"):
    go = _px()
    common = actual.index.intersection(predicted.index)
    sa, sp = thin_pair(actual.loc[common], predicted.loc[common])
    a, p = sa.to_numpy(), sp.to_numpy()
    lo, hi = float(np.nanmin([a.min(), p.min()])), float(np.nanmax([a.max(), p.max()]))
    Line = _line(go, len(a))
    fig = go.Figure()
    fig.add_trace(Line(x=a, y=p, mode="markers", name="시점",
                       marker=dict(size=4, color=ACTUAL, opacity=0.45)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완전일치",
                             line=dict(color=MUTED, width=1, dash="dash")))
    fig = apply_layout(fig, title, height=380, xlabel="실측", ylabel="예측")
    fig.update_layout(hovermode="closest")
    return fig


def leaderboard_bar(board: pd.DataFrame, metric: str, top_n: int = 12):
    go = _px()
    col = f"holdout_{metric}"
    d = board[board.get("status", "ok") == "ok"].head(top_n).iloc[::-1]
    colors = [PREDICTED if "Ensemble" in str(m) else ACTUAL for m in d["model"]]
    fig = go.Figure(go.Bar(
        x=d[col], y=d["model"], orientation="h", marker_color=colors,
        text=[f"{v:.4g}" for v in d[col]], textposition="outside", cliponaxis=False))
    fig = apply_layout(fig, f"홀드아웃 {metric}", height=60 + 32 * len(d), xlabel=metric)
    fig.update_layout(hovermode="closest")
    return fig


def shap_importance_bar(imp: pd.DataFrame, top_n: int = 20):
    go = _px()
    d = imp.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["mean_abs_shap"], y=d["feature"], orientation="h", marker_color=ACTUAL,
        text=[f"{v:.1f}%" for v in d["contribution_pct"]], textposition="outside",
        cliponaxis=False))
    fig = apply_layout(fig, "평균 절대 SHAP", height=80 + 26 * len(d), xlabel="mean |SHAP|")
    fig.update_layout(hovermode="closest")
    return fig


def _trendline(go, fig, d: pd.DataFrame, color: str, name: str, dash: str = "solid"):
    """구간 평균 추세선. 산점만으로는 형태가 안 보여서 겹쳐 그린다."""
    d = d.dropna(subset=["feature_value", "shap_value"])
    if len(d) <= 50:
        return
    q = min(20, int(d["feature_value"].nunique()))
    if q < 3:
        return
    bins = pd.qcut(d["feature_value"], q=q, duplicates="drop")
    trend = d.groupby(bins, observed=True).agg(
        x=("feature_value", "median"), y=("shap_value", "mean")).dropna()
    if len(trend) > 2:
        fig.add_trace(go.Scatter(x=trend["x"], y=trend["y"], mode="lines", name=name,
                                 line=dict(color=color, width=2.2, dash=dash)))


def shap_dependence(
    dep: pd.DataFrame,
    feature: str,
    interaction: str | None = None,
    color_mode: str = "interaction",
    subtitle: str = "",
):
    """SHAP dependence plot. x=피처값, y=SHAP값.

    color_mode
        "interaction" — 상호작용 피처 값으로 색을 입힌다 (기본)
        "time"        — 시점으로 색을 입힌다. 기간 안에서 관계가 흘러가는지 본다
        "period"      — 'period' 열의 구간별로 색을 나눠 겹쳐 그린다
        "none"        — 단색
    """
    go = _px()
    # 산점도는 점 하나가 곧 마커라 개수가 그대로 부담이 된다. 추세선은 솎아내기
    # 전 원본으로 그려야 형태가 유지되므로, 계산을 먼저 하고 표시만 줄인다.
    dep_full = dep
    dep = thin(dep, MAX_POINTS // 2)
    ts = pd.to_datetime(dep["timestamp"])
    title = f"{feature} — SHAP dependence"
    if subtitle:
        title += f"  ·  {subtitle}"

    if color_mode == "period" and "period" in dep.columns:
        fig = go.Figure()
        labels = list(dict.fromkeys(dep["period"]))
        for i, label in enumerate(labels):
            part = dep[dep["period"] == label]
            color = SERIES[i % len(SERIES)]
            fig.add_trace(go.Scatter(
                x=part["feature_value"], y=part["shap_value"], mode="markers",
                marker=dict(size=5, opacity=0.5, color=color),
                name=str(label),
                customdata=np.stack([part["timestamp"].astype(str)], axis=-1),
                hovertemplate=(f"{feature}=%{{x:.4g}}<br>SHAP=%{{y:.4g}}"
                               f"<br>%{{customdata[0]}}<extra>{label}</extra>"),
                legendgroup=str(label)))
            _trendline(go, fig, dep_full[dep_full["period"] == label], color,
                       f"{label} 평균")
            if fig.data:
                fig.data[-1].legendgroup = str(label)
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
        fig = apply_layout(fig, title, height=430,
                           xlabel=f"{feature} 값", ylabel="SHAP 값")
        fig.update_layout(hovermode="closest")
        return fig

    marker = dict(size=5, opacity=0.6)
    if color_mode == "time":
        marker.update(color=ts.astype("int64"), colorscale="Viridis", showscale=True,
                      colorbar=dict(title=dict(text="시점", side="right"), thickness=12,
                                    tickvals=[ts.astype("int64").min(), ts.astype("int64").max()],
                                    ticktext=[f"{ts.min():%m-%d}", f"{ts.max():%m-%d}"]))
    elif color_mode == "interaction" and "interaction_value" in dep.columns and interaction:
        marker.update(color=dep["interaction_value"], colorscale="Cividis",
                      colorbar=dict(title=dict(text=interaction, side="right"), thickness=12),
                      showscale=True)
    else:
        marker.update(color=ACTUAL)

    fig = go.Figure(_line(go, len(dep))(
        x=dep["feature_value"], y=dep["shap_value"], mode="markers", marker=marker,
        customdata=np.stack([dep["timestamp"].astype(str)], axis=-1),
        hovertemplate=f"{feature}=%{{x:.4g}}<br>SHAP=%{{y:.4g}}<br>%{{customdata[0]}}<extra></extra>",
        name=feature))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    _trendline(go, fig, dep_full, PREDICTED, "구간 평균")

    fig = apply_layout(fig, title, height=430,
                       xlabel=f"{feature} 값", ylabel="SHAP 값")
    fig.update_layout(hovermode="closest")
    return fig


def shap_period_shift(shift: pd.DataFrame, labels: list[str]):
    """구간별 기여 비중 비교 막대."""
    go = _px()
    d = shift.iloc[::-1]
    fig = go.Figure()
    for i, label in enumerate(labels):
        if label not in d.columns:
            continue
        fig.add_trace(go.Bar(x=d[label], y=d["feature"], orientation="h",
                             name=str(label), marker_color=SERIES[i % len(SERIES)]))
    fig = apply_layout(fig, "구간별 기여 비중", height=110 + 30 * len(d),
                       xlabel="mean |SHAP| 비중 (%)")
    fig.update_layout(barmode="group", hovermode="closest", bargap=0.28)
    return fig


def shap_contribution_stream(values: pd.DataFrame, top_n: int = 6, freq: str | None = None):
    """시간에 따른 피처별 기여도 추이. 기간 안에서 주도 인자가 바뀌는지 본다."""
    go = _px()
    order = values.abs().mean().sort_values(ascending=False).head(top_n).index
    d = values[order]
    if freq:
        d = d.resample(freq).mean()
    fig = go.Figure()
    for i, col in enumerate(order):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=str(col),
                                 line=dict(width=1.5, color=SERIES[i % len(SERIES)])))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    fig = apply_layout(fig, "기여도 추이", height=380, ylabel="SHAP 값")
    return fig


def local_waterfall(local: pd.DataFrame, base_value: float, prediction: float | None = None):
    go = _px()
    d = local.iloc[::-1]
    colors = [POSITIVE if v > 0 else NEGATIVE for v in d["shap_value"]]
    labels = [f"{f}  ({v:.4g})" for f, v in zip(d["feature"], d["feature_value"])]
    fig = go.Figure(go.Bar(x=d["shap_value"], y=labels, orientation="h",
                           marker_color=colors, text=[f"{v:+.4g}" for v in d["shap_value"]],
                           textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    sub = f"기준값 {base_value:.4g}"
    if prediction is not None:
        sub += f" → 예측 {prediction:.4g}"
    fig = apply_layout(fig, sub, height=100 + 26 * len(d), xlabel="SHAP 기여")
    fig.update_layout(hovermode="closest")
    return fig


def whatif_compare(res: pd.DataFrame, title: str = "What-if 비교"):
    go = _px()
    fig = go.Figure()
    res = thin(res)
    fig.add_trace(_line(go, len(res))(x=res.index, y=res["baseline"], name="현재 조건",
                             mode="lines", line=dict(color=ACTUAL, width=1.5)))
    fig.add_trace(_line(go, len(res))(x=res.index, y=res["scenario"], name="변경 조건",
                             mode="lines", line=dict(color=PREDICTED, width=1.5)))
    return apply_layout(fig, title, ylabel="예측 Y")


def pdp_curve(curve: pd.DataFrame, feature: str, ice: pd.DataFrame | None = None):
    go = _px()
    fig = go.Figure()
    if ice is not None and not ice.empty:
        for _, g in ice.groupby("row"):
            fig.add_trace(go.Scatter(x=g[feature], y=g["prediction"], mode="lines",
                                     line=dict(color=MUTED, width=0.7), opacity=0.35,
                                     showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=curve[feature], y=curve["p90"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=curve[feature], y=curve["p10"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(27,58,92,0.10)", name="10~90 분위"))
    fig.add_trace(go.Scatter(x=curve[feature], y=curve["prediction"], mode="lines",
                             name="평균 반응", line=dict(color=PREDICTED, width=2.4)))
    fig = apply_layout(fig, f"{feature} 를 바꾸면 Y 는", xlabel=feature, ylabel="예측 Y")
    fig.update_layout(hovermode="closest")
    return fig


def anomaly_timeline(score: pd.Series, flag: pd.Series, title: str = "이상 점수"):
    go = _px()
    fig = go.Figure()
    score = thin(score)
    fig.add_trace(_line(go, len(score))(x=score.index, y=score.to_numpy(), mode="lines",
                             line=dict(color=ACTUAL, width=1.2), name="점수"))
    hit = score[flag.reindex(score.index).fillna(False).to_numpy()]
    if len(hit):
        fig.add_trace(go.Scatter(x=hit.index, y=hit.to_numpy(), mode="markers",
                                 marker=dict(color=NEGATIVE, size=6, symbol="x"), name="이상"))
    return apply_layout(fig, title)


def cluster_timeline(labels: pd.Series, title: str = "군집 배정"):
    go = _px()
    fig = go.Figure()
    for i, c in enumerate(sorted(labels.dropna().unique())):
        sub = labels[labels == c]
        sub = thin(sub, MAX_POINTS // 2)
        fig.add_trace(_line(go, len(sub))(x=sub.index, y=[c] * len(sub), mode="markers",
                                 marker=dict(size=4, color=SERIES[i % len(SERIES)]),
                                 name=f"cluster {c}"))
    fig = apply_layout(fig, title, height=300, ylabel="cluster")
    fig.update_layout(hovermode="closest")
    return fig


def scatter_2d(df: pd.DataFrame, x: str, y: str, color: pd.Series | None = None, title: str = ""):
    go = _px()
    fig = go.Figure()
    if color is None:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers",
                                 marker=dict(size=5, color=ACTUAL, opacity=0.6)))
    else:
        for i, c in enumerate(sorted(pd.Series(color).dropna().unique())):
            m = pd.Series(color).to_numpy() == c
            fig.add_trace(go.Scatter(x=df[x][m], y=df[y][m], mode="markers", name=str(c),
                                     marker=dict(size=5, color=SERIES[i % len(SERIES)], opacity=0.65)))
    fig = apply_layout(fig, title, xlabel=x, ylabel=y)
    fig.update_layout(hovermode="closest")
    return fig


def missing_heat(df: pd.DataFrame, max_cols: int = 40, bins: int = 200):
    """결측 분포를 시간×컬럼 격자로 본다."""
    go = _px()
    cols = list(df.columns[:max_cols])
    d = df[cols].isna().astype(float)
    if len(d) > bins:
        grp = np.array_split(np.arange(len(d)), bins)
        z = np.vstack([d.iloc[g].mean().to_numpy() for g in grp]).T
        xs = [df.index[g[0]] for g in grp]
    else:
        z, xs = d.to_numpy().T, df.index
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=cols, colorscale="Greys",
                               colorbar=dict(title=dict(text="결측률"), thickness=12)))
    return apply_layout(fig, "결측 분포", height=60 + 16 * len(cols))
