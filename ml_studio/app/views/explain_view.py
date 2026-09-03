"""6단계. SHAP 해석.

기간 설계
---------
SHAP 값은 시점마다 독립적으로 계산된다. 그래서 한 번 계산해 두면 기간을
바꿔 볼 때 다시 돌릴 필요가 없다 — 계산 결과를 잘라 쓰기만 한다.
화면도 그 구조를 따른다.

  계산 범위   무거운 작업. 버튼을 눌러야 돈다.
  표시 기간   가벼운 작업. 슬라이더를 움직이면 바로 다시 그린다.

구간 비교는 별도 모드로 뒀다. 설비 개조 전후처럼 같은 피처가 구간마다
다르게 작동하는지 겹쳐 보는 용도다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state, theme
from core import explain, plots

PRESETS = {
    "전체": None,
    "최근 7일": pd.Timedelta(days=7),
    "최근 30일": pd.Timedelta(days=30),
    "최근 90일": pd.Timedelta(days=90),
}


def render() -> None:
    S = st.session_state
    theme.page_head(
        "SHAP 해석",
        "학습에 쓴 구간에 챔피언 모델을 대입해 시점별 기여도를 구합니다. "
        "SHAP 값은 모델이 학습한 통계적 관계이며, 설비의 인과 관계와는 다를 수 있습니다.")

    if not state.guard("explain", "먼저 챔피언 모델을 확정해 주세요."):
        return
    pipe = state.champion_pipeline()
    if pipe is None:
        st.error("챔피언 모델을 찾지 못했습니다. 4단계에서 다시 학습해 주세요.")
        return

    _compute_panel(pipe)

    if S.shap_result is None:
        return

    res = S.shap_result
    lo, hi = explain.period_bounds(res)

    st.header("표시 기간")
    theme.caption("계산은 다시 하지 않습니다. 아래 결과는 고른 기간의 시점만 담습니다.")
    view, label = _period_picker(res, lo, hi)

    st.caption(f"{res['explainer']} · 계산 {res['n_samples']:,}개 시점 · "
               f"표시 {view['n_samples']:,}개 · 기준값 {res['base_value']:.4g}")

    imp = explain.importance(view)
    st.header("기여도 순위")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(plots.shap_importance_bar(imp), use_container_width=True)
    with c2:
        st.markdown("**선택 기간 기준**")
        st.dataframe(
            imp[["feature", "mean_abs_shap", "contribution_pct"]],
            use_container_width=True, hide_index=True, height=440,
            column_config={
                "feature": "피처",
                "mean_abs_shap": st.column_config.NumberColumn("mean |SHAP|", format="%.4g"),
                "contribution_pct": st.column_config.NumberColumn("비중 %", format="%.1f"),
            })

    st.header("Dependence plot")
    _dependence(res, view, imp, label, lo, hi)

    st.header("기여도 추이")
    theme.caption("기간 안에서 주도 인자가 바뀌는지 봅니다. 선이 교차하면 운전 국면이 달라진 지점입니다.")
    freq = st.selectbox("집계 간격", ["원본", "1h", "6h", "1D"], index=2,
                        help="점이 많으면 뭉개져 보입니다. 간격을 늘려 평균을 냅니다.")
    st.plotly_chart(
        plots.shap_contribution_stream(view["values"], top_n=6,
                                       freq=None if freq == "원본" else freq),
        use_container_width=True)

    st.header("특정 시점 분해")
    _local(view)


# ── 계산 ────────────────────────────────────────────────────────────────
def _compute_panel(pipe) -> None:
    S = st.session_state
    X = S.X
    # 구간 이름을 분할과 정확히 맞춘다. 3분할 도입 후 test_idx 는 '검증' 구간이지
    # '미접촉' 구간이 아니다. 여기서 이름이 틀리면 해석 자체를 오독하게 된다.
    scopes = {
        "학습 구간 (in-sample)": X.iloc[S.train_idx],
        "검증 구간 (모델 선택에 사용됨)": X.iloc[S.test_idx],
    }
    if S.unseen_idx is not None and len(S.unseen_idx):
        scopes["Final Unseen (최종 보고 구간)"] = X.iloc[S.unseen_idx]
    scopes["전체"] = X

    with st.expander("계산 범위", expanded=S.shap_result is None):
        c1, c2 = st.columns([2, 1])
        scope = c1.selectbox("대상", list(scopes), index=0)
        target_X = scopes[scope]

        if scope.startswith("검증"):
            theme.caption("이 구간은 모델을 고르는 데 이미 쓰였습니다. 여기서 본 기여도는 "
                          "챔피언이 선택된 이유와 얽혀 있으므로, 일반화된 설명으로 읽지 마세요.")
        elif scope.startswith("Final Unseen"):
            st.warning("Final Unseen 은 성능 보고용 구간입니다. SHAP 계산은 성능 평가가 "
                       "아니므로 접근 횟수에 포함되지 않지만, 여기서 본 결과를 근거로 "
                       "모델을 바꾸면 그 구간은 더 이상 미접촉이 아닙니다.")
        elif scope == "전체":
            theme.caption("세 구간이 섞입니다. 구간별로 관계가 달라지는지 보려면 "
                          "학습 구간과 Final Unseen 을 따로 계산해 비교하세요.")
        # 어떤 방법으로 갈지는 **core 가 정한다.** 화면이 짐작하면 안내와 실제가
        # 갈린다 — 예전에 그래서 몇 시간짜리 계산을 "1분" 이라고 안내했다.
        probe = explain.plan(pipe, len(target_X))
        # 구간이 200행보다 짧으면 기본값이 하한 아래로 내려가 위젯이 예외를 던진다.
        n_hi = max(200, min(50000, len(target_X)))
        n_default = min(int(probe["n"]) or 200, n_hi)
        n_max = c2.number_input("표본 상한", 200, n_hi, max(200, n_default),
                                step=500,
                                help="많이 넣을수록 정확하지만 오래 걸립니다. "
                                     "기본값은 이 모델에 맞춰 잡아 뒀습니다.")

        c3, c4 = st.columns(2)
        d_lo, d_hi = target_X.index.min().date(), target_X.index.max().date()
        use_range = c3.checkbox("계산 범위를 날짜로 좁히기", value=False,
                                help="아주 긴 데이터에서 특정 구간만 볼 때 계산 시간을 줄입니다.")
        if use_range:
            picked = c4.date_input("계산 구간", (d_lo, d_hi),
                                   min_value=d_lo, max_value=d_hi)
            if isinstance(picked, tuple) and len(picked) == 2:
                s, e = picked
                target_X = target_X.loc[
                    str(s):str(pd.Timestamp(e) + pd.Timedelta(days=1))]

        theme.caption(
            f"대상 {len(target_X):,}개 시점 중 최대 {int(n_max):,}개를 균등 표본으로 씁니다.")

        # 얼마나 걸리는지 모르면 멈춘 줄 알고 새로고침한다. core 가 정한 방법과
        # 그 방법의 성격을 그대로 옮긴다.
        n = int(min(int(n_max), len(target_X)))
        msg = f"**{probe['label']}** — {probe['note']}"
        if probe["slow"]:
            st.warning(f"{msg}\n\n표본을 {n:,}개로 잡았습니다. 더 늘리면 시간이 "
                       "비례해서 늘어납니다. 오래 걸리는 게 곤란하면 아래 "
                       "**순열 중요도**로도 어떤 인자가 중요한지는 볼 수 있습니다.")
        else:
            st.info(f"{msg} 지금 설정은 {n:,}개 시점입니다. "
                    "계산 중에는 브라우저를 새로 고치지 마세요.")
        if probe["slow"] and st.button("순열 중요도로 대신 보기"):
            _fallback(pipe, target_X)

        if st.button("SHAP 계산", type="primary"):
            if target_X.empty:
                st.error("고른 구간에 데이터가 없습니다.")
                return
            try:
                with st.spinner(f"{len(target_X):,}개 시점 계산 중"):
                    S.shap_result = explain.compute_shap(
                        pipe, target_X, explain.ShapConfig(max_samples=int(n_max)))
                st.rerun()
            except explain.ShapUnavailable as e:
                st.error(str(e))
                _fallback(pipe, target_X)
            except Exception as e:  # noqa: BLE001
                st.error(f"계산 실패 — {type(e).__name__}: {e}")
                _fallback(pipe, target_X)


# ── 기간 선택 ───────────────────────────────────────────────────────────
def _period_picker(res: dict, lo, hi) -> tuple[dict, str]:
    c1, c2 = st.columns([1, 3])
    preset = c1.selectbox("범위", list(PRESETS), index=0)

    if PRESETS[preset] is not None:
        start = max(lo, hi - PRESETS[preset])
        end = hi
        c2.markdown(
            f'<p class="caption">{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}</p>',
            unsafe_allow_html=True)
    elif lo >= hi:
        start, end = lo, hi
        c2.markdown('<p class="caption">단일 시점입니다.</p>', unsafe_allow_html=True)
    else:
        span = hi - lo
        step = pd.Timedelta(hours=1) if span > pd.Timedelta(days=3) else pd.Timedelta(minutes=10)
        start, end = c2.slider(
            "기간", min_value=lo.to_pydatetime(), max_value=hi.to_pydatetime(),
            value=(lo.to_pydatetime(), hi.to_pydatetime()),
            step=step.to_pytimedelta(), format="YYYY-MM-DD HH:mm",
            label_visibility="collapsed")

    try:
        view = explain.slice_period(res, start, end)
    except ValueError as e:
        st.warning(str(e))
        return res, "전체"

    return view, f"{pd.Timestamp(start):%Y-%m-%d} ~ {pd.Timestamp(end):%Y-%m-%d}"


# ── dependence ──────────────────────────────────────────────────────────
def _dependence(res: dict, view: dict, imp: pd.DataFrame, label: str, lo, hi) -> None:
    theme.caption("가로축은 피처 값, 세로축은 그 피처가 예측을 얼마나 밀었는지. "
                  "0 위쪽이면 값을 올리는 방향, 아래쪽이면 내리는 방향입니다.")

    features = list(imp["feature"])
    if not features:
        st.info("표시할 피처가 없습니다.")
        return

    mode = st.radio("보기", ["단일 기간", "구간 비교"], horizontal=True,
                    label_visibility="collapsed")

    if mode == "단일 기간":
        c1, c2, c3 = st.columns([3, 2, 1.4])
        pick = c1.multiselect("피처", features, default=features[:3])
        color = c3.selectbox("색", ["상호작용", "시점", "단색"], index=0)
        auto = color == "상호작용"
        inter = c2.selectbox("상호작용 피처", ["자동 선택"] + features, disabled=not auto)

        for f in pick:
            if color == "상호작용":
                i = explain.auto_interaction(view, f) if inter == "자동 선택" else inter
                cmode = "interaction"
            else:
                i, cmode = None, ("time" if color == "시점" else "none")
            try:
                dep = explain.dependence_data(view, f, i)
                st.plotly_chart(
                    plots.shap_dependence(dep, f, i, color_mode=cmode, subtitle=label),
                    use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.caption(f"{f} — 그리지 못했습니다: {e}")
        return

    _compare(res, features, lo, hi)


def _compare(res: dict, features: list[str], lo, hi) -> None:
    theme.caption("두 구간을 겹쳐 그립니다. 같은 피처 값인데 SHAP 값이 다르면, "
                  "그 사이에 설비나 운전 조건이 달라졌다는 신호입니다.")
    mid = (lo + (hi - lo) / 2).date()
    c1, c2 = st.columns(2)
    a = c1.date_input("구간 A", (lo.date(), mid),
                      min_value=lo.date(), max_value=hi.date(), key="shap_pa")
    b = c2.date_input("구간 B", (mid, hi.date()),
                      min_value=lo.date(), max_value=hi.date(), key="shap_pb")
    if not (isinstance(a, tuple) and len(a) == 2
            and isinstance(b, tuple) and len(b) == 2):
        st.info("두 구간의 시작일과 종료일을 모두 골라 주세요.")
        return

    day = pd.Timedelta(days=1)
    periods = [
        (f"A · {a[0]:%m-%d}~{a[1]:%m-%d}", pd.Timestamp(a[0]), pd.Timestamp(a[1]) + day),
        (f"B · {b[0]:%m-%d}~{b[1]:%m-%d}", pd.Timestamp(b[0]), pd.Timestamp(b[1]) + day),
    ]
    labels = [p[0] for p in periods]

    try:
        shift = explain.period_shift(res, periods)
    except ValueError as e:
        st.warning(str(e))
        return

    st.plotly_chart(plots.shap_period_shift(shift, labels), use_container_width=True)
    if "변화" in shift.columns and len(shift):
        top = shift.iloc[0]
        direction = "커졌습니다" if top["변화"] > 0 else "작아졌습니다"
        st.markdown(
            f'<p class="caption">기여 비중이 가장 크게 움직인 피처는 '
            f'<code>{top["feature"]}</code> 로, B 구간에서 '
            f'{abs(top["변화"]):.1f}%p {direction}.</p>', unsafe_allow_html=True)

    pick = st.multiselect("겹쳐 볼 피처", features, default=features[:2],
                          key="shap_cmp_feats")
    for f in pick:
        try:
            dep = explain.dependence_by_periods(res, f, periods)
            st.plotly_chart(plots.shap_dependence(dep, f, None, color_mode="period"),
                            use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.caption(f"{f} — 그리지 못했습니다: {e}")


# ── 국소 해석 ───────────────────────────────────────────────────────────
def _local(view: dict) -> None:
    S = st.session_state
    theme.caption("한 시점의 예측이 왜 그 값이 되었는지 항목별로 나눕니다.")

    idx = view["values"].index
    if len(idx) == 0:
        st.info("선택 기간에 시점이 없습니다.")
        return
    pos = st.slider("시점", 0, max(len(idx) - 1, 1), len(idx) // 2,
                    help="선택 기간 안에서 이동합니다.")
    pos = min(pos, len(idx) - 1)
    ts = idx[pos]
    st.caption(f"{ts:%Y-%m-%d %H:%M}")

    local = explain.local_explanation(view, ts)
    pipe = state.champion_pipeline()
    pred = None
    try:
        pred = float(pipe.predict(S.X.loc[[ts]])[0])
    except Exception:  # noqa: BLE001
        pass

    st.plotly_chart(plots.local_waterfall(local, view["base_value"], pred),
                    use_container_width=True)

    if S.y is not None and ts in S.y.index:
        c1, c2 = st.columns(2)
        c1.metric("실측", f"{S.y.loc[ts]:,.4g}")
        if pred is not None:
            c2.metric("예측", f"{pred:,.4g}", f"{pred - S.y.loc[ts]:+,.4g}")


def _fallback(pipe, X) -> None:
    S = st.session_state
    st.info("대신 순열 중요도로 대체합니다. 방향성은 알 수 없고 크기만 나옵니다.")
    try:
        y = S.y.reindex(X.index)
        imp = explain.permutation_importance_fallback(pipe, X, y, n_repeats=3)
        st.dataframe(imp.head(25), use_container_width=True, hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"대체 계산도 실패했습니다 — {e}")
