"""7단계. What-if 시나리오."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import state, theme
from core import plots, whatif


def render() -> None:
    S = st.session_state
    st.title("7. What-if")
    if not state.guard("whatif", "먼저 챔피언 모델을 확정해 주세요."):
        return

    pipe = state.champion_pipeline()
    if pipe is None:
        st.error("챔피언 모델을 찾지 못했습니다.")
        return

    st.markdown('<p class="caption">X 를 바꾸면 예측 Y 가 어떻게 움직이는지 봅니다. '
                'Y 의 과거값을 X 로 쓰지 않았기 때문에 이 화면의 값을 그대로 읽을 수 있습니다.</p>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["조건 변경", "단일 피처 반응곡선"])
    with tab1:
        _scenario(pipe)
    with tab2:
        _sweep(pipe)


def _base_frame() -> pd.DataFrame:
    S = st.session_state
    X = S.X
    st.markdown("**적용 구간**")
    c1, c2, c3 = st.columns([2, 2, 1])
    lo, hi = X.index[0].date(), X.index[-1].date()
    start = c1.date_input("시작", value=max(lo, (X.index[-1] - pd.Timedelta(days=7)).date()),
                          min_value=lo, max_value=hi, key="wi_start")
    end = c2.date_input("종료", value=hi, min_value=lo, max_value=hi, key="wi_end")
    sub = X.loc[(X.index >= pd.Timestamp(start)) &
                (X.index < pd.Timestamp(end) + pd.Timedelta(days=1))]
    c3.metric("대상 행", f"{len(sub):,}")
    return sub


def _scenario(pipe) -> None:
    S = st.session_state
    Xb = _base_frame()
    if Xb.empty:
        st.warning("선택 구간에 데이터가 없습니다.")
        return

    numeric = [c for c in Xb.columns if pd.api.types.is_numeric_dtype(Xb[c])]
    order = numeric
    if S.shap_result is not None:
        from core import explain
        ranked = list(explain.importance(S.shap_result)["feature"])
        order = [c for c in ranked if c in numeric] + [c for c in numeric if c not in ranked]

    picks = st.multiselect("바꿀 피처", order, default=order[:2])

    changes = []
    skipped = []
    for f in picks:
        s = pd.to_numeric(Xb[f], errors="coerce")
        cur = float(s.mean())
        lo, hi = float(s.min()), float(s.max())
        # 전부 결측이면 셋 다 NaN 이 되고, NaN 을 슬라이더에 넣으면 예외가 난다.
        if not (np.isfinite(cur) and np.isfinite(lo) and np.isfinite(hi)):
            skipped.append(f)
            continue
        c1, c2, c3 = st.columns([1, 2, 1])
        mode = c1.selectbox("방식", [whatif.DELTA, whatif.PCT, whatif.SET], key=f"m_{f}",
                            format_func=lambda m: {"delta": "증감", "pct": "비율(%)",
                                                   "set": "고정값"}[m])
        if mode == whatif.SET:
            span = (hi - lo) or 1.0
            val = c2.slider(f, lo - span * 0.1, hi + span * 0.1, cur, key=f"v_{f}")
        elif mode == whatif.PCT:
            val = c2.slider(f"{f} (%)", -50.0, 50.0, 0.0, 1.0, key=f"v_{f}")
        else:
            span = (hi - lo) or 1.0
            val = c2.slider(f"{f} (증감)", -span / 2, span / 2, 0.0, key=f"v_{f}")
        c3.caption(f"현재 평균 {cur:,.4g}  \n관측 {lo:,.4g}~{hi:,.4g}")
        changes.append(whatif.Change(feature=f, mode=mode, value=float(val)))

    if skipped:
        st.warning(f"선택 구간에서 값이 전부 비어 있어 조정할 수 없는 피처: {', '.join(skipped)}")

    with st.expander("연동 피처 (물리적으로 함께 움직이는 값)"):
        st.caption("예: 유량을 올리면 차압도 따라 오릅니다. 계수는 단위 변화당 반응량입니다. "
                   "설정하지 않으면 나머지 피처는 그대로 둔 채 계산합니다.")
        linked: dict = {}
        for f in picks:
            others = [c for c in Xb.columns if c != f and pd.api.types.is_numeric_dtype(Xb[c])]
            dep = st.multiselect(f"{f} 를 바꿀 때 함께 움직일 피처", others, key=f"l_{f}")
            pairs = []
            for d in dep:
                coef = st.number_input(f"{f} → {d} 계수", value=0.0, step=0.1, key=f"c_{f}_{d}")
                pairs.append((d, float(coef)))
            if pairs:
                linked[f] = pairs

    if not changes:
        st.info("바꿀 피처를 하나 이상 골라 주세요.")
        return

    cfg = whatif.ScenarioConfig(changes=changes, linked=linked)

    flags = whatif.extrapolation_flag(Xb, cfg)
    if not flags.empty:
        outside = flags[flags["범위 밖 비율"] > 0]
        if not outside.empty:
            st.warning("학습 데이터에 없던 범위입니다. 모델이 본 적 없는 조건이므로 "
                       "예측값의 신뢰도가 떨어집니다.")
        st.dataframe(flags, use_container_width=True, hide_index=True)

    if st.button("시나리오 실행", type="primary"):
        with st.spinner("계산 중"):
            S.scenario = whatif.run_scenario(pipe, Xb, cfg)

    if S.scenario is None:
        return

    res = S.scenario
    summary = whatif.scenario_summary(res)
    cols = st.columns(len(summary))
    for c, (k, v) in zip(cols, summary.items()):
        c.metric(k, "—" if pd.isna(v) else f"{v:,.4g}")

    st.plotly_chart(plots.whatif_compare(res), use_container_width=True)
    with st.expander("시점별 결과"):
        st.dataframe(res, use_container_width=True, height=320)
        theme.csv_download("CSV 내려받기", res, "whatif.csv", "whatif")


def _sweep(pipe) -> None:
    S = st.session_state
    X = S.X
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    c1, c2, c3 = st.columns([2, 1, 1])
    f = c1.selectbox("피처", numeric, key="sweep_f")
    n_grid = c2.number_input("격자 수", 5, 100, 25)
    n_rows = c3.number_input("표본 행", 100, 5000, 500, step=100,
                             help="전 구간을 다 쓰면 느립니다. 무작위 표본으로 평균 반응을 봅니다.")

    show_ice = st.checkbox("개별 시점 곡선(ICE) 겹쳐 보기", value=True,
                           help="평균 뒤에 가려진 이질성을 봅니다. 곡선 모양이 제각각이면 "
                                "조건에 따라 반응이 달라진다는 뜻입니다.")

    if st.button("반응곡선 계산", type="primary", key="sweep_run"):
        rng = np.random.default_rng(0)
        pos = np.sort(rng.choice(len(X), size=min(int(n_rows), len(X)), replace=False))
        Xs = X.iloc[pos]
        values = whatif.suggest_range(X, f, n=int(n_grid))
        with st.spinner("계산 중"):
            curve = whatif.sweep(pipe, Xs, f, values)
            ice = whatif.ice_curves(pipe, Xs, f, values, n_lines=30) if show_ice else None
        st.plotly_chart(plots.pdp_curve(curve, f, ice), use_container_width=True)

        d = curve["prediction"]
        c1, c2, c3 = st.columns(3)
        c1.metric("반응 폭", f"{d.max() - d.min():,.4g}")
        c2.metric("최대 지점", f"{curve.loc[d.idxmax(), f]:,.4g}")
        c3.metric("최소 지점", f"{curve.loc[d.idxmin(), f]:,.4g}")
        st.dataframe(curve, use_container_width=True, hide_index=True, height=280)
