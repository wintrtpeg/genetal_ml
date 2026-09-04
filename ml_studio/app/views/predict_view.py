"""5단계. 선택 구간 예측과 실측 대비 시각화."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state, theme
from core import diagnostics, plots, train


def render() -> None:
    S = st.session_state
    st.title("5. 예측")
    if not state.guard("predict", "먼저 지도학습으로 챔피언 모델을 확정해 주세요."):
        return

    pipe = state.champion_pipeline()
    if pipe is None:
        st.error("챔피언 모델을 찾지 못했습니다. 학습을 다시 실행해 주세요.")
        return

    X, y = S.X, S.y
    valid_start = X.index[S.test_idx[0]]
    has_unseen = S.unseen_idx is not None and len(S.unseen_idx) > 0
    unseen_start = X.index[S.unseen_idx[0]] if has_unseen else None

    st.header("구간 선택")
    c1, c2, c3 = st.columns([2, 2, 1.4])
    lo, hi = X.index[0].date(), X.index[-1].date()
    start = c1.date_input("시작", value=lo, min_value=lo, max_value=hi)
    end = c2.date_input("종료", value=hi, min_value=lo, max_value=hi)
    quick = c3.selectbox("빠른 지정", ["직접 지정", "학습 구간", "검증 구간"]
                         + (["Final Unseen"] if has_unseen else []))

    if quick == "학습 구간":
        start, end = lo, (valid_start - pd.Timedelta(days=1)).date()
    elif quick == "검증 구간":
        start = valid_start.date()
        end = ((unseen_start - pd.Timedelta(days=1)).date() if has_unseen
               else X.index[-1].date())
    elif quick == "Final Unseen":
        start, end = unseen_start.date(), X.index[-1].date()

    # 예측 자체는 Unseen 접근 횟수에 안 잡히지만, 여기 그림을 보고 모델을 바꾸면
    # 그 구간은 더 이상 미접촉이 아니다. 그 경계를 화면에서 짚어 준다.
    if has_unseen and pd.Timestamp(end) >= unseen_start.normalize():
        st.warning(
            "선택 구간에 **Final Unseen** 이 들어 있습니다. 예측을 그려 보는 것은 "
            "접근 횟수에 포함되지 않지만, 여기서 본 결과를 근거로 모델·피처·하이퍼파라미터를 "
            "바꾸면 그 구간은 더 이상 미접촉 구간이 아닙니다. "
            "그때는 보고 성능도 낙관 편향된 값이 됩니다.")

    if st.button("예측 실행", type="primary"):
        with st.spinner("계산 중"):
            pred = train.predict_range(pipe, X, start, pd.Timestamp(end) + pd.Timedelta(days=1))
        S.predictions = pd.DataFrame({"actual": y.reindex(pred.index), "predicted": pred})
        st.rerun()

    if S.predictions is None:
        st.info("구간을 정하고 예측을 실행해 주세요.")
        return

    res = S.predictions.dropna()
    if res.empty:
        st.warning("선택 구간에 유효한 값이 없습니다.")
        return

    st.divider()
    _metrics(res, valid_start, unseen_start)

    st.plotly_chart(
        plots.actual_vs_pred(res["actual"], res["predicted"], train_end=valid_start,
                             title=f"{S.target} — 실측 대비 예측", ylabel=S.target),
        **theme.WIDE)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(plots.residual_series(res["actual"], res["predicted"]),
                        **theme.WIDE)
    with c2:
        st.plotly_chart(plots.scatter_actual_pred(res["actual"], res["predicted"]),
                        **theme.WIDE)

    st.divider()
    _residual_diagnostics(res)

    with st.expander("예측값 표"):
        st.dataframe(res.assign(residual=res["actual"] - res["predicted"]),
                     **theme.WIDE, height=340)
        theme.csv_download("CSV 내려받기", res, f"predictions_{S.target}.csv", "predict")


def _residual_diagnostics(res: pd.DataFrame) -> None:
    """잔차 진단 — R2 한 숫자로는 안 보이는 '어디서 어떻게 틀렸는가'."""
    st.header("잔차 진단")
    st.markdown('<p class="caption">잔차가 백색잡음에 가까우면 뽑아낼 구조를 다 뽑은 것입니다. '
                '특정 구간에 편향이 몰리거나 자기상관이 남아 있으면 아직 여지가 있습니다.</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    window = c1.number_input("rolling 창 (행)", 6, 5000, min(96, max(6, len(res) // 10)))
    n_seg = c2.slider("drift 구간 수", 2, 12, 6)
    sigma = c3.slider("이상점 기준 (robust z)", 2.0, 6.0, 3.0, 0.5)
    cfg = diagnostics.ResidualConfig(window=int(window), n_segments=int(n_seg),
                                     outlier_sigma=float(sigma))

    r = diagnostics.residuals(res["actual"], res["predicted"])
    if len(r) < 10:
        st.info("잔차가 너무 적어 진단을 건너뜁니다.")
        return

    s = diagnostics.summary(r, cfg)
    cols = st.columns(5)
    cols[0].metric("평균", f"{s['mean']:,.4f}", help="0 에서 멀면 계통 편향입니다.")
    cols[1].metric("표준편차", f"{s['std']:,.4f}")
    cols[2].metric("MAE", f"{s['MAE']:,.4f}")
    cols[3].metric("lag1 자기상관", f"{s['lag1_acf']:,.3f}",
                   help="0 에 가까워야 합니다. 크면 못 뽑아낸 시간 구조가 남아 있습니다.")
    cols[4].metric("이상점", f"{s['outliers']:,}")

    if abs(s["bias_ratio"]) > 0.2:
        st.warning(f"잔차 평균이 절대평균의 {s['bias_ratio']:+.0%} 입니다. "
                   f"모델이 전반적으로 {'과소' if s['bias_ratio'] > 0 else '과대'}예측하고 있습니다.")

    out = diagnostics.outliers(r, cfg)
    st.plotly_chart(plots.residual_band(diagnostics.rolling_stats(r, cfg), out),
                    **theme.WIDE)

    drift = diagnostics.drift_table(r, cfg)
    if not drift.empty:
        verdict = diagnostics.drift_verdict(drift)
        (st.warning if verdict["drift"] else st.info)(verdict["message"])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(plots.residual_drift(drift), **theme.WIDE)
        with c2:
            st.dataframe(drift[["구간", "행수", "mean", "std", "MAE", "MAE_배율"]].round(4),
                         **theme.WIDE, hide_index=True, height=300)

    acf = diagnostics.autocorrelation(r, cfg)
    c1, c2 = st.columns([3, 2])
    with c1:
        if not acf.empty:
            st.plotly_chart(plots.residual_acf(acf, len(r)), **theme.WIDE)
    with c2:
        st.markdown(f"**이상점 {len(out):,}건**")
        st.caption("중앙값·MAD 기준입니다. 평균을 쓰면 이상점이 스스로를 정상으로 만듭니다.")
        if out.empty:
            st.info("기준을 넘는 이상점이 없습니다.")
        else:
            st.dataframe(out.head(200), **theme.WIDE, height=300)


def _metrics(res: pd.DataFrame, valid_start, unseen_start=None) -> None:
    """구간별 성능. 이름을 분할과 정확히 맞춘다.

    3분할에서 '검증 구간' 은 모델을 고르는 데 이미 쓰인 구간이다. 이것을 '홀드아웃
    (unseen)' 이라 부르면 미접촉 구간으로 오해하게 되고, 그 점수를 최종 성능으로
    보고하는 사고로 이어진다.
    """
    S = st.session_state
    task = S.task or "regression"

    parts = [
        ("학습 구간 (in-sample)", res[res.index < valid_start],
         "모델이 직접 본 구간입니다. 성능이 좋은 것은 당연합니다."),
    ]
    if unseen_start is not None:
        parts.append(("검증 구간 (모델 선택에 사용됨)",
                      res[(res.index >= valid_start) & (res.index < unseen_start)],
                      "이 점수로 챔피언을 골랐습니다. 모델 수만큼 선택 편향이 들어 있습니다."))
        parts.append(("Final Unseen (최종 보고)", res[res.index >= unseen_start],
                      "학습·선별·모델선택 어디에도 쓰이지 않은 구간입니다. 이 값을 보고하세요."))
    else:
        parts.append(("홀드아웃 (2분할 — 선택과 보고 겸용)", res[res.index >= valid_start],
                      "Final Unseen 이 없어 이 점수가 모델 선택과 보고를 겸합니다. 낙관 편향됩니다."))

    st.header("성능")
    st.markdown(f'<p class="caption">챔피언 <b>{S.champion}</b> · '
                f'검증 시작 {valid_start:%Y-%m-%d %H:%M}'
                + (f' · Final Unseen 시작 {unseen_start:%Y-%m-%d %H:%M}'
                   if unseen_start is not None else '')
                + '</p>', unsafe_allow_html=True)

    for label, part, note in parts:
        if part.empty:
            continue
        s = train.score(task, part["actual"], part["predicted"])
        cols = st.columns(len(s) + 1)
        cols[0].markdown(f"**{label}**  \n<span class='caption'>{len(part):,}행</span>",
                         unsafe_allow_html=True)
        for c, (k, v) in zip(cols[1:], s.items()):
            c.metric(k, "—" if pd.isna(v) else f"{v:,.4f}")
        st.markdown(f'<p class="caption" style="margin-top:-8px">{note}</p>',
                    unsafe_allow_html=True)
