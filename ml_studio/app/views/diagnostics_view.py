"""8단계. 진단 — 결과를 믿어도 되는지 확인하는 화면.

학습 화면은 "무엇이 제일 잘 맞았나"를 보고, 이 화면은 "그 숫자를 믿어도 되나"를 본다.
누수 점검·폴드 안정성·잔차·시기별 성능·분할 방식 격차를 한자리에 모았다 (SPEC §24-7).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state
from core import diagnostics, features, models, plots, preprocess, train, validation


def render() -> None:
    S = st.session_state
    st.title("8. 진단")
    if not state.guard("diagnostics", "먼저 학습을 실행해 주세요."):
        return

    st.markdown('<p class="caption">학습 화면이 "무엇이 제일 잘 맞았나"를 본다면 '
                '이 화면은 "그 숫자를 믿어도 되나"를 봅니다.</p>', unsafe_allow_html=True)

    zoo = models.get_model_zoo(S.task or "regression",
                               include_heavy=S.X is not None and len(S.X) <= 100_000)
    metric = (S.train_config.champion_metric if S.train_config else "R2")

    tabs = st.tabs(["누수·분할", "폴드 안정성", "잔차", "Rolling Backtest", "Random vs Time"])
    with tabs[0]:
        _leakage_panel()
    with tabs[1]:
        _stability_panel()
    with tabs[2]:
        _residual_panel()
    with tabs[3]:
        _backtest_panel(metric, zoo)
    with tabs[4]:
        _split_diagnosis_panel(metric, zoo)


def _leakage_panel() -> None:
    """누수 점검표와 구간 경계를 한자리에서 다시 확인한다."""
    S = st.session_state
    if S.X is None or S.split is None:
        st.info("분할 정보가 없습니다.")
        return

    st.markdown("**구간 분할**")
    st.dataframe(S.split.describe(S.X.index), use_container_width=True, hide_index=True)

    lookback = (features.warmup_rows(S.feature_config, S.X.index)
                if S.feature_config else 0)
    check = validation.leakage_checklist(
        S.X.index, S.train_idx, S.test_idx, list(S.X.columns), S.target,
        S.provenance, S.split_config.gap if S.split_config else 0, lookback,
        selection_idx=S.selection_train_idx, unseen_idx=S.unseen_idx)

    failed = int((check["결과"] == "실패").sum())
    (st.error if failed else st.success)(
        f"{failed}개 항목이 실패했습니다." if failed else
        f"{len(check)}개 항목 전부 통과했습니다.")
    st.dataframe(check, use_container_width=True, hide_index=True)

    if S.unseen_guard is not None:
        st.caption(f"Final Unseen 접근 {S.unseen_guard.access_count}회 "
                   f"({', '.join(S.unseen_guard.accessed_by) or '아직 없음'})")

    with st.expander("교차검증 폴드 경계"):
        if S.split_config is None:
            st.caption("분할 설정이 없습니다. 3단계에서 분할을 다시 확정해 주세요.")
        else:
            try:
                st.dataframe(
                    validation.audit_splits(S.X.index,
                                            validation.make_cv(S.split_config),
                                            len(S.train_idx)),
                    use_container_width=True, hide_index=True)
            except validation.LeakageError as e:
                st.error(str(e))


def _stability_panel() -> None:
    """폴드 내부 선별이 폴드마다 얼마나 같은 피처를 고르는가."""
    S = st.session_state
    st.markdown("**폴드 간 선별 중복도 (Jaccard)**")
    st.caption("1.0 에 가까우면 선별이 안정적입니다. 낮으면 '어떤 피처가 중요한가'의 답이 "
               "구간마다 달라진다는 뜻이라 SHAP 해석까지 조심해야 합니다.")

    tbl = S.fold_stability
    if tbl is None or tbl.empty:
        st.info("폴드 내부 선별을 끄고 학습했거나 기록이 없습니다. "
                "학습 화면의 '폴드 내부에서 피처 재선별' 을 켜고 다시 학습하세요.")
        return

    worst = tbl["폴드간 Jaccard"].min()
    if worst < 0.5:
        st.warning(f"가장 낮은 Jaccard 가 {worst:.3f} 입니다. 폴드마다 다른 피처를 고르고 "
                   "있으므로, 폴드 밖에서 한 번만 선별했다면 CV 점수가 낙관 편향됐을 조건입니다.")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    pick = st.selectbox("모델별 폴드 상세", list(tbl["model"]))
    rec = S.detail.get(pick, {})
    jt = rec.get("_fold_jaccard_table")
    if jt is not None and not jt.empty:
        st.dataframe(jt, use_container_width=True, hide_index=True)
    sets = rec.get("_fold_feature_sets")
    if sets:
        with st.expander("폴드별 선택 피처"):
            for i, s in enumerate(sets, 1):
                st.caption(f"fold{i} · {len(s)}개")
                st.code(", ".join(sorted(s))[:2000] or "—")


def _residual_panel() -> None:
    """잔차 요약. 자세한 조정은 5단계 예측 화면에서 한다."""
    S = st.session_state
    if S.predictions is None:
        st.info("5단계 예측을 먼저 실행하면 잔차 진단이 여기에도 표시됩니다.")
        return

    res = S.predictions.dropna()
    r = diagnostics.residuals(res["actual"], res["predicted"])
    if len(r) < 10:
        st.info("잔차가 너무 적습니다.")
        return

    cfg = diagnostics.ResidualConfig(window=min(96, max(6, len(r) // 10)))
    s = diagnostics.summary(r, cfg)
    cols = st.columns(5)
    cols[0].metric("평균", f"{s['mean']:,.4f}")
    cols[1].metric("표준편차", f"{s['std']:,.4f}")
    cols[2].metric("MAE", f"{s['MAE']:,.4f}")
    cols[3].metric("lag1 자기상관", f"{s['lag1_acf']:,.3f}")
    cols[4].metric("이상점", f"{s['outliers']:,}")

    drift = diagnostics.drift_table(r, cfg)
    if not drift.empty:
        v = diagnostics.drift_verdict(drift)
        (st.warning if v["drift"] else st.info)(v["message"])
        st.plotly_chart(plots.residual_drift(drift), use_container_width=True)

    st.plotly_chart(plots.residual_band(diagnostics.rolling_stats(r, cfg),
                                        diagnostics.outliers(r, cfg)),
                    use_container_width=True)
    acf = diagnostics.autocorrelation(r, cfg)
    if not acf.empty:
        st.plotly_chart(plots.residual_acf(acf, len(r)), use_container_width=True)


def _backtest_panel(metric: str, zoo: dict) -> None:
    """Rolling Backtest — 한 번의 점수가 운이었는지 실력이었는지 가른다."""
    S = st.session_state
    st.header("Rolling Backtest")
    st.markdown('<p class="caption">시기를 굴리며 같은 절차로 재학습·평가합니다. '
                '구간 간 편차가 크면 그 모델은 시기를 탄다는 뜻입니다.</p>',
                unsafe_allow_html=True)

    ok = S.leaderboard[S.leaderboard["status"] == "ok"]
    pool = [m for m in ok["model"] if m in zoo]
    if not pool:
        st.info("backtest 를 돌릴 단일 모델이 없습니다. 앙상블은 대상이 아닙니다.")
        return

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    default_i = pool.index(S.champion) if S.champion in pool else 0
    pick = c1.selectbox("대상 모델", pool, index=default_i)
    n_folds = c2.number_input("구간 수", 2, 12, 5)
    mode = c3.selectbox("학습 구간", ["누적 (expanding)", "고정 (sliding)"])
    c4.caption("앙상블은 base 재학습이 필요해 제외합니다.")

    if st.button("Backtest 실행", disabled=S.X is None):
        num, cat = preprocess.split_column_types(S.X)
        pre = preprocess.build_preprocessor(num, cat, S.prep_config or preprocess.PreprocessConfig())
        bar, label = st.progress(0.0), st.empty()

        def tick(i, total, when):
            bar.progress(i / total)
            label.caption(f"{i}/{total} · {when} 구간")

        try:
            table, stitched = train.rolling_backtest(
                S.X, S.y, pre, zoo[pick], S.train_config,
                n_folds=int(n_folds), expanding=mode.startswith("누적"), progress=tick)
            bar.progress(1.0)
            label.empty()
            S.backtest = {"model": pick, "table": table, "pred": stitched}
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    bt = S.backtest
    if not bt:
        return

    table = bt["table"]
    summ = train.backtest_summary(table, metric)
    if summ:
        cols = st.columns(5)
        cols[0].metric("구간 수", summ["구간수"])
        cols[1].metric(f"{metric} 평균", f"{summ['평균']:.4f}")
        cols[2].metric("표준편차", f"{summ['표준편차']:.4f}",
                       help="크면 시기를 탑니다. 한 번의 홀드아웃 점수를 믿기 어렵습니다.")
        cols[3].metric("최저", f"{summ['최저']:.4f}")
        cols[4].metric("최악 구간", f"{summ['최악구간']}구간",
                       help=f"{pd.Timestamp(summ['최악시작']):%Y-%m-%d} 시작")

    st.caption(f"대상 **{bt['model']}**")
    st.plotly_chart(plots.backtest_series(table, metric), use_container_width=True)
    show = [c for c in ("구간", "학습", "평가시작", "평가끝", "n_train", "n_test",
                        metric, "RMSE", "MAE", "fit_seconds", "status")
            if c in table.columns]
    st.dataframe(table[show], use_container_width=True, hide_index=True, height=280)


def _split_diagnosis_panel(metric: str, zoo: dict) -> None:
    """Random vs Time — 진단 전용. 챔피언 선정에는 절대 쓰지 않는다 (G-3)."""
    S = st.session_state
    st.header("Random vs Time 진단")
    st.markdown('<p class="caption">같은 모델을 무작위 분할로도 평가해 격차를 봅니다. '
                '무작위 분할은 검증 행의 바로 앞뒤를 학습에 넣으므로 그 점수는 '
                '미래 성능이 아닙니다. 아래 결과는 진단용이며 리더보드·챔피언·리포트 '
                '어디에도 반영되지 않습니다.</p>', unsafe_allow_html=True)

    ok = S.leaderboard[S.leaderboard["status"] == "ok"]
    pool = [m for m in ok["model"] if m in zoo]
    if not pool:
        st.info("진단할 단일 모델이 없습니다.")
        return

    c1, c2 = st.columns([3, 1])
    picks = c1.multiselect("진단할 모델", pool, default=pool[:3])
    th = c2.slider("격차 임계", 0.05, 0.5, 0.15, 0.05)

    if st.button("진단 실행", disabled=not picks):
        num, cat = preprocess.split_column_types(S.X)
        pre = preprocess.build_preprocessor(num, cat, S.prep_config or preprocess.PreprocessConfig())
        bar, label = st.progress(0.0), st.empty()

        def tick(i, total, name):
            bar.progress(i / total)
            label.caption(f"{i}/{total} · {name}")

        try:
            S.split_diag = train.random_vs_time(
                S.X, S.y, pre, zoo, picks, S.train_config,
                threshold=float(th), progress=tick)
        except Exception as e:  # noqa: BLE001
            # 모델 하나가 터져도 화면 전체가 빨간 트레이스로 덮이면 안 된다.
            bar.empty()
            label.empty()
            st.error(f"진단 실행 실패 — {type(e).__name__}: {e}")
            return
        bar.progress(1.0)
        label.empty()
        st.rerun()

    diag = S.get("split_diag")
    if not diag:
        return

    v = diag["verdict"]
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 격차", f"{diag['mean_gap']:+.4f}",
              help="Random 이 Time 보다 얼마나 좋게 나왔는지입니다.")
    c2.metric("타겟 lag1 자기상관", f"{v.get('lag1_acf', float('nan')):.3f}")
    c3.metric("타겟 분포 이동", f"{v.get('y_shift_sd', float('nan')):.2f}σ")

    m = diag["metric"]
    cols = ["model", f"time_{m}", f"random_{m}", "격차"]
    st.dataframe(diag["table"][[c for c in cols if c in diag["table"].columns]],
                 use_container_width=True, hide_index=True)

    if v["significant"]:
        st.warning(f"격차가 임계({v['threshold']:.2f})를 넘습니다. "
                   "Time 쪽 숫자만 실제 성능으로 보세요.")
        st.markdown("**원인 후보**")
        st.dataframe(pd.DataFrame(v["causes"]), use_container_width=True, hide_index=True)
    else:
        st.info(f"격차가 임계({v['threshold']:.2f}) 안입니다. "
                "시간 의존이 강하지 않은 데이터로 보입니다.")
