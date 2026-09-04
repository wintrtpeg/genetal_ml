"""4단계. 지도학습 또는 비지도학습."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app import state, theme
from core import (
    ensemble, models, plots, preprocess, train, tuning, unsupervised, validation,
)


def render() -> None:
    S = st.session_state
    st.title("4. 학습")
    if not state.guard("train", "먼저 파생변수를 생성해 주세요."):
        return

    mode = st.radio("학습 방식", ["지도학습", "비지도학습"], horizontal=True,
                    help="지도학습은 타겟(Y)을 맞히는 모델을 만듭니다. "
                         "비지도학습은 Y 없이 군집·이상탐지·주성분을 봅니다.",
                    index=0 if S.learning_mode == "지도학습" else 1)
    S.learning_mode = mode
    st.divider()
    (_supervised if mode == "지도학습" else _unsupervised)()


# ─────────────────────────────────────────────────────────────
def _supervised() -> None:
    S = st.session_state
    # X 와 y 는 3단계에서 함께 만들어진다. 한쪽만 있으면 상태가 깨진 것이므로
    # 같은 안내로 돌려보낸다 — detect_task(None) 은 AttributeError 를 낸다.
    if S.X is None or S.y is None:
        if S.feature_review is not None:
            st.warning("3단계에서 품질 리포트까지는 만들었지만 아직 **확정 전**입니다. "
                       "파생변수 화면으로 돌아가 X 피처를 확정해 주세요.")
        else:
            st.info("파생변수 화면에서 품질 리포트를 만들고 X 피처를 확정해 주세요.")
        return

    X, y = S.X, S.y
    task = models.detect_task(y)
    S.task = task
    metric_options = (train.REGRESSION_METRICS if task == models.TASK_REGRESSION
                      else train.CLASSIFICATION_METRICS)

    st.header("검증 설계")
    if S.split is None:
        st.info("파생변수 화면에서 선별과 분할을 먼저 실행해 주세요.")
        return

    sp = S.split
    tr, te = sp.train, sp.valid
    base_cfg = S.split_config or validation.SplitConfig()

    st.markdown('<p class="caption">구간 분할은 3단계에서 확정됩니다. 여기서는 바꾸지 않습니다 — '
                '학습 화면에서 다시 나누면 선별에 쓴 구간이 평가 구간으로 넘어갑니다.</p>',
                unsafe_allow_html=True)
    st.dataframe(sp.describe(X.index), **theme.WIDE, hide_index=True)

    c1, c2, c3 = st.columns(3)
    # 불러온 설정의 폴드 수가 범위를 벗어나 있으면 위젯이 예외를 던진다.
    n_splits = c1.number_input("교차검증 폴드", 2, 10, min(10, max(2, int(base_cfg.n_splits))),
                               help="학습 구간 안에서 시간순으로 몇 번 나눠 평가할지입니다. "
                                    "많을수록 안정적이지만 그만큼 오래 걸립니다.")
    metric = c2.selectbox("챔피언 기준", metric_options,
                          help="R2 는 1에 가까울수록 좋고, RMSE·MAE 는 0에 가까울수록 "
                               "좋습니다. 보통 R2 로 고르고 RMSE 를 함께 봅니다.")
    # 이 버튼은 여기서 분할을 바꾸는 게 아니라 3단계로 되돌린다. 이름이 그렇게 읽혀야 한다.
    if c3.button("3단계로 돌아가 다시 나누기",
                 help="지금까지의 학습 결과를 지우고 피처 선별 화면으로 되돌립니다."):
        state.invalidate("split")
        st.warning("3단계로 돌아가 선별을 다시 실행해 주세요.")
        st.stop()

    gap = int(base_cfg.gap)
    split = validation.SplitConfig(
        holdout_ratio=base_cfg.holdout_ratio, unseen_ratio=base_cfg.unseen_ratio,
        n_splits=int(n_splits), gap=gap,
        valid_cut=base_cfg.valid_cut, unseen_cut=base_cfg.unseen_cut)
    S.split_config, S.train_idx, S.test_idx = split, tr, te

    check = validation.leakage_checklist(
        X.index, tr, te, list(X.columns), S.target, S.provenance,
        gap, _max_lookback(),
        selection_idx=S.selection_train_idx, unseen_idx=S.unseen_idx)
    failed = (check["결과"] == "실패").any()

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**누수 점검**")
        st.dataframe(check, **theme.WIDE, hide_index=True)
    with c2:
        st.markdown("**교차검증 폴드**")
        try:
            st.dataframe(validation.audit_splits(X.index, validation.make_cv(split), len(tr)),
                         **theme.WIDE, hide_index=True, height=240)
        except validation.LeakageError as e:
            st.error(str(e))
            failed = True

    if failed:
        st.error("점검을 통과하지 못했습니다. 학습을 진행하지 않습니다.")
        return

    st.header("모델")
    zoo = models.get_model_zoo(task, include_heavy=len(X) <= 100_000)
    default = models.default_selection(zoo, len(X))
    selected = st.multiselect("학습할 모델", list(zoo), default=default)

    c1, c2, c3 = st.columns([1, 1, 2])
    n_jobs = c1.selectbox("병렬", [-1, 1, 2, 4, 8], index=0,
                          format_func=lambda v: "전체 코어" if v == -1 else f"{v} 프로세스")
    show_progress = c2.checkbox("진행상황 표시", value=True,
                                help="어느 모델까지 끝났는지 막대로 보여줍니다. "
                                     "켜도 병렬은 그대로 쓰므로 느려지지 않습니다.")
    missing = [n for n in ("XGBoost", "LightGBM", "CatBoost") if n not in zoo]
    if missing:
        c3.caption(f"미설치로 제외: {', '.join(missing)}")

    # 고급 설정은 Expert 모드에서만 노출한다. 노출하지 않아도 기본값은 같다 —
    # 폴드 내부 선별은 켜져 있고, 앙상블 임계값은 3% 다.
    fold_sel, sel_top_k, ens_th = True, 0, 0.03
    tune_cfg = None
    if state.at_least("Expert"):
        with st.expander("선별·앙상블 설정", expanded=False):
            c1, c2, c3 = st.columns(3)
            fold_sel = c1.checkbox(
                "폴드 내부에서 피처 재선별", value=True,
                help="켜면 CV 폴드마다 선별을 다시 합니다. 끄면 CV 점수가 낙관 편향됩니다. "
                     "비용은 후보 피처 수에 비례합니다 — 후보가 이미 추려져 있으면 거의 공짜지만, "
                     "200개에서 40개를 고르는 조건에서는 학습시간이 몇 배가 됩니다.")
            sel_top_k = c2.number_input("폴드별 상위 개수 (0=제한없음)", 0, 500,
                                        0, disabled=not fold_sel)
            ens_th = c3.slider(
                "앙상블 자동채택 임계값", 0.0, 0.20, 0.03, 0.01, format="%.2f",
                help="단일 최고 모델 대비 이만큼 좋아져야 앙상블을 챔피언으로 삼습니다. "
                     "SPEC §17 — 미미한 개선이면 복잡한 모델을 자동 선택하지 않습니다.")

        with st.expander("하이퍼파라미터 탐색 (nested CV)", expanded=False):
            st.caption("파라미터를 고르는 데 쓴 구간으로 성능까지 보고하면 홀드아웃을 "
                       "겸용했을 때와 같은 편향이 생깁니다. 그래서 폴드 안에서 한 겹 더 "
                       "나눠 파라미터를 고르고, 바깥 폴드로만 점수를 냅니다.")
            c1, c2, c3 = st.columns(3)
            do_tune = c1.checkbox("탐색 켜기", value=False)
            n_iter = c2.number_input("조합 수", 4, 60, 12, disabled=not do_tune,
                                     help="조합 수 × 안쪽 폴드 수만큼 학습이 늘어납니다.")
            inner = c3.number_input("안쪽 폴드", 2, 6, 3, disabled=not do_tune)
            tune_targets = st.multiselect(
                "탐색할 모델 (비우면 전부)",
                [m for m in selected if tuning.tunable(m)],
                default=[], disabled=not do_tune)
            if do_tune:
                cost = int(n_iter) * int(inner)
                st.warning(f"모델당 학습 횟수가 약 {cost}배가 됩니다. "
                           "모델을 좁혀서 쓰는 편이 낫습니다.")
                tune_cfg = tuning.TuneConfig(
                    enabled=True, n_iter=int(n_iter), inner_splits=int(inner),
                    models=tune_targets)
    else:
        st.caption("폴드 내부 선별 ON · 앙상블 자동채택 임계 3% · 하이퍼파라미터 탐색 OFF "
                   "(Expert 모드에서 조절)")

    if st.button("학습 시작", type="primary", disabled=not selected):
        num, cat = preprocess.split_column_types(X)
        pre = preprocess.build_preprocessor(num, cat, S.prep_config or preprocess.PreprocessConfig())
        cfg = train.TrainConfig(
            task=task, split=split, n_jobs=n_jobs, champion_metric=metric,
            fold_selection=bool(fold_sel),
            selection_top_k=int(sel_top_k) or None,
            ensemble_threshold=float(ens_th),
            tune=tune_cfg)
        S.train_config = cfg

        bar, label = st.progress(0.0), st.empty()
        started = time.monotonic()

        def tick(i, total, name):
            # i 는 **끝난 개수**다. 예전에는 시작 전에 세서 막대가 실제보다
            # 한 칸 앞서 있었다 — 첫 모델을 붙잡고 있는데 10% 가 차 있었다.
            #
            # 경과 시간을 같이 적는다. 모델 하나가 몇 분씩 걸리는 일이 흔한데
            # (RandomForest 는 나무 400 그루다) 숫자가 안 움직이면 멈춘 것처럼
            # 보인다. 시간이 흐르는 것만 보여도 "돌고 있다" 가 전해진다.
            bar.progress(i / total)
            el = int(time.monotonic() - started)
            label.caption(f"{i}/{total} 완료 · {name} · {el // 60}분 {el % 60}초 경과")

        try:
            board, detail = train.train_all(
                X, y, tr, te, pre, zoo, selected, cfg,
                progress=tick if show_progress else None)
            bar.progress(1.0)
            label.empty()
            S.leaderboard, S.detail = board, detail
            S.champion = train.pick_champion(board, metric)
            S.fold_stability = _stability_table(detail)
            state.invalidate("train")
            # 여기서 guard 를 새로 만들면 안 된다. 학습을 다시 돌릴 때마다 접근
            # 횟수가 0 으로 돌아가서, 같은 분할로 Final Unseen 을 몇 번이든 열어
            # 모델을 고를 수 있게 된다. guard 는 3단계에서 분할과 함께 만든다.
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"학습 실패 — {type(e).__name__}: {e}")

    if S.leaderboard is None:
        return

    st.divider()
    _leaderboard_panel(metric)
    st.divider()
    _ensemble_panel(metric, zoo)
    st.divider()
    _unseen_panel(metric)
    st.caption("Rolling Backtest 와 Random vs Time 진단은 8단계 진단 화면에 있습니다.")


def _stability_table(detail: dict) -> pd.DataFrame:
    """폴드 내부 선별의 안정성 — 폴드 간 Jaccard 중복도."""
    rows = []
    for name, rec in detail.items():
        if rec.get("status") != "ok" or "fold_jaccard" not in rec:
            continue
        rows.append({"model": name,
                     "폴드평균 피처수": round(rec.get("fold_features_mean", float("nan")), 1),
                     "폴드간 Jaccard": round(rec["fold_jaccard"], 4)})
    return pd.DataFrame(rows)


def _unseen_panel(metric: str) -> None:
    """Final Unseen — 챔피언 확정 뒤 단 한 번만 연다."""
    S = st.session_state
    st.header("Final Unseen 평가")
    if S.unseen_idx is None or not len(S.unseen_idx):
        st.info("2분할(구버전 호환) 모드입니다. Final Unseen 구간이 없어 홀드아웃 점수가 "
                "모델 선택과 최종 보고를 겸합니다. 3단계에서 Unseen 비율을 주면 분리됩니다.")
        return

    st.markdown('<p class="caption">여기까지 오는 동안 이 구간은 학습·선별·모델선택 어디에도 '
                '쓰이지 않았습니다. 한 번 열면 다시 열 수 없습니다.</p>', unsafe_allow_html=True)

    guard = S.unseen_guard
    spent = guard is not None and getattr(guard, "access_count", 0) >= 1

    if S.unseen_scores is None and spent:
        # 평가한 뒤 다시 학습하면 점수는 지워지지만 접근권은 돌아오지 않는다.
        # 버튼을 눌러 보고 예외로 알게 하는 대신 여기서 미리 말해 준다.
        who = (guard.accessed_by or ["?"])[0]
        st.error(
            f"이 분할의 Final Unseen 은 이미 **{who}** 로 한 번 열렸습니다. "
            "그 뒤 학습을 다시 돌리셨기 때문에 점수 표시는 지워졌지만, 접근권은 "
            "돌아오지 않습니다.\n\n"
            "지금 챔피언의 최종 성능을 보려면 **3단계로 돌아가 분할부터 다시** 하세요. "
            "그렇게 하지 않고 같은 구간을 다시 열면, 그 구간은 모델을 고르는 데 쓰인 "
            "것이 되어 더 이상 최종 보고에 쓸 수 없습니다.")
        st.caption("검증 점수는 위 리더보드에 그대로 있습니다. 다만 그 값에는 "
                   "모델 수만큼의 선택 편향이 들어 있습니다.")
        return

    if S.unseen_scores is None:
        if guard is None:
            st.warning("이 분할의 Unseen 접근권이 없습니다. 3단계에서 "
                       "[이 목록으로 확정] 을 다시 눌러 분할을 확정해 주세요.")
            return
        st.warning(f"챔피언 **{S.champion}** 으로 최종 평가를 실행합니다. "
                   "실행 후 모델을 바꾸려면 분할부터 다시 해야 합니다.")
        if st.button("Final Unseen 평가 실행", type="primary"):
            pipe = state.champion_pipeline()
            try:
                S.unseen_scores = train.evaluate_unseen(
                    pipe, S.X, S.y, S.unseen_idx, S.train_config,
                    S.unseen_guard, who=S.champion)
                st.rerun()
            except train.UnseenAccessError as e:
                st.error(str(e))
        return

    sc = S.unseen_scores
    cols = st.columns(min(4, max(1, len(sc) - 1)))
    for c, (k, v) in zip(cols, [(k, v) for k, v in sc.items() if k != "unseen_rows"]):
        c.metric(k.replace("unseen_", ""), f"{v:.4f}")
    st.caption(f"{int(sc.get('unseen_rows', 0)):,}행 · 접근 1회")

    bias = train.selection_bias_report(S.leaderboard, S.champion, sc, metric)
    st.markdown("**선택 편향 크기**")
    st.caption(
        "검증 점수와 Final Unseen 점수의 차이입니다. 검증 구간은 여러 모델 중 하나를 "
        "고르는 데 이미 쓰였으므로, 그 점수에는 '운 좋게 잘 맞은 정도'가 섞여 있습니다. "
        "그 거품이 얼마였는지를 여기서 처음 보게 됩니다.")
    st.dataframe(bias, **theme.WIDE, hide_index=True)
    st.caption(
        "읽는 법 — 차이가 작으면(대략 검증 점수의 5% 안쪽) 모델 선택이 안정적이었다는 "
        "뜻입니다. 차이가 크면 검증 구간에만 맞춘 모델일 수 있으니, 후보 모델 수를 줄이거나 "
        "폴드 수를 늘려 다시 돌려 보세요. **보고할 숫자는 언제나 Final Unseen 쪽입니다.**")


def _max_lookback() -> int:
    from core import features
    S = st.session_state
    return (features.warmup_rows(S.feature_config, S.X.index if S.X is not None else None)
            if S.feature_config else 0)


def _leaderboard_panel(metric: str) -> None:
    S = st.session_state
    board = S.leaderboard
    st.header("리더보드")
    three_way = S.unseen_idx is not None and len(S.unseen_idx) > 0
    st.markdown('<p class="caption">cv_ 는 학습 구간 내 교차검증 평균입니다. '
                + ('holdout_ 은 <b>검증 구간</b> 성능이며, 아래에서 챔피언을 고르는 데 '
                   '쓰입니다 — 최종 성능은 Final Unseen 쪽을 보세요.'
                   if three_way else
                   'holdout_ 은 마지막 구간 성능이며, 2분할이라 모델 선택과 보고를 겸합니다.')
                + '</p>', unsafe_allow_html=True)

    ok = board[board["status"] == "ok"]
    if ok.empty:
        st.error("성공한 모델이 없습니다.")
        st.dataframe(board, **theme.WIDE, hide_index=True)
        return

    cols = ["rank", "model", "family"]
    cols += [c for c in board.columns if c.startswith("cv_") and not c.endswith("_std")]
    cols += [c for c in board.columns if c.startswith("holdout_")]
    cols += [c for c in ("insample_R2", "fit_seconds") if c in board.columns]
    st.dataframe(board[[c for c in cols if c in board.columns]],
                 **theme.WIDE, hide_index=True, height=340)

    st.plotly_chart(plots.leaderboard_bar(board, metric), **theme.WIDE)

    gap_rows = []
    for _, r in ok.iterrows():
        ins, hold = r.get(f"insample_{metric}"), r.get(f"holdout_{metric}")
        if pd.notna(ins) and pd.notna(hold):
            gap_rows.append({"model": r["model"], "학습": round(ins, 4),
                             "검증": round(hold, 4), "차이": round(ins - hold, 4)})
    if gap_rows:
        with st.expander("과적합 점검 (학습 대비 검증)"):
            st.caption("차이가 크면 학습 구간에만 맞춰진 모델입니다.")
            st.dataframe(pd.DataFrame(gap_rows), **theme.WIDE, hide_index=True)

    tuned = [n for n, r in S.detail.items() if r.get("tuned")]
    if tuned:
        with st.expander(f"하이퍼파라미터 탐색 결과 {len(tuned)}개 모델"):
            st.caption("아래 cv_ 점수는 nested CV 로 낸 값입니다. 파라미터를 고른 구간과 "
                       "점수를 낸 구간이 분리돼 있어 낙관 편향이 없습니다.")
            for n in tuned:
                rec = S.detail[n]
                st.markdown(f"**{n}** — 최종 선택: `{rec.get('best_params', '-')}`")
                ps = rec.get("_param_stability")
                if ps is not None and not ps.empty:
                    st.dataframe(ps, **theme.WIDE, hide_index=True)
                    if (ps["안정성"] == "흔들림").any():
                        st.caption("폴드마다 다른 값이 뽑힌 파라미터가 있습니다. "
                                   "그 축은 데이터가 결정해 주지 않는다는 뜻이라 "
                                   "튜닝 결과를 그대로 믿기 어렵습니다.")

    if S.fold_stability is not None and not S.fold_stability.empty:
        with st.expander("폴드 내부 선별 안정성"):
            st.caption("폴드마다 다시 선별한 결과의 중복도입니다. Jaccard 가 낮으면 어떤 피처를 "
                       "고르는지가 구간마다 흔들린다는 뜻이라, 피처 해석을 조심해야 합니다.")
            st.dataframe(S.fold_stability, **theme.WIDE, hide_index=True)

    failed = board[board["status"] != "ok"]
    if not failed.empty:
        with st.expander(f"실패한 모델 {len(failed)}개"):
            st.dataframe(failed[["model", "error"]], **theme.WIDE, hide_index=True)

    names = list(ok["model"])
    pick = st.selectbox("챔피언", names, index=names.index(S.champion) if S.champion in names else 0)
    if pick != S.champion:
        S.champion = pick
        state.invalidate("train")
    seg = "검증" if (S.unseen_idx is not None and len(S.unseen_idx)) else "홀드아웃"
    st.success(f"챔피언 **{S.champion}** · {seg} {metric} "
               f"{ok[ok['model'] == S.champion][f'holdout_{metric}'].iloc[0]:.4f}")


def _ensemble_panel(metric: str, zoo: dict) -> None:
    S = st.session_state
    st.header("앙상블 · 스태킹")
    st.markdown('<p class="caption">여러 모델의 예측을 합쳐 하나로 만듭니다. '
                '서로 다른 방식으로 틀리는 모델끼리 합칠 때 효과가 큽니다.</p>',
                unsafe_allow_html=True)

    with st.expander("합치는 방식 세 가지"):
        st.markdown(
            "- **보팅** — 구성 모델의 예측을 단순 평균합니다. 가장 단순하고, "
            "특정 모델이 크게 틀린 구간을 나머지가 덜어 줍니다.\n"
            "- **가중** — 평균 대신 모델마다 다른 비중을 줍니다. 비중은 nnls 로 "
            "구하며 음수가 없고 합이 1 이라, '이 모델을 몇 % 믿는다'로 읽힙니다.\n"
            "- **스태킹** — 구성 모델의 예측을 입력으로 받는 모델을 하나 더 얹습니다. "
            "이때 그 입력은 **OOF 예측**이어야 합니다.\n\n"
            "**OOF(out-of-fold) 란** — 구성 모델이 학습에 쓰지 않은 구간에 대해 낸 "
            "예측입니다. 학습에 썼던 구간의 예측을 쓰면 이미 정답을 본 뒤의 예측이라 "
            "실제보다 훨씬 정확해 보이고, 위에 얹는 모델이 그 거품을 그대로 배웁니다. "
            "이 도구는 시간순 분할로 OOF 를 만듭니다 — 기본 KFold 를 쓰면 미래 구간이 "
            "과거 예측에 섞입니다.")

    ok = S.leaderboard[S.leaderboard["status"] == "ok"]
    base_pool = [m for m in ok["model"] if m in zoo]
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    bases = c1.multiselect("구성 모델", base_pool, default=base_pool[:3],
                           help="2개 이상 골라야 합칠 수 있습니다. 성격이 다른 모델을 "
                                "섞을수록 효과가 큽니다.")
    voting = c2.checkbox("보팅", value=True, help="예측을 단순 평균합니다.")
    weighted = c3.checkbox("가중", value=True,
                           help="모델별 비중을 nnls 로 구합니다. 음수 없이 합이 1 입니다.")
    stacking = c4.checkbox("스태킹", value=True,
                           help="구성 모델의 OOF 예측을 입력으로 받는 모델을 하나 더 얹습니다.")

    if st.button("앙상블 학습", disabled=len(bases) < 2):
        num, cat = preprocess.split_column_types(S.X)
        pre = preprocess.build_preprocessor(num, cat, S.prep_config)
        with st.spinner("OOF 재사용 중 — base 는 다시 학습하지 않습니다"):
            eb, ed = train.build_ensembles(
                S.X, S.y, S.train_idx, S.test_idx, pre, zoo, bases,
                S.train_config, voting, stacking, detail=S.detail,
                include_weighted=weighted)
        if eb.empty:
            st.warning("앙상블을 만들지 못했습니다. 성공한 base 모델이 2개 이상 필요합니다.")
        else:
            S.leaderboard = train.sort_leaderboard(
                pd.concat([S.leaderboard, eb], ignore_index=True), metric)
            S.detail = {**S.detail, **ed}
            # SPEC §17 — 임계값 미달이면 앙상블을 챔피언으로 삼지 않는다
            champ, report = ensemble.adopt_ensemble(
                S.leaderboard, metric,
                threshold=getattr(S.train_config, "ensemble_threshold", 0.03),
                prefix="holdout_")
            S.champion = champ or train.pick_champion(S.leaderboard, metric)
            S.ensemble_report = report
            state.invalidate("train")
            st.rerun()

    if S.ensemble_report is not None and not S.ensemble_report.empty:
        st.markdown("**자동채택 판정**")
        st.caption(f"단일 최고 모델 대비 "
                   f"{getattr(S.train_config, 'ensemble_threshold', 0.03):.0%} 이상 "
                   "좋아진 앙상블만 챔피언이 됩니다 (SPEC §17).")
        st.dataframe(S.ensemble_report, **theme.WIDE, hide_index=True)
        w = S.detail.get("Ensemble_Weighted", {}).get("_pipeline")
        if w is not None and getattr(w, "weights_", None) is not None:
            with st.expander("가중앙상블 가중치"):
                st.caption("nnls 로 구한 음수 없는 가중치입니다. 합은 1 입니다.")
                st.dataframe(w.weight_table(), **theme.WIDE, hide_index=True)



# ─────────────────────────────────────────────────────────────
def _unsupervised() -> None:
    S = st.session_state
    st.markdown('<p class="caption">Y 없이 도는 분석입니다. 군집·이상탐지·차원축소 결과는 '
                '예측·SHAP 화면 대신 이 화면에서 확인합니다.</p>', unsafe_allow_html=True)

    feat = S.feat_df
    confirmed = bool(S.selected_features)
    cols = S.selected_features or [c for c in feat.columns if c != S.target]
    X = feat[cols]

    if not confirmed:
        # 확정 전이면 파생 전체가 들어간다. 군집·이상탐지 결과가 3단계에서 고른
        # 피처 기준이라고 오해하기 쉬운 자리다.
        st.warning(f"3단계에서 피처를 확정하지 않아 **생성된 파생 {len(cols):,}개 전체**로 "
                   "계산합니다. 확정한 피처만 쓰려면 3단계에서 '이 목록으로 확정'을 "
                   "먼저 누르세요.")

    c1, c2 = st.columns([1, 3])
    mode = c1.selectbox("분석", ["군집", "이상탐지", "주성분"])
    c2.caption(f"대상 {len(cols)}개 피처 · {len(X):,}행"
               + (" · 3단계 확정 목록" if confirmed else " · 확정 전 (파생 전체)"))

    pre_cfg = S.prep_config or preprocess.PreprocessConfig()
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, pre_cfg)

    if mode == "군집":
        c1, c2, c3 = st.columns(3)
        k_lo = c1.number_input("k 최소", 2, 20, 2)
        k_hi = c2.number_input("k 최대", int(k_lo), 30, max(int(k_lo), 6))
        algos = c3.multiselect("알고리즘", ["KMeans", "GaussianMixture", "Agglomerative", "DBSCAN"],
                               default=["KMeans", "GaussianMixture"])
        cfg = unsupervised.UnsupervisedConfig(
            mode=unsupervised.CLUSTERING, k_range=(int(k_lo), int(k_hi)), selected=algos)
        if st.button("실행", type="primary"):
            with st.spinner("군집 비교 중"):
                board, detail = unsupervised.run_clustering(X, pre, cfg)
            S.unsup_board, S.unsup_detail, S.unsup_config = board, detail, cfg
            st.rerun()

        if S.unsup_board is not None and not S.unsup_board.empty:
            st.dataframe(S.unsup_board, **theme.WIDE, hide_index=True)
            keys = [k for k in S.unsup_detail]
            pick = st.selectbox("상세", keys)
            labels = S.unsup_detail[pick]["labels"]
            st.plotly_chart(plots.cluster_timeline(labels), **theme.WIDE)
            st.markdown("**군집별 평균**")
            st.dataframe(unsupervised.cluster_profile(X, labels),
                         **theme.WIDE, hide_index=True, height=320)

    elif mode == "이상탐지":
        c1, c2 = st.columns(2)
        cont = c1.slider("예상 이상 비율", 0.001, 0.2, 0.01, 0.001)
        algos = c2.multiselect("알고리즘", ["IsolationForest", "LocalOutlierFactor"],
                               default=["IsolationForest"])
        cfg = unsupervised.UnsupervisedConfig(mode=unsupervised.ANOMALY,
                                              contamination=cont, selected=algos)
        if st.button("실행", type="primary"):
            with st.spinner("이상 점수 계산 중"):
                board, detail = unsupervised.run_anomaly(X, pre, cfg)
            S.unsup_board, S.unsup_detail, S.unsup_config = board, detail, cfg
            st.rerun()

        if S.unsup_board is not None and not S.unsup_board.empty:
            st.dataframe(S.unsup_board, **theme.WIDE, hide_index=True)
            pick = st.selectbox("상세", list(S.unsup_detail))
            d = S.unsup_detail[pick]
            st.plotly_chart(plots.anomaly_timeline(d["score"], d["flag"]),
                            **theme.WIDE)
            hits = d["flag"][d["flag"]]
            st.caption(f"이상 판정 {len(hits):,}건")
            if len(hits):
                st.dataframe(X.loc[hits.index].head(200), **theme.WIDE, height=300)

    else:
        # 피처가 2~3개뿐이면 상한이 기본값(3)보다 작아져 슬라이더가 예외를 던진다.
        pc_max = min(10, max(X.shape[1], 2))
        if X.shape[1] < 2:
            st.warning("주성분 분석에는 피처가 2개 이상 필요합니다.")
            return
        n_comp = st.slider("주성분 개수", 2, pc_max, min(3, pc_max))
        if st.button("실행", type="primary"):
            cfg = unsupervised.UnsupervisedConfig(mode=unsupervised.REDUCTION,
                                                  n_components=int(n_comp))
            with st.spinner("계산 중"):
                S.pca = unsupervised.run_pca(X, pre, cfg)
            st.rerun()

        if S.pca:
            p = S.pca
            cols_ = st.columns(len(p["explained_variance_ratio"]))
            for i, (c, v) in enumerate(zip(cols_, p["explained_variance_ratio"])):
                c.metric(f"PC{i+1}", f"{v:.1%}", f"누적 {p['cumulative'][i]:.1%}")
            st.plotly_chart(plots.scatter_2d(p["scores"], "PC1", "PC2", title="주성분 공간"),
                            **theme.WIDE)
            st.markdown("**적재량 (loadings)**")
            st.dataframe(p["loadings"].reset_index(names="feature"),
                         **theme.WIDE, hide_index=True, height=320)
