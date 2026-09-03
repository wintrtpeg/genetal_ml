"""3단계. 룰 기반 파생변수 생성과 선별."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import advice_ui, state
from core import advisor, features, models, preprocess


def render() -> None:
    S = st.session_state
    st.title("3. 파생변수")
    if not state.guard("features", "먼저 2단계에서 X 후보를 확정해 주세요."):
        return

    st.markdown('<p class="caption">지금 값만으로는 설비 상태를 설명할 수 없습니다. '
                '<b>lag · 이동통계 · 차분</b> 을 만들어 붙입니다. 모든 파생변수는 '
                '<b>시점 t 까지의 정보만</b> 씁니다 — Y 에서 나온 피처는 만들지 '
                '않습니다.</p>', unsafe_allow_html=True)

    cfg = _controls()
    est = _estimate(cfg)
    lookback = features.warmup_rows(cfg, S.df.index)
    c1, c2, c3 = st.columns(3)
    step = features.step_minutes(S.df.index)
    c1.metric("예상 생성 개수", f"{est:,}")
    c2.metric("최대 lookback", f"{lookback}행  ({_span(step, lookback)})",
              help="가장 멀리 돌아보는 거리입니다. 앞부분은 참고할 과거가 없어 "
                   "버리고, 구간 분할의 gap 도 이 값을 기준으로 잡습니다.")
    c3.metric("남는 행", f"{max(len(S.df) - lookback, 0):,}")

    if st.button("생성", type="primary"):
        try:
            with st.spinner("생성 중"):
                feat, prov = features.generate(S.df, S.target, S.kept, cfg)
                feat = features.drop_warmup(feat, cfg)
            S.feat_df, S.provenance, S.feature_config = feat, prov, cfg
            state.invalidate("features")
            st.success(f"X 후보 **{feat.shape[1] - 1:,}개** · {len(feat):,}행. "
                       "아래에서 무엇을 쓸지 고릅니다.")
        except features.TargetLeakage as e:
            st.error(f"차단됨 — {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"생성 실패 — {type(e).__name__}: {e}")

    if S.feat_df is None:
        return

    st.divider()
    _selection_panel()


def _span(step: float | None, rows: int) -> str:
    """스텝 수를 사람이 아는 시간으로. '12' 보다 '1시간 전' 이 훨씬 빨리 읽힌다."""
    if not step:
        return f"{rows}스텝"
    m = step * rows
    if m < 60:
        return f"{m:g}분"
    if m < 1440:
        h = m / 60
        return f"{h:g}시간" if h != int(h) else f"{int(h)}시간"
    d = m / 1440
    return f"{d:g}일" if d != int(d) else f"{int(d)}일"


def _lag_roll_advice(df, target: str, kept: list[str], step: float | None,
                     limits: advisor.PhysicalLimits):
    """lag·rolling 추천. 상관 스캔이 무거워서 데이터가 그대로면 다시 안 돈다.

    한계치가 바뀌면 다시 계산해야 한다 — 상한이 바뀌면 답도 바뀌기 때문에
    stamp 에 한계치를 넣는다.
    """
    stamp = advice_ui.frame_stamp(
        df, (target, tuple(kept), step,
             limits.max_lag_minutes, limits.max_rolling_minutes))
    cols = [c for c in kept if c != target] or None
    return advice_ui.cached(
        "lag_roll", stamp,
        lambda: (advisor.recommend_lags(df, target, cols, step, limits),
                 advisor.recommend_rolling(df, target, step, limits)))


def _controls() -> features.FeatureConfig:
    S = st.session_state
    step = features.step_minutes(S.df.index)

    with st.expander("왜 파생변수가 필요한가", expanded=False):
        st.markdown(
            "설비의 지금 상태는 **지금 값만으로 결정되지 않습니다.**\n\n"
            "> 30분 전에 유량을 올렸고, 최근 한 시간 평균 온도가 높았고, "
            "압력이 계속 떨어지는 추세였다 — 그래서 지금 수율이 이렇다.\n\n"
            "그런데 원본 데이터에는 **그 시점의 값 하나씩만** 들어 있습니다. "
            "'30분 전 유량', '최근 1시간 평균 온도', '압력 변화량' 같은 컬럼은 "
            "따로 만들어 줘야 모델이 볼 수 있습니다. 그걸 여기서 만듭니다.\n\n"
            "**기본값 그대로 두고 [생성] 을 누르셔도 됩니다.** 공정에서 "
            "'이 정도 시간 뒤에 반응이 온다' 는 감이 있으시면 그 시간대를 추가하세요.\n\n"
            "---\n"
            "**lag** — 원인이 결과보다 먼저 옵니다. 밸브를 열면 유량이 바뀌고 "
            "한참 뒤에 온도가 따라옵니다.\n\n"
            "**이동통계(rolling)** — 계측 노이즈를 걷어내고 흐름을 봅니다. "
            "한 점의 값보다 최근 한 시간의 평균·변동이 더 안정적입니다.\n\n"
            "**차분** — 값 자체보다 변화가 중요한 경우입니다. 온도가 80도인 것보다 "
            "'10분 사이 5도 올랐다' 가 신호일 때가 많습니다.")

    if step:
        st.caption(f"이 데이터는 약 **{step:g}분**마다 기록됩니다. "
                   "아래 괄호 안이 실제 시간입니다.")

    # ── 물리적 한계 → 추천 → 기본값. 순서가 중요하다.
    # 한계를 먼저 받아야 추천이 그 안에서 나오고, 추천이 나와야 기본값을 채운다.
    limits = advice_ui.limits_form(step, key="feat_limits")
    lag_cap = limits.lag_rows(step)
    roll_cap = limits.rolling_rows(step)

    with st.spinner("반응 지연을 찾는 중"):
        lag_adv, roll_adv = _lag_roll_advice(S.df, S.target, S.kept, step, limits)

    if state.at_least("Guided") and step:
        unit = st.radio(
            "지정 단위", ["행 (스텝)", "물리 시간 (분)"], horizontal=True,
            help="결과는 같습니다. 편한 쪽으로 쓰세요.")
    else:
        unit = "행 (스텝)"
    by_minutes = unit.startswith("물리") and step is not None

    lag_min: list[float] = []
    roll_min: list[float] = []

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**시간 지연 (lag)**")
        rec_lags = [int(v) for v in (lag_adv.value or [])]
        if by_minutes:
            opts = advice_ui.within([5, 10, 15, 30, 60, 120, 240, 480, 1440],
                                    int(limits.max_lag_minutes or 0) or None)
            rec_min = sorted({m for m in opts
                              if any(abs(m - r * step) < step / 2 for r in rec_lags)})
            lag_min = st.multiselect(
                "lag (분)", opts, default=rec_min or [m for m in opts[:4]],
                format_func=lambda m: _span(1, m),
                help="공정 반응이 늦게 나타나는 시간대를 넣으세요. "
                     "기본값은 아래 근거대로 데이터에서 찾은 지연입니다.")
            lags = []
        else:
            opts = advice_ui.within(
                sorted({1, 2, 3, 6, 12, 24, 48, 96, *rec_lags}), lag_cap)
            lags = st.multiselect(
                "lag (스텝)", opts,
                default=[r for r in rec_lags if r in opts] or [o for o in opts[:4]],
                format_func=lambda r: f"{r}  ({_span(step, r)})",
                help="공정 반응이 늦게 나타나는 시간대를 넣으세요. "
                     "기본값은 아래 근거대로 데이터에서 찾은 지연입니다. "
                     "괄호 안이 실제 시간입니다.")
        advice_ui.why(
            lag_adv, "태그별 최적 지연 보기",
            detail_caption="각 태그를 조금씩 밀어 보며 타겟과 가장 잘 맞는 "
                           "지점을 찾은 결과입니다. **'오차 감소율'** 이 핵심입니다 — "
                           "지연시켰을 때 설명 못 하던 오차가 몇 % 줄었는지이고, "
                           "상관 숫자 자체보다 이쪽이 실제 도움을 잘 나타냅니다. "
                           "상관은 인과가 아니므로 물리적으로 말이 되는지는 "
                           "확인이 필요합니다.")
        if not by_minutes:
            advice_ui.deviation(lag_adv, lags, name="lag",
                                fmt=lambda v: ", ".join(_span(step, r) for r in sorted(v))
                                or "없음")

        diffs = st.multiselect(
            "차분", [1, 2, 3, 6], default=[1],
            format_func=lambda r: f"{r}  ({_span(step, r)})",
            help="값 자체보다 변화가 중요할 때 씁니다. 온도 80도보다 "
                 "'10분 사이 5도 상승' 이 신호일 때가 많습니다.")
        roc = st.checkbox(
            "변화율(%) 추가", value=False,
            help="절대 변화량 대신 비율로. 태그마다 단위가 크게 다를 때 유용합니다.")

    with c2:
        st.markdown("**이동통계 (rolling)**")
        rec_wins = [int(v) for v in (roll_adv.value or [])]
        if by_minutes:
            opts = advice_ui.within([15, 30, 60, 120, 240, 480, 1440],
                                    int(limits.max_rolling_minutes or 0) or None)
            rec_rm = sorted({m for m in opts
                             if any(abs(m - r * step) < step / 2 for r in rec_wins)})
            roll_min = st.multiselect(
                "rolling 창 (분)", opts, default=rec_rm or [m for m in opts[:3]],
                format_func=lambda m: _span(1, m),
                help="계측 노이즈를 걷어내고 추세를 봅니다. "
                     "기본값은 타겟이 얼마나 천천히 움직이는지를 보고 잡았습니다.")
            wins = []
        else:
            opts = advice_ui.within(
                sorted({3, 6, 12, 24, 48, 96, 288, *rec_wins}), roll_cap)
            wins = st.multiselect(
                "rolling 창 (스텝)", opts,
                default=[r for r in rec_wins if r in opts] or [o for o in opts[:3]],
                format_func=lambda r: f"{r}  ({_span(step, r)})",
                help="계측 노이즈를 걷어내고 추세를 봅니다. "
                     "기본값은 타겟이 얼마나 천천히 움직이는지를 보고 잡았습니다. "
                     "괄호 안이 실제 시간입니다.")
        advice_ui.why(
            roll_adv, "타겟 자기상관 보기",
            detail_caption="타겟이 자기 과거와 얼마나 닮았는지입니다. 이 값이 "
                           "0.5 아래로 떨어지는 지점보다 긴 창으로 평균을 내면 "
                           "지금과 무관한 과거까지 섞입니다.")
        if not by_minutes:
            advice_ui.deviation(roll_adv, wins, name="rolling 창",
                                fmt=lambda v: ", ".join(_span(step, r) for r in sorted(v))
                                or "없음")

        stats = st.multiselect(
            "통계량", ["mean", "std", "min", "max", "median"],
            default=["mean", "std"],
            format_func=lambda k: {
                "mean": "mean (평균)", "std": "std (표준편차)",
                "min": "min (최솟값)", "max": "max (최댓값)",
                "median": "median (중앙값)",
            }[k],
            help="std 는 그 구간에서 얼마나 흔들렸는지입니다 — 평균만큼 유용한 "
                 "경우가 많습니다.")
        ewm = st.multiselect(
            "지수이동평균 span", [3, 6, 12, 24, 48], default=[12],
            format_func=lambda r: f"{r}  ({_span(step, r)})",
            help="단순 이동평균은 창 안의 모든 시점을 똑같이 취급합니다. "
                 "EWM 은 최근 값에 더 큰 가중치를 줍니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**시간 파생**")
        tf = st.checkbox(
            "시각·요일·월", value=True,
            help="주야 교대, 주말, 계절에 따라 운전이 달라진다면 켜 두세요.")
        cyc = st.checkbox(
            "주기성을 sin/cos 로", value=True, disabled=not tf,
            help="시각을 0~23 정수로 주면 모델이 23시와 0시를 가장 먼 값으로 "
                 "봅니다. 실제로는 이웃한 시각이죠. 원형으로 바꿔 그 문제를 없앱니다.")
    with c2:
        # 태그끼리 조합하는 것은 공정 지식이 필요하다. Expert 에서만 연다.
        st.markdown("**피처 간 조합**")
        if state.at_least("Expert"):
            numeric = [c for c in S.kept if pd.api.types.is_numeric_dtype(S.df[c])]
            picks = st.multiselect(
                "조합할 피처 (2개 이상)", numeric, default=[],
                help="고른 피처끼리 비율과 차이를 만듭니다. 예: 입구온도·출구온도를 "
                     "고르면 온도차가 생깁니다. 물리적으로 의미 있는 쌍만 고르세요.")
            ops = st.multiselect(
                "연산", ["ratio", "diff"], default=["ratio", "diff"],
                format_func=lambda k: {"ratio": "ratio (비율)",
                                       "diff": "diff (차이)"}[k],
                disabled=len(picks) < 2)
        else:
            picks, ops = [], ["ratio", "diff"]
            st.caption("입구온도 − 출구온도 같은 조합입니다. 물리적으로 의미 있는 "
                       "쌍을 골라야 해서 **Expert** 모드에서만 엽니다.")

    pairs = [(a, b) for i, a in enumerate(picks) for b in picks[i + 1:]]

    st.info("**Y 의 과거값은 X 로 만들지 않습니다.** 넣으면 정확도는 오르지만 "
            "'직전 Y 가 이랬으니 지금도 비슷할 것' 을 학습한 모델이 되어, "
            "X 를 바꿔 Y 를 움직이려는 What-if 와 인과 해석에 쓸 수 없습니다. "
            "코드 수준에서 막혀 있습니다.")

    cfg = features.FeatureConfig(
        lags=lags, rolling_windows=wins, rolling_stats=stats, ewm_spans=ewm,
        diffs=diffs, rate_of_change=roc, time_features=tf, cyclical=cyc,
        interactions=pairs, interaction_ops=ops, allow_target_derived=False,
        lag_minutes=lag_min, rolling_minutes=roll_min,
    )
    if by_minutes:
        spec = features.describe_time_spec(cfg, S.df.index)
        if not spec.empty:
            with st.expander(f"분 → 행 환산 (샘플링 간격 {step:g}분)", expanded=False):
                st.caption("행 수로 환산된 값이 lookback 이 되고, 그대로 gap 점검의 기준이 됩니다.")
                st.dataframe(spec, use_container_width=True, hide_index=True)
    return cfg


def _estimate(cfg: features.FeatureConfig) -> int:
    S = st.session_state
    n = len([c for c in S.kept if pd.api.types.is_numeric_dtype(S.df[c])])
    per = (len(cfg.lags) + len(cfg.rolling_windows) * len(cfg.rolling_stats)
           + len(cfg.ewm_spans) + len(cfg.diffs) + (1 if cfg.rate_of_change else 0))
    t = (10 if cfg.cyclical else 4) if cfg.time_features else 0
    return n * per + len(cfg.interactions) * len(cfg.interaction_ops) + t


def _selection_panel() -> None:
    S = st.session_state
    feat = S.feat_df
    st.header("선별")
    st.markdown('<p class="caption">파생변수가 수백 개면 전부 넣기보다 추리는 편이 '
                '낫습니다 — 느리고 해석도 어려워집니다. 자동 선별은 <b>추천</b>이고 '
                '<b>최종 확정은 아래 검토 화면에서 직접</b> 하십니다.<br>'
                '추천도 검토용 통계도 학습 구간에서만 계산합니다 — 홀드아웃을 보고 '
                '고르면 그 자체가 누수입니다.</p>', unsafe_allow_html=True)

    from core import validation

    c1, c2 = st.columns(2)
    # 후보가 5개도 안 되면 min>max 가 되어 위젯이 예외를 던진다.
    # (원본 컬럼이 적고 파생을 껐을 때 실제로 일어난다)
    n_cand = max(1, feat.shape[1] - 1)
    k_max = max(5, min(500, n_cand))
    k_adv = advisor.recommend_top_k(len(feat), n_cand)
    with c1:
        top_k = st.number_input(
            "상위 몇 개까지", 1, k_max, min(int(k_adv.value), k_max),
            help="상호정보량 순으로 이만큼만 남깁니다. 기본값은 학습 행 수와 후보 "
                 "수를 보고 잡았습니다. 최종 확정은 아래에서 하시니 넉넉하게 "
                 "두셔도 됩니다.")
        advice_ui.why(k_adv)
        advice_ui.deviation(k_adv, int(top_k), name="상위 개수")
    corr_th = c2.slider(
        "중복 제거 상관", 0.90, 1.0, 0.98, 0.01, format="%.2f",
        help="이 값 이상으로 상관된 피처 쌍에서 하나만 남깁니다. "
             "roll6_mean 과 roll12_mean 처럼 거의 겹치는 파생을 걸러냅니다.")

    st.markdown("**구간 분할**")
    st.caption("**Train** 학습 · **Validation** 모델 선택 · **Final Unseen** 최종 보고. "
               "Unseen 은 챔피언 확정 뒤 딱 한 번만 엽니다 — 미리 보고 고르면 "
               "그 점수는 실제 성능이 아니게 됩니다. 항상 시간순으로 자릅니다. "
               "분할은 여기서 한 번만 정하고 이후 단계는 읽기만 합니다.")
    mode = st.radio("지정 방식", ["비율로", "날짜로"], horizontal=True,
                    help="설비 개조·촉매 교체처럼 운전 조건이 바뀐 시점이 있으면 "
                         "날짜로 그 경계를 잡으세요.")

    idx = feat.index
    if S.feature_config is None:
        # feat_df 는 있는데 설정이 없으면 lookback 을 알 수 없고, 그러면 gap 을
        # 정할 수 없다. gap 을 모르는 채 분할하면 누수 점검이 무의미해진다.
        st.warning("파생 설정이 없어 gap 을 계산할 수 없습니다. "
                   "위에서 [생성] 을 다시 눌러 주세요.")
        return
    gap = features.warmup_rows(S.feature_config, feat.index)
    valid_cut = unseen_cut = None
    if mode == "비율로":
        c1, c2, c3 = st.columns(3)
        ratio = c1.slider(
            "Validation 비율", 0.05, 0.5, 0.2, 0.05, format="%.2f",
            help="여러 모델을 견줘 챔피언을 고르는 구간입니다.")
        unseen = c2.slider(
            "Final Unseen 비율", 0.0, 0.4, 0.15, 0.05, format="%.2f",
            help="챔피언 확정 뒤 한 번만 여는 구간. 여기서 나온 값이 보고할 "
                 "성능입니다. 0 이면 2분할로 동작하고, 그때는 보고 성능이 "
                 "낙관 편향됩니다.")
        c3.metric("gap", f"{gap}행",
                  help="학습 끝과 검증 시작 사이를 이만큼 비웁니다. 안 비우면 "
                       "학습 마지막 행의 rolling 창이 검증 구간 값을 이미 "
                       "포함합니다. 파생 lookback 만큼 자동으로 잡힙니다.")
    else:
        # select_slider 에 전체 인덱스를 넘기면 옵션이 행 수만큼 생긴다.
        # 12,000행이면 옵션 12,000개가 브라우저로 전송돼 화면이 멈춘다.
        # 날짜 입력 두 개로 바꾼다 — 경계는 어차피 하루 단위로 잡는다.
        ratio, unseen = 0.2, 0.15
        lo, hi = idx[0].date(), idx[-1].date()
        c1, c2 = st.columns(2)
        valid_cut = c1.date_input("Validation 시작 날짜", value=idx[int(len(idx) * 0.65)].date(),
                                  min_value=lo, max_value=hi)
        unseen_cut = c2.date_input("Final Unseen 시작 날짜",
                                   value=idx[int(len(idx) * 0.85)].date(),
                                   min_value=lo, max_value=hi)
        valid_cut = pd.Timestamp(valid_cut)
        unseen_cut = pd.Timestamp(unseen_cut)
        if unseen_cut <= valid_cut:
            st.error("Final Unseen 시작은 Validation 시작보다 뒤여야 합니다.")
            return

    try:
        cfg_split = validation.SplitConfig(
            holdout_ratio=ratio, unseen_ratio=unseen if mode == "비율" else 0.15,
            gap=gap, valid_cut=valid_cut, unseen_cut=unseen_cut)
        preview = validation.build_split(cfg_split, idx)
        st.dataframe(preview.describe(idx), use_container_width=True, hide_index=True)
        if not preview.three_way:
            st.warning("Final Unseen 이 없습니다. 홀드아웃이 모델 선택과 최종 보고를 겸하므로 "
                       "보고되는 성능이 낙관 편향됩니다. (2분할 — 구버전 호환 모드)")
        else:
            st.caption("Train 은 학습·선별에, Validation 은 모델 선택에, "
                       "Final Unseen 은 최종 보고에만 씁니다. Unseen 은 챔피언 확정 뒤 한 번만 열립니다.")
    except ValueError as e:
        st.error(f"분할 불가 — {e}")
        return

    if st.button("품질 리포트 만들기", type="primary"):
        cols = [c for c in feat.columns if c != S.target]
        X_all, y_all = preprocess.prepare_xy(feat, S.target, cols)
        # 미리보기는 feat.index 로 계산했지만 실제 분할은 결측행이 빠진 X_all.index 로
        # 다시 잡는다. 행이 줄면서 gap 을 못 벌리는 경우가 있어 여기서도 막아야 한다.
        try:
            split = validation.build_split(cfg_split, X_all.index)
            validation.assert_disjoint(split)
        except ValueError as e:
            st.error(f"분할 불가 — {e}\n\n결측 제거 후 남은 행이 {len(X_all):,}행입니다. "
                     "파생 lookback 을 줄이거나 검증 비율을 낮춰 보세요.")
            return
        with st.spinner("학습 구간에서만 진단 중"):
            sel, rep = features.select_features(
                X_all.iloc[split.train], y_all.iloc[split.train],
                task=models.detect_task(y_all), top_k=int(top_k),
                corr_threshold=corr_th)
            review = features.feature_report(rep, S.provenance,
                                             X_train=X_all.iloc[split.train])
        # 아직 확정하지 않는다. 사람이 검토하고 확정 버튼을 눌러야 X 가 만들어진다.
        S.feature_review = review
        S.review_picks = list(sel)
        S.selection_report = rep
        S.X_pool, S.y = X_all, y_all
        S.split = split
        S.train_idx, S.test_idx = split.train, split.valid
        S.unseen_idx = split.unseen
        S.selection_train_idx = split.train      # 선별에 실제로 쓴 구간을 기록
        S.split_config = cfg_split
        S.selected_features, S.X = [], None
        state.invalidate("train")
        st.rerun()

    if S.feature_review is None:
        return

    st.divider()
    _review_gate()


# ─────────────────────────────────────────────────────────────
def _risks_cached(picks: list[str], review, X_train):
    """위험 점검은 상관행렬을 만든다. 선택이 안 바뀌었으면 다시 계산하지 않는다.

    체크박스 하나 누를 때마다 8,000행 × 40열 상관행렬을 다시 구하면 클릭이
    씹히는 것처럼 느껴진다. 선택 집합을 키로 삼아 결과를 재사용한다.
    """
    S = st.session_state
    key = (tuple(picks), len(X_train))
    if S.get("_risk_key") != key:
        S["_risk_key"] = key
        S["_risk_val"] = features.selection_risks(picks, review, X_train)
    return S["_risk_val"]


def _bulk(picks: list[str]) -> None:
    """일괄 선택. data_editor 는 자기 편집 상태를 key 로 붙들고 있어서,
    바깥에서 값을 바꾸려면 key 를 갈아 새로 그리게 해야 한다."""
    S = st.session_state
    S.review_picks = sorted(picks)
    S.review_gen = S.get("review_gen", 0) + 1
    st.rerun()


def _review_gate() -> None:
    """학습에 넣기 전 사람이 X 피처를 최종 확정하는 관문.

    자동 선별은 **추천**이고 확정은 사람이 한다. 도메인 판단이 통계보다 옳은 경우가
    많기 때문이다. 다만 판단 근거로 보여주는 통계는 전부 학습 구간에서만 계산한다 —
    전체 구간 통계를 보여주면 알고리즘 대신 사람이 누수 경로가 된다.
    """
    S = st.session_state
    review = S.feature_review
    X_pool = S.X_pool
    X_train = X_pool.iloc[S.train_idx]
    total = len(review)

    st.header("피처 품질 리포트")
    st.markdown('<p class="caption">자동 추천을 그대로 쓰셔도 되고 직접 켜고 끄셔도 '
                '됩니다. <b>도메인 판단이 통계보다 옳은 경우가 많습니다.</b><br>'
                '여기 숫자는 전부 <b>학습 구간에서만</b> 계산했습니다 — 알고리즘이 '
                '홀드아웃을 못 보게 막아 놓고 사람에게만 보여주면 앞뒤가 안 맞습니다.</p>',
                unsafe_allow_html=True)

    auto = set(review.loc[review["kept"] == True, "feature"])      # noqa: E712
    picks = set(S.review_picks or auto)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("생성된 피처", f"{total:,}")
    c2.metric("자동 추천", f"{len(auto):,}")
    c3.metric("현재 선택", f"{len(picks):,}",
              delta=f"{len(picks) - len(auto):+d}" if len(picks) != len(auto) else None)
    c4.metric("학습 구간", f"{len(X_train):,}행")

    # ── 원본 컬럼별 요약 ──
    roll = features.origin_rollup(review.assign(kept=review["feature"].isin(picks)))
    with st.expander("원본 컬럼별 요약", expanded=True):
        st.caption("파생이 많을 때는 이 단위가 빠릅니다. **한 원본에서 나온 파생이 "
                   "통째로 0개**라면 그 태그를 다시 보세요 — 센서가 죽었거나 "
                   "정말 무관하다는 뜻입니다.")
        st.dataframe(roll, use_container_width=True, hide_index=True, height=240)

    # ── 일괄 선택 ──
    st.markdown("**일괄 선택**")
    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("자동 추천대로", use_container_width=True):
        _bulk(sorted(auto))
    if b2.button("전체 선택", use_container_width=True):
        _bulk(list(review["feature"]))
    if b3.button("전체 해제", use_container_width=True):
        _bulk([])
    if b4.button("원본 컬럼만", use_container_width=True,
                 help="lag·rolling 같은 파생 없이 원본 그대로인 피처만 남깁니다."):
        _bulk(list(review.loc[review["transform"] == "raw", "feature"]))
    if b5.button("MI 상위 30", use_container_width=True):
        top = review.sort_values("mutual_info", ascending=False, na_position="last")
        _bulk(list(top["feature"].head(30)))

    # ── 필터 ──
    c1, c2, c3 = st.columns([2, 2, 3])
    origins = ["(전체)"] + sorted(review["origin"].dropna().astype(str).unique())
    f_origin = c1.selectbox("원본 컬럼", origins)
    transforms = ["(전체)"] + sorted(review["transform"].dropna().astype(str).unique())
    f_transform = c2.selectbox("변환 종류", transforms)
    keyword = c3.text_input("이름 검색", placeholder="예: temp, roll, lag")

    view = review.copy()
    if f_origin != "(전체)":
        view = view[view["origin"].astype(str) == f_origin]
    if f_transform != "(전체)":
        view = view[view["transform"].astype(str) == f_transform]
    if keyword.strip():
        view = view[view["feature"].str.contains(keyword.strip(), case=False, regex=False)]

    only_auto = st.checkbox("자동 추천에서 탈락한 것만 보기", value=False)
    if only_auto:
        view = view[~view["feature"].isin(auto)]

    # ── 편집표 ──
    st.markdown(f"**피처 목록** · {len(view):,}개 표시")
    st.caption("선택 열의 체크박스를 직접 켜고 끌 수 있습니다. "
               "MI(상호정보량)는 절대 크기보다 **순위**를 보시면 됩니다.")

    show = view.assign(선택=view["feature"].isin(picks))
    if "결측률" in show.columns:
        # NumberColumn 의 format 은 sprintf 형식이라 "%.2%" 는 잘못된 문자열이다.
        # 비율을 퍼센트 값으로 바꾸고 리터럴 %% 를 붙인다.
        show = show.assign(결측률=show["결측률"].astype(float) * 100.0)
    cols = ["선택", "feature", "mutual_info", "MI순위", "origin", "transform",
            "lookback", "variance", "결측률", "reason"]
    show = show[[c for c in cols if c in show.columns]]

    edited = st.data_editor(
        show, use_container_width=True, hide_index=True, height=420,
        key=f"feat_editor_{S.get('review_gen', 0)}_{f_origin}_{f_transform}"
            f"_{keyword}_{only_auto}",
        column_config={
            "선택": st.column_config.CheckboxColumn("선택", width="small"),
            "feature": st.column_config.TextColumn("피처", disabled=True,
                                                   width="medium"),
            "mutual_info": st.column_config.NumberColumn(
                "MI", disabled=True, format="%.4f",
                help="상호정보량. 선형 상관과 달리 '어느 온도를 넘으면 급변한다' "
                     "같은 비선형 관계도 잡습니다. 0 이면 무관, 클수록 관련이 "
                     "큽니다. 절대 크기보다 순위를 보세요."),
            "MI순위": st.column_config.NumberColumn(
                "MI순위", disabled=True, help="MI 가 큰 순서. 1등이 가장 관련이 큽니다."),
            "origin": st.column_config.TextColumn("원본", disabled=True),
            "transform": st.column_config.TextColumn("변환", disabled=True),
            "lookback": st.column_config.NumberColumn(
                "lookback", disabled=True,
                help="이 피처가 과거 몇 행까지 참고하는지. 분할 gap 이 이 값보다 "
                     "커야 검증 구간 정보가 새어 들어오지 않습니다."),
            "variance": st.column_config.NumberColumn(
                "분산", disabled=True, format="%.3g",
                help="0 에 가까우면 거의 안 움직이는 피처입니다 — 설명력이 없습니다."),
            "결측률": st.column_config.NumberColumn(
                "결측률", disabled=True, format="%.1f%%",
                help="높으면 실제 계측값이 아니라 대치값이 대부분입니다."),
            "reason": st.column_config.TextColumn("자동 판정 사유", disabled=True, width="large"),
        },
    )

    # 편집 결과를 상태에 반영하되 **재실행을 강제하지 않는다.**
    # st.rerun() 을 부르면 체크박스 하나 누를 때마다 화면 전체가 다시 그려지고,
    # 그때마다 상관행렬·롤업을 다시 계산해 클릭이 눌리지 않는 것처럼 느껴진다.
    # 위젯 조작 자체가 이미 재실행을 일으키므로 상태만 갱신하면 충분하다.
    shown = set(view["feature"])
    checked = set(edited.loc[edited["선택"] == True, "feature"])   # noqa: E712
    picks = (picks - shown) | checked              # 필터로 가려진 선택은 유지한다
    S.review_picks = sorted(picks)

    # ── 위험 경고 ──
    risks = _risks_cached(sorted(picks), review, X_train)
    if not risks.empty:
        with st.expander(f"검토 필요 {len(risks)}건", expanded=False):
            st.caption("**막지 않습니다.** '이 태그는 물리적으로 반드시 들어가야 "
                       "한다' 는 판단을 도구가 뒤집으면 안 되니까요. 다만 무엇을 "
                       "감수하시는지는 알려 드립니다.")
            st.dataframe(risks, use_container_width=True, hide_index=True)

    changed_add = sorted(picks - auto)
    changed_del = sorted(auto - picks)
    if changed_add or changed_del:
        st.info(f"자동 추천 대비 **추가 {len(changed_add)}개 · 제외 {len(changed_del)}개**. "
                "바꾼 내용은 감사 이력과 재현 기록에 사유와 함께 남습니다.")

    # ── 확정 ──
    st.divider()
    c1, c2 = st.columns([1, 3])
    if c1.button("이 목록으로 확정", type="primary", disabled=not picks,
                 use_container_width=True):
        from core import train as _train

        chosen, updated = features.apply_manual_selection(review, sorted(picks))
        S.selected_features = chosen
        S.selection_report = updated
        S.feature_review = updated
        S.X = X_pool[chosen]
        state.invalidate("train")
        # Final Unseen 접근권은 **이 분할에 한 번**이다. 그래서 분할이 확정되는
        # 여기서 만든다. 학습 화면에서 만들면 학습을 다시 돌릴 때마다 접근권이
        # 되살아나서, 마음에 드는 점수가 나올 때까지 열어 볼 수 있게 된다.
        S.unseen_guard = (_train.UnseenGuard(S.unseen_idx)
                          if S.unseen_idx is not None and len(S.unseen_idx) else None)
        st.rerun()
    if not picks:
        c2.warning("피처를 하나 이상 선택해야 합니다.")
    elif S.X is None:
        c2.warning(f"아직 확정 전입니다. {len(picks):,}개를 확정하면 학습으로 넘어갑니다.")
    else:
        c2.success(f"X 피처 **{len(S.selected_features):,}개** 확정. 4단계로 넘어가세요.")

    with st.expander("피처 출처 대장"):
        st.caption("SHAP 결과를 원본 태그까지 되짚을 때 씁니다.")
        prov = S.provenance
        if prov is None or prov.empty:
            st.caption("출처 대장이 없습니다. 파생변수를 다시 생성하면 만들어집니다.")
        else:
            st.dataframe(prov[prov["feature"].isin(sorted(picks))],
                         use_container_width=True, hide_index=True, height=300)
