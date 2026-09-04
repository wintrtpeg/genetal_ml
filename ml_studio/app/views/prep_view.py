"""2단계. 품질 진단 → 제외 확정 → 전처리 방식 결정."""

from __future__ import annotations

import streamlit as st

from app import advice_ui, state, theme
from core import advisor, datasource, plots, preprocess, profiling

CORR_SAMPLE_ROWS = 50_000


def _pairs_cached(df, candidates, threshold: float):
    """중복 피처 탐색은 상관행렬을 만든다. 슬라이더를 건드릴 때마다 돌면 안 된다.

    Streamlit 은 위젯을 하나 만질 때마다 스크립트 전체를 다시 실행한다. 여기서
    매번 상관행렬을 새로 만들면 결측 허용 한도 슬라이더 하나가 수십 초를 먹는다.
    입력이 바뀌지 않았으면 지난 결과를 그대로 쓴다.
    """
    S = st.session_state
    key = (tuple(candidates), round(float(threshold), 4), len(df), id(df))
    if S.get("_pairs_key") != key:
        S["_pairs_key"] = key
        S["_pairs_val"] = profiling.find_correlated_pairs(
            df[candidates], threshold, max_rows=CORR_SAMPLE_ROWS)
    return S["_pairs_val"]


def render() -> None:
    S = st.session_state
    st.title("2. 품질·전처리")
    st.markdown('<p class="caption">고장난 센서, 늘 같은 값만 나오는 태그, 서로 '
                '똑같이 움직이는 중복 태그를 찾아 <b>제외 후보로 제안</b>합니다. '
                '최종 판단은 하십니다 — 도메인 판단이 통계보다 옳은 경우가 많습니다.</p>',
                unsafe_allow_html=True)
    if not state.guard("prep", "먼저 1단계에서 데이터를 불러오고 타겟을 지정해 주세요."):
        return

    df, target = S.df, S.target
    rule = _rule_controls()

    if S.quality_profile is None or st.button("다시 진단"):
        with st.spinner("태그별로 살펴보는 중"):
            S.quality_profile = profiling.profile(df)
    prof = S.quality_profile

    pairs = _pairs_cached(df, S.candidates, rule.max_corr)
    suggested = profiling.suggest_drops(prof, rule, protect=[target], corr_pairs=pairs)

    st.header("컬럼별 상태")
    st.caption("각 컬럼의 자료형·결측률·분산·고유값 수입니다. "
               "이상해 보이는 태그가 있으면 아래에서 빼시면 됩니다.")
    st.dataframe(prof, **theme.WIDE, height=320)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("제외 후보")
        if suggested.empty:
            st.success("규칙에 걸리는 컬럼이 없습니다.")
        else:
            st.dataframe(suggested, **theme.WIDE, hide_index=True)
    with c2:
        st.subheader("중복 후보 (상관 높은 쌍)")
        if pairs.empty:
            st.caption(f"상관 {rule.max_corr} 이상인 쌍이 없습니다.")
        else:
            st.dataframe(pairs, **theme.WIDE, hide_index=True, height=240)
        if len(df) > CORR_SAMPLE_ROWS:
            st.caption(f"데이터가 많아 {CORR_SAMPLE_ROWS:,}줄만 골라 비교했습니다. "
                       "제안용 수치이고, 학습에는 전체를 씁니다.")

    st.header("학습에 쓸 컬럼 확정")
    st.caption("위 제안을 반영해 미리 골라 뒀습니다. **현장 판단이 우선입니다** — "
               "'이 태그는 물리적으로 반드시 들어가야 한다' 싶으면 다시 넣으세요. "
               "반대로 통계상 멀쩡해도 못 믿을 센서면 빼시면 됩니다.")
    default_keep = [c for c in S.candidates if c not in set(suggested["column"])] \
        if not suggested.empty else list(S.candidates)
    kept = st.multiselect("X 후보", options=S.candidates,
                          default=S.kept or default_keep,
                          help="x 를 눌러 빼거나, 칸을 눌러 다시 넣을 수 있습니다.")
    if kept != S.kept:
        S.kept = kept
        state.invalidate("prep")

    with st.expander("시간축 결손 구간"):
        st.caption("평소 간격의 2배 넘게 기록이 비어 있는 구간입니다. "
                   "설비 정지·통신 장애·정기보수 때 생깁니다.")
        gaps = profiling.gap_report(df.index, datasource.infer_freq(df.index))
        if gaps.empty:
            st.caption("끊긴 구간이 없습니다.")
        else:
            st.caption(f"{len(gaps)}개 구간에서 기록이 끊겼습니다.")
            st.dataframe(gaps, **theme.WIDE, hide_index=True, height=200)

    with st.expander("결측 분포"):
        if kept:
            try:
                st.plotly_chart(plots.missing_heat(df[kept]),
                                **theme.WIDE)
            except Exception as e:  # noqa: BLE001
                st.caption(f"차트를 그리지 못했습니다 — {e}")

    st.divider()
    _prep_controls()

    if S.kept:
        st.success(f"X 후보 **{len(S.kept)}개** 확정. 다음은 파생변수입니다.")


def _at(options: list, value) -> int:
    """추천값이 선택지 어디에 있는지. 없으면 첫 번째로 — 죽지는 않게."""
    try:
        return options.index(value)
    except ValueError:
        return 0


def features_step(index) -> float | None:
    """샘플링 간격(분). 추천 문장이 '12행' 대신 '1시간' 이라고 말하게 하는 값."""
    from core import features as _f
    try:
        return _f.step_minutes(index)
    except Exception:  # noqa: BLE001
        return None


def _rule_controls() -> profiling.QualityRule:
    with st.expander("제외 후보 판정 기준 (기본값 권장)", expanded=False):
        st.caption("아래 기준에 걸리는 컬럼을 제외 후보로 **제안만** 합니다. "
                   "실제로 뺄지는 아래에서 직접 정하십니다.")
        c1, c2, c3, c4 = st.columns(4)
        miss = c1.slider(
            "결측 허용 한도", 0.0, 1.0, 0.30, 0.05, format="%.2f",
            help="0.30 이면 '값이 30% 넘게 비어 있으면 빼자' 는 뜻입니다. "
                 "채워 넣은 값이 절반 가까이 되면 그 태그는 믿기 어렵습니다.")
        dom = c2.slider(
            "단일값 편중 한도", 0.5, 1.0, 0.98, 0.01, format="%.2f",
            help="0.98 이면 '전체의 98% 가 똑같은 값이면 빼자' 는 뜻입니다. "
                 "안 움직이는 태그는 아무것도 설명하지 못합니다 "
                 "(고장난 센서, 항상 닫혀 있는 밸브 등).")
        corr = c3.slider(
            "중복 판정 상관", 0.80, 1.0, 0.95, 0.01, format="%.2f",
            help="0.95 면 '거의 똑같이 움직이는 두 태그 중 하나는 빼자' 는 뜻입니다. "
                 "예: 입구온도와 출구온도가 늘 붙어 다니는 경우. "
                 "둘 다 넣으면 어느 쪽이 중요한지 해석이 흐려집니다.")
        lvl = c4.number_input(
            "범주 수준 한도", 2, 500, 50,
            help="상태값 컬럼에 서로 다른 글자가 50가지 넘게 있으면, "
                 "그건 상태가 아니라 ID 나 메모일 가능성이 큽니다.")
    return profiling.QualityRule(max_missing_ratio=miss, max_dominant_ratio=dom,
                                 max_corr=corr, max_categorical_levels=int(lvl))


_VALUE_LABELS = {
    "impute": {"ffill": "직전 값 유지", "median": "중앙값", "mean": "평균",
               "zero": "0", "interpolate": "앞뒤 보간"},
    "scaler": {"standard": "Standard", "robust": "Robust", "none": "없음"},
    "encoding": {"onehot": "One-Hot", "ordinal": "Ordinal"},
    "clip": {True: "켬", False: "끔"},
}
_ITEM_LABELS = {"impute": "결측 대치", "scaler": "스케일링",
                "encoding": "범주 인코딩", "clip": "극단값 처리"}


def _advice(df, kept: list[str], step_min: float | None) -> dict:
    """네 가지 전처리 추천을 한 번에. 데이터가 그대로면 다시 계산하지 않는다."""
    cols = list(kept) or None
    return advice_ui.cached(
        "prep", advice_ui.frame_stamp(df, (tuple(kept), step_min)),
        lambda: advisor.recommend_preprocess(df, cols, step_min))


def _prep_controls() -> None:
    S = st.session_state
    st.header("전처리 방식")
    st.markdown('<p class="caption">여기서 고른 처리는 학습 파이프라인 안에 들어가 '
                '폴드마다 다시 계산됩니다. 전체 데이터로 미리 처리해 두지 않습니다. '
                '<b>아래 값은 이 데이터를 실제로 읽어 본 뒤 채워 넣은 것입니다</b> — '
                '각 항목 밑에 그렇게 본 이유가 붙어 있습니다.</p>',
                unsafe_allow_html=True)

    step_min = features_step(S.df.index)
    with st.spinner("데이터를 읽고 적절한 방식을 고르는 중"):
        adv = _advice(S.df, S.kept or list(S.candidates), step_min)

    advice_ui.summary(
        advisor.summary_table(adv, _ITEM_LABELS, _VALUE_LABELS),
        "이 데이터에 대한 추천입니다. **확신**이 '낮음'인 항목은 판단 근거가 "
        "약하다는 뜻이니 현장 지식으로 직접 정하시는 편이 낫습니다.")

    with st.expander("각 항목이 하는 일", expanded=False):
        st.markdown(
            "**결측 대치** — 계측이 끊기거나 통신이 튀면 값이 비어 있습니다. "
            "모델은 빈 칸을 못 읽으니 채워야 합니다. 기본은 직전 값 유지 — "
            "설비 신호는 통신이 끊겨도 실제 값은 유지되고 있었을 테니까요.\n\n"
            "**스케일링** — 유량은 수백, 압력은 한 자리, 밸브개도는 0~100. "
            "이대로 넣으면 선형·거리 기반 모델이 **숫자가 큰 태그를 더 중요하게** "
            "봅니다. 트리 계열은 영향이 없지만 Ridge·SVR·신경망은 이게 없으면 "
            "결과가 크게 나빠집니다.\n\n"
            "**범주 인코딩** — `NORMAL` / `WARN` / `TRIP` 같은 상태값, 제품 코드처럼 "
            "숫자가 아닌 컬럼을 모델이 읽을 수 있는 형태로 바꿉니다. 순서가 없으면 "
            "One-Hot, 등급처럼 순서가 있으면 Ordinal 입니다.\n\n"
            "**극단값 처리** — 계측 오류로 찍히는 값(온도 −9999 등)이 있으면 모델이 "
            "거기에 끌려다닙니다. 상·하위 분위수를 경계값으로 바꿔 영향을 줄입니다. "
            "경계는 **학습 구간에서만** 계산해 폴드마다 다시 잡습니다.")

    impute_opts = list(preprocess.IMPUTE_METHODS)
    scaler_opts = ["standard", "robust", "none"]
    enc_opts = ["onehot", "ordinal"]

    c1, c2 = st.columns(2)
    with c1:
        impute = st.selectbox(
            "결측 대치", impute_opts,
            index=_at(impute_opts, adv["impute"].value),
            format_func=lambda k: preprocess.IMPUTE_METHODS[k],
            help="설비 신호는 보통 직전 값 유지가 맞습니다 — 통신이 끊겨도 실제 값은 "
                 "유지되고 있었을 테니까요. 값이 자주 튀는 태그라면 중앙값이 낫습니다.")
        advice_ui.why(adv["impute"], "태그별 결측 모양 보기",
                      detail_caption="비율만으로는 방법을 못 정합니다. 같은 5%라도 "
                                     "2~3행씩 끊긴 것과 500행이 통으로 빈 것은 "
                                     "다르게 다뤄야 합니다.")
        advice_ui.deviation(adv["impute"], impute, name="결측 대치",
                            fmt=lambda v: _VALUE_LABELS["impute"].get(v, v))
    with c2:
        scaler = st.selectbox(
            "스케일링", scaler_opts,
            index=_at(scaler_opts, adv["scaler"].value),
            format_func=lambda k: {
                "standard": "Standard — 평균 0, 표준편차 1",
                "robust": "Robust — 중앙값·IQR 기준 (이상값에 덜 민감)",
                "none": "없음 — 원래 크기 그대로",
            }[k],
            help="단위가 다른 태그를 같은 눈금으로 맞춥니다. 이상값이 많은 데이터면 "
                 "Robust 가 낫습니다. 트리 계열만 쓸 거면 없음도 무방합니다.")
        advice_ui.why(adv["scaler"], "태그별 크기·이상값 보기",
                      detail_caption="'최대 robust z' 는 중앙값에서 얼마나 떨어진 "
                                     "값이 있는지입니다. 10을 넘으면 계측 오류를 "
                                     "의심할 만합니다.")
        advice_ui.deviation(adv["scaler"], scaler, name="스케일링",
                            fmt=lambda v: _VALUE_LABELS["scaler"].get(v, v))

    c1, c2 = st.columns(2)
    with c1:
        enc = st.selectbox(
            "범주 인코딩", enc_opts,
            index=_at(enc_opts, adv["encoding"].value),
            format_func=lambda k: {
                "onehot": "One-Hot — 값마다 별도 컬럼",
                "ordinal": "Ordinal — 순서 있는 정수",
            }[k],
            help="NORMAL/WARN/TRIP 처럼 순서가 없으면 One-Hot, "
                 "1·2·3등급처럼 크기 순서가 있으면 Ordinal 을 쓰세요.")
        advice_ui.why(adv["encoding"], "글자 컬럼 목록 보기")
        advice_ui.deviation(adv["encoding"], enc, name="범주 인코딩",
                            fmt=lambda v: _VALUE_LABELS["encoding"].get(v, v))
    with c2:
        clip = st.checkbox(
            "극단값 처리", value=bool(adv["clip"].value),
            help="계측 오류로 찍힌 값(온도 -9999 등)을 경계값으로 바꿔 영향을 줄입니다.")
        lo, hi = st.slider(
            "clip 분위수", 0.0, 1.0, (0.001, 0.999), 0.001, disabled=not clip,
            format="%.3f",
            help="0.001 ~ 0.999 면 상·하위 0.1% 를 경계값으로 대체합니다. 값을 버리는 게 "
                 "아니라 잘라내는 것입니다. 계측 오류가 많으면 0.01 ~ 0.99 로 넓히세요.")
        advice_ui.why(adv["clip"], "극단값이 있는 태그 보기")
        if clip:
            st.caption(f"하위 {lo:.1%} · 상위 {1 - hi:.1%} 를 경계값으로 대체합니다. "
                       "기준은 **학습 구간에서만** 계산해 폴드마다 다시 잡습니다.")
        advice_ui.deviation(adv["clip"], clip, name="극단값 처리",
                            fmt=lambda v: "켬" if v else "끔")

    cfg = preprocess.PreprocessConfig(
        impute_numeric=impute, scaler=scaler, categorical_encoding=enc,
        clip_outliers=clip, clip_quantiles=(lo, hi),
    )
    if (warn := preprocess.impute_warning(cfg)):
        st.warning(warn)
    S.prep_config = cfg
