"""추천을 화면에 붙이는 공통 부품.

왜 이 파일이 따로 있나
----------------------
"데이터를 먼저 보고 추천한다" 는 2단계·3단계에 똑같이 필요하고, 앞으로도
설정 화면이 늘어나면 또 필요해진다. 화면마다 각자 그리면 **표기가 갈라진다** —
한쪽은 사유를 접어 두고 다른 쪽은 펼쳐 두면 사용자는 두 화면을 다른 기능으로
읽는다. 여기 한 곳에서 모양을 정한다.

추천을 다루는 원칙 세 가지
--------------------------
1. **추천값을 기본값으로 넣는다.** 추천해 놓고 위젯은 딴 값으로 시작하면
   추천을 읽지 않은 사람에게는 아무 효과가 없다.
2. **사유를 항상 보이게 둔다.** 접어 두면 아무도 안 편다. 근거 '표' 는 접어도
   되지만 사유 '문장' 은 접으면 안 된다.
3. **바꾸면 바꿨다고 말해 준다.** 나무라는 게 아니라, 추천과 다른 상태로
   진행 중이라는 사실이 화면에 남아야 나중에 결과를 해석할 수 있다.

계산은 무겁다 (상관 스캔·자기상관). 재실행마다 다시 돌면 위젯 하나 만질 때마다
수 초가 사라지므로 데이터가 그대로면 지난 결과를 그대로 쓴다.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from core.advisor import Advice


# ── 캐시 ─────────────────────────────────────────────────────
def cached(key: str, stamp: tuple, fn: Callable[[], Any]) -> Any:
    """stamp 가 그대로면 지난 결과를 쓴다.

    st.cache_data 를 안 쓰는 이유 — 인자로 DataFrame 이 통째로 들어가면
    streamlit 이 해시하려고 전체를 훑는다. 50만 행이면 그 해싱 자체가 비싸다.
    무엇이 바뀌면 다시 계산해야 하는지는 호출부가 제일 잘 안다.
    """
    box = st.session_state.setdefault("_advice_cache", {})
    hit = box.get(key)
    if hit is None or hit[0] != stamp:
        hit = (stamp, fn())
        box[key] = hit
    return hit[1]


def frame_stamp(df, extra: tuple = ()) -> tuple:
    """데이터가 '그대로인가' 를 싸게 판단한다. 내용 전체를 해싱하지 않는다."""
    if df is None:
        return ("none",) + extra
    return (df.shape, tuple(map(str, df.columns)),
            str(df.index[0]) if len(df) else "",
            str(df.index[-1]) if len(df) else "") + extra


# ── 표시 ─────────────────────────────────────────────────────
_CONF_KIND = {"높음": "ok", "보통": "warn", "낮음": "bad"}


def why(advice: Advice, label: str = "근거 보기", *,
        detail_caption: str = "", key: str = "") -> None:
    """사유 한 줄 + 접어 둔 근거 표.

    사유는 접지 않는다 — 접어 두면 아무도 안 펴고, 그러면 근거 없이 고르라는
    처음 상태로 되돌아간다.
    """
    conf = _CONF_KIND.get(advice.confidence, "idle")
    st.markdown(
        f'<div class="advice advice-{conf}">'
        f'<span class="advice-tag">추천 근거</span> {advice.reason}</div>',
        unsafe_allow_html=True)

    for n in advice.notes:
        st.caption(f"· {n}")

    if advice.detail is not None and len(advice.detail):
        with st.expander(label, expanded=False):
            if detail_caption:
                st.caption(detail_caption)
            st.dataframe(advice.detail, use_container_width=True,
                         hide_index=True, height=min(320, 40 + 28 * len(advice.detail)))


def deviation(advice: Advice, chosen: Any, *, name: str,
              fmt: Callable[[Any], str] | None = None) -> bool:
    """추천과 다른 값을 골랐으면 조용히 알려 준다. 막지는 않는다.

    **막으면 안 된다.** 현장 판단이 통계보다 옳은 경우가 많고, 이 도구 전체가
    "제안하고 결정은 사람이" 로 돌아간다. 다만 다른 상태로 진행 중이라는 사실은
    화면에 남아야 한다 — 나중에 결과가 이상할 때 여기부터 볼 수 있게.
    """
    f = fmt or (lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple, set))
                else str(v))
    a, b = advice.value, chosen
    if isinstance(a, (list, tuple, set)) and isinstance(b, (list, tuple, set)):
        same = set(a) == set(b)
    else:
        same = a == b
    if same:
        return False
    st.caption(f"↳ {name}: 추천 **{f(a)}** 대신 **{f(b)}** 로 진행합니다. "
               "현장 판단이 우선입니다 — 그대로 두셔도 됩니다.")
    return True


def summary(table, caption: str = "") -> None:
    """추천 묶음을 한 장으로. 화면 맨 위에서 '무엇을 왜 이렇게 잡았는지' 를 준다."""
    if table is None or table.empty:
        return
    if caption:
        st.caption(caption)
    st.dataframe(table, use_container_width=True, hide_index=True,
                 height=min(280, 40 + 34 * len(table)))


def limits_form(step_min: float | None, *, key: str,
                default_lag_min: float = 0.0,
                default_roll_min: float = 0.0):
    """물리적 한계 입력. 통계 추천이 절대 넘지 못하는 상한이다.

    왜 필요한가 — 상관은 우연히도 높아진다. 노이즈 태그가 8시간 지연에서
    우연히 최대가 되는 일이 실제로 있었다. "이 공정에서 반응이 4시간 뒤에
    온다는 건 물리적으로 말이 안 된다" 는 **사람만 아는 지식**이고, 그걸
    걸어 두면 통계가 뭐라 하든 그 너머는 후보에서 빠진다.

    0 을 넣으면 한계 없음이다.
    """
    from core.advisor import PhysicalLimits

    with st.expander("물리적 한계 지정 — 이 시간을 넘는 추천은 나오지 않습니다",
                     expanded=False):
        st.caption(
            "공정을 아는 사람만 정할 수 있는 값입니다. **통계는 우연히도 "
            "상관이 높아지므로**, 물리적으로 말이 안 되는 지연을 추천할 수 "
            "있습니다. 여기에 상한을 걸면 그 너머는 추천에서도, 선택지에서도 "
            "빠집니다. 모르시면 0 으로 두세요 (한계 없음).")
        c1, c2 = st.columns(2)
        lag_min = c1.number_input(
            "최대 반응 지연 (분)", min_value=0.0, max_value=100_000.0,
            value=float(default_lag_min), step=5.0, key=f"{key}_lag",
            help="원인이 결과에 나타나기까지 걸릴 수 있는 **최대** 시간입니다. "
                 "예를 들어 반응기 체류시간이 40분이면 그보다 훨씬 긴 지연은 "
                 "물리적으로 설명이 안 됩니다. 0 = 한계 없음.")
        roll_min = c2.number_input(
            "최대 평균 구간 (분)", min_value=0.0, max_value=100_000.0,
            value=float(default_roll_min), step=10.0, key=f"{key}_roll",
            help="의미 있게 묶어서 볼 수 있는 **최대** 구간입니다. 배치 하나가 "
                 "2시간이면 그보다 긴 평균은 다른 배치까지 섞습니다. 0 = 한계 없음.")

        lim = PhysicalLimits(max_lag_minutes=lag_min or None,
                             max_rolling_minutes=roll_min or None)
        if step_min:
            parts = []
            if lim.max_lag_minutes:
                parts.append(f"지연 최대 **{lim.lag_rows(step_min)}행**")
            if lim.max_rolling_minutes:
                parts.append(f"평균 구간 최대 **{lim.rolling_rows(step_min)}행**")
            if parts:
                st.caption(f"이 데이터({step_min:g}분 간격)에서는 "
                           + " · ".join(parts) + " 입니다.")
    return lim


def within(options: list, cap: int | None) -> list:
    """상한을 넘는 선택지를 아예 지운다.

    비활성화가 아니라 제거다 — 고를 수 없는 항목을 회색으로 남겨 두면
    "왜 이건 못 고르지" 를 다시 설명해야 한다. 한계를 걸었으면 없는 게 맞다.
    """
    if not cap:
        return list(options)
    kept = [o for o in options if o <= cap]
    return kept or [min(options)] if options else []
