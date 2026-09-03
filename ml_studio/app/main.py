"""시계열 ML 스튜디오 — 실행 진입점.

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import nav, state, theme  # noqa: E402
from app.views import (  # noqa: E402
    config_view, data_view, diagnostics_view, explain_view, features_view,
    predict_view, prep_view, report_view, train_view, whatif_view,
)

st.set_page_config(page_title="시계열 ML 스튜디오", page_icon="◧",
                   layout="wide", initial_sidebar_state="expanded")
theme.inject()
state.init()
S = st.session_state


def _rail() -> str:
    """좌측 파이프라인 레일. 단계마다 완료 여부를 점으로 표시한다."""
    with st.sidebar:
        st.markdown('<div class="rail-title">시계열 ML 스튜디오</div>'
                    '<div class="rail-sub">적재 · 학습 · 해석 · 시나리오</div>',
                    unsafe_allow_html=True)

        # 모드는 **맨 위**에 둔다. 예전에는 단계 목록 아래, 초기화 버튼 근처에
        # 있었는데 "왼쪽 밑 구석이라 안 보인다" 는 지적을 받았다. 화면에 어떤
        # 설정이 뜨는지를 통째로 가르는 값이라 제일 먼저 눈에 들어와야 한다.
        st.markdown('<div class="rail-section">작업 모드</div>', unsafe_allow_html=True)
        picked_mode = st.radio(
            "모드", state.MODES,
            index=state.MODES.index(state.mode()),
            format_func=lambda m: state.MODE_LABEL[m],
            label_visibility="collapsed",
            help="노출되는 설정의 범위만 다릅니다. 누수 방지 장치는 "
                 "어느 모드에서도 꺼지지 않습니다.")
        if picked_mode != S.mode:
            S.mode = picked_mode
            st.rerun()
        st.markdown(f'<div class="rail-note">{state.MODE_HELP[state.mode()]}</div>',
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="rail-section">단계</div>', unsafe_allow_html=True)
        labels = [label for label, _ in state.STEPS]
        keys = [key for _, key in state.STEPS]

        def fmt(i: int) -> str:
            return f"{nav.mark(keys[i])}  {labels[i]}"

        # '다음 단계' 버튼과 사이드바가 같은 값을 보도록 _step 을 기준으로 삼는다.
        # key 가 붙은 위젯은 index 를 무시하고 session_state 값을 따르므로,
        # 위젯을 만들기 **전에** 그 값을 맞춰 둬야 버튼으로 옮긴 단계가 반영된다.
        cur = nav.current()
        idx = keys.index(cur) if cur in keys else 0
        if S.get("_rail") != idx:
            S["_rail"] = idx
        choice = st.radio("단계", options=range(len(labels)),
                          format_func=fmt, label_visibility="collapsed",
                          key="_rail")
        if keys[choice] != cur:
            S["_step"] = keys[choice]
            cur = keys[choice]

        st.caption("● 완료   ◉ 현재   ○ 대기")

        st.divider()
        if S.df is not None:
            rows = [f"<span>행</span> {len(S.df):,}",
                    f"<span>열</span> {S.df.shape[1]}"]
            if S.target:
                rows.append(f"<span>타겟</span> {S.target}")
            if S.champion:
                rows.append(f"<span>챔피언</span> {S.champion}")
            st.markdown(f'<div class="rail-meta">{"<br>".join(rows)}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="rail-meta"><span>데이터가 아직 없습니다</span></div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="rail-note">Y 의 과거값은 X 로 쓰지 않습니다. '
                    '인과 해석과 What-if 를 우선하기 위한 설정입니다.</div>',
                    unsafe_allow_html=True)

        # 한 번 누르면 데이터·선별·학습 결과가 전부 사라진다. 두 번 누르게 한다.
        if S.get("_reset_armed"):
            st.warning("데이터·피처 선별·학습 결과가 모두 지워집니다.")
            c1, c2 = st.columns(2)
            if c1.button("지웁니다", type="primary", **theme.WIDE):
                for k, v in state.DEFAULTS.items():
                    S[k] = v
                S["_reset_armed"] = False
                st.rerun()
            if c2.button("취소", **theme.WIDE):
                S["_reset_armed"] = False
                st.rerun()
        elif st.button("전체 초기화", **theme.WIDE):
            S["_reset_armed"] = True
            st.rerun()

    return cur


def _runbar() -> None:
    """어느 단계에 있든 실행 상태를 한 줄로 보여준다."""
    df = S.df
    # 모드를 맨 앞에 둔다 — 사이드바를 접어 둔 사람에게는 여기가 유일한 표시다.
    items = [
        ("모드", state.mode(), True),
        ("데이터", f"{len(df):,}행 × {df.shape[1]}열" if df is not None else "미적재",
         df is not None),
        ("출처", (S.source_desc or "—")[:34], bool(S.source_desc)),
        ("타겟", S.target or "미선택", bool(S.target)),
    ]
    if S.feat_df is not None:
        items.append(("X 후보", f"{len(S.selected_features or []):,}개",
                      bool(S.selected_features)))
    if S.train_idx is not None and S.test_idx is not None:
        items.append(("분할", f"{len(S.train_idx):,} / {len(S.test_idx):,}", True))
    items.append(("챔피언", S.champion or "미확정", bool(S.champion)))

    if S.champion:
        flag = ("누수 가드", theme.badge(
            "통과", "ok",
            help="시간 순서·gap·Y 파생 차단·선별 구간 격리·Final Unseen 격리 "
                 "점검을 모두 통과한 상태에서 챔피언이 정해졌습니다."))
    elif S.X is not None:
        flag = ("누수 가드", theme.badge(
            "분할 완료", "warn",
            help="구간은 나뉘었지만 아직 학습 전입니다. 학습을 시작하면 "
                 "점검표를 다시 돌리고, 하나라도 실패하면 학습을 막습니다."))
    else:
        flag = ("누수 가드", theme.badge(
            "대기", "idle",
            help="3단계에서 피처를 확정하면 구간을 나누고 누수 점검을 시작합니다."))
    theme.runbar(items, flag)


step = _rail()
_runbar()

# invalidate() 가 남긴 안내를 한 번만 띄우고 지운다. toast 를 못 쓰는 구버전
# streamlit 에서도 무엇이 지워졌는지는 보여야 한다.
_lost = S.pop("_invalidated", None)
if _lost:
    st.caption(_lost)

nav.header(step)

{
    "data": data_view.render,
    "prep": prep_view.render,
    "features": features_view.render,
    "train": train_view.render,
    "predict": predict_view.render,
    "explain": explain_view.render,
    "whatif": whatif_view.render,
    "diagnostics": diagnostics_view.render,
    "report": report_view.render,
    "config": config_view.render,
}[step]()

nav.footer(step)
