"""단계 이동.

사이드바 라디오만으로는 "지금 어디이고, 다음에 뭘 해야 하는지"가 안 보인다.
그래서 세 가지를 둔다.

1. 화면 맨 위 — 몇 단계 중 몇 번째이고 이 단계에서 뭘 하는지
2. 화면 맨 아래 — 이전 / 다음 버튼. 다음이 막혀 있으면 무엇이 모자란지 적는다
3. 사이드바 — 완료(●) / 현재(◉) / 대기(○) 표시

'다음' 이 회색이면 이 단계에서 할 일이 남았다는 뜻이다. 그 이유를 버튼 옆에
그대로 적어 두어, 사용자가 화면을 뒤지지 않아도 되게 한다.
"""

from __future__ import annotations

import streamlit as st

from app import state

# 단계별 한 줄 설명과, 다음으로 넘어가기 위한 조건
GUIDE: dict[str, str] = {
    "data": "CSV 를 올리거나 SQL 로 가져온 뒤, 시간축과 타겟(Y)을 정합니다.",
    "prep": "컬럼 품질을 보고 X 후보를 확정하고, 전처리 방식을 정합니다.",
    "features": "lag·이동통계·차분을 만들고, 품질 리포트를 검토해 X 를 확정합니다.",
    "train": "여러 모델을 같은 조건으로 학습해 챔피언을 고릅니다.",
    "predict": "선택 구간을 예측하고 실측과 비교합니다.",
    "explain": "챔피언 모델의 피처 기여도를 SHAP 으로 봅니다.",
    "whatif": "조건을 바꿨을 때 예측이 어떻게 달라지는지 봅니다.",
    "diagnostics": "결과를 믿어도 되는지 — 누수·안정성·잔차·시기별 성능을 점검합니다.",
    "report": "단독 HTML 리포트를 만들고 실행 결과를 저장합니다.",
    "config": "모드 전환, 설정 저장·불러오기, 지난 실행과 비교.",
}


def _blocker(key: str) -> str | None:
    """이 단계에서 아직 안 끝난 일. 없으면 None."""
    S = st.session_state
    if key == "data":
        if S.df is None:
            return "[시계열로 변환] 을 눌러 시간축을 만드세요."
        if not S.target:
            return "타겟(Y)을 골라야 합니다."
    elif key == "prep":
        if not S.kept:
            return "X 후보를 하나 이상 남겨야 합니다."
    elif key == "features":
        if S.feat_df is None:
            return "파생변수를 먼저 [생성] 하세요."
        if S.feature_review is None:
            return "[품질 리포트 만들기] 를 눌러 검토 화면을 여세요."
        if S.X is None:
            return "검토를 마쳤으면 [이 목록으로 확정] 을 누르세요."
    elif key == "train":
        if S.champion is None and S.unsup_board is None:
            return "[학습 시작] 을 눌러 챔피언을 정해야 합니다."
    return None


def _advice(key: str) -> str | None:
    """막지는 않지만 권하는 일. 다음 단계로 넘어가는 것을 방해하지 않는다."""
    S = st.session_state
    if key == "predict" and S.predictions is None:
        # SHAP 은 챔피언만 있으면 계산된다. 예측을 안 돌렸다고 해석 화면을 막으면
        # 필요도 없는 단계를 강제로 거치게 된다.
        return "예측을 한 번 실행해 두면 잔차 진단과 What-if 비교까지 이어집니다."
    if key == "explain" and S.shap_result is None:
        return "SHAP 을 계산해 두면 What-if 화면에서 기여도 순으로 피처가 정렬됩니다."
    return None


def _index(key: str) -> int:
    for i, (_, k) in enumerate(state.STEPS):
        if k == key:
            return i
    return 0


def header(key: str) -> None:
    """화면 맨 위 — 지금 몇 단계이고 무엇을 하는 곳인지."""
    steps = [(label, k) for label, k in state.STEPS if k != "config"]
    i = _index(key)
    label = dict((k, lb) for lb, k in state.STEPS).get(key, key)

    if key == "config":
        st.markdown(f'<p class="caption">{GUIDE.get(key, "")}</p>', unsafe_allow_html=True)
        return

    done = sum(1 for _, k in steps if state.ready(k) and _blocker(k) is None)
    st.progress(min(done / len(steps), 1.0))
    st.markdown(
        f'<p class="caption"><b>{label}</b> · 전체 {len(steps)}단계 중 '
        f'{i + 1}번째 · 완료 {done}개 &nbsp;—&nbsp; {GUIDE.get(key, "")}</p>',
        unsafe_allow_html=True)


def footer(key: str) -> None:
    """화면 맨 아래 — 이전 / 다음. 막혀 있으면 이유를 적는다."""
    steps = [k for _, k in state.STEPS if k != "config"]
    if key not in steps:
        return
    i = steps.index(key)
    labels = dict((k, lb) for lb, k in state.STEPS)

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 3])

    if i > 0:
        prev = steps[i - 1]
        if c1.button(f"← {labels[prev]}", use_container_width=True, key=f"nav_prev_{key}"):
            _go(prev)

    if i < len(steps) - 1:
        nxt = steps[i + 1]
        blocker = _blocker(key)
        ok = blocker is None and state.ready(nxt)
        if c2.button(f"{labels[nxt]} →", type="primary", use_container_width=True,
                     disabled=not ok, key=f"nav_next_{key}"):
            _go(nxt)
        if not ok:
            c3.markdown(
                f'<p class="caption" style="padding-top:6px">— '
                f'{blocker or "앞 단계 결과가 있어야 넘어갈 수 있습니다."}</p>',
                unsafe_allow_html=True)
        else:
            tip = _advice(key)
            if tip:
                c3.markdown(f'<p class="caption" style="padding-top:6px">{tip}</p>',
                            unsafe_allow_html=True)
    else:
        c2.markdown('<p class="caption" style="padding-top:6px">마지막 단계입니다.</p>',
                    unsafe_allow_html=True)


def _go(key: str) -> None:
    st.session_state["_step"] = key
    st.rerun()


def current() -> str:
    """지금 단계. 사이드바 선택과 다음 버튼이 같은 값을 본다."""
    return st.session_state.get("_step", "data")


def mark(key: str) -> str:
    """사이드바 표식 — 완료 / 현재 / 대기."""
    if key == current():
        return "◉"
    if key == "config":
        return "·"
    return "●" if (state.ready(key) and _blocker(key) is None) else "○"
