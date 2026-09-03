"""화면 간 상태 공유.

단계는 앞에서 뒤로만 흐른다. 앞 단계 입력이 바뀌면 뒤 결과는 무효로 만든다.
그렇게 해두지 않으면 예전 학습 결과 위에 새 데이터를 그리는 사고가 난다.
"""

from __future__ import annotations

import streamlit as st

DEFAULTS: dict = {
    # 1~2 데이터
    "raw": None, "df": None, "source_desc": "", "target": None,
    "time_col": None, "sql_query": "SELECT\n\nFROM\n\nWHERE\n",
    "sql_url": "", "candidates": [],
    # 3~4 전처리·파생
    "quality_profile": None, "dropped": [], "kept": [],
    "feat_df": None, "provenance": None, "feature_config": None,
    "prep_config": None, "selected_features": [], "selection_report": None,
    # 3단계 검토 게이트 — 확정 전 상태
    "feature_review": None, "review_picks": None, "X_pool": None,
    "review_gen": 0,
    # 5~7 학습
    "learning_mode": "지도학습", "task": None,
    "X": None, "y": None, "train_idx": None, "test_idx": None,
    "split_config": None, "leaderboard": None, "detail": {},
    "champion": None, "train_config": None,
    # 3분할 · 선별구간 추적 (P0-2, P0-3)
    "split": None,             # validation.Split — 3단계에서만 만든다
    "unseen_idx": None,
    "selection_train_idx": None,   # 선별에 실제로 쓴 행. 누수 점검이 이걸 본다
    "unseen_guard": None,          # Final Unseen 접근 횟수 감시
    "unseen_scores": None,
    "ensemble_report": None,
    "fold_stability": None,
    # 비지도
    "unsup_board": None, "unsup_detail": {}, "unsup_config": None, "pca": None,
    # 8~11
    "predictions": None, "shap_result": None, "scenario": None,
    "run_dir": None, "saved": {}, "manifest": None, "report_html": "",
    # Rolling Backtest (P2-11) · Random vs Time 진단 (P2-7, 진단 전용)
    "backtest": None, "split_diag": None,
    # 모드 · 설정 · Champion-Challenger
    "mode": "Guided", "studio_config": None, "challenger": None,
    "auto_decisions": None,
    "_step": "data",
}

# 단계 이름은 **정확한 용어**로 둔다. 엔지니어가 아는 말을 굳이 풀어 쓰면
# 정확도만 잃고 유치해진다. 설명이 필요한 것은 화면 안 도움말로 붙인다.
STEPS = [
    ("1. 데이터", "data"),
    ("2. 품질·전처리", "prep"),
    ("3. 파생변수", "features"),
    ("4. 학습", "train"),
    ("5. 예측", "predict"),
    ("6. SHAP 해석", "explain"),
    ("7. What-if", "whatif"),
    ("8. 진단", "diagnostics"),
    ("9. 리포트", "report"),
    ("설정", "config"),
]

# ── 모드 (SPEC §3) ──────────────────────────────────────────
# Auto   기본값으로 끝까지. 노출은 최소한만.
# Guided 주요 선택지만 보여주고 나머지는 기본값. 기본 모드.
# Expert 전부 노출.
MODES = ["Auto", "Guided", "Expert"]
MODE_LEVEL = {"Auto": 0, "Guided": 1, "Expert": 2}

# 사이드바 라디오에 그대로 뜬다. 이름은 그대로 두고 한 줄 요약만 붙인다 —
# 영어 한 단어만으로는 뭐가 다른지 알 수 없지만, 이름을 갈아버리면
# 문서·CHANGELOG·설정파일과 어긋난다.
MODE_LABEL = {
    "Auto": "Auto · 한 번에",
    "Guided": "Guided · 단계별 (권장)",
    "Expert": "Expert · 전 항목 노출",
}
MODE_HELP = {
    "Auto": "타겟만 고르면 챔피언까지 자동으로 갑니다. 무엇이 어떻게 정해졌는지는 "
            "결정 표로 남습니다. 처음 둘러보거나 데이터가 쓸 만한지 빠르게 볼 때.",
    "Guided": "결과를 크게 좌우하는 선택만 노출하고 나머지는 기본값으로 둡니다. "
              "평소 분석은 대부분 여기서 합니다.",
    "Expert": "폴드 내부 선별 토글·앙상블 임계값·하이퍼파라미터 탐색까지 전부 "
              "열립니다. 탐색을 켜면 학습 시간이 수십 배가 될 수 있습니다.",
}

# 화면에서 길게 설명할 때 쓴다.
MODE_DETAIL = {
    "Auto": ("**언제 쓰나** — 처음 둘러볼 때, 새 데이터가 쓸 만한지 빠르게 볼 때.\n\n"
             "1단계에서 타겟을 고르면 실행 버튼이 나타나고 챔피언까지 한 번에 갑니다. "
             "**자동화되는 것은 '버튼을 누르는 일' 이지 '검증을 생략하는 일' 이 "
             "아닙니다** — 3분할·gap 점검·선별구간 추적이 그대로 돌고, 누수 점검을 "
             "통과하지 못하면 학습 전에 멈춥니다."),
    "Guided": ("**언제 쓰나** — 평소. 실제 분석은 대부분 여기서 합니다.\n\n"
               "결과를 크게 좌우하는 선택만 노출합니다. lag·rolling 창처럼 공정마다 "
               "다른 것은 물어보고, 앙상블 임계값처럼 기본값이 무난한 것은 채워 둡니다."),
    "Expert": ("**언제 쓰나** — 결과를 이미 보셨고, 특정 설정을 바꿔 비교할 때.\n\n"
               "추가로 열리는 것: 피처 간 조합, 폴드 내부 재선별 토글, 앙상블 "
               "자동채택 임계값, nested CV 하이퍼파라미터 탐색. 탐색을 켜면 "
               "학습 시간이 수십 배가 될 수 있으니 모델을 좁혀서 쓰세요."),
}


def mode() -> str:
    return st.session_state.get("mode", "Guided")


def at_least(level: str) -> bool:
    """현재 모드가 지정 레벨 이상인지. 고급 설정 노출 여부를 가른다.

    노출을 줄일 뿐 동작은 바꾸지 않는다. 누수 방지 장치(3분할·gap 점검·
    선별구간 추적·Unseen 1회 접근)는 모드와 무관하게 항상 켜져 있다.
    """
    return MODE_LEVEL.get(mode(), 1) >= MODE_LEVEL.get(level, 1)


def init() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def invalidate(after: str) -> None:
    """지정 단계 이후의 산출물을 비운다."""
    # 완료된 실행 하나를 가리키는 것들. 앞 단계를 건드리면 이것들이 가리키는 대상이
    # 더 이상 '지금 실행'이 아니게 된다. 디스크에 저장된 run 폴더는 기록이므로
    # 그대로 두고, 세션이 '지금 것' 이라고 붙들고 있는 손잡이만 놓는다.
    #
    # 이걸 안 지우면 config_view 의 Champion-Challenger 가 예전 manifest 의 데이터
    # 지문을 '지금 지문' 으로 삼아 비교한다. 터지지 않고 틀린 답을 내므로 더 나쁘다.
    report_out = ["run_dir", "saved", "manifest", "report_html", "challenger"]
    train_out = ["leaderboard", "detail", "champion", "predictions", "shap_result",
                 "scenario", "unseen_scores", "ensemble_report",
                 "fold_stability", "backtest", "split_diag", "auto_decisions",
                 *report_out]
    # unseen_guard 는 **분할에 딸린 것**이지 학습 실행에 딸린 것이 아니다.
    # train_out 에 두면 학습을 다시 돌릴 때마다 접근 횟수가 0 으로 돌아가서,
    # 같은 분할로 Final Unseen 을 몇 번이든 열어 볼 수 있게 된다. 그러면
    # "미접촉 구간" 이라는 말 자체가 성립하지 않는다. 분할이 바뀔 때만 새로 만든다.
    select_out = ["selected_features", "selection_report", "selection_train_idx",
                  "X", "y", "split", "train_idx", "test_idx", "unseen_idx",
                  "feature_review", "review_picks", "X_pool", "review_gen",
                  "unseen_guard"]
    chains = {
        "data": ["quality_profile", "dropped", "kept", "feat_df", "provenance",
                 *select_out, *train_out, "unsup_board", "unsup_detail", "pca"],
        "prep": ["feat_df", "provenance", *select_out, *train_out,
                 "unsup_board", "unsup_detail", "pca"],
        # 선별 결과와 split 은 한 몸이다. 하나가 바뀌면 둘 다 무효로 만든다.
        "features": [*select_out, *train_out],
        "split": [*select_out, *train_out],
        # 챔피언이 바뀌면 리포트·manifest 도 그 챔피언 얘기가 아니게 된다.
        # unseen_guard 는 일부러 뺐다 — 위 주석 참고.
        "train": ["predictions", "shap_result", "scenario", "unseen_scores",
                  *report_out],
    }
    # 무엇이 지워졌는지 말해 준다. 조용히 사라지면 "아까 있던 챔피언이 왜 없지" 가
    # 되고, 사용자는 자기가 뭘 잘못 눌렀는지 알 수 없다.
    labels = {"leaderboard": "리더보드", "champion": "챔피언", "predictions": "예측",
              "shap_result": "SHAP", "unseen_scores": "Final Unseen 평가",
              "X": "확정 피처", "feat_df": "파생변수", "backtest": "백테스트"}
    keys = chains.get(after, [])
    lost = [labels[k] for k in labels
            if k in keys and st.session_state.get(k) is not None
            and not (hasattr(st.session_state.get(k), "empty")
                     and st.session_state[k].empty)]

    for k in keys:
        st.session_state[k] = DEFAULTS[k]

    if lost:
        msg = "다시 실행해야 합니다 — " + ", ".join(lost)
        try:
            st.toast(msg)
        except Exception:      # noqa: BLE001  (구버전 streamlit 에는 toast 가 없다)
            pass
        st.session_state["_invalidated"] = msg


def ready(step: str) -> bool:
    """이 단계 화면이 안전하게 그려질 수 있는가.

    **화면이 실제로 쓰는 것을 그대로 확인해야 한다.** 예전에는 5~7단계를
    `champion is not None` 하나로만 봤는데, 그 화면들은 champion 뿐 아니라
    X · y · split · detail 을 바로 꺼내 쓴다. 평소에는 이것들이 같이 채워지고
    같이 지워지니 문제가 없지만, 그건 invalidate 체인이 지금 그렇게 짜여 있다는
    것에 기댄 것이지 여기서 보장한 게 아니다. 체인을 한 번 손대거나 설정을
    불러오는 경로가 생기면 바로 AttributeError 트레이스가 뜬다.
    대리 조건 대신 진짜 조건을 본다.
    """
    S = st.session_state
    trained = (S.champion is not None and S.X is not None and S.y is not None
               and S.detail is not None and S.split is not None)
    return {
        "data": True,
        "prep": S.df is not None and S.target is not None,
        "features": S.df is not None and bool(S.kept),
        "train": S.feat_df is not None,
        "predict": trained,
        "explain": trained,
        "whatif": trained,
        "diagnostics": S.leaderboard is not None and S.X is not None,
        "report": S.leaderboard is not None or S.unsup_board is not None,
        "config": True,
    }.get(step, False)


def guard(step: str, message: str) -> bool:
    """준비가 안 됐으면 안내만 띄우고 False 를 돌려준다."""
    if ready(step):
        return True
    st.info(message)
    return False


def champion_pipeline():
    S = st.session_state
    if not S.champion:
        return None
    rec = S.detail.get(S.champion)
    return rec.get("_pipeline") if rec else None
