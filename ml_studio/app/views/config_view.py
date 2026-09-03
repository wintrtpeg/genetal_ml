"""설정 — 저장·불러오기와 Champion–Challenger.

화면에서 고른 값을 파일 하나로 떨어뜨려 두면 두 가지가 된다.
  1. Dataiku Scenario 처럼 UI 가 없는 곳에서 같은 설정으로 돌릴 수 있다 (SPEC §2)
  2. 사람 사이에 "그때 그 설정"을 그대로 주고받을 수 있다
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state, theme
from core import config, housekeeping as hk, persist, train


def render() -> None:
    S = st.session_state
    st.title("설정")

    tabs = st.tabs(["모드", "설정 파일", "Champion–Challenger", "저장공간"])
    with tabs[0]:
        _mode_panel()
    with tabs[1]:
        _config_panel()
    with tabs[2]:
        _challenger_panel()
    with tabs[3]:
        _storage_panel()


# ─────────────────────────────────────────────────────────────
def _mode_panel() -> None:
    S = st.session_state
    st.markdown("**작업 모드**")
    st.markdown('<p class="caption">달라지는 것은 <b>노출되는 설정의 범위</b>뿐입니다. '
                '3분할·gap 점검·선별구간 추적·Unseen 1회 접근은 어느 모드에서도 '
                '그대로 돕니다 — <b>"Auto 라서 검증을 덜 한다" 같은 건 없습니다.</b></p>',
                unsafe_allow_html=True)

    picked = st.radio(
        "모드", state.MODES,
        index=state.MODES.index(state.mode()),
        format_func=lambda m: f"{m} — {state.MODE_HELP[m]}")
    if picked != S.mode:
        S.mode = picked
        st.rerun()

    st.info(state.MODE_DETAIL[state.mode()])

    st.markdown("**모드별 노출 범위**")
    st.dataframe(pd.DataFrame([
        {"설정": "타겟·데이터 선택",
         "Auto": "직접", "Guided": "직접", "Expert": "직접"},
        {"설정": "제외 컬럼 조정",
         "Auto": "자동", "Guided": "직접", "Expert": "직접"},
        {"설정": "전처리 방식 (대치·스케일링·클리핑)",
         "Auto": "자동", "Guided": "직접", "Expert": "직접"},
        {"설정": "파생변수 종류·창 크기",
         "Auto": "자동", "Guided": "직접", "Expert": "직접"},
        {"설정": "분할 비율·날짜",
         "Auto": "자동", "Guided": "직접", "Expert": "직접"},
        {"설정": "X 피처 최종 확정",
         "Auto": "자동", "Guided": "직접", "Expert": "직접"},
        {"설정": "피처 간 조합 (ratio·diff)",
         "Auto": "안 함", "Guided": "안 함", "Expert": "직접"},
        {"설정": "폴드 내부 선별 토글",
         "Auto": "항상 켬", "Guided": "항상 켬", "Expert": "직접"},
        {"설정": "앙상블 자동채택 임계값",
         "Auto": "3%", "Guided": "3%", "Expert": "직접"},
        {"설정": "하이퍼파라미터 탐색 (nested CV)",
         "Auto": "안 함", "Guided": "안 함", "Expert": "직접"},
        {"설정": "Rolling Backtest·분할 진단",
         "Auto": "숨김", "Guided": "보임", "Expert": "보임"},
        {"설정": "누수 방지 장치 전부",
         "Auto": "항상 켬", "Guided": "항상 켬", "Expert": "항상 켬"},
    ]), use_container_width=True, hide_index=True, height=460)
    st.caption("'자동' 은 기본값으로 대신한다는 뜻이고, 무엇을 어떻게 정했는지는 "
               "실행 뒤 결정 표에 근거와 함께 남습니다. '항상 켬' 은 끌 수 없습니다.")


# ─────────────────────────────────────────────────────────────
def _current() -> config.StudioConfig:
    S = st.session_state
    from core import features, preprocess, validation
    return config.StudioConfig(
        features=S.feature_config or features.FeatureConfig(),
        preprocess=S.prep_config or preprocess.PreprocessConfig(),
        split=S.split_config or validation.SplitConfig(),
        train=S.train_config or train.TrainConfig(),
        meta={"target": S.target, "time_col": S.time_col,
              "kept_columns": S.kept, "mode": S.mode,
              "source": S.source_desc},
    )


def _config_panel() -> None:
    S = st.session_state
    st.markdown("**현재 설정 내려받기**")
    st.markdown('<p class="caption">파생변수·전처리·분할·학습 설정이 한 파일에 담깁니다. '
                '데이터와 접속정보는 들어가지 않습니다.</p>', unsafe_allow_html=True)

    cfg = _current()
    text = config.dumps(cfg)
    fmt = "yaml" if config._has_yaml() else "json"
    c1, c2 = st.columns([1, 3])
    c1.download_button(f"설정 내려받기 (.{fmt})", text.encode("utf-8"),
                       file_name=f"ml_studio_config.{fmt}",
                       mime="text/plain", type="primary")
    if fmt == "json":
        c2.caption("pyyaml 이 없어 JSON 으로 떨어집니다. 불러오기는 양쪽 다 됩니다.")

    with st.expander("내용 보기"):
        st.code(text, language=fmt)

    st.divider()
    st.markdown("**설정 불러오기**")
    up = st.file_uploader("설정 파일", type=["yaml", "yml", "json"])
    pasted = st.text_area("또는 붙여넣기", height=140, placeholder="split:\n  unseen_ratio: 0.2")

    raw = None
    if up is not None:
        raw = up.getvalue().decode("utf-8")
    elif pasted.strip():
        raw = pasted

    if raw and st.button("적용", type="primary"):
        try:
            loaded, warns = config.loads(raw)
        except Exception as e:  # noqa: BLE001
            st.error(f"읽지 못했습니다 — {type(e).__name__}: {e}")
            return

        for w in warns:
            st.warning(w)

        d = config.diff(_current(), loaded)
        if d.empty:
            st.info("현재 설정과 같습니다. 바뀌는 것이 없습니다.")
            return

        st.markdown("**바뀌는 항목**")
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.warning("설정을 바꾸면 파생변수부터 다시 실행해야 합니다. "
                   "이전 학습 결과는 무효가 됩니다.")
        if st.button("확인 — 적용하고 3단계부터 다시"):
            S.feature_config = loaded.features
            S.prep_config = loaded.preprocess
            S.split_config = loaded.split
            S.train_config = loaded.train
            S.studio_config = loaded
            if loaded.meta.get("mode") in state.MODES:
                S.mode = loaded.meta["mode"]
            state.invalidate("prep")
            st.success("적용했습니다. 3단계 파생변수부터 다시 실행해 주세요.")


# ─────────────────────────────────────────────────────────────
def _load_run_cached(name: str) -> dict:
    """지난 run 을 디스크에서 읽는 일은 한 번만 한다.

    임계 슬라이더를 한 칸 옮길 때마다 run 폴더 전체를 다시 읽으면 조작이 굼뜨다.
    고른 run 이 그대로면 지난번에 읽은 것을 쓴다.
    """
    S = st.session_state
    if S.get("_run_cache_key") != name:
        S["_run_cache_key"] = name
        S["_run_cache_val"] = persist.load_run(persist.RUNS_DIR / name)
    return S["_run_cache_val"]


def _challenger_panel() -> None:
    """SPEC §18 — 새 모델이 의미 있게 나아졌을 때만 교체한다."""
    S = st.session_state
    st.markdown("**Champion–Challenger**")
    st.markdown('<p class="caption">지난 run 의 챔피언과 지금 챔피언을 견줍니다. '
                '운영 모델을 바꾸는 데는 재검증 비용이 따르므로, 소수점 뒤 개선으로 '
                '갈아타지 않습니다.</p>', unsafe_allow_html=True)

    if not S.champion or not S.unseen_scores:
        st.info("지금 run 의 Final Unseen 평가를 먼저 실행해 주세요. "
                "같은 성격의 구간에서 잰 점수끼리만 비교합니다.")
        return

    runs = persist.list_runs()
    if not len(runs):
        st.info("비교할 지난 run 이 없습니다. 리포트 화면에서 저장하면 쌓입니다.")
        return

    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("지난 run (challenger 로 볼 대상)", list(runs["run"]))
    th = c2.slider("교체 임계", 0.0, 0.20, 0.02, 0.01, format="%.2f")

    loaded = _load_run_cached(pick)
    man = loaded.get("manifest")
    if not man:
        st.warning("그 run 에는 manifest.json 이 없습니다 (구버전 실행). "
                   "unseen 점수를 확인할 수 없어 비교하지 않습니다.")
        return

    past_scores = man.get("final_unseen") or {}
    if not past_scores:
        st.warning("그 run 에는 Final Unseen 점수가 없습니다. "
                   "2분할로 돌렸거나 unseen 평가를 실행하지 않은 run 입니다.")
        return

    metric = S.train_config.champion_metric if S.train_config else "R2"
    ds_now = (S.manifest or {}).get("dataset", {}).get("sha256")
    ds_past = man.get("dataset", {}).get("sha256")
    if ds_now and ds_past and ds_now != ds_past:
        st.warning("두 run 의 데이터 지문이 다릅니다. 구간 난이도 차이가 성능 차이로 "
                   "보일 수 있으니 결과를 그대로 믿지 마세요.")

    verdict = persist.challenge(
        champion_name=f"{man.get('champion', pick)} ({pick})",
        champion_scores=past_scores,
        challenger_name=S.champion,
        challenger_scores=S.unseen_scores,
        metric=metric, threshold=float(th))

    if verdict["decision"] == "교체":
        st.success(verdict["reason"])
    elif verdict["decision"] == "유지":
        st.info(verdict["reason"])
    else:
        st.warning(verdict["reason"])
        return

    st.dataframe(pd.DataFrame([{
        "역할": "기존 (지난 run)", "모델": verdict["champion"],
        f"unseen {metric}": verdict["champion_score"],
    }, {
        "역할": "신규 (지금 run)", "모델": verdict["challenger"],
        f"unseen {metric}": verdict["challenger_score"],
    }]), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("개선율", f"{verdict['개선율']:+.2%}")
    c2.metric("판정", verdict["decision"])
    S.challenger = verdict


# ─────────────────────────────────────────────────────────────
def _storage_panel() -> None:
    """저장공간 — 무엇이 얼마나 차지하는지, 무엇을 지울지.

    가장 큰 것은 실행마다 남는 `champion_model.joblib` 이다. RandomForest·앙상블
    이면 한 번에 수백 MB 라, 며칠 쓰면 기가 단위가 된다.

    **말없이 지우지 않는다.** 실행 결과는 사용자 산출물이므로, 정책을 눈에
    보이게 두고 무엇이 지워질지 먼저 보여준 뒤에 지운다.
    """
    S = st.session_state
    st.subheader("저장공간")

    u = hk.usage()
    c1, c2, c3 = st.columns(3)
    c1.metric("실행 결과", u["runs"], help="runs 폴더 전체 크기입니다.")
    c2.metric("정리 가능한 찌꺼기", u["junk"],
              help="캐시·중간에 죽은 실행 폴더·점검 임시파일입니다. "
                   "지워도 아무것도 잃지 않습니다.")
    c3.metric("드라이브 여유", u["free"])

    theme.caption("실행 하나가 큰 이유는 <b>champion_model.joblib</b> 입니다 — "
                  "RandomForest·앙상블이면 수백 MB 가 나옵니다. 나머지(csv·json·"
                  "html)는 합쳐도 대개 수 MB 입니다.")

    st.markdown("**자동 정리 기준**")
    st.caption("화면을 실행할 때마다 이 기준으로 자동 정리합니다. "
               "**보관으로 지정한 실행은 기준과 무관하게 남습니다.**")
    c1, c2, c3 = st.columns(3)
    keep_n = c1.number_input(
        "최신 몇 개를 남길지", 0, 200, int(S.get("keep_runs", 10)),
        help="0 이면 개수로는 제한하지 않습니다.")
    budget = c2.number_input(
        "총 용량 상한 (MB)", 0, 500_000, int(S.get("keep_mb", 2000)), step=500,
        help="runs 폴더 전체가 이 크기를 넘으면 오래된 것부터 지웁니다. "
             "0 이면 용량으로는 제한하지 않습니다.")
    days = c3.number_input(
        "며칠까지 보관", 0, 3650, int(S.get("keep_days", 0)),
        help="이보다 오래된 실행은 정리합니다. 0 이면 기간은 보지 않습니다.")
    S["keep_runs"], S["keep_mb"], S["keep_days"] = int(keep_n), int(budget), int(days)

    policy = hk.RetentionPolicy(keep_runs=int(keep_n),
                                max_total_mb=float(budget),
                                keep_days=int(days))
    st.caption(f"지금 기준 — {policy.describe()}")

    protect = tuple(x for x in [S.get("run_dir") and str(S["run_dir"]).split("/")[-1]] if x)
    p = hk.plan(policy, protect=protect)

    st.markdown("**지금 정리하면**")
    if not p:
        st.success("정리할 것이 없습니다.")
    else:
        lines = []
        if p.junk:
            lines.append(f"찌꺼기 {len(p.junk)}개 ({hk.mb(p.junk_bytes)})")
        if p.runs:
            lines.append(f"실행 {len(p.runs)}개 ({hk.mb(p.runs_bytes)})")
        st.warning(f"{' · '.join(lines)} → 총 **{hk.mb(p.total_bytes)}** 확보")
        if p.runs:
            st.dataframe(pd.DataFrame(
                [{"실행": d.name, "사유": p.reasons.get(d.name, "")} for d in p.runs]),
                use_container_width=True, hide_index=True, height=180)
        for k in p.kept[:5]:
            st.caption(f"· 남깁니다 — {k}")

        c1, c2 = st.columns([1, 3])
        if c1.button("지금 정리", type="primary"):
            res = hk.apply(p)
            (st.success if not res.failed else st.warning)(res.summary())
            for f in res.failed[:5]:
                st.caption(f"· {f}")
            st.rerun()
        c2.caption("실행 결과 폴더가 통째로 지워집니다. 보관할 것은 아래에서 "
                   "먼저 지정하세요.")

    st.divider()
    st.markdown("**실행별 보관 지정**")
    st.caption("보관으로 지정하면 **어떤 기준에도 걸리지 않습니다.** "
               "나중에 다시 열어 볼 결과에 표시해 두세요.")
    table = hk.scan()
    if table.empty:
        st.caption("저장된 실행이 없습니다.")
        return

    show = table[["실행", "용량", "생성", "일수", "보관", "모델", "리포트"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=260)

    c1, c2, c3 = st.columns([2, 1, 1])
    pick = c1.selectbox("실행 선택", list(table["실행"]))
    if c2.button("보관 지정", use_container_width=True):
        hk.pin(hk.RUNS_DIR / pick)
        st.rerun()
    if c3.button("보관 해제", use_container_width=True):
        hk.unpin(hk.RUNS_DIR / pick)
        st.rerun()
