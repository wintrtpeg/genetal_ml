"""화면 10개를 **실제로 실행해** 본다.

지금까지 화면 코드는 소스를 눈으로 읽어서만 점검했다. streamlit·plotly 설치가
막혀 있어서 달리 방법이 없었기 때문이다. 그 방식으로 10차까지 여러 건을 잡았지만,
한 줄씩 훑는 일이라 반드시 새는 곳이 생긴다.

여기서는 대역(tests/fake_streamlit.py · tests/fake_plotly.py)을 꽂고 render() 를
진짜로 부른다. 그러면 두 부류가 자동으로 드러난다.

  1. 파이썬 수준 오류 — KeyError, IndexError, AttributeError, TypeError,
     없는 컬럼 참조, None 연산. 라이브러리 유무와 무관한 진짜 결함이다.
  2. 진짜 streamlit 이 던지는 예외 — 위젯 기본값 범위 밖, 잘못된 icon,
     잘못된 format 문자열, 없는 컬럼을 가리키는 column_config.
     지금까지 손으로 하나씩 찾아낸 결함이 전부 이 부류였다.

**여기서 통과한다고 화면이 예쁘다는 뜻은 아니다. 터지지 않는다는 뜻이다.**
레이아웃·색·간격은 브라우저가 있어야 보인다.

상태는 가짜로 만들지 않고 **진짜 파이프라인을 돌려서** 채운다. 손으로 만든
상태는 실제와 어긋나기 마련이고, 그러면 통과해도 의미가 없다.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import fake_plotly, fake_streamlit  # noqa: E402
from tests.fake_streamlit import (  # noqa: E402
    FakeStreamlit, Recorder, RerunException, SessionState, StopException,
)

VIEWS = ["data_view", "prep_view", "features_view", "train_view", "predict_view",
         "explain_view", "whatif_view", "diagnostics_view", "report_view",
         "config_view"]


# ─────────────────────────────────────────────────────────────
# 대역 설치
# ─────────────────────────────────────────────────────────────
def _install(rec: Recorder, state: SessionState, clicks=None, values=None) -> FakeStreamlit:
    """sys.modules 에 가짜 streamlit·plotly 를 꽂는다.

    app.* 는 모듈 최상단에서 `import streamlit as st` 를 하므로, 이미 임포트된
    모듈이 있으면 지워서 다시 임포트되게 해야 한다.
    """
    fake_plotly.install()
    fs = FakeStreamlit(rec, state, clicks, values)

    mod = types.ModuleType("streamlit")
    for name in dir(fs):
        if not name.startswith("__"):
            setattr(mod, name, getattr(fs, name))
    mod.session_state = state
    mod.column_config = fs.column_config
    mod.sidebar = fs.sidebar          # `with st.sidebar:` 를 받아야 한다
    mod.errors = types.SimpleNamespace(
        StreamlitAPIException=fake_streamlit.StreamlitAPIException)
    sys.modules["streamlit"] = mod
    sys.modules["streamlit.errors"] = mod.errors

    for name in list(sys.modules):
        if name.startswith("app.") or name == "app":
            del sys.modules[name]
    return fs


def _uninstall() -> None:
    for name in list(sys.modules):
        if name.startswith(("app", "streamlit", "plotly")):
            del sys.modules[name]


# ─────────────────────────────────────────────────────────────
# 진짜 파이프라인으로 상태 만들기
# ─────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _pipeline_state() -> dict:
    """1~4단계를 실제로 돌려 화면이 기대하는 상태를 만든다. 한 번만 계산한다."""
    if "state" in _CACHE:
        return dict(_CACHE["state"])

    from core import (
        datasource, features, models, preprocess, profiling, train, validation,
    )

    # 소규모 합성 데이터 — 데모 CSV 가 없는 환경에서도 돌아야 한다
    n = 1200
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    flow = 50 + np.cumsum(rng.normal(scale=0.3, size=n))
    temp = 60 + np.sin(np.linspace(0, 30, n)) * 8 + rng.normal(scale=0.4, size=n)
    press = 3 + rng.normal(scale=0.1, size=n)
    y = 0.8 * flow + 0.5 * np.maximum(temp - 65, 0) + rng.normal(scale=0.5, size=n)
    raw = pd.DataFrame({
        "timestamp": idx, "flow": flow, "temp": temp, "pressure": press,
        "status": rng.choice(["NORMAL", "WARN"], n), "y_output": y,
    })

    df = datasource.to_timeseries(raw, "timestamp")
    target = "y_output"
    candidates = [c for c in df.columns if c != target]
    kept = [c for c in candidates if c != "status"]

    fcfg = features.FeatureConfig(lags=(1, 3), rolling_windows=(6,),
                                  rolling_stats=("mean",), diffs=(1,))
    feat, prov = features.generate(df, target, kept, fcfg)

    cols = [c for c in feat.columns if c != target]
    X_all, y_all = preprocess.prepare_xy(feat, target, cols)
    gap = features.warmup_rows(fcfg, feat.index)
    scfg = validation.SplitConfig(holdout_ratio=0.2, unseen_ratio=0.15, gap=gap)
    split = validation.build_split(scfg, X_all.index)

    task = models.detect_task(y_all)
    sel, rep = features.select_features(
        X_all.iloc[split.train], y_all.iloc[split.train], task=task, top_k=12)
    review = features.feature_report(rep, prov, X_train=X_all.iloc[split.train])
    chosen, updated = features.apply_manual_selection(review, sel)
    X = X_all[chosen]

    pcfg = preprocess.PreprocessConfig()
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, pcfg)
    zoo = models.get_model_zoo(task, include_heavy=False)
    picked = [m for m in ("Ridge", "DecisionTree") if m in zoo] or list(zoo)[:2]
    tcfg = train.TrainConfig(task=task, n_jobs=1, fold_selection=False,
                             split=validation.SplitConfig(n_splits=2, gap=gap))
    board, detail = train.train_all(X, y_all, split.train, split.valid,
                                    pre, zoo, picked, tcfg)
    champ = train.pick_champion(board, tcfg.champion_metric)
    guard = train.UnseenGuard(split.unseen)
    unseen = train.evaluate_unseen(detail[champ]["_pipeline"], X, y_all,
                                   split.unseen, tcfg, guard, who=champ)

    pipe = detail[champ]["_pipeline"]
    pred = pd.Series(pipe.predict(X.iloc[split.valid]), index=X.index[split.valid])
    predictions = pd.DataFrame({"actual": y_all.iloc[split.valid], "predicted": pred})

    st_ = {
        "raw": raw, "df": df, "target": target, "time_col": "timestamp",
        "source_desc": "합성 데이터", "candidates": candidates, "kept": kept,
        "quality_profile": profiling.profile(df),
        "feat_df": feat, "provenance": prov, "feature_config": fcfg,
        "prep_config": pcfg, "selection_report": updated,
        "feature_review": updated, "review_picks": list(chosen),
        "selected_features": chosen, "X_pool": X_all, "X": X, "y": y_all,
        "split": split, "split_config": scfg,
        "train_idx": split.train, "test_idx": split.valid,
        "unseen_idx": split.unseen, "selection_train_idx": split.train,
        "task": task, "train_config": tcfg,
        "leaderboard": board, "detail": detail, "champion": champ,
        "unseen_scores": unseen, "unseen_guard": guard,
        "predictions": predictions,
        "learning_mode": "지도학습",
    }
    _CACHE["state"] = st_
    return dict(st_)


def _state(**over) -> SessionState:
    """DEFAULTS 로 채운 뒤 파이프라인 결과를 얹는다."""
    import importlib

    # state 모듈은 streamlit 을 임포트하므로 대역이 꽂힌 뒤에 읽어야 한다
    st_mod = importlib.import_module("app.state")
    s = SessionState()
    for k, v in st_mod.DEFAULTS.items():
        s[k] = v.copy() if isinstance(v, (dict, list)) else v
    s.update(over)
    return s


# ─────────────────────────────────────────────────────────────
# 실행 헬퍼
# ─────────────────────────────────────────────────────────────
def _render(view_name: str, state_over: dict, clicks=None, values=None) -> Recorder:
    """화면 하나를 실제로 그려 본다. rerun/stop 은 정상 종료로 본다."""
    rec = Recorder()
    _install(rec, SessionState(), clicks, values)
    try:
        import importlib
        s = _state(**state_over)
        _install(rec, s, clicks, values)          # 상태를 확정해 다시 꽂는다
        mod = importlib.import_module(f"app.views.{view_name}")
        try:
            mod.render()
        except (RerunException, StopException):
            pass                                   # 진짜 streamlit 도 여기서 끊는다
    finally:
        _uninstall()
    return rec


# ─────────────────────────────────────────────────────────────
# 1. 아무것도 없는 상태 — 첫 실행
# ─────────────────────────────────────────────────────────────
def test_every_view_renders_on_a_cold_start():
    """데이터가 없는 상태에서 10개 화면이 전부 안내만 띄우고 멈춰야 한다.

    사용자가 사이드바로 아무 단계나 바로 누를 수 있다. 그때 트레이스가 뜨면 안 된다.
    """
    broken = []
    for v in VIEWS:
        try:
            _render(v, {})
        except Exception as e:  # noqa: BLE001
            broken.append(f"{v}: {type(e).__name__}: {e}")
    assert not broken, "빈 상태에서 죽는 화면:\n" + "\n".join(broken)


# ─────────────────────────────────────────────────────────────
# 2. 완전한 상태 — 끝까지 진행한 뒤
# ─────────────────────────────────────────────────────────────
def test_every_view_renders_with_a_full_pipeline():
    """1~4단계를 실제로 돌린 상태에서 10개 화면이 전부 그려져야 한다."""
    full = _pipeline_state()
    broken = []
    for v in VIEWS:
        try:
            rec = _render(v, full)
            assert rec.calls, f"{v}: 아무것도 그리지 않았습니다"
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-4:]
            broken.append(f"{v}: {type(e).__name__}: {e}\n      " + "\n      ".join(tb))
    assert not broken, "완전한 상태에서 죽는 화면:\n" + "\n".join(broken)


def test_full_pipeline_views_actually_draw_charts():
    """그려졌다고 하면서 차트가 0개면 조건 분기가 잘못 걸린 것이다."""
    full = _pipeline_state()
    for v in ("predict_view", "train_view"):
        rec = _render(v, full)
        assert rec.charts > 0, f"{v}: 완전한 상태인데 차트를 하나도 안 그렸습니다"


# ─────────────────────────────────────────────────────────────
# 3. 중간 상태 — 사용자가 실제로 멈춰 있는 지점들
# ─────────────────────────────────────────────────────────────
def test_views_render_at_every_intermediate_stage():
    """단계별로 상태를 하나씩 지워 가며 그려 본다.

    사용자는 늘 중간 어딘가에 있다. "완전한 상태" 만 확인하면 정작 실제로
    보게 되는 화면을 한 번도 안 그려 본 셈이 된다.
    """
    full = _pipeline_state()
    stages = {
        "데이터만": {k: full[k] for k in ("raw", "df", "target", "time_col",
                                       "candidates", "source_desc")},
        "전처리까지": {k: full[k] for k in ("raw", "df", "target", "time_col",
                                       "candidates", "source_desc", "kept",
                                       "quality_profile")},
        "파생만 생성": {**{k: full[k] for k in ("raw", "df", "target", "time_col",
                                          "candidates", "source_desc", "kept",
                                          "quality_profile", "feat_df",
                                          "provenance", "feature_config")}},
        "검토 전": {**{k: full[k] for k in
                    ("raw", "df", "target", "time_col", "candidates", "source_desc",
                     "kept", "quality_profile", "feat_df", "provenance",
                     "feature_config", "prep_config")}},
        "확정 전": {**{k: v for k, v in full.items()
                    if k not in ("X", "selected_features", "leaderboard", "detail",
                                 "champion", "unseen_scores", "unseen_guard",
                                 "predictions")}},
        "학습 전": {**{k: v for k, v in full.items()
                    if k not in ("leaderboard", "detail", "champion",
                                 "unseen_scores", "unseen_guard", "predictions")}},
        "예측 전": {**{k: v for k, v in full.items() if k != "predictions"}},
        "unseen 전": {**{k: v for k, v in full.items()
                       if k not in ("unseen_scores",)}},
    }
    broken = []
    for stage, s in stages.items():
        for v in VIEWS:
            try:
                _render(v, s)
            except Exception as e:  # noqa: BLE001
                broken.append(f"[{stage}] {v}: {type(e).__name__}: {e}")
    assert not broken, "중간 상태에서 죽는 화면:\n" + "\n".join(broken)


# ─────────────────────────────────────────────────────────────
# 4. 극단적인 데이터 모양
# ─────────────────────────────────────────────────────────────
def test_views_survive_degenerate_data():
    """빈 프레임 · 한 행 · 상수 컬럼 · 전 결측 — 실제로 나오는 모양들이다."""
    idx = pd.date_range("2025-01-01", periods=3, freq="5min")
    cases = {
        "빈 raw": {"raw": pd.DataFrame(), "source_desc": "빈 쿼리 결과"},
        "컬럼만 있고 행 없음": {
            "raw": pd.DataFrame(columns=["timestamp", "a", "y"]),
            "source_desc": "조건이 좁은 쿼리"},
        "한 행": {
            "raw": pd.DataFrame({"timestamp": [idx[0]], "a": [1.0], "y": [2.0]}),
            "df": pd.DataFrame({"a": [1.0], "y": [2.0]}, index=idx[:1]),
            "target": "y", "candidates": ["a"], "kept": ["a"]},
        "전 결측 컬럼": {
            "raw": pd.DataFrame({"timestamp": idx, "a": [np.nan] * 3, "y": [1.0, 2, 3]}),
            "df": pd.DataFrame({"a": [np.nan] * 3, "y": [1.0, 2, 3]}, index=idx),
            "target": "y", "candidates": ["a"], "kept": ["a"]},
        # 세로형(long) — 돌려 세우기 화면이 뜨는 경로
        "세로형 원본": {
            "raw": pd.concat([
                pd.DataFrame({"tag_time": pd.date_range("2025-01-01", periods=100,
                                                        freq="5min"),
                              "tag_name": t,
                              "value": np.random.default_rng(0).normal(size=100)})
                for t in ("FLOW", "TEMP", "Y")], ignore_index=True),
            "source_desc": "세로형"},
        "상수 컬럼": {
            "raw": pd.DataFrame({"timestamp": idx, "a": [5.0] * 3, "y": [1.0, 2, 3]}),
            "df": pd.DataFrame({"a": [5.0] * 3, "y": [1.0, 2, 3]}, index=idx),
            "target": "y", "candidates": ["a"], "kept": ["a"]},
    }
    broken = []
    for name, s in cases.items():
        for v in VIEWS:
            try:
                _render(v, s)
            except Exception as e:  # noqa: BLE001
                broken.append(f"[{name}] {v}: {type(e).__name__}: {e}")
    assert not broken, "극단적 데이터에서 죽는 화면:\n" + "\n".join(broken)


def test_views_survive_partially_missing_state():
    """상태가 한 조각씩 빠진 조합을 전부 지나가 본다.

    평소에는 이 값들이 같이 채워지고 같이 지워진다. 하지만 그건 invalidate
    체인이 지금 그렇게 짜여 있다는 것에 기댄 것이지 화면이 보장한 게 아니다.
    체인을 한 번 손대거나 설정을 불러오는 경로가 생기면 바로 트레이스가 뜬다.

    실제로 이 테스트가 처음 돌았을 때 9건이 나왔고, ready() 가 champion 하나만
    보고 X·y·split 을 안 보던 것이 그 뿌리였다.
    """
    full = _pipeline_state()
    probes = {
        "비지도학습 모드": {**full, "learning_mode": "비지도학습"},
        "비지도·확정 전": {**{k: v for k, v in full.items()
                          if k not in ("X", "selected_features")},
                        "learning_mode": "비지도학습"},
        "Expert 모드": {**full, "mode": "Expert"},
        "Auto 모드": {**full, "mode": "Auto"},
        "2분할(unseen 없음)": {**full, "unseen_idx": None, "unseen_scores": None,
                            "unseen_guard": None},
        "챔피언 없음": {**full, "champion": None},
        "detail 비어있음": {**full, "detail": {}},
        "provenance 없음": {**full, "provenance": None},
        "feature_config 없음": {**full, "feature_config": None},
        "prep_config 없음": {**full, "prep_config": None},
        "train_config 없음": {**full, "train_config": None},
        "split_config 없음": {**full, "split_config": None},
        "split 없음": {**full, "split": None},
        "kept 비어있음": {**full, "kept": []},
        "X 없음": {**full, "selected_features": [], "X": None},
        "y 없음": {**full, "y": None},
        "leaderboard 없음": {**full, "leaderboard": None},
    }
    broken = []
    for name, s in probes.items():
        for v in VIEWS:
            try:
                _render(v, s)
            except Exception as e:  # noqa: BLE001
                broken.append(f"[{name}] {v}: {type(e).__name__}: {e}")
    assert not broken, "상태가 빠진 조합에서 죽습니다:\n" + "\n".join(broken)


def test_ready_checks_what_the_views_actually_use():
    """ready() 가 대리 조건이 아니라 진짜 조건을 봐야 한다.

    5~7단계 화면은 champion 뿐 아니라 X·y·split·detail 을 바로 꺼내 쓴다.
    champion 하나만 보고 통과시키면 나머지가 None 일 때 트레이스가 뜬다.
    """
    src = (ROOT / "app" / "state.py").read_text(encoding="utf-8")
    body = src[src.index("def ready("):src.index("def guard(")]
    for key in ("S.X is not None", "S.y is not None", "S.detail is not None",
                "S.split is not None"):
        assert key in body, f"ready() 가 {key} 를 확인하지 않습니다"
    for step in ("predict", "explain", "whatif"):
        i = body.index(f'"{step}"')
        line = body[i:body.index("\n", i)]
        assert "trained" in line, f"'{step}' 이 여전히 대리 조건을 씁니다: {line}"


# ─────────────────────────────────────────────────────────────
# 4-b. 버튼을 눌렀을 때
# ─────────────────────────────────────────────────────────────
def test_button_paths_execute():
    """버튼 안쪽 코드는 평소에 실행되지 않는다. 눌린 것으로 만들어 지나가 본다.

    화면을 그리는 것만으로는 버튼 블록이 한 번도 안 돌아 본다. 정작 사용자가
    누르는 곳이고, 무거운 계산과 상태 변경이 전부 그 안에 있다.
    """
    full = _pipeline_state()
    cases = [
        ("predict_view", full, "예측 실행"),
        ("diagnostics_view", full, "진단 실행"),
        ("report_view", full, "리포트 만들기"),
        ("features_view", full, "품질 리포트 만들기"),
        ("features_view", full, "이 목록으로 확정"),
        ("features_view", full, "자동 추천대로"),
        ("features_view", full, "전체 선택"),
        ("features_view", full, "전체 해제"),
        ("features_view", full, "MI 상위 30"),
        ("train_view", full, "3단계로 돌아가 다시 나누기"),
        ("prep_view", full, "다시 진단"),
    ]
    broken, missing = [], []
    for view, state, label in cases:
        try:
            rec = _render(view, state, clicks={label})
            # 라벨이 안 맞으면 아무 버튼도 안 눌린다. 그러면 이 테스트는
            # 조용히 통과하면서 **아무것도 검사하지 않는다.** 화면 문구를
            # 다듬을 때마다 생길 수 있는 구멍이라 여기서 막는다.
            if f"button:{label}" not in rec.widgets:
                seen = [w[7:] for w in rec.widgets if w.startswith("button:")]
                missing.append(f"{view}: '{label}' 버튼이 없습니다. 실제 버튼: {seen[:8]}")
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-3:]
            broken.append(f"{view} [{label}]: {type(e).__name__}: {e}\n      "
                          + "\n      ".join(tb))
    assert not missing, ("버튼 이름이 화면과 다릅니다 — 이 테스트가 헛돌고 있었습니다:\n"
                         + "\n".join(missing))
    assert not broken, "버튼 경로에서 죽습니다:\n" + "\n".join(broken)


def test_unseen_evaluation_button_respects_a_spent_guard():
    """이미 쓴 접근권으로 평가 버튼을 누르면 두 번째 평가가 나오면 안 된다."""
    full = _pipeline_state()
    guard = full["unseen_guard"]
    assert guard.access_count == 1, "준비 상태부터 한 번 열려 있어야 합니다"

    # 점수만 지운 상태 = 학습을 다시 돌린 뒤의 화면
    spent = {**full, "unseen_scores": None}
    rec = _render("train_view", spent, clicks={"Final Unseen 평가 실행"})
    assert guard.access_count == 1, "접근권이 두 번 열렸습니다"
    joined = " ".join(rec.errors)
    assert "이미" in joined and "분할" in joined, \
        f"이미 열렸다는 안내가 없습니다: {rec.errors[:2]}"


def test_reset_button_needs_two_presses():
    """전체 초기화는 한 번 눌러서는 지워지지 않아야 한다."""
    import importlib

    full = _pipeline_state()
    rec = Recorder()
    s = SessionState()
    _install(rec, s, clicks={"전체 초기화"})
    try:
        s2 = _state(**full)
        _install(rec, s2, clicks={"전체 초기화"})
        main = importlib.import_module("app.main")   # noqa: F841
    except RerunException:
        pass
    except Exception:
        pass
    finally:
        _uninstall()
    # 첫 번째 누름은 확인 단계로만 넘어간다
    assert s2.get("_reset_armed") is True, "확인 단계 없이 바로 지웁니다"
    assert s2.get("champion") is not None, "첫 누름에 이미 지워졌습니다"


# ─────────────────────────────────────────────────────────────
# 5. 대역 자체가 제구실을 하는지
# ─────────────────────────────────────────────────────────────
def test_the_fake_catches_out_of_range_widget_defaults():
    """대역이 아무것도 안 잡으면 위 테스트들이 전부 무의미해진다."""
    import pytest

    rec, s = Recorder(), SessionState()
    fs = FakeStreamlit(rec, s)
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.number_input("표본 상한", 200, 50000, 50)        # 기본값 < 하한
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.slider("주성분 개수", 2, 2, 3)                    # 기본값 > 상한
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.number_input("상위 몇 개까지", 5, 2, 3)           # min > max
    _uninstall()


def test_the_fake_catches_bad_icons_and_formats():
    import pytest

    rec, s = Recorder(), SessionState()
    fs = FakeStreamlit(rec, s)
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.info("안내", icon="◧")                            # 이모지가 아니다
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.column_config.NumberColumn("결측률", format="%.2%")
    fs.info("안내")                                          # icon 없으면 통과
    fs.column_config.NumberColumn("MI", format="%.4f")
    fs.column_config.NumberColumn("결측률", format="%.1f%%")
    _uninstall()


def test_the_fake_catches_removed_plotly_attributes():
    import pytest

    from tests import fake_plotly as fp
    fig = fp.Figure()
    with pytest.raises(ValueError, match="plotly 5"):
        fig.update_layout(yaxis2=dict(title="x", titlefont=dict(color="#000")))
    fig.update_layout(yaxis2=dict(title=dict(text="x", font=dict(color="#000"))))


def test_the_fake_catches_oversized_option_lists():
    """12,948개 옵션을 넘기던 그 결함이 다시 오면 잡아야 한다."""
    import pytest

    rec, s = Recorder(), SessionState()
    fs = FakeStreamlit(rec, s)
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.select_slider("경계", options=list(range(12948)))
    fs.select_slider("경계", options=list(range(50)))
    _uninstall()


def test_the_fake_catches_column_config_pointing_at_missing_columns():
    import pytest

    rec, s = Recorder(), SessionState()
    fs = FakeStreamlit(rec, s)
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(fake_streamlit.StreamlitAPIException):
        fs.dataframe(df, column_config={"없는컬럼": fs.column_config.TextColumn("x")})
    fs.dataframe(df, column_config={"a": fs.column_config.NumberColumn("A")})
    _uninstall()


# ─────────────────────────────────────────────────────────────
# 6. 차트 함수 전수 실행
# ─────────────────────────────────────────────────────────────
def test_every_plot_function_executes():
    """차트 22종을 대역 위에서 전부 실행한다.

    진짜 plotly 스키마까지는 못 보지만, 파이썬 수준 오류와 제거된 속성은 잡힌다.
    진짜로 그려 보는 것은 scripts/verify_env.py 가 설치된 PC 에서 한다.
    """
    fake_plotly.install()
    try:
        import importlib

        plots = importlib.import_module("core.plots")
        importlib.reload(plots)

        n = 300
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        rng = np.random.default_rng(0)
        a = pd.Series(np.sin(np.linspace(0, 20, n)) * 10 + 50, index=idx)
        b = a + rng.normal(scale=0.5, size=n)
        resid = a - b

        big = pd.date_range("2025-01-01", periods=5000, freq="min")
        ba = pd.Series(rng.normal(size=5000), index=big)

        calls = {
            "actual_vs_pred": lambda: plots.actual_vs_pred(a, b, train_end=idx[200]),
            "actual_vs_pred(WebGL)": lambda: plots.actual_vs_pred(ba, ba * 1.01),
            "residual_series": lambda: plots.residual_series(a, b),
            "residual_band": lambda: plots.residual_band(pd.DataFrame({
                "residual": resid.to_numpy(),
                "roll_mean": resid.rolling(12).mean().to_numpy(),
                "roll_std": resid.rolling(12).std().to_numpy()}, index=idx)),
            "residual_drift": lambda: plots.residual_drift(pd.DataFrame(
                {"구간": [1, 2, 3], "MAE": [.4, .5, .45], "std": [.6, .8, .7]})),
            "backtest_series": lambda: plots.backtest_series(pd.DataFrame({
                "구간": [1, 2, 3], "평가시작": idx[[0, 100, 200]],
                "R2": [.9, .85, .88], "n_train": [100, 200, 300],
                "n_test": [50, 50, 50], "status": ["ok"] * 3})),
            "residual_acf": lambda: plots.residual_acf(pd.DataFrame(
                {"lag": range(1, 21), "acf": rng.normal(scale=.1, size=20)}), n),
            "scatter_actual_pred": lambda: plots.scatter_actual_pred(a, b),
            "leaderboard_bar": lambda: plots.leaderboard_bar(pd.DataFrame({
                "model": ["Ridge", "RF"], "holdout_R2": [.81, .93],
                "status": ["ok"] * 2, "family": ["linear", "ensemble"]}), "R2"),
            "shap_importance_bar": lambda: plots.shap_importance_bar(pd.DataFrame({
                "feature": [f"f{i}" for i in range(8)],
                "mean_abs_shap": np.linspace(1, .1, 8),
                "contribution_pct": np.linspace(30, 2, 8)})),
            # 컬럼 이름은 explain.dependence_data 가 내는 것과 같아야 한다
            "shap_dependence": lambda: plots.shap_dependence(pd.DataFrame({
                "timestamp": idx, "feature_value": rng.normal(size=n),
                "shap_value": rng.normal(size=n),
                "interaction_value": rng.normal(size=n)}), "f0"),
            "shap_dependence(시점색)": lambda: plots.shap_dependence(pd.DataFrame({
                "timestamp": idx, "feature_value": rng.normal(size=n),
                "shap_value": rng.normal(size=n)}), "f0", color_mode="time"),
            "shap_dependence(단색)": lambda: plots.shap_dependence(pd.DataFrame({
                "timestamp": idx, "feature_value": rng.normal(size=n),
                "shap_value": rng.normal(size=n)}), "f0", color_mode="none"),
            "shap_dependence(구간)": lambda: plots.shap_dependence(pd.DataFrame({
                "timestamp": idx, "feature_value": rng.normal(size=n),
                "shap_value": rng.normal(size=n),
                "period": ["A"] * 150 + ["B"] * 150}), "f0", color_mode="period"),
            "shap_period_shift": lambda: plots.shap_period_shift(pd.DataFrame({
                "feature": [f"f{i}" for i in range(5)],
                "P1": np.linspace(1, .2, 5), "P2": np.linspace(.9, .1, 5)}),
                ["P1", "P2"]),
            "shap_contribution_stream": lambda: plots.shap_contribution_stream(
                pd.DataFrame(rng.normal(size=(n, 6)),
                             columns=[f"f{i}" for i in range(6)], index=idx)),
            "local_waterfall": lambda: plots.local_waterfall(pd.DataFrame({
                "feature": [f"f{i}" for i in range(6)],
                "shap_value": rng.normal(size=6),
                "feature_value": rng.normal(size=6)}), 50.0, 52.3),
            # whatif.scenario / whatif.pdp 가 내는 컬럼과 같아야 한다
            "whatif_compare": lambda: plots.whatif_compare(pd.DataFrame({
                "baseline": a.to_numpy(), "scenario": b.to_numpy(),
                "delta": (b - a).to_numpy()}, index=idx)),
            "pdp_curve": lambda: plots.pdp_curve(pd.DataFrame({
                "f0": np.linspace(0, 10, 25),
                "prediction": np.linspace(1, 5, 25),
                "p10": np.linspace(.8, 4.6, 25),
                "p90": np.linspace(1.2, 5.4, 25)}), "f0"),
            "pdp_curve(ICE)": lambda: plots.pdp_curve(
                pd.DataFrame({"f0": np.linspace(0, 10, 25),
                              "prediction": np.linspace(1, 5, 25),
                              "p10": np.linspace(.8, 4.6, 25),
                              "p90": np.linspace(1.2, 5.4, 25)}), "f0",
                ice=pd.DataFrame({"row": np.repeat(np.arange(5), 25),
                                  "f0": np.tile(np.linspace(0, 10, 25), 5),
                                  "prediction": rng.normal(size=125)})),
            "anomaly_timeline": lambda: plots.anomaly_timeline(
                pd.Series(rng.normal(size=n), index=idx),
                pd.Series(rng.normal(size=n) > 1.5, index=idx)),
            "cluster_timeline": lambda: plots.cluster_timeline(
                pd.Series(rng.integers(0, 3, size=n), index=idx)),
            "scatter_2d": lambda: plots.scatter_2d(pd.DataFrame(
                {"PC1": rng.normal(size=n), "PC2": rng.normal(size=n)}, index=idx),
                "PC1", "PC2"),
            "missing_heat": lambda: plots.missing_heat(pd.DataFrame(
                {f"c{i}": np.where(rng.random(n) < .1, np.nan, rng.normal(size=n))
                 for i in range(5)}, index=idx)),
        }
        broken = []
        for name, fn in calls.items():
            try:
                fig = fn()
                assert fig is not None, f"{name}: None 을 돌려줬습니다"
            except Exception as e:  # noqa: BLE001
                broken.append(f"{name}: {type(e).__name__}: {e}")
        assert not broken, "차트 함수 실행 실패:\n" + "\n".join(broken)
    finally:
        for m in list(sys.modules):
            if m.startswith(("plotly", "core.plots")):
                del sys.modules[m]


# ─────────────────────────────────────────────────────────────
# 8. 추천이 실제로 화면에 도달하는가
#
# 추천을 계산해 놓고 위젯은 딴 값으로 시작하면 **추천을 읽지 않는 사람에게는
# 아무 효과가 없다.** 그런데 그 상태에서도 화면은 멀쩡히 그려지므로 앞의
# 테스트들은 전부 통과한다. 여기서만 잡힌다.
# ─────────────────────────────────────────────────────────────
def _missing_state(kind: str) -> dict:
    """결측 모양을 일부러 만든 상태. kind: short(순단) / long(설비 정지)."""
    full = _pipeline_state()
    df = full["df"].copy()
    col = "flow"
    if kind == "short":
        df.iloc[np.arange(50, 1100, 37), df.columns.get_loc(col)] = np.nan
    else:
        df.iloc[200:600, df.columns.get_loc(col)] = np.nan
    full["df"] = df
    full["quality_profile"] = None
    return full


def test_prep_view_defaults_to_the_recommended_impute():
    """결측 모양이 다르면 위젯 시작값도 달라져야 한다.

    같은 값으로 시작한다면 추천을 계산은 하되 쓰지는 않는다는 뜻이다.
    """
    short = _render("prep_view", _missing_state("short")).defaults.get("결측 대치")
    long_ = _render("prep_view", _missing_state("long")).defaults.get("결측 대치")
    assert short == "ffill", f"짧은 순단인데 {short} 로 시작합니다"
    assert long_ == "median", f"긴 정지인데 {long_} 로 시작합니다"


def test_prep_view_turns_clip_on_when_the_data_has_spikes():
    """극단값이 있으면 체크박스가 켜진 채로 시작해야 한다."""
    clean = _pipeline_state()
    spiky = dict(clean)
    df = clean["df"].copy()
    df.iloc[[10, 700], df.columns.get_loc("temp")] = 1e6
    spiky["df"] = df
    spiky["quality_profile"] = None

    assert _render("prep_view", clean).defaults.get("극단값 처리") is False
    assert _render("prep_view", spiky).defaults.get("극단값 처리") is True


def test_prep_view_shows_the_reason_not_just_the_value():
    """사유가 화면에 안 나오면 그냥 남이 정해 준 값이다 — 검증할 방법이 없다."""
    rec = _render("prep_view", _pipeline_state())
    text = " ".join(str(a) for name, a, _ in rec.calls if name == "markdown")
    assert "추천 근거" in text, "추천 사유 블록이 화면에 없습니다"
    assert text.count("추천 근거") >= 4, "네 항목 모두에 사유가 붙어야 합니다"


def test_features_view_defaults_lags_to_what_the_data_shows():
    """지연이 실제로 있는 데이터에서는 그 지연이 기본값에 들어와야 한다."""
    full = _pipeline_state()
    df = full["df"].copy()
    # flow 를 12행(1시간) 늦게 따라가는 타겟으로 바꾼다
    df["y_output"] = (df["flow"].shift(12) * 1.5).bfill() + 1.0
    full = dict(full, df=df, feat_df=None)

    rec = _render("features_view", full)
    default = rec.defaults.get("lag (스텝)")
    assert default, "lag 기본값이 비었습니다"
    assert 12 in default, f"12행 지연을 찾았는데 기본값에 없습니다: {default}"


def test_physical_limit_removes_options_beyond_the_cap():
    """한계를 걸면 그 너머는 **선택지에서 사라져야** 한다.

    비활성화로 남겨 두면 "왜 이건 못 고르지" 를 다시 설명해야 한다.
    """
    full = _pipeline_state()
    rec = _render("features_view", dict(full, feat_df=None),
                  values={"최대 반응 지연 (분)": 30.0})     # 5분 간격 → 6행
    opts = rec.options.get("lag (스텝)", [])
    assert opts, "lag 선택지가 비었습니다"
    assert max(opts) <= 6, f"한계 6행을 넘는 선택지가 남아 있습니다: {opts}"
    assert max(rec.defaults.get("lag (스텝)", [0])) <= 6


def test_mode_selector_comes_before_the_step_rail():
    """모드는 화면에 뜨는 설정 전체를 가른다 — 단계 목록보다 먼저 나와야 한다.

    예전에는 초기화 버튼 근처 맨 밑에 있어서 "안 보인다" 는 지적을 받았다.
    """
    import importlib

    rec = Recorder()
    _install(rec, SessionState())              # app.state 를 읽으려면 먼저 꽂아야 한다
    try:
        s = _state(**_pipeline_state())
        _install(rec, s)
        try:
            importlib.import_module("app.main")   # main 은 임포트 시점에 그려진다
        except (RerunException, StopException):
            pass
        order = [w for w in rec.widgets if w.startswith("radio:")]
    finally:
        _uninstall()
    assert order, "사이드바 라디오가 하나도 없습니다"
    assert order[0] == "radio:모드", f"모드가 첫 번째가 아닙니다: {order}"
