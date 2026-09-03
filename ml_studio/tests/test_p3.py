"""P3 회귀 테스트.

 13. Diagnostics 탭        — core 함수가 UI 없이 다 돌아가는가
 14. 설정 YAML 외부화      — 왕복 동일성, 오타 감지, 하위호환
 15. Champion-Challenger   — 미달이면 유지하는가
 16. Auto/Guided/Expert    — 모드가 누수 방지 장치를 끄지 않는가

그리고 구조 보존 — core 가 streamlit 을 import 하지 않는가 (Dataiku 이식 전제).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, features, persist, preprocess, train, validation  # noqa: E402


# ── 14. 설정 외부화 ─────────────────────────────────────────
def test_config_roundtrip_is_lossless():
    c = config.StudioConfig()
    c.split.unseen_ratio = 0.18
    c.split.gap = 24
    c.features.lag_minutes = [60, 120]
    c.features.rolling_stats = ["mean", "std", "max"]
    c.train.fold_selection = False
    c.train.ensemble_threshold = 0.07
    c.preprocess.clip_outliers = True
    c.meta = {"target": "y", "mode": "Expert"}

    back, warns = config.loads(config.dumps(c))
    assert warns == []
    assert back == c
    assert back.split.unseen_ratio == 0.18
    assert back.features.lag_minutes == [60, 120]
    assert back.train.ensemble_threshold == 0.07
    assert back.preprocess.clip_outliers is True


def test_config_keeps_train_split_in_sync():
    """TrainConfig.split 이 split 섹션과 어긋나면 학습이 다른 분할을 쓴다."""
    c = config.StudioConfig()
    c.split.unseen_ratio = 0.25
    c.split.n_splits = 7
    back, _ = config.loads(config.dumps(c))
    assert back.train.split.unseen_ratio == 0.25
    assert back.train.split.n_splits == 7


def test_config_warns_on_unknown_key():
    """오타가 조용히 무시되면 '설정을 바꿨는데 결과가 같다'가 된다."""
    _, warns = config.from_dict({"split": {"unseen_ratioo": 0.3}})
    assert any("unseen_ratioo" in w for w in warns)

    _, warns2 = config.from_dict({"nonsense_section": {"a": 1}})
    assert any("nonsense_section" in w for w in warns2)


def test_config_warns_on_schema_version_mismatch():
    _, warns = config.from_dict({"schema_version": 999})
    assert any("schema_version" in w for w in warns)


def test_config_missing_sections_fall_back_to_defaults():
    cfg, warns = config.from_dict({"split": {"gap": 5}})
    assert cfg.split.gap == 5
    assert cfg.features.lags == features.FeatureConfig().lags
    assert warns == []


def test_config_json_path_works_without_yaml():
    c = config.StudioConfig()
    c.split.gap = 11
    text = config.dumps(c, prefer_yaml=False)
    assert text.lstrip().startswith("{")
    back, warns = config.loads(text)
    assert back.split.gap == 11 and warns == []


def test_config_diff_lists_only_changes():
    a = config.StudioConfig()
    b = config.StudioConfig()
    b.split.unseen_ratio = 0.3
    b.train.seed = 99
    d = config.diff(a, b)
    assert set(d["항목"]) == {"unseen_ratio", "seed"}


def test_config_dump_load_file(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    c = config.StudioConfig()
    c.features.lags = [1, 5, 9]
    p = config.dump(c, tmp / "cfg.yaml")
    assert p.exists()
    back, warns = config.load(p)
    assert back.features.lags == [1, 5, 9] and warns == []


# ── 15. Champion-Challenger ─────────────────────────────────
def test_challenger_kept_out_on_marginal_gain():
    """SPEC §18 — 소수점 뒤 개선으로 운영 모델을 갈아타지 않는다."""
    v = persist.challenge("old", {"unseen_R2": 0.900},
                          "new", {"unseen_R2": 0.905},
                          metric="R2", threshold=0.02)
    assert v["decision"] == "유지"
    assert v["champion"] == "old"
    assert 0 < v["개선율"] < 0.02


def test_challenger_swaps_on_real_gain():
    v = persist.challenge("old", {"unseen_R2": 0.80},
                          "new", {"unseen_R2": 0.90},
                          metric="R2", threshold=0.02)
    assert v["decision"] == "교체"
    assert v["개선율"] > 0.02


def test_challenger_handles_lower_is_better_metric():
    """RMSE 는 낮을수록 좋다. 부호를 뒤집어 판정해야 한다."""
    v = persist.challenge("old", {"unseen_RMSE": 10.0},
                          "new", {"unseen_RMSE": 8.0},
                          metric="RMSE", threshold=0.05)
    assert v["decision"] == "교체"

    worse = persist.challenge("old", {"unseen_RMSE": 10.0},
                              "new", {"unseen_RMSE": 12.0},
                              metric="RMSE", threshold=0.05)
    assert worse["decision"] == "유지"
    assert worse["개선율"] < 0


def test_challenger_refuses_without_both_scores():
    v = persist.challenge("old", {}, "new", {"unseen_R2": 0.9}, metric="R2")
    assert v["decision"] == "판정 불가"


# ── 16. 모드 ────────────────────────────────────────────────
def _state_constant(name: str):
    """app/state.py 는 streamlit 을 import 하므로 소스에서 상수만 읽어 온다.

    이 테스트가 streamlit 설치 여부에 좌우되면, 폐쇄망에서 누수 검증만 돌릴 때
    조용히 건너뛰게 된다. 그래서 import 대신 파싱한다.
    """
    tree = ast.parse((ROOT / "app" / "state.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"app/state.py 에 {name} 이 없습니다")


def test_mode_levels_are_ordered():
    modes = _state_constant("MODES")
    level = _state_constant("MODE_LEVEL")
    help_ = _state_constant("MODE_HELP")
    assert level["Auto"] < level["Guided"] < level["Expert"]
    assert set(modes) == set(level)
    assert set(help_) == set(modes)


def test_mode_defaults_keep_leakage_guards_on():
    """모드는 노출만 줄인다. 누수 방지 장치를 끄면 안 된다.

    Expert 가 아닐 때 train_view 가 쓰는 기본값이 안전한 쪽인지 확인한다.
    """
    cfg = train.TrainConfig()
    assert cfg.fold_selection is True, "기본값에서 폴드 내부 선별이 꺼져 있습니다"
    assert cfg.ensemble_threshold > 0, "앙상블 자동채택 임계가 0 이면 무조건 채택됩니다"
    assert validation.SplitConfig().gap >= 0


# ── 13. Diagnostics — core 만으로 동작 ──────────────────────
def test_diagnostics_pipeline_runs_without_ui():
    """진단 화면이 쓰는 core 함수가 전부 UI 없이 돈다 (Dataiku 이식 전제)."""
    from core import diagnostics

    n = 600
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(13)
    X = pd.DataFrame({"a": np.cumsum(rng.normal(0, 1, n)),
                      "b": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series(X["a"] * 0.7 + rng.normal(0, 0.5, n), index=idx, name="y")

    sp = validation.three_way_split(n, 0.2, 0.15, gap=0)
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = __import__("core.models", fromlist=["*"]).get_model_zoo(
        "regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3),
                            fold_selection=True, selection_top_k=1)

    board, detail = train.train_all(X, y, sp.train, sp.valid, pre, zoo, ["Ridge"], cfg)
    assert (board["status"] == "ok").all()

    # 누수 점검
    chk = validation.leakage_checklist(
        X.index, sp.train, sp.valid, list(X.columns), "y", None, 0, 0,
        selection_idx=sp.train, unseen_idx=sp.unseen)
    assert (chk["결과"] == "통과").all()

    # 폴드 안정성
    assert "fold_jaccard" in detail["Ridge"]
    assert detail["Ridge"]["_fold_jaccard_table"] is not None

    # 잔차
    pred = train.predict_range(detail["Ridge"]["_pipeline"], X)
    r = diagnostics.residuals(y, pred)
    assert len(diagnostics.rolling_stats(r)) == len(r)
    assert not diagnostics.drift_table(r).empty
    assert diagnostics.summary(r)["n"] == len(r)

    # backtest
    bt, _ = train.rolling_backtest(X, y, pre, zoo["Ridge"], cfg, n_folds=3)
    assert len(bt) == 3

    # Random vs Time
    diag = train.random_vs_time(X, y, pre, zoo, ["Ridge"], cfg)
    assert diag["purpose"] == "diagnostic"


# ── G-6 하이퍼파라미터 탐색 (nested CV) ────────────────────
def _tiny():
    n = 500
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(21)
    X = pd.DataFrame({"a": np.cumsum(rng.normal(0, 1, n)),
                      "b": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series(X["a"] * 0.7 + rng.normal(0, 0.5, n), index=idx, name="y")
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    from core.models import get_model_zoo
    return X, y, pre, get_model_zoo("regression", include_heavy=False)


def test_search_uses_timeseries_split_not_kfold():
    """KFold 로 파라미터를 고르면 미래 구간을 보고 고른 것이 된다."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from core import tuning

    pipe = Pipeline([("est", Ridge())])
    s = tuning.make_search(pipe, "Ridge", tuning.TuneConfig(enabled=True, n_iter=3))
    assert isinstance(s.cv, TimeSeriesSplit), "탐색 cv 가 TimeSeriesSplit 이 아닙니다"
    assert s.refit is True


def test_search_returns_none_for_unknown_model():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from core import tuning
    assert tuning.make_search(Pipeline([("est", Ridge())]), "없는모델",
                              tuning.TuneConfig(enabled=True)) is None


def test_tune_config_targets_only_selected_models():
    from core import tuning
    cfg = tuning.TuneConfig(enabled=True, models=["Ridge"])
    assert cfg.applies_to("Ridge") is True
    assert cfg.applies_to("RandomForest") is False
    assert tuning.TuneConfig(enabled=True).applies_to("RandomForest") is True
    assert tuning.TuneConfig(enabled=False).applies_to("Ridge") is False


def test_nested_tuning_runs_and_records_per_fold_params():
    from core import tuning
    X, y, pre, zoo = _tiny()
    sp = validation.three_way_split(len(X), 0.2, 0.15, gap=0)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3),
                            fold_selection=False,
                            tune=tuning.TuneConfig(enabled=True, n_iter=4, inner_splits=2))
    board, detail = train.train_all(X, y, sp.train, sp.valid, pre, zoo,
                                    ["DecisionTree"], cfg)
    rec = detail["DecisionTree"]
    assert rec["status"] == "ok"
    assert rec.get("tuned") is True
    assert len(rec["_fold_params"]) == 3, "폴드마다 파라미터를 고르지 않았습니다"
    assert "best_params" in rec
    assert not rec["_param_stability"].empty


def test_tuning_never_touches_validation_or_unseen():
    """탐색이 검증·Unseen 구간을 보지 않는지 — 그 구간을 오염시켜 확인.

    바깥 검증 구간을 극단으로 바꿔도 학습 구간에서 고른 파라미터는 그대로여야 한다.
    """
    from core import tuning
    X, y, pre, zoo = _tiny()
    sp = validation.three_way_split(len(X), 0.2, 0.15, gap=0)
    tc = tuning.TuneConfig(enabled=True, n_iter=5, inner_splits=2, seed=1)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3),
                            fold_selection=False, tune=tc)

    _, d1 = train.train_all(X, y, sp.train, sp.valid, pre, zoo, ["DecisionTree"], cfg)

    y2 = y.copy()
    y2.iloc[sp.valid] += 1e5          # 검증 구간만 오염
    y2.iloc[sp.unseen] += 1e5         # unseen 도 오염
    _, d2 = train.train_all(X, y2, sp.train, sp.valid, pre, zoo, ["DecisionTree"], cfg)

    assert d1["DecisionTree"]["best_params"] == d2["DecisionTree"]["best_params"], \
        "검증·Unseen 오염이 파라미터 선택을 바꿨습니다 — 탐색이 미래를 봅니다"
    assert d1["DecisionTree"]["_fold_params"] == d2["DecisionTree"]["_fold_params"]


def test_tuning_off_by_default():
    """G-6 — 기본은 꺼져 있어야 한다. 켜지 않은 사람이 비용을 물면 안 된다."""
    assert train.TrainConfig().tune is None
    from core import tuning
    assert tuning.TuneConfig().enabled is False


def test_param_grids_only_target_estimator_step():
    """그리드 키가 전처리나 선별 단계를 건드리면 파이프라인이 깨진다."""
    from core import tuning
    for name in ("Ridge", "RandomForest", "HistGradientBoosting", "MLP"):
        grid = tuning.param_grid(name)
        assert grid
        assert all(k.startswith("est__") for k in grid), f"{name} 그리드 키가 잘못됐습니다"


def test_search_n_iter_capped_to_grid_size():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from core import tuning
    s = tuning.make_search(Pipeline([("est", Ridge())]), "Ridge",
                           tuning.TuneConfig(enabled=True, n_iter=999))
    assert s.n_iter == 5, "조합 수보다 많이 뽑으려 합니다"


# ── 구조 보존 ───────────────────────────────────────────────
def test_core_never_imports_streamlit():
    """HANDOFF 6절 원칙 1. 문자열이 아니라 실제 import 문을 본다."""
    offenders = []
    for path in sorted((ROOT / "core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            if "streamlit" in names:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"core 가 streamlit 을 import 합니다: {offenders}"


def test_core_never_imports_app():
    offenders = []
    for path in sorted((ROOT / "core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]]
            if "app" in mods:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"core 가 app 을 import 합니다: {offenders}"


def test_every_step_has_a_view():
    """레일에 있는 단계가 전부 렌더러를 갖고 있는지."""
    main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    steps = _state_constant("STEPS")
    for _, key in steps:
        assert f'"{key}":' in main_src, f"{key} 단계의 렌더러가 없습니다"
        assert (ROOT / "app" / "views" / f"{key}_view.py").exists(), \
            f"{key}_view.py 가 없습니다"
