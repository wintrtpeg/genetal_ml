"""누수 방지 장치가 실제로 동작하는지 확인한다.

이 테스트가 깨지면 결과 전체를 믿을 수 없다.
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import features, preprocess, validation  # noqa: E402


@pytest.fixture
def ts() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=600, freq="5min")
    rng = np.random.default_rng(0)
    x1 = np.cumsum(rng.normal(0, 1, 600)) + 50
    x2 = rng.normal(10, 2, 600)
    y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, 600)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}, index=idx)


# ── 파생변수 ────────────────────────────────────────────────
def test_target_lag_is_blocked(ts):
    cfg = features.FeatureConfig(lags=[1, 2], rolling_windows=[3], allow_target_derived=False)
    feat, prov = features.generate(ts, "y", ["x1", "x2"], cfg)
    assert not [c for c in feat.columns if c.startswith("y__")]
    assert (prov[prov["feature"] != "y"]["origin"] != "y").all()


def test_guard_raises_on_target_derived():
    prov = pd.DataFrame([{"feature": "y__lag1", "origin": "y", "transform": "lag(1)"}])
    with pytest.raises(features.TargetLeakage):
        features.assert_no_target_derived(["y__lag1"], "y", prov)


def test_lag_looks_backward_only(ts):
    cfg = features.FeatureConfig(lags=[3], rolling_windows=[], ewm_spans=[], diffs=[],
                                 time_features=False)
    feat, _ = features.generate(ts, "y", ["x1"], cfg)
    made = feat["x1__lag3"].to_numpy()
    orig = ts["x1"].to_numpy()
    assert np.allclose(made[3:], orig[:-3], equal_nan=True)
    assert np.isnan(made[:3]).all()


def test_rolling_never_uses_future(ts):
    """맨 뒤 값을 바꿔도 앞쪽 rolling 값은 변하지 않아야 한다."""
    cfg = features.FeatureConfig(lags=[], rolling_windows=[6], rolling_stats=["mean"],
                                 ewm_spans=[], diffs=[], time_features=False)
    a, _ = features.generate(ts, "y", ["x1"], cfg)
    tampered = ts.copy()
    tampered.iloc[-50:, tampered.columns.get_loc("x1")] += 1000
    b, _ = features.generate(tampered, "y", ["x1"], cfg)
    assert np.allclose(a["x1__roll6_mean"].iloc[:-50].to_numpy(),
                       b["x1__roll6_mean"].iloc[:-50].to_numpy(), equal_nan=True)


# ── 분할 ────────────────────────────────────────────────────
def test_holdout_is_last_segment(ts):
    tr, te = validation.time_holdout(len(ts), 0.2, gap=0)
    assert tr.max() < te.min()
    assert len(te) == pytest.approx(len(ts) * 0.2, abs=2)


def test_gap_is_respected(ts):
    tr, te = validation.time_holdout(len(ts), 0.2, gap=10)
    assert te.min() - tr.max() >= 10


def test_cv_folds_never_look_ahead(ts):
    cfg = validation.SplitConfig(n_splits=4, gap=5)
    cv = validation.make_cv(cfg)
    audit = validation.audit_splits(ts.index, cv, len(ts))
    assert len(audit) == 4
    for _, r in audit.iterrows():
        assert r["train_end"] < r["valid_start"]


def test_temporal_violation_is_caught(ts):
    with pytest.raises(validation.LeakageError):
        validation.assert_temporal_order(ts.index, np.arange(100, 200), np.arange(0, 50))


def test_overlap_is_caught(ts):
    with pytest.raises(validation.LeakageError):
        validation.assert_temporal_order(ts.index, np.arange(0, 100), np.arange(90, 150))


# ── 전처리 ──────────────────────────────────────────────────
def test_scaler_fits_on_train_only(ts):
    """홀드아웃 값을 극단으로 바꿔도 학습 구간 변환 결과는 그대로여야 한다."""
    from sklearn.pipeline import Pipeline

    X = ts[["x1", "x2"]]
    tr, te = validation.time_holdout(len(X), 0.2)
    pre = preprocess.build_preprocessor(["x1", "x2"], [], preprocess.PreprocessConfig())

    pipe = Pipeline([("prep", pre)]).fit(X.iloc[tr])
    before = pipe.transform(X.iloc[tr])

    tampered = X.copy()
    tampered.iloc[te] = tampered.iloc[te] * 1e6
    pipe2 = Pipeline([("prep", preprocess.build_preprocessor(
        ["x1", "x2"], [], preprocess.PreprocessConfig()))]).fit(tampered.iloc[tr])
    after = pipe2.transform(tampered.iloc[tr])

    assert np.allclose(before, after)


def test_ffill_imputer_does_not_use_future():
    imp = preprocess.ForwardFillImputer()
    X = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
    out = imp.fit(X).transform(X)
    assert out[1, 0] == 1.0 and out[2, 0] == 1.0


# ── 선별 ────────────────────────────────────────────────────
def test_feature_selection_drops_constants(ts):
    X = ts[["x1", "x2"]].copy()
    X["constant"] = 3.0
    keep, rep = features.select_features(X, ts["y"])
    assert "constant" not in keep


def test_feature_selection_drops_duplicates(ts):
    X = ts[["x1", "x2"]].copy()
    X["x1_copy"] = X["x1"]
    keep, _ = features.select_features(X, ts["y"], corr_threshold=0.98)
    assert not ("x1" in keep and "x1_copy" in keep)


# ── 쿼리 검증 ───────────────────────────────────────────────
@pytest.mark.parametrize("sql", [
    "DELETE FROM t",
    "SELECT * FROM t; DROP TABLE t",
    "UPDATE t SET a=1",
    "INSERT INTO t VALUES (1)",
    "",
])
def test_bad_queries_rejected(sql):
    from core.datasource import QueryNotAllowed, validate_select
    with pytest.raises(QueryNotAllowed):
        validate_select(sql)


@pytest.mark.parametrize("sql", [
    "SELECT a, create_dt FROM t WHERE b > 1",
    "WITH c AS (SELECT 1 AS x) SELECT * FROM c",
    "SELECT * FROM t ORDER BY ts;",
    "-- 주석\nSELECT a FROM t",
])
def test_good_queries_pass(sql):
    from core.datasource import validate_select
    assert validate_select(sql)
