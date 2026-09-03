"""P1·P2 회귀 테스트.

  6. Residual 분석      — drift·outlier·자기상관이 실제 신호에 반응하는가
  9. Lag 물리시간       — 분 → 행 환산이 맞고, lookback 에 반영되는가
 10. 재현성 필드        — 저장·로드 왕복과 지문 민감도
 11. Rolling Backtest   — 구간이 시간순이고 겹치지 않는가
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    diagnostics, features, models, persist, preprocess, train, validation,
)


@pytest.fixture
def ts() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=900, freq="5min")
    rng = np.random.default_rng(11)
    x1 = np.cumsum(rng.normal(0, 1, 900)) + 50
    x2 = rng.normal(10, 2, 900)
    y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, 900)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}, index=idx)


# ── 6. Residual 분석 ────────────────────────────────────────
def test_residuals_align_on_common_index(ts):
    a = ts["y"]
    p = ts["y"] * 0.9
    r = diagnostics.residuals(a, p.iloc[:500])
    assert len(r) == 500
    assert r.index.equals(p.iloc[:500].index)


def test_drift_table_detects_widening_error():
    """뒤로 갈수록 오차를 키우면 MAE 배율이 커져야 한다."""
    idx = pd.date_range("2025-01-01", periods=600, freq="5min")
    rng = np.random.default_rng(3)
    scale = np.linspace(1.0, 6.0, 600)          # 오차가 6배로 벌어짐
    res = pd.Series(rng.normal(0, 1, 600) * scale, index=idx)
    tbl = diagnostics.drift_table(res, diagnostics.ResidualConfig(n_segments=6))
    assert len(tbl) == 6
    assert tbl["MAE_배율"].iloc[-1] > tbl["MAE_배율"].iloc[0]
    v = diagnostics.drift_verdict(tbl, threshold=1.5)
    assert v["drift"] is True


def test_drift_verdict_quiet_on_stable_residual():
    idx = pd.date_range("2025-01-01", periods=600, freq="5min")
    res = pd.Series(np.random.default_rng(4).normal(0, 1, 600), index=idx)
    v = diagnostics.drift_verdict(diagnostics.drift_table(res), threshold=1.5)
    assert v["drift"] is False


def test_outliers_use_robust_scale():
    """이상점이 많아도 스스로를 정상으로 만들지 못해야 한다."""
    idx = pd.date_range("2025-01-01", periods=300, freq="5min")
    v = np.random.default_rng(5).normal(0, 1, 300)
    v[[10, 50, 120]] = 40.0                      # 명백한 이상점
    res = pd.Series(v, index=idx)
    out = diagnostics.outliers(res, diagnostics.ResidualConfig(outlier_sigma=3.0))
    assert set(idx[[10, 50, 120]]).issubset(set(out.index))
    assert (out["방향"] == "과소예측").all()


def test_autocorrelation_flags_structured_residual():
    """AR(1) 잔차는 lag1 자기상관이 크게 남아야 한다."""
    n = 800
    rng = np.random.default_rng(6)
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = 0.8 * v[i - 1] + rng.normal(0, 0.3)
    res = pd.Series(v, index=pd.date_range("2025-01-01", periods=n, freq="5min"))
    acf = diagnostics.autocorrelation(res, diagnostics.ResidualConfig(max_lag=5))
    assert acf["acf"].iloc[0] > 0.6

    white = pd.Series(rng.normal(0, 1, n), index=res.index)
    acf_w = diagnostics.autocorrelation(white, diagnostics.ResidualConfig(max_lag=5))
    assert abs(acf_w["acf"].iloc[0]) < 0.15


def test_summary_reports_bias_direction():
    idx = pd.date_range("2025-01-01", periods=400, freq="5min")
    res = pd.Series(np.random.default_rng(7).normal(3.0, 0.5, 400), index=idx)
    s = diagnostics.summary(res)
    assert s["mean"] > 2.5
    assert s["bias_ratio"] > 0.8          # 거의 전부 한쪽으로 쏠림


# ── 9. Lag 물리시간 ─────────────────────────────────────────
def test_step_minutes_reads_sampling_interval(ts):
    assert features.step_minutes(ts.index) == pytest.approx(5.0)


def test_minutes_to_rows_converts_correctly():
    assert features.minutes_to_rows([60], 5.0) == [12]          # 5min × 12 = 60min
    assert features.minutes_to_rows([5, 15, 60], 5.0) == [1, 3, 12]
    assert features.minutes_to_rows([60], 15.0) == [4]
    assert features.minutes_to_rows([2], 5.0) == []             # 1행 미만은 버림
    assert features.minutes_to_rows([60], None) == []           # 간격 모르면 환산 안 함


def test_resolve_config_merges_minute_spec(ts):
    cfg = features.FeatureConfig(lags=[1], rolling_windows=[6],
                                 lag_minutes=[60], rolling_minutes=[120])
    out = features.resolve_config(cfg, ts.index)
    assert 12 in out.lags                       # 60분 → 12행
    assert 1 in out.lags                        # 기존 행 지정도 유지
    assert 24 in out.rolling_windows            # 120분 → 24행
    assert cfg.lags == [1], "원본 config 가 변경됐습니다"


def test_resolve_config_is_idempotent(ts):
    cfg = features.FeatureConfig(lags=[1], lag_minutes=[60])
    once = features.resolve_config(cfg, ts.index)
    twice = features.resolve_config(once, ts.index)
    assert once.lags == twice.lags


def test_minute_lag_reaches_lookback_and_gap(ts):
    """분 지정을 빠뜨리면 gap 점검이 무력해진다. lookback 에 반영되는지 확인."""
    cfg = features.FeatureConfig(lags=[], rolling_windows=[], ewm_spans=[], diffs=[],
                                 lag_minutes=[60], time_features=False)
    assert features.warmup_rows(cfg) == 0                     # index 없으면 못 봄
    assert features.warmup_rows(cfg, ts.index) == 12          # index 주면 반영


def test_generate_creates_minute_based_lag(ts):
    cfg = features.FeatureConfig(lags=[], rolling_windows=[], ewm_spans=[], diffs=[],
                                 lag_minutes=[60], time_features=False)
    feat, prov = features.generate(ts, "y", ["x1"], cfg)
    assert "x1__lag12" in feat.columns
    made = feat["x1__lag12"].to_numpy()
    assert np.allclose(made[12:], ts["x1"].to_numpy()[:-12], equal_nan=True)


def test_describe_time_spec(ts):
    cfg = features.FeatureConfig(lag_minutes=[60, 30], rolling_minutes=[])
    spec = features.describe_time_spec(cfg, ts.index)
    assert len(spec) == 2
    assert set(spec["환산(행)"]) == {12, 6}


# ── 10. 재현성 필드 ─────────────────────────────────────────
def test_fingerprint_changes_when_a_value_changes(ts):
    a = persist.dataset_fingerprint(ts)
    b = ts.copy()
    b.iloc[3, 0] += 1e-6
    assert a["sha256"] != persist.dataset_fingerprint(b)["sha256"]


def test_fingerprint_stable_for_same_frame(ts):
    assert (persist.dataset_fingerprint(ts)["sha256"]
            == persist.dataset_fingerprint(ts.copy())["sha256"])


def test_manifest_records_split_bounds_and_packages(ts):
    sp = validation.three_way_split(len(ts), 0.2, 0.15, gap=0)
    man = persist.build_manifest(
        run_id="test_run", target="y", df=ts, split=sp, index=ts.index,
        seed=42, champion="Ridge")
    assert man["run_id"] == "test_run"
    assert man["seed"] == 42
    assert set(man["split_bounds"]) == {"train", "valid", "unseen"}
    assert man["split_bounds"]["unseen"]["rows"] == len(sp.unseen)
    assert "pandas" in man["packages"] and "python" in man["packages"]
    assert "sha256" in man["dataset"]


def test_manifest_roundtrip_survives_save_and_load(ts):
    sp = validation.three_way_split(len(ts), 0.2, 0.15, gap=0)
    man = persist.build_manifest(run_id="rt", target="y", df=ts, split=sp,
                                 index=ts.index, seed=7, champion="Ridge")
    tmp = Path(tempfile.mkdtemp())
    try:
        persist.save_run(tmp, manifest=man)
        loaded = persist.load_run(tmp)["manifest"]
        assert loaded["dataset"]["sha256"] == man["dataset"]["sha256"]
        assert loaded["seed"] == 7
        assert loaded["split_bounds"] == man["split_bounds"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_records_exclusion_reasons(ts):
    X = ts[["x1", "x2"]].copy()
    X["constant"] = 1.0
    _, rep = features.select_features(X, ts["y"])
    man = persist.build_manifest(run_id="r", selection_report=rep)
    excluded = man["features_excluded"]
    assert any(e["feature"] == "constant" and e["reason"] for e in excluded)


def test_compare_manifests_flags_difference(ts):
    sp = validation.three_way_split(len(ts), 0.2, 0.15, gap=0)
    a = persist.build_manifest(run_id="a", target="y", df=ts, split=sp,
                               index=ts.index, seed=1)
    other = ts.copy()
    other.iloc[0, 0] += 5.0
    b = persist.build_manifest(run_id="b", target="y", df=other, split=sp,
                               index=ts.index, seed=1)
    cmp = persist.compare_manifests(a, b)
    row = cmp[cmp["항목"] == "dataset sha256"]
    assert row["일치"].iloc[0] == "✕"


# ── 11. Rolling Backtest ────────────────────────────────────
def test_rolling_windows_are_ordered_and_disjoint():
    ws = validation.rolling_windows(1000, n_folds=5, gap=10)
    assert len(ws) == 5
    prev_end = -1
    for tr, te in ws:
        assert tr.max() < te.min(), "학습이 평가 구간을 침범했습니다"
        assert te.min() - tr.max() > 10, "gap 이 지켜지지 않았습니다"
        assert np.intersect1d(tr, te).size == 0
        assert te.min() > prev_end, "평가 구간이 시간순이 아닙니다"
        prev_end = te.min()


def test_rolling_windows_expanding_vs_sliding():
    exp = validation.rolling_windows(1000, n_folds=4, expanding=True)
    sli = validation.rolling_windows(1000, n_folds=4, expanding=False, min_train=200)
    assert all(tr[0] == 0 for tr, _ in exp), "expanding 은 항상 0 에서 시작해야 합니다"
    assert len({len(tr) for tr, _ in exp}) > 1, "expanding 은 학습 구간이 커져야 합니다"
    assert all(len(tr) <= 200 for tr, _ in sli), "sliding 은 학습 길이가 고정이어야 합니다"


def test_rolling_backtest_produces_per_segment_scores(ts):
    X, y = ts[["x1", "x2"]], ts["y"]
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)

    table, stitched = train.rolling_backtest(X, y, pre, zoo["Ridge"], cfg, n_folds=4)
    assert len(table) == 4
    assert (table["status"] == "ok").all()
    assert table["R2"].notna().all()
    assert list(table["구간"]) == [1, 2, 3, 4]
    assert stitched.index.is_monotonic_increasing

    summ = train.backtest_summary(table, "R2")
    assert summ["구간수"] == 4
    assert summ["최저"] <= summ["평균"] <= summ["최고"]


def test_random_split_is_disjoint_and_covers_all():
    tr, te = validation.random_split(1000, 0.2, seed=42)
    assert len(te) == 200 and len(tr) == 800
    assert np.intersect1d(tr, te).size == 0
    assert np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(1000))


def test_random_split_is_seed_stable():
    a, _ = validation.random_split(500, 0.2, seed=7)
    b, _ = validation.random_split(500, 0.2, seed=7)
    c, _ = validation.random_split(500, 0.2, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_split_actually_interleaves():
    """무작위 분할은 검증 행이 학습 구간 사이사이에 섞인다 — 그게 문제의 근원이다."""
    tr, te = validation.random_split(1000, 0.2, seed=1)
    assert te.min() < tr.max(), "무작위인데 뒤쪽에만 몰렸습니다"
    assert tr.min() < te.max()


def test_random_vs_time_gap_appears_on_autocorrelated_data():
    """자기상관이 강한 데이터에서 Random 이 Time 보다 좋게 나와야 한다."""
    n = 900
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(0, 1, n))
    X = pd.DataFrame({"x1": walk, "x2": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series(walk + rng.normal(0, 0.4, n), index=idx, name="y")

    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)

    out = train.random_vs_time(X, y, pre, zoo, ["DecisionTree"], cfg)
    assert out["purpose"] == "diagnostic"
    assert "격차" in out["table"].columns
    assert out["verdict"]["lag1_acf"] > 0.5
    assert any(c["원인 후보"] == "자기상관" for c in out["verdict"]["causes"])


def test_diagnostic_result_cannot_enter_evaluation_path():
    """G-3 — 진단 결과가 모델 선택으로 새면 예외로 막아야 한다."""
    fake = {"purpose": "diagnostic", "table": pd.DataFrame()}
    with pytest.raises(ValueError):
        train.assert_not_diagnostic(fake)
    train.assert_not_diagnostic({"purpose": "evaluation"})   # 통과해야 함


def test_split_gap_causes_quiet_when_no_gap(ts):
    v = diagnostics.split_gap_causes(0.01, ts["y"], ts[["x1", "x2"]], threshold=0.15)
    assert v["significant"] is False


def test_distribution_drift_detects_shift():
    idx = pd.date_range("2025-01-01", periods=600, freq="5min")
    rng = np.random.default_rng(9)
    y = pd.Series(np.r_[rng.normal(0, 1, 300), rng.normal(6, 1, 300)], index=idx)
    X = pd.DataFrame({"a": rng.normal(0, 1, 600)}, index=idx)
    d = diagnostics.distribution_drift(X, y)
    assert d["y_shift_sd"] > 1.0


def test_rolling_backtest_never_trains_on_future(ts):
    """각 구간의 학습 끝 시각이 평가 시작보다 앞서야 한다."""
    X, y = ts[["x1", "x2"]], ts["y"]
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3, gap=5),
                            fold_selection=False)
    table, _ = train.rolling_backtest(X, y, pre, zoo["Ridge"], cfg, n_folds=3)
    for _, r in table.iterrows():
        train_end = pd.Timestamp(r["학습"].split(" ~ ")[1])
        assert train_end <= pd.Timestamp(r["평가시작"]).normalize() + pd.Timedelta(days=1)
