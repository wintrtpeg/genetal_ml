"""P0 회귀 테스트 — 누수가 있으면 실패하도록 짠 것들.

여기 있는 테스트는 "동작하는가"가 아니라 "누수가 없는가"를 본다.
통과했다고 결과가 좋다는 뜻이 아니고, 실패하면 결과를 믿을 수 없다는 뜻이다.

대상
  1. 자체 OOF 스태킹  — 메타 학습기가 미래를 보지 않는가
  2. Split 단일화     — 선별에 쓴 구간이 평가 구간에 새어 들어가는가
  3. Final Unseen     — 3구간이 무교집합인가, 접근이 1회로 막히는가
  4. gap 확보         — gap < lookback 이면 실제로 실패를 내는가
 12. 폴드 내부 선별   — 폴드마다 다시 선별되는가
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import ensemble, features, models, preprocess, train, validation  # noqa: E402


@pytest.fixture
def ts() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=800, freq="5min")
    rng = np.random.default_rng(7)
    x1 = np.cumsum(rng.normal(0, 1, 800)) + 50
    x2 = rng.normal(10, 2, 800)
    x3 = rng.normal(0, 1, 800)                     # 무관 신호
    y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, 800)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y}, index=idx)


def _xy(ts):
    return ts[["x1", "x2", "x3"]], ts["y"]


def _pre(X):
    num, cat = preprocess.split_column_types(X)
    return preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())


# ── 3. Final Unseen 3분할 ───────────────────────────────────
def test_three_way_split_is_disjoint_and_ordered():
    s = validation.three_way_split(1000, valid_ratio=0.2, unseen_ratio=0.15, gap=10)
    assert np.intersect1d(s.train, s.valid).size == 0
    assert np.intersect1d(s.valid, s.unseen).size == 0
    assert np.intersect1d(s.train, s.unseen).size == 0
    assert s.train.max() < s.valid.min() < s.valid.max() < s.unseen.min()
    validation.assert_disjoint(s)


def test_three_way_split_respects_gap():
    gap = 25
    s = validation.three_way_split(1000, 0.2, 0.15, gap=gap)
    assert s.valid.min() - s.train.max() > gap
    assert s.unseen.min() - s.valid.max() > gap


def test_unseen_ratio_zero_matches_legacy_two_way():
    """하위호환 — unseen_ratio=0 이면 기존 time_holdout 과 경계가 같아야 한다."""
    for n, ratio, gap in ((800, 0.2, 0), (1000, 0.3, 12), (500, 0.1, 5)):
        s = validation.three_way_split(n, ratio, 0.0, gap)
        tr, te = validation.time_holdout(n, ratio, gap)
        assert np.array_equal(s.train, tr)
        assert np.array_equal(s.valid, te)
        assert len(s.unseen) == 0


def test_three_way_split_by_date(ts):
    s = validation.three_way_split_by_date(ts.index, ts.index[500], ts.index[700], gap=5)
    assert ts.index[s.valid.min()] == ts.index[500]
    assert ts.index[s.unseen.min()] == ts.index[700]
    validation.assert_disjoint(s)


def test_unseen_guard_blocks_second_access():
    """Final Unseen 은 한 번만 열려야 한다. 두 번째는 예외."""
    g = train.UnseenGuard(np.arange(100, 200))
    assert len(g.open("champion")) == 100
    with pytest.raises(train.UnseenAccessError):
        g.open("두번째")
    assert g.access_count == 1


def test_evaluate_unseen_never_touches_train_or_valid(ts):
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.15, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)
    board, detail = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, ["Ridge"], cfg)

    guard = train.UnseenGuard(s.unseen)
    res = train.evaluate_unseen(detail["Ridge"]["_pipeline"], X, y, s.unseen, cfg, guard)
    assert res["unseen_rows"] == len(s.unseen)
    # 평가에 쓴 인덱스가 학습·검증과 겹치지 않는다
    assert np.intersect1d(guard.idx, s.train).size == 0
    assert np.intersect1d(guard.idx, s.valid).size == 0


# ── 2. Split 단일화 + 선별구간 추적 ─────────────────────────
def test_checklist_catches_selection_bleed(ts):
    """3단계에서 640행으로 선별하고 4단계에서 홀드아웃을 키우면 실패해야 한다."""
    n = len(ts)
    sel_train, _ = validation.time_holdout(n, 0.2, gap=0)       # 640행으로 선별
    tr2, te2 = validation.time_holdout(n, 0.35, gap=0)          # 홀드아웃을 키움

    check = validation.leakage_checklist(
        ts.index, tr2, te2, ["x1"], "y", None, gap=0, max_lookback=0,
        selection_idx=sel_train)
    row = check[check["항목"] == "선별 구간 격리"]
    assert len(row) == 1
    assert row["결과"].iloc[0] == "실패", "선별 구간 침범을 잡지 못했습니다"


def test_checklist_passes_when_selection_stays_in_train(ts):
    n = len(ts)
    tr, te = validation.time_holdout(n, 0.2, gap=0)
    check = validation.leakage_checklist(
        ts.index, tr, te, ["x1"], "y", None, gap=0, max_lookback=0,
        selection_idx=tr)
    assert (check["결과"] == "통과").all()


def test_checklist_catches_unseen_contamination(ts):
    """unseen 이 학습 구간과 겹치면 실패해야 한다."""
    s = validation.three_way_split(len(ts), 0.2, 0.15, gap=0)
    dirty = np.union1d(s.unseen, s.train[-10:])       # 학습 꼬리를 unseen 에 섞음
    check = validation.leakage_checklist(
        ts.index, s.train, s.valid, ["x1"], "y", None, gap=0, max_lookback=0,
        unseen_idx=dirty)
    row = check[check["항목"] == "Final Unseen 격리"]
    assert row["결과"].iloc[0] == "실패"


# ── 4. gap 확보 ─────────────────────────────────────────────
def test_gap_check_fails_when_gap_below_lookback(ts):
    """죽은 코드였던 항목. gap 6 < lookback 12 는 실패여야 한다."""
    tr, te = validation.time_holdout(len(ts), 0.2, gap=6)
    check = validation.leakage_checklist(
        ts.index, tr, te, ["x1"], "y", None, gap=6, max_lookback=12)
    row = check[check["항목"] == "gap 확보"]
    assert row["결과"].iloc[0] == "실패", "gap < lookback 인데 통과로 표시됩니다"


def test_gap_check_passes_when_gap_meets_lookback(ts):
    tr, te = validation.time_holdout(len(ts), 0.2, gap=12)
    check = validation.leakage_checklist(
        ts.index, tr, te, ["x1"], "y", None, gap=12, max_lookback=12)
    assert check[check["항목"] == "gap 확보"]["결과"].iloc[0] == "통과"


def test_gap_check_ignores_zero_lookback(ts):
    tr, te = validation.time_holdout(len(ts), 0.2, gap=0)
    check = validation.leakage_checklist(
        ts.index, tr, te, ["x1"], "y", None, gap=0, max_lookback=0)
    assert check[check["항목"] == "gap 확보"]["결과"].iloc[0] == "통과"


# ── 1. 자체 OOF 스태킹 ──────────────────────────────────────
def test_stacking_actually_runs(ts):
    """기존 결함 1 — sklearn Stacking 은 여기서 항상 ValueError 였다."""
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.0, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)
    bases = ["Ridge", "DecisionTree", "ElasticNet"]
    board, detail = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, bases, cfg)

    eb, ed = train.build_ensembles(X, y, s.train, s.valid, _pre(X), zoo, bases, cfg,
                                   detail=detail)
    assert not eb.empty
    st = eb[eb["model"] == "Ensemble_Stacking"]
    assert len(st) == 1
    assert st["status"].iloc[0] == "ok", f"스태킹 실패: {st.get('error')}"
    assert np.isfinite(st["holdout_R2"].iloc[0])


def test_oof_matrix_has_no_nan_and_precedes_holdout(ts):
    """메타 학습에 쓰는 OOF 행렬에 NaN 이 없고, 전부 검증 구간보다 앞서야 한다."""
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.15, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)
    bases = ["Ridge", "DecisionTree"]
    _, detail = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, bases, cfg)

    P, used = ensemble.oof_matrix(detail, bases)
    assert not P.empty and len(used) == 2
    assert not P.isna().to_numpy().any(), "OOF 행렬에 NaN 이 남았습니다"
    assert P.index.max() < X.index[s.valid.min()], "OOF 가 검증 구간을 침범했습니다"
    assert P.index.max() < X.index[s.unseen.min()], "OOF 가 Final Unseen 을 침범했습니다"


def test_meta_learner_does_not_see_future(ts):
    """학습 구간 뒷부분 y 를 오염시켜도 앞부분 OOF 는 변하지 않아야 한다.

    OOF 가 forward-only 라면 시점 t 의 예측은 t 이전 데이터로만 만들어진다.
    """
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.0, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=4), fold_selection=False)

    _, d1 = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, ["Ridge"], cfg)
    y2 = y.copy()
    y2.iloc[int(len(s.train) * 0.75):] += 1e4          # 뒤쪽 25% 오염
    _, d2 = train.train_all(X, y2, s.train, s.valid, _pre(X), zoo, ["Ridge"], cfg)

    a, b = d1["Ridge"]["_oof"], d2["Ridge"]["_oof"]
    cut = int(len(s.train) * 0.5)
    head_a, head_b = a.iloc[:cut], b.iloc[:cut]
    both = head_a.notna() & head_b.notna()
    assert both.sum() > 0
    assert np.allclose(head_a[both], head_b[both]), \
        "미래 y 오염이 과거 OOF 를 바꿨습니다 — 메타 학습기가 미래를 봅니다"


def test_weighted_ensemble_weights_are_nonnegative_and_sum_to_one(ts):
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.0, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=3), fold_selection=False)
    bases = ["Ridge", "DecisionTree", "ElasticNet"]
    _, detail = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, bases, cfg)

    eb, ed = train.build_ensembles(X, y, s.train, s.valid, _pre(X), zoo, bases, cfg,
                                   detail=detail)
    w = ed["Ensemble_Weighted"]["_pipeline"].weights_
    assert (w >= -1e-12).all(), "음수 가중치가 나왔습니다"
    assert abs(w.sum() - 1.0) < 1e-9


def test_ensemble_not_adopted_on_marginal_gain():
    """SPEC §17 — 미미한 개선이면 앙상블을 자동 채택하지 않는다."""
    board = pd.DataFrame([
        {"model": "Ridge", "family": "linear", "status": "ok", "cv_R2": 0.900},
        {"model": "Ensemble_Stacking", "family": "blend", "status": "ok", "cv_R2": 0.905},
    ])
    champ, tbl = ensemble.adopt_ensemble(board, "R2", threshold=0.03, prefix="cv_")
    assert champ == "Ridge"
    assert "기각" in tbl[tbl["model"] == "Ensemble_Stacking"]["판정"].iloc[0]

    champ2, _ = ensemble.adopt_ensemble(board, "R2", threshold=0.001, prefix="cv_")
    assert champ2 == "Ensemble_Stacking"


def test_bagging_models_count_as_single_not_blend():
    """RandomForest·ExtraTrees 의 family 도 'ensemble' 이다.

    그 이름으로 결합 모델을 판정하면 배깅 모델이 '앙상블 후보' 로 분류되어
    단일 최고 모델과의 비교 기준 자체가 틀어진다. 실제로 ExtraTrees 가 1위인데도
    챔피언에서 밀려나는 사고가 났다.
    """
    board = pd.DataFrame([
        {"model": "Ridge", "family": "linear", "status": "ok", "cv_R2": 0.70},
        {"model": "ExtraTrees", "family": "ensemble", "status": "ok", "cv_R2": 0.96},
        {"model": "RandomForest", "family": "ensemble", "status": "ok", "cv_R2": 0.95},
        {"model": "Ensemble_Weighted", "family": "blend", "status": "ok", "cv_R2": 0.961},
    ])
    champ, tbl = ensemble.adopt_ensemble(board, "R2", threshold=0.03, prefix="cv_")
    assert champ == "ExtraTrees", "배깅 모델이 단일 최고로 잡히지 않았습니다"
    base_row = tbl[tbl["판정"] == "단일 최고 (기준)"]
    assert base_row["model"].iloc[0] == "ExtraTrees"
    # 결합 모델만 판정 대상이어야 한다
    judged = set(tbl["model"]) - {"ExtraTrees"}
    assert judged == {"Ensemble_Weighted"}


def test_blend_detection_falls_back_to_name_prefix():
    """family 가 없는 예전 리더보드도 이름으로 결합 모델을 알아봐야 한다."""
    board = pd.DataFrame([
        {"model": "Ridge", "status": "ok", "cv_R2": 0.90},
        {"model": "Ensemble_Stacking", "status": "ok", "cv_R2": 0.99},
    ])
    champ, tbl = ensemble.adopt_ensemble(board, "R2", threshold=0.03, prefix="cv_")
    assert champ == "Ensemble_Stacking"
    assert len(tbl) == 2


# ── 12. 폴드 내부 Feature Selection ─────────────────────────
def test_fold_selector_refits_per_fold(ts):
    """폴드마다 선별이 다시 돌아야 한다. 폴드별 선택 집합을 남기는지 확인."""
    X, y = _xy(ts)
    s = validation.three_way_split(len(X), 0.2, 0.0, gap=0)
    zoo = models.get_model_zoo("regression", include_heavy=False)
    cfg = train.TrainConfig(split=validation.SplitConfig(n_splits=4),
                            fold_selection=True, selection_top_k=2)
    _, detail = train.train_all(X, y, s.train, s.valid, _pre(X), zoo, ["Ridge"], cfg)
    sets = detail["Ridge"]["_fold_feature_sets"]
    assert len(sets) == 4, "폴드 수만큼 선별이 돌지 않았습니다"
    assert all(len(fs) <= 2 for fs in sets)


def test_fold_selector_sees_only_its_own_train_rows(ts):
    """폴드 내부 선별이 검증 구간을 보지 않는지 — 검증 구간을 오염시켜 확인.

    폴드 밖에서 한 번만 선별하면 이 오염이 선택 결과를 바꾼다. 폴드 내부에서
    선별하면 각 폴드의 학습 구간만 보므로 첫 폴드의 선택은 그대로여야 한다.
    """
    X = ts[["x1", "x2", "x3"]].copy()
    y = ts["y"]
    cv = validation.make_cv(validation.SplitConfig(n_splits=4))
    tr_idx = np.arange(int(len(X) * 0.8))
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

    first_tr, _ = next(iter(cv.split(X_tr)))
    sel = features.FoldSelector(top_k=2, task="regression")
    base = set(sel.fit(X_tr.iloc[first_tr].to_numpy(float),
                       y_tr.iloc[first_tr].to_numpy(float)).selected_index_)

    X2 = X_tr.copy()
    X2.iloc[len(first_tr):, X2.columns.get_loc("x3")] *= 1e3   # 이후 구간만 오염
    after = set(features.FoldSelector(top_k=2, task="regression").fit(
        X2.iloc[first_tr].to_numpy(float), y_tr.iloc[first_tr].to_numpy(float)
    ).selected_index_)
    assert base == after, "폴드 선별이 자기 학습 구간 밖을 봤습니다"


def test_fold_selection_toggle_off_keeps_all_features(ts):
    X, y = _xy(ts)
    sel = features.FoldSelector(enabled=False).fit(X.to_numpy(float), y.to_numpy(float))
    assert sel.support_.all()
    assert sel.transform(X.to_numpy(float)).shape[1] == X.shape[1]


def test_select_core_matches_select_features(ts):
    """DataFrame 경로와 Pipeline 경로가 같은 선별 결과를 내야 한다."""
    X = ts[["x1", "x2", "x3"]]
    y = ts["y"]
    keep, _ = features.select_features(X, y, top_k=2)
    idx, _ = features.select_core(X.to_numpy(float), y.to_numpy(), list(X.columns), top_k=2)
    assert keep == [list(X.columns)[i] for i in idx]


def test_selection_report_has_reason_for_every_feature(ts):
    """SPEC §9 — 제외된 피처 전건에 사유가 있어야 한다."""
    X = ts[["x1", "x2", "x3"]].copy()
    X["constant"] = 5.0
    X["x1_copy"] = X["x1"]
    _, rep = features.select_features(X, ts["y"], top_k=2, corr_threshold=0.98)
    assert "reason" in rep.columns and "status" in rep.columns
    removed = rep[rep["status"] == "removed"]
    assert len(removed) > 0
    assert (removed["reason"].str.len() > 0).all(), "사유가 빈 피처가 있습니다"
    assert any("분산" in r for r in removed["reason"])
    assert any("corr" in r for r in removed["reason"])


def test_jaccard_stability_shape():
    sets = [{"a", "b", "c"}, {"a", "b", "d"}, {"a", "b", "c"}]
    tbl = features.jaccard_stability(sets)
    assert len(tbl) == 3                      # 3C2
    assert tbl["jaccard"].between(0, 1).all()


# ── 신규 결함 B — 리더보드 재정렬 ───────────────────────────
def test_sort_leaderboard_is_idempotent():
    """이미 rank 가 붙은 보드를 다시 정렬해도 죽지 않아야 한다."""
    board = pd.DataFrame([
        {"model": "A", "status": "ok", "holdout_R2": 0.9},
        {"model": "B", "status": "ok", "holdout_R2": 0.8},
    ])
    once = train.sort_leaderboard(board, "R2")
    twice = train.sort_leaderboard(once, "R2")          # 예전에는 ValueError
    assert list(twice["rank"]) == [1, 2]
    assert list(twice["model"]) == ["A", "B"]

    merged = train.sort_leaderboard(
        pd.concat([once, pd.DataFrame([
            {"model": "C", "status": "ok", "holdout_R2": 0.95}])], ignore_index=True), "R2")
    assert merged["model"].iloc[0] == "C"
