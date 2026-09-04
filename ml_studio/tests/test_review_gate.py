"""3단계 검토 게이트 회귀 테스트.

사람이 X 피처를 최종 확정하는 관문이다. 여기서 중요한 것은 두 가지다.

1. 검토 화면이 보여주는 통계가 **학습 구간에서만** 계산되는가.
   알고리즘이 홀드아웃을 못 보게 막아 놓고 사람에게는 보여주면, 사람이 누수 경로가 된다.
2. 사람이 자동 추천을 바꾼 것이 감사 이력에 남는가.
   "왜 이 피처가 들어갔나"에 사람이 바꾼 경우도 답할 수 있어야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import features, persist, preprocess, validation  # noqa: E402


@pytest.fixture
def ts() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=800, freq="5min")
    rng = np.random.default_rng(31)
    x1 = np.cumsum(rng.normal(0, 1, 800)) + 50
    x2 = rng.normal(10, 2, 800)
    x3 = rng.normal(0, 1, 800)               # 무관 신호
    y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, 800)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y}, index=idx)


def _prepared(ts):
    cfg = features.FeatureConfig(lags=[1, 3], rolling_windows=[6], rolling_stats=["mean"],
                                 ewm_spans=[], diffs=[1], time_features=False)
    feat, prov = features.generate(ts, "y", ["x1", "x2", "x3"], cfg)
    feat = features.drop_warmup(feat, cfg)
    cols = [c for c in feat.columns if c != "y"]
    X, y = preprocess.prepare_xy(feat, "y", cols)
    sp = validation.three_way_split(len(X), 0.2, 0.15, gap=6)
    return X, y, prov, sp


# ── 1. 검토 통계가 학습 구간에서만 나오는가 ────────────────
def test_review_report_ignores_validation_and_unseen(ts):
    """검증·Unseen 구간을 극단으로 바꿔도 검토 표가 변하지 않아야 한다.

    변한다면 사람이 그 표를 보고 피처를 고르는 순간 미래를 본 것이 된다.
    """
    X, y, prov, sp = _prepared(ts)

    sel_a, rep_a = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=6)
    view_a = features.feature_report(rep_a, prov, X_train=X.iloc[sp.train])

    X2 = X.copy()
    X2.iloc[sp.valid] *= 1e4          # 검증 구간 오염
    X2.iloc[sp.unseen] *= 1e4         # Unseen 구간 오염
    y2 = y.copy()
    y2.iloc[sp.valid] += 1e4
    y2.iloc[sp.unseen] += 1e4

    sel_b, rep_b = features.select_features(X2.iloc[sp.train], y2.iloc[sp.train], top_k=6)
    view_b = features.feature_report(rep_b, prov, X_train=X2.iloc[sp.train])

    assert sel_a == sel_b, "오염이 자동 추천을 바꿨습니다"
    for col in ("mutual_info", "variance", "결측률", "학습구간_평균"):
        if col in view_a.columns:
            a = view_a.set_index("feature")[col]
            b = view_b.set_index("feature")[col]
            assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                               equal_nan=True), f"{col} 이 검증·Unseen 오염에 반응했습니다"


def test_review_report_stats_match_train_slice_only(ts):
    """표의 학습구간 평균이 실제 학습 구간 평균과 일치해야 한다."""
    X, y, prov, sp = _prepared(ts)
    _, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=5)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])

    got = view.set_index("feature")["학습구간_평균"]
    want = X.iloc[sp.train].select_dtypes("number").mean()
    common = [c for c in want.index if c in got.index]
    assert len(common) > 3
    assert np.allclose(got.loc[common].to_numpy(dtype=float),
                       want.loc[common].to_numpy(dtype=float), equal_nan=True)


# ── 2. 표 구성 ──────────────────────────────────────────────
def test_review_report_joins_provenance(ts):
    X, y, prov, sp = _prepared(ts)
    _, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=5)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])

    for col in ("feature", "origin", "transform", "lookback", "reason", "status", "MI순위"):
        assert col in view.columns, f"{col} 이 검토 표에 없습니다"

    row = view[view["feature"] == "x1__lag3"]
    assert len(row) == 1
    assert row["origin"].iloc[0] == "x1"
    assert row["transform"].iloc[0] == "lag(3)"
    assert int(row["lookback"].iloc[0]) == 3


def test_review_report_survives_missing_provenance(ts):
    X, y, _, sp = _prepared(ts)
    _, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=5)
    view = features.feature_report(rep, None, X_train=X.iloc[sp.train])
    assert "origin" in view.columns and len(view) == len(rep)


def test_origin_rollup_counts_per_source(ts):
    X, y, prov, sp = _prepared(ts)
    _, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=5)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])
    roll = features.origin_rollup(view)

    assert set(roll.columns) >= {"원본", "생성", "선택", "선택률"}
    assert roll["생성"].sum() == len(view)
    assert (roll["선택"] <= roll["생성"]).all()
    assert "x1" in set(roll["원본"])


# ── 3. 수동 선택이 이력에 남는가 ────────────────────────────
def test_manual_addition_is_recorded_with_reason(ts):
    X, y, prov, sp = _prepared(ts)
    sel, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=3)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])

    dropped = [f for f in view["feature"] if f not in sel]
    assert dropped
    chosen = sorted(set(sel) | {dropped[0]})

    final, updated = features.apply_manual_selection(view, chosen)
    assert final == chosen
    row = updated[updated["feature"] == dropped[0]].iloc[0]
    assert row["status"] == "selected(수동추가)"
    assert "사용자가 직접 추가" in row["reason"]
    assert row["kept"] is True or row["kept"] == True   # noqa: E712


def test_manual_removal_is_recorded_with_reason(ts):
    X, y, prov, sp = _prepared(ts)
    sel, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=4)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])

    chosen = sorted(set(sel) - {sel[0]})
    final, updated = features.apply_manual_selection(view, chosen)
    assert sel[0] not in final
    row = updated[updated["feature"] == sel[0]].iloc[0]
    assert row["status"] == "removed(수동제외)"
    assert "사용자가 직접 제외" in row["reason"]


def test_untouched_features_keep_automatic_status(ts):
    """사람이 건드리지 않은 피처는 자동 판정 사유가 그대로 남아야 한다."""
    X, y, prov, sp = _prepared(ts)
    sel, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=4)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])

    _, updated = features.apply_manual_selection(view, list(sel))
    assert not updated["status"].astype(str).str.contains("수동").any()
    keep = updated[updated["feature"] == sel[0]].iloc[0]
    assert keep["status"] == "selected"
    assert "MI 상위" in str(keep["reason"])


def test_manual_overrides_reach_the_manifest(ts):
    """재현 기록에 사람이 바꾼 내용이 사유와 함께 남아야 한다."""
    X, y, prov, sp = _prepared(ts)
    sel, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=3)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])
    dropped = [f for f in view["feature"] if f not in sel][0]
    chosen, updated = features.apply_manual_selection(view, sorted(set(sel) | {dropped}))

    man = persist.build_manifest(run_id="r", selection_report=updated)
    assert dropped in man["features_selected"], "수동 추가분이 선택 목록에서 빠졌습니다"
    assert "manual_overrides" in man
    assert any(o["feature"] == dropped for o in man["manual_overrides"])
    assert all(o["reason"] for o in man["manual_overrides"])


# ── 4. 위험 경고 ────────────────────────────────────────────
def test_risk_flags_correlated_pair_chosen_together(ts):
    X = ts[["x1", "x2"]].copy()
    X["x1_copy"] = X["x1"]                     # 완전 중복
    y = ts["y"]
    _, rep = features.select_features(X, y)
    view = features.feature_report(rep, None, X_train=X)

    risks = features.selection_risks(["x1", "x1_copy", "x2"], view, X)
    assert (risks["위험"] == "중복").any()
    assert any("x1" in p for p in risks["피처"])


def test_risk_flags_zero_variance_and_high_missing(ts):
    X = ts[["x1", "x2"]].copy()
    X["const"] = 3.0
    X["holey"] = np.where(np.arange(len(X)) % 3 == 0, X["x2"], np.nan)
    y = ts["y"]
    _, rep = features.select_features(X, y)
    view = features.feature_report(rep, None, X_train=X)

    risks = features.selection_risks(["const", "holey", "x1"], view, X, max_missing=0.3)
    kinds = set(risks["위험"])
    assert "분산 0" in kinds
    assert "결측 과다" in kinds


def test_risks_are_empty_for_a_clean_pick(ts):
    X, y, prov, sp = _prepared(ts)
    sel, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=3)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])
    risks = features.selection_risks(list(sel), view, X.iloc[sp.train])
    assert risks.empty or "중복" not in set(risks["위험"])


def test_risks_never_block_the_choice(ts):
    """위험이 있어도 선택 자체는 통과해야 한다. 도메인 판단이 우선이다."""
    X = ts[["x1", "x2"]].copy()
    X["const"] = 3.0
    _, rep = features.select_features(X, ts["y"])
    view = features.feature_report(rep, None, X_train=X)
    chosen, updated = features.apply_manual_selection(view, ["const", "x1"])
    assert chosen == ["const", "x1"]           # 예외 없이 반영된다
    assert updated[updated["feature"] == "const"]["kept"].iloc[0]


# ── 5. 2단계 제외가 3단계에서 되살아나지 않는가 ────────────
def test_excluded_columns_do_not_return_as_candidates(ts):
    """2단계에서 뺀 컬럼이 3단계 X 후보로 돌아오면 사용자의 제외 결정이 무시된 것이다.

    검토 게이트를 만들면서 드러난 결함. generate 가 df.copy() 를 쓰는 바람에
    제외 컬럼이 그대로 남아 있었다.
    """
    cfg = features.FeatureConfig(lags=[1], rolling_windows=[], ewm_spans=[],
                                 diffs=[], time_features=False)
    keep = ["x1", "x2"]                       # x3 는 2단계에서 제외했다고 가정
    feat, prov = features.generate(ts, "y", keep, cfg)

    assert "x3" not in feat.columns, "제외한 컬럼이 파생 결과에 남아 있습니다"
    assert not [c for c in feat.columns if c.startswith("x3__")]
    assert "x1" in feat.columns and "y" in feat.columns
    assert "x1__lag1" in feat.columns


def test_every_candidate_has_provenance(ts):
    """X 후보 전건에 출처가 있어야 한다. 없으면 어디서 온 피처인지 못 되짚는다."""
    cfg = features.FeatureConfig(lags=[1, 3], rolling_windows=[6], rolling_stats=["mean"],
                                 ewm_spans=[], diffs=[], time_features=True)
    feat, prov = features.generate(ts, "y", ["x1", "x2"], cfg)
    known = set(prov["feature"])
    orphan = [c for c in feat.columns if c != "y" and c not in known]
    assert not orphan, f"출처 대장에 없는 X 후보: {orphan}"


def test_review_report_has_no_unknown_origin(ts):
    """검토 표의 원본 컬럼이 전부 실제 태그여야 한다 ('—' 가 없어야 한다)."""
    X, y, prov, sp = _prepared(ts)
    _, rep = features.select_features(X.iloc[sp.train], y.iloc[sp.train], top_k=5)
    view = features.feature_report(rep, prov, X_train=X.iloc[sp.train])
    assert "—" not in set(view["origin"].astype(str)), \
        "출처를 모르는 피처가 검토 표에 있습니다"


# ── 6. 게이트가 실제로 막는가 ───────────────────────────────
def test_state_defaults_have_gate_keys():
    """확정 전에는 X 가 없어야 학습이 막힌다."""
    import ast
    tree = ast.parse((ROOT / "app" / "state.py").read_text(encoding="utf-8"))
    defaults = None
    for node in tree.body:
        # DEFAULTS 는 `DEFAULTS: dict = {...}` 형태라 AnnAssign 이다
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "DEFAULTS" and node.value is not None:
            defaults = ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets):
            defaults = ast.literal_eval(node.value)
    assert defaults is not None, "app/state.py 에서 DEFAULTS 를 읽지 못했습니다"
    for k in ("feature_review", "review_picks", "X_pool"):
        assert k in defaults, f"{k} 상태 키가 없습니다"
    assert defaults["X"] is None, "확정 전 X 가 비어 있어야 합니다"


def test_selection_invalidation_clears_review():
    """분할이나 선별을 다시 하면 검토 상태도 함께 비워져야 한다."""
    src = (ROOT / "app" / "state.py").read_text(encoding="utf-8")
    i = src.index("select_out = [")
    block = src[i:src.index("]", i)]
    for k in ("feature_review", "review_picks", "X_pool", "X", "selected_features"):
        assert f'"{k}"' in block, f"{k} 가 무효화 목록에 없습니다"
