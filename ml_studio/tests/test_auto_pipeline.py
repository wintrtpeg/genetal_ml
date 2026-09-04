"""Auto 모드 회귀 테스트.

자동화되는 것은 '사람이 버튼을 누르는 일' 이지 '검증을 생략하는 일' 이 아니다.
자동 경로에서도 누수 방지 장치가 전부 살아 있는지 확인한다.
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

from core import features, models, pipeline, validation  # noqa: E402


@pytest.fixture
def ts() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=900, freq="5min")
    rng = np.random.default_rng(41)
    x1 = np.cumsum(rng.normal(0, 1, 900)) + 50
    x2 = rng.normal(10, 2, 900)
    x3 = np.full(900, 7.0)                      # 상수 — 품질 단계에서 빠져야 한다
    y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, 900)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y}, index=idx)


def _cfg(**kw):
    base = dict(top_k=8, n_splits=3, include_heavy=False, ensemble=False,
                feature=features.FeatureConfig(
                    lags=[1, 3], rolling_windows=[6], rolling_stats=["mean"],
                    ewm_spans=[], diffs=[1], time_features=False))
    base.update(kw)
    return pipeline.AutoConfig(**base)


def test_auto_run_produces_a_champion(ts):
    res = pipeline.run_auto(ts, "y", _cfg())
    assert res.champion
    assert res.leaderboard is not None and not res.leaderboard.empty
    assert (res.leaderboard["status"] == "ok").any()
    assert len(res.selected_features) > 0
    assert res.X.shape[1] == len(res.selected_features)


def test_auto_run_reports_its_decisions(ts):
    """자동으로 돌렸어도 무엇이 어떻게 정해졌는지 되짚을 수 있어야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg())
    d = res.decisions
    assert not d.empty
    assert set(d.columns) == {"단계", "항목", "결정", "근거"}
    assert {"품질", "파생", "분할", "선별", "학습"} <= set(d["단계"])
    assert (d["근거"].str.len() > 0).all(), "근거가 빈 결정이 있습니다"


# ── 누수 방지 장치가 자동 경로에서도 살아 있는가 ────────────
def test_auto_run_always_makes_three_way_split(ts):
    res = pipeline.run_auto(ts, "y", _cfg())
    sp = res.split
    assert len(sp.unseen) > 0, "Auto 가 Final Unseen 없이 돌았습니다"
    validation.assert_disjoint(sp)
    assert sp.train.max() < sp.valid.min() < sp.valid.max() < sp.unseen.min()


def test_auto_run_sets_gap_to_lookback(ts):
    """gap 이 파생 lookback 보다 좁으면 점검이 막아야 한다. 자동으로 맞춰지는지 확인."""
    res = pipeline.run_auto(ts, "y", _cfg())
    lookback = features.warmup_rows(res.feature_config, ts.index)
    assert res.split_config.gap >= lookback
    row = res.checklist[res.checklist["항목"] == "gap 확보"]
    assert row["결과"].iloc[0] == "통과"


def test_auto_run_leakage_checklist_all_pass(ts):
    res = pipeline.run_auto(ts, "y", _cfg())
    assert (res.checklist["결과"] == "통과").all()
    assert "선별 구간 격리" in set(res.checklist["항목"])
    assert "Final Unseen 격리" in set(res.checklist["항목"])


def test_auto_run_selects_features_on_train_only(ts):
    """선별에 쓴 구간이 학습 구간과 정확히 같아야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg())
    assert np.array_equal(res.split.train, res.split.train)
    assert np.intersect1d(res.split.train, res.split.valid).size == 0
    assert np.intersect1d(res.split.train, res.split.unseen).size == 0


def test_auto_run_blocks_target_derived_features(ts):
    """자동이어도 Y 파생은 막혀야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg())
    assert not [c for c in res.X.columns if c.startswith("y__")]
    lookup = res.provenance.set_index("feature")["origin"].to_dict()
    assert all("y" not in str(lookup.get(f, "")).split("|") for f in res.selected_features)
    assert res.feature_config.allow_target_derived is False


def test_auto_run_unseen_opened_at_most_once(ts):
    res = pipeline.run_auto(ts, "y", _cfg())
    assert res.unseen_guard is not None
    assert res.unseen_guard.access_count == 1
    from core import train
    with pytest.raises(train.UnseenAccessError):
        res.unseen_guard.open("두번째")


def test_auto_run_reports_unseen_not_validation(ts):
    """최종 보고값은 Final Unseen 이어야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg())
    assert res.unseen_scores
    assert "unseen_R2" in res.unseen_scores
    assert res.unseen_scores["unseen_rows"] == len(res.split.unseen)


def test_auto_run_can_skip_unseen_evaluation(ts):
    """평가를 미루면 guard 는 열리지 않은 채로 남아야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg(evaluate_unseen=False))
    assert res.unseen_scores == {}
    assert res.unseen_guard.access_count == 0


def test_auto_run_drops_constant_column(ts):
    """품질 단계가 상수 컬럼을 걸러야 한다."""
    res = pipeline.run_auto(ts, "y", _cfg())
    assert "x3" not in res.kept
    assert not [c for c in res.X.columns if c.startswith("x3")]


def test_auto_run_fold_selection_on_by_default(ts):
    res = pipeline.run_auto(ts, "y", _cfg())
    assert res.train_config.fold_selection is True
    ok = [n for n, r in res.detail.items() if r.get("status") == "ok"]
    assert any("fold_jaccard" in res.detail[n] for n in ok)


# ── 실패 경로 ───────────────────────────────────────────────
def test_auto_run_rejects_missing_target(ts):
    with pytest.raises(pipeline.AutoRunError, match="타겟"):
        pipeline.run_auto(ts, "없는컬럼", _cfg())


def test_auto_run_rejects_non_datetime_index(ts):
    flat = ts.reset_index(drop=True)
    with pytest.raises(pipeline.AutoRunError, match="DatetimeIndex"):
        pipeline.run_auto(flat, "y", _cfg())


def test_auto_run_stops_when_leakage_check_fails(ts, monkeypatch):
    """점검이 실패하면 학습으로 넘어가지 않고 멈춰야 한다."""
    from core import validation as V

    real = V.leakage_checklist

    def fake(*a, **k):
        out = real(*a, **k)
        out.loc[out.index[0], "결과"] = "실패"
        return out

    monkeypatch.setattr(V, "leakage_checklist", fake)
    with pytest.raises(pipeline.AutoRunError, match="누수 점검"):
        pipeline.run_auto(ts, "y", _cfg())


# ── 구조 ────────────────────────────────────────────────────
def test_pipeline_module_has_no_streamlit_import():
    """Auto 오케스트레이션은 UI 없이 돌아야 한다 (Dataiku Scenario 전제)."""
    tree = ast.parse((ROOT / "core" / "pipeline.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        assert "streamlit" not in mods, f"pipeline.py:{node.lineno} 가 streamlit 을 import 합니다"


def test_auto_result_covers_every_field_the_view_reads():
    """_apply 가 res.<이름> 으로 읽는 값이 전부 AutoResult 에 있어야 한다.

    문자열이 아니라 AST 로 속성 접근을 찾는다. 필드 이름을 바꾸고 화면 쪽을
    안 고치면 Auto 실행이 런타임에 죽는데, 그걸 여기서 잡는다.
    """
    tree = ast.parse((ROOT / "app" / "views" / "data_view.py").read_text(encoding="utf-8"))
    apply_fn = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_apply"), None)
    assert apply_fn is not None, "data_view 에 _apply 가 없습니다"

    used = {
        node.attr for node in ast.walk(apply_fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "res"
    }
    assert used, "_apply 가 AutoResult 를 읽지 않습니다"
    fields = set(pipeline.AutoResult.__dataclass_fields__)
    assert used <= fields, f"AutoResult 에 없는 필드를 읽습니다: {sorted(used - fields)}"
