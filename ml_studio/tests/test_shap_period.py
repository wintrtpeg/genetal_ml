"""SHAP 기간 기능 검증.

shap/plotly 없이도 돌도록, 계산 결과 구조를 흉내 낸 표로 순수 로직만 확인한다.
pytest 로도 돌고, 직접 실행하면 결과를 표로 찍는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import explain  # noqa: E402

FEATS = ["flow", "temp", "valve"]


def _fixture():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=600, freq="1h")
    data = pd.DataFrame(rng.normal(size=(600, 3)), index=idx, columns=FEATS)
    vals = pd.DataFrame(rng.normal(size=(600, 3)), index=idx, columns=FEATS)
    # 후반 구간에서 valve 기여를 키워, 구간 비교가 이를 잡아내는지 본다
    vals.loc[idx[300:], "valve"] *= 6
    res = {"values": vals, "data": data, "base_value": 12.5,
           "explainer": "TreeExplainer", "n_samples": 600}
    return res, idx, vals


def test_slice_period_keeps_original_intact():
    res, idx, _ = _fixture()
    lo, hi = explain.period_bounds(res)
    assert lo == idx[0] and hi == idx[-1]

    v = explain.slice_period(res, idx[100], idx[199])
    assert v["n_samples"] == 100
    assert v["base_value"] == res["base_value"]      # 기준값은 기간과 무관
    assert res["n_samples"] == 600                   # 원본 불변
    assert len(res["values"]) == 600


def test_empty_period_raises():
    res, _, _ = _fixture()
    try:
        explain.slice_period(res, "2030-01-01", "2030-02-01")
    except ValueError:
        return
    raise AssertionError("빈 구간인데 예외가 나지 않았습니다.")


def test_dependence_period_matches_source():
    res, idx, vals = _fixture()
    d_all = explain.dependence_data(res, "flow", "temp")
    d_sub = explain.dependence_data(res, "flow", "temp", idx[100], idx[199])
    assert len(d_all) == 600 and len(d_sub) == 100
    assert "interaction_value" in d_sub.columns
    assert np.allclose(d_sub["shap_value"].to_numpy(),
                       vals["flow"].iloc[100:200].to_numpy())


def test_dependence_by_periods():
    res, idx, _ = _fixture()
    periods = [("A", idx[0], idx[299]), ("B", idx[300], idx[-1])]
    dp = explain.dependence_by_periods(res, "valve", periods)
    assert len(dp) == 600
    assert set(dp["period"]) == {"A", "B"}


def test_period_shift_detects_moved_contribution():
    res, idx, _ = _fixture()
    periods = [("A", idx[0], idx[299]), ("B", idx[300], idx[-1])]
    sh = explain.period_shift(res, periods)
    assert "변화" in sh.columns
    assert sh.iloc[0]["feature"] == "valve"   # 가장 크게 움직인 피처
    assert sh.iloc[0]["변화"] > 0             # B 구간에서 커졌다


def test_importance_reflects_period():
    res, idx, _ = _fixture()
    a = explain.importance(explain.slice_period(res, idx[0], idx[299]))
    b = explain.importance(explain.slice_period(res, idx[300], idx[-1]))
    assert b.iloc[0]["feature"] == "valve"

    # A 는 세 피처가 고만고만하고, B 에서 valve 만 크게 벌어져야 한다
    pct = lambda t, f: float(t.loc[t["feature"] == f, "contribution_pct"].iloc[0])  # noqa: E731
    assert pct(a, "valve") < 40 < pct(b, "valve")
    assert a["contribution_pct"].max() - a["contribution_pct"].min() < 5

    for t in (a, b):
        assert abs(t["contribution_pct"].sum() - 100) < 1e-6


def test_local_explanation_within_view():
    res, idx, _ = _fixture()
    v = explain.slice_period(res, idx[100], idx[199])
    lc = explain.local_explanation(v, idx[150])
    assert len(lc) == 3
    assert lc.attrs["timestamp"] == idx[150]


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    width = max(len(f.__name__) for f in fns)
    fails = 0
    for f in fns:
        try:
            f()
            print(f"  PASS  {f.__name__:<{width}}  {(f.__doc__ or '').strip()}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL  {f.__name__:<{width}}  {type(e).__name__}: {e}")

    res, idx, _ = _fixture()
    periods = [("A", idx[0], idx[299]), ("B", idx[300], idx[-1])]
    print("\n구간별 기여 비중(%)")
    print(explain.period_shift(res, periods).to_string(index=False,
                                                       float_format=lambda x: f"{x:7.2f}"))
    print(f"\n{len(fns) - fails}/{len(fns)} 통과")
    sys.exit(1 if fails else 0)


# ── 앙상블 SHAP — 20분 기다려도 안 끝나던 그것 ────────────────
"""챔피언이 앙상블이면 SHAP 이 사실상 안 끝났다.

`_estimator_kind` 가 앙상블을 트리로 못 봐서 KernelExplainer 로 떨어졌고,
표본 4,000 × 피처 46 이면 모델 호출이 860만 번이다. 그 한 번마다 base 5개가
전부 돈다. 사용자가 20분 기다리다 포기했다.

게다가 **화면은 "수 초~1분" 이라고 안내하고 있었다** — 챔피언 이름에
"Ensemble" 이 있으면 빠른 경로라고 짐작했기 때문이다. 안내와 실제가 갈렸다.

고친 방법: 이 도구의 앙상블 셋은 전부 base 예측의 **선형결합**이므로,
SHAP 도 같은 가중치의 선형결합이다. base 를 각각 빠르게 풀어 합치면
**근사가 아니라 정확히 같은 값**이 나온다.
"""


def _fake_pipe(k):
    class P:
        def predict(self, X):
            return X["a"].to_numpy() * k
    return P()


def test_blend_weights_reproduce_the_prediction():
    """분해한 가중치로 base 예측을 합치면 앙상블 예측과 같아야 한다.

    이게 성립해야 'SHAP 도 같은 가중치로 합치면 된다' 가 성립한다.
    여기가 틀리면 화면에 **조용히 틀린 기여도**가 뜬다.
    """
    from core import ensemble, explain

    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0), "m3": _fake_pipe(3.0)}
    X = pd.DataFrame({"a": np.arange(10.0)})

    mean = ensemble.MeanBlend(pipes)
    weighted = ensemble.WeightedBlend(pipes)
    weighted.weights_ = np.array([0.2, 0.3, 0.5])

    for blend in (mean, weighted):
        pipes_, names, w, b = explain.blend_parts(blend)
        combined = sum(wi * pipes_[n].predict(X) for n, wi in zip(names, w)) + b
        assert np.allclose(blend.predict(X), combined), type(blend).__name__


def test_stacking_decomposes_through_its_linear_meta():
    from sklearn.linear_model import Ridge

    from core import ensemble, explain

    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0), "m3": _fake_pipe(3.0)}
    P = pd.DataFrame({"m1": [1., 2, 3], "m2": [2., 4, 6], "m3": [3., 6, 9]})
    stack = ensemble.OofStack(pipes, Ridge().fit(P, [1., 2, 3]))
    X = pd.DataFrame({"a": np.arange(10.0)})

    pipes_, names, w, b = explain.blend_parts(stack)
    combined = sum(wi * pipes_[n].predict(X) for n, wi in zip(names, w)) + b
    assert np.allclose(stack.predict(X), combined)


def test_nonlinear_meta_is_refused_not_silently_wrong():
    """메타가 비선형이면 분해가 성립하지 않는다. 그때는 **거부해야** 한다.

    억지로 합치면 틀린 기여도가 조용히 화면에 뜬다 — 느린 것보다 나쁘다.
    """
    from sklearn.ensemble import RandomForestRegressor

    from core import ensemble, explain

    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0)}
    P = pd.DataFrame({"m1": [1., 2, 3], "m2": [2., 4, 6]})
    stack = ensemble.OofStack(pipes,
                              RandomForestRegressor(n_estimators=3).fit(P, [1., 2, 3]))
    assert explain.blend_parts(stack) is None


def test_plan_never_promises_a_fast_path_for_a_slow_one():
    """**안내와 실제가 갈리면 안 된다.** 20분을 기다리게 만든 원인이다.

    plan() 이 말하는 방법과 compute_shap 이 실제로 쓰는 경로가 같아야 한다.
    """
    from core import ensemble, explain

    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0)}
    blend = ensemble.MeanBlend(pipes)

    p = explain.plan(blend, 4000)
    assert p["method"] == "blend", p
    assert p["slow"] is False
    assert "앙상블" in p["label"]

    class Opaque:                      # 트리도 선형도 앙상블도 아닌 것
        def predict(self, X):
            return np.zeros(len(X))

    q = explain.plan(Opaque(), 4000)
    assert q["method"] == "kernel", q
    assert q["slow"] is True, "느린 경로를 느리다고 말하지 않습니다"
    assert q["n"] <= 300, (f"근사 경로인데 표본을 {q['n']}개나 씁니다 — "
                           "행마다 모델을 수천 번 부릅니다")


def test_kernel_path_is_capped_hard():
    """근사 경로는 사용자가 4,000을 넣어도 상한이 걸려야 한다.

    상한이 없으면 '끝나지 않는 계산' 이 된다. 실제로 그랬다.
    """
    from core import explain

    class Opaque:
        def predict(self, X):
            return np.zeros(len(X))

    cfg = explain.ShapConfig(max_samples=4000)
    assert explain.plan(Opaque(), 100000, cfg)["n"] <= cfg.kernel_max_samples


def test_blend_shap_is_the_weighted_sum_of_member_shap(monkeypatch):
    """합치는 계산 자체를 확인한다 — shap 없이.

    멤버별 결과를 알고 있는 값으로 바꿔치기하고, 나온 합이 가중합인지 본다.
    """
    from core import ensemble, explain

    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    cols = ["f1", "f2"]
    canned = {
        "m1": pd.DataFrame([[1.0, 2.0]] * 5, index=idx, columns=cols),
        "m2": pd.DataFrame([[10.0, 20.0]] * 5, index=idx, columns=cols),
    }
    bases = {"m1": 1.0, "m2": 5.0}

    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0)}
    blend = ensemble.WeightedBlend(pipes)
    blend.weights_ = np.array([0.25, 0.75])

    order = list(canned)

    def fake_explain(pipeline, Xs, cfg, how):
        name = order[list(pipes.values()).index(pipeline)]
        return {"values": canned[name], "data": canned[name],
                "base_value": bases[name], "explainer": "fake",
                "n_samples": len(Xs)}

    monkeypatch.setattr(explain, "_explain", fake_explain)
    out = explain._explain_blend(blend, pd.DataFrame({"a": range(5)}, index=idx),
                                 explain.ShapConfig())

    assert np.allclose(out["values"]["f1"], 0.25 * 1.0 + 0.75 * 10.0)
    assert np.allclose(out["values"]["f2"], 0.25 * 2.0 + 0.75 * 20.0)
    assert out["base_value"] == pytest.approx(0.25 * 1.0 + 0.75 * 5.0)
    assert "앙상블" in out["explainer"]


def test_zero_weight_members_are_not_computed(monkeypatch):
    """가중치 0 인 멤버를 계산하는 건 순수한 낭비다."""
    from core import ensemble, explain

    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    frame = pd.DataFrame([[1.0]] * 3, index=idx, columns=["f1"])
    pipes = {"m1": _fake_pipe(1.0), "m2": _fake_pipe(2.0), "m3": _fake_pipe(3.0)}
    blend = ensemble.WeightedBlend(pipes)
    blend.weights_ = np.array([0.5, 0.0, 0.5])

    seen = []

    def fake_explain(pipeline, Xs, cfg, how):
        seen.append(pipeline)
        return {"values": frame, "data": frame, "base_value": 0.0,
                "explainer": "fake", "n_samples": 3}

    monkeypatch.setattr(explain, "_explain", fake_explain)
    explain._explain_blend(blend, pd.DataFrame({"a": range(3)}, index=idx),
                           explain.ShapConfig())
    assert len(seen) == 2, f"가중치 0 인 멤버까지 계산했습니다 ({len(seen)}개)"
