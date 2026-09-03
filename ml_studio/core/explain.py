"""SHAP 기반 해석.

학습에 쓴 구간(in-sample)에 챔피언 모델을 대입해 시점별 기여도를 구한다.
파이프라인을 (전처리 / 추정기) 로 분해해서, 전처리된 행렬 위에서 설명한다.
그래야 SHAP 값의 축과 화면에 그리는 피처 값의 축이 일치한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

TREE_FAMILIES = ("RandomForest", "ExtraTrees", "DecisionTree", "GradientBoosting",
                 "HistGradientBoosting", "XGB", "LGBM", "CatBoost", "Booster")
LINEAR_FAMILIES = ("Ridge", "Lasso", "ElasticNet", "LinearRegression", "LogisticRegression")


@dataclass
class ShapConfig:
    max_samples: int = 3000        # 계산 표본 상한
    background_size: int = 200     # 커널/선형 설명기 배경 표본
    seed: int = 42
    # 근사(커널) 경로는 **행마다 모델을 수천 번** 부른다. 상한을 안 걸면
    # 사용자가 4,000행을 그대로 넣고 몇 시간을 기다리게 된다 (실제로 그랬다).
    kernel_max_samples: int = 300


# 커널 설명기가 한 행에 쓰는 모델 호출 수 = 2*피처 + 2048 (shap 의 "auto").
# 시간 예측과 상한 판단에 쓴다.
KERNEL_OVERHEAD = 2048


class ShapUnavailable(RuntimeError):
    pass


def split_pipeline(pipeline) -> tuple[object | None, object]:
    """(전처리, 추정기) 로 분해. 파이프라인이 아니면 (None, 모델)."""
    if isinstance(pipeline, Pipeline):
        est = pipeline.steps[-1][1]
        pre = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
        return pre, est
    return None, pipeline


def transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """전처리를 통과한 행렬을 컬럼명이 붙은 DataFrame 으로."""
    pre, _ = split_pipeline(pipeline)
    if pre is None:
        return X.copy()
    Z = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except (AttributeError, ValueError):
        names = list(X.columns) if Z.shape[1] == X.shape[1] else [f"f{i}" for i in range(Z.shape[1])]
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return pd.DataFrame(np.asarray(Z), index=X.index, columns=names)


def blend_parts(est):
    """앙상블을 (멤버 파이프라인, 가중치, 절편) 으로 분해한다. 아니면 None.

    **왜 이게 결정적인가**

    이 도구의 앙상블 셋은 전부 base 예측의 **선형결합**이다.

        MeanBlend     : 1/k 씩
        WeightedBlend : nnls 가 구한 가중치
        OofStack      : 메타 모델이 선형이면 그 계수

    SHAP 은 가법적이므로, 예측이 선형결합이면 **SHAP 도 같은 가중치의
    선형결합**이다. 즉 base 각각을 빠른 트리 경로로 풀어서 합치면
    근사가 아니라 **정확히 같은 값**이 나온다.

    이걸 안 하면 앙상블 전체를 블랙박스로 보고 KernelExplainer 로 간다.
    4,000행 × 46피처 = 모델 호출 860만 번이고, 그 한 번마다 base 5개가
    전부 돈다. 실제로 사용자가 20분을 기다리다 포기했다.
    """
    pipes = getattr(est, "pipelines", None)
    names = getattr(est, "member_names_", None)
    if not pipes or not names:
        return None
    k = len(names)

    w = getattr(est, "weights_", None)
    if w is not None:                                  # WeightedBlend
        return pipes, list(names), np.asarray(w, dtype=float).ravel(), 0.0

    meta = getattr(est, "meta", None)
    if meta is not None:                               # OofStack
        coef = getattr(meta, "coef_", None)
        if coef is None:
            return None                                # 비선형 메타 — 분해 불가
        b = getattr(meta, "intercept_", 0.0)
        b = float(np.ravel(b)[0]) if np.ndim(b) else float(b)
        return pipes, list(names), np.asarray(coef, dtype=float).ravel(), b

    if hasattr(est, "predict"):                        # MeanBlend
        return pipes, list(names), np.full(k, 1.0 / k), 0.0
    return None


def _estimator_kind(est) -> str:
    name = type(est).__name__
    if any(k in name for k in TREE_FAMILIES):
        return "tree"
    if any(k in name for k in LINEAR_FAMILIES):
        return "linear"
    return "other"


def plan(pipeline, n_rows: int, cfg: ShapConfig | None = None) -> dict:
    """어떤 방법으로, 얼마나 걸릴지 **미리** 알려준다.

    **화면이 직접 짐작하면 안 된다.** 예전에는 explain_view 가 챔피언 이름에
    "Ensemble" 이 있으면 "트리 계열이라 수 초~1분" 이라고 안내했는데, 정작
    core 는 앙상블을 트리로 보지 않아 커널 근사로 갔다. 안내와 실제가 갈린
    탓에 사용자는 몇 시간짜리 계산을 1분짜리로 알고 20분을 기다렸다.
    **판단은 여기 한 곳에서만 한다.**

    반환: method · label · n(실제 계산 행수) · 설명 · 느린가 · 모델 호출 수
    """
    cfg = cfg or ShapConfig()
    _, est = split_pipeline(pipeline)

    if blend_parts(est) is not None:
        members = len(blend_parts(est)[1])
        n = min(n_rows, cfg.max_samples)
        return {"method": "blend", "label": f"앙상블 분해 ({members}개 모델)",
                "n": n, "slow": False, "calls": 0,
                "note": (f"base 모델 {members}개를 각각 정확히 풀어 가중평균합니다. "
                         "근사가 아니라 정확한 값이고, 보통 수십 초 안에 끝납니다.")}

    kind = _estimator_kind(est)
    if kind == "tree":
        n = min(n_rows, cfg.max_samples)
        return {"method": "tree", "label": "TreeExplainer", "n": n,
                "slow": False, "calls": 0,
                "note": "트리 계열이라 정확 계산이 가능합니다. 보통 수 초~1분입니다."}
    if kind == "linear":
        n = min(n_rows, cfg.max_samples)
        return {"method": "linear", "label": "LinearExplainer", "n": n,
                "slow": False, "calls": 0,
                "note": "선형 모델이라 즉시 끝납니다."}

    n = min(n_rows, cfg.kernel_max_samples)
    return {"method": "kernel", "label": "KernelExplainer", "n": n,
            "slow": True, "calls": None,
            "note": (f"이 모델은 정확 계산 경로가 없어 **근사**를 씁니다. 한 행마다 "
                     f"모델을 수천 번 부르므로 표본을 {cfg.kernel_max_samples:,}개로 "
                     "제한합니다. 그래도 수 분 걸릴 수 있습니다.")}


def compute_shap(pipeline, X: pd.DataFrame, cfg: ShapConfig | None = None) -> dict:
    """SHAP 값을 구한다.

    반환: {'values': DataFrame, 'data': DataFrame, 'base_value': float, 'explainer': str}
    values 와 data 는 같은 인덱스·컬럼을 갖는다.
    """
    cfg = cfg or ShapConfig()
    how = plan(pipeline, len(X), cfg)

    # 표본 추출은 **여기 한 번만** 한다. 앙상블 분해가 멤버마다 다시 뽑으면
    # 멤버끼리 다른 행을 설명하게 되어 합칠 수 없다.
    Xs = X
    if len(X) > how["n"]:
        rng = np.random.default_rng(cfg.seed)
        pos = np.sort(rng.choice(len(X), size=how["n"], replace=False))
        Xs = X.iloc[pos]

    out = _explain(pipeline, Xs, cfg, how)
    out["capped"] = how["n"] < len(X)
    out["plan"] = how["label"]
    return out


def _explain(pipeline, Xs: pd.DataFrame, cfg: ShapConfig, how: dict) -> dict:
    """표본이 이미 정해진 상태에서 실제 계산을 한다."""
    try:
        import shap
    except ImportError as e:
        raise ShapUnavailable("shap 이 설치되어 있지 않습니다. pip install shap") from e

    _, est = split_pipeline(pipeline)

    if how["method"] == "blend":
        return _explain_blend(est, Xs, cfg)

    Z = transformed_frame(pipeline, Xs)

    if how["method"] == "tree":
        explainer = shap.TreeExplainer(est)
        raw = explainer.shap_values(Z, check_additivity=False)
        base = explainer.expected_value
        label = "TreeExplainer"
    elif how["method"] == "linear":
        bg = Z.sample(min(cfg.background_size, len(Z)), random_state=cfg.seed)
        explainer = shap.LinearExplainer(est, bg)
        raw = explainer.shap_values(Z)
        base = explainer.expected_value
        label = "LinearExplainer"
    else:
        bg = shap.sample(Z, min(cfg.background_size, len(Z)), random_state=cfg.seed)
        fn = est.predict_proba if hasattr(est, "predict_proba") else est.predict
        explainer = shap.KernelExplainer(fn, bg)
        raw = explainer.shap_values(Z, nsamples="auto", silent=True)
        base = explainer.expected_value
        label = "KernelExplainer"

    values, base = _flatten(raw, base)
    if values.shape[1] != Z.shape[1]:
        raise RuntimeError(
            f"SHAP 출력 형태가 예상과 다릅니다: {values.shape} vs 피처 {Z.shape[1]}개"
        )

    return {
        "values": pd.DataFrame(values, index=Z.index, columns=Z.columns),
        "data": Z,
        "base_value": float(base),
        "explainer": label,
        "n_samples": len(Z),
    }


def _explain_blend(est, Xs: pd.DataFrame, cfg: ShapConfig) -> dict:
    """앙상블을 base 별로 풀어 가중평균한다. **근사가 아니라 정확한 값이다.**

    예측이 base 예측의 선형결합이므로 SHAP 도 같은 가중치의 선형결합이 된다.
    base 는 대개 트리라 각각은 빠른 경로를 탄다.
    """
    pipes, names, weights, intercept = blend_parts(est)

    total = None
    base_sum = float(intercept)
    data = None
    used = []
    for name, w in zip(names, weights):
        if abs(float(w)) < 1e-9:
            continue                       # 가중치 0 인 멤버는 계산할 이유가 없다
        member = pipes[name]
        part = _explain(member, Xs, cfg, plan(member, len(Xs), cfg))
        v = part["values"] * float(w)
        if total is None:
            total, data = v, part["data"]
        else:
            # 멤버끼리 전처리가 같아야 축이 맞는다. 이 도구는 같은 prep_config 를
            # 쓰므로 항상 같지만, 어긋나면 조용히 틀린 값을 내느니 멈춘다.
            if list(v.columns) != list(total.columns):
                raise RuntimeError(
                    f"앙상블 멤버 '{name}' 의 피처 축이 다릅니다 — 합칠 수 없습니다.")
            total = total + v
        base_sum += float(w) * float(part["base_value"])
        used.append(f"{name}×{float(w):.3f}")

    if total is None:
        raise RuntimeError("가중치가 0 이 아닌 멤버가 없습니다.")

    return {
        "values": total,
        "data": data,
        "base_value": base_sum,
        "explainer": f"앙상블 분해 ({', '.join(used)})",
        "n_samples": len(total),
    }


def _flatten(raw, base):
    """다중 클래스 출력이면 첫 번째(또는 양성) 클래스를 쓴다."""
    if isinstance(raw, list):
        idx = 1 if len(raw) == 2 else 0
        values = np.asarray(raw[idx])
        base = base[idx] if isinstance(base, (list, np.ndarray)) else base
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            idx = 1 if values.shape[2] == 2 else 0
            values = values[:, :, idx]
            if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0:
                base = np.asarray(base).ravel()[idx]
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    return values, float(base)


def importance(shap_result: dict, top_n: int | None = None) -> pd.DataFrame:
    """평균 절대 SHAP 기준 중요도."""
    v = shap_result["values"]
    out = (v.abs().mean().rename("mean_abs_shap").to_frame()
           .assign(mean_shap=v.mean(), std_shap=v.std())
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(names="feature"))
    out["contribution_pct"] = 100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.head(top_n) if top_n else out


def period_bounds(shap_result: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """SHAP 결과가 덮고 있는 시간 범위."""
    idx = shap_result["values"].index
    return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())


def slice_period(shap_result: dict, start=None, end=None) -> dict:
    """계산된 SHAP 결과를 기간으로 자른다.

    SHAP 값은 시점마다 독립적으로 계산되므로, 한 번 구해 두면 기간을 바꿔
    다시 계산할 필요가 없다. 잘라 쓰기만 하면 된다. 기준값(base_value)은
    모델 전체의 기대값이라 기간과 무관하게 그대로 둔다.
    """
    v, d = shap_result["values"], shap_result["data"]
    lo = pd.Timestamp(start) if start is not None else v.index.min()
    hi = pd.Timestamp(end) if end is not None else v.index.max()
    mask = (v.index >= lo) & (v.index <= hi)
    if not mask.any():
        raise ValueError(f"{lo:%Y-%m-%d %H:%M} ~ {hi:%Y-%m-%d %H:%M} 구간에 계산된 시점이 없습니다.")
    out = dict(shap_result)
    out["values"] = v.loc[mask]
    out["data"] = d.loc[mask]
    out["n_samples"] = int(mask.sum())
    out["period"] = (lo, hi)
    return out


def dependence_data(
    shap_result: dict,
    feature: str,
    interaction: str | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """dependence plot 용 표: 피처 값 vs 해당 피처의 SHAP 값.

    start/end 를 주면 그 기간의 시점만 담는다.
    """
    v, d = shap_result["values"], shap_result["data"]
    if feature not in v.columns:
        raise KeyError(f"'{feature}' 는 SHAP 결과에 없습니다.")
    if start is not None or end is not None:
        sub = slice_period(shap_result, start, end)
        v, d = sub["values"], sub["data"]
    out = pd.DataFrame({
        "timestamp": d.index,
        "feature_value": d[feature].to_numpy(),
        "shap_value": v[feature].to_numpy(),
    })
    if interaction and interaction in d.columns:
        out["interaction_value"] = d[interaction].to_numpy()
        out.attrs["interaction"] = interaction
    out.attrs["feature"] = feature
    out.attrs["period"] = (out["timestamp"].min(), out["timestamp"].max()) if len(out) else None
    return out


def dependence_by_periods(
    shap_result: dict,
    feature: str,
    periods: list[tuple[str, object, object]],
) -> pd.DataFrame:
    """여러 구간을 한 표에 쌓는다. 'period' 열로 구분한다.

    설비 개조 전후나 운전조건 변경 전후처럼, 같은 피처가 다른 구간에서
    다르게 작동하는지 겹쳐 보기 위한 것이다.
    """
    frames = []
    for label, start, end in periods:
        try:
            part = dependence_data(shap_result, feature, None, start, end)
        except ValueError:
            continue
        if part.empty:
            continue
        part = part.copy()
        part["period"] = label
        frames.append(part)
    if not frames:
        raise ValueError("선택한 구간 어디에도 계산된 시점이 없습니다.")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["feature"] = feature
    return out


def period_shift(
    shap_result: dict,
    periods: list[tuple[str, object, object]],
    top_n: int = 15,
) -> pd.DataFrame:
    """구간별 평균 절대 SHAP 을 나란히 놓아 기여도가 옮겨간 피처를 찾는다."""
    cols = {}
    for label, start, end in periods:
        try:
            sub = slice_period(shap_result, start, end)
        except ValueError:
            continue
        cols[label] = sub["values"].abs().mean()
    if not cols:
        raise ValueError("선택한 구간 어디에도 계산된 시점이 없습니다.")
    out = pd.DataFrame(cols)
    out = out.div(out.sum(axis=0), axis=1) * 100  # 구간별 기여 비중(%)
    if out.shape[1] >= 2:
        first, last = out.columns[0], out.columns[-1]
        out["변화"] = out[last] - out[first]
        out = out.reindex(out["변화"].abs().sort_values(ascending=False).index)
    else:
        out = out.sort_values(out.columns[0], ascending=False)
    return out.head(top_n).reset_index(names="feature")


def auto_interaction(shap_result: dict, feature: str) -> str | None:
    """해당 피처의 SHAP 값 변동을 가장 잘 설명하는 다른 피처를 고른다."""
    v, d = shap_result["values"], shap_result["data"]
    if feature not in v.columns:
        return None
    s = v[feature]
    others = [c for c in d.columns if c != feature]
    if not others:
        return None
    corr = d[others].corrwith(s).abs()
    corr = corr.dropna()
    return str(corr.idxmax()) if not corr.empty else None


def local_explanation(shap_result: dict, timestamp, top_n: int = 12) -> pd.DataFrame:
    """특정 시점 하나에 대한 기여도 분해."""
    v, d = shap_result["values"], shap_result["data"]
    ts = pd.Timestamp(timestamp)
    if ts not in v.index:
        pos = int(np.argmin(np.abs(v.index.values - np.datetime64(ts))))
        ts = v.index[pos]
    row = v.loc[ts]
    out = pd.DataFrame({
        "feature": row.index,
        "shap_value": row.to_numpy(),
        "feature_value": d.loc[ts].to_numpy(),
    })
    out["abs"] = out["shap_value"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs").head(top_n)
    out.attrs["timestamp"] = ts
    out.attrs["base_value"] = shap_result["base_value"]
    return out


def permutation_importance_fallback(pipeline, X, y, scoring=None, n_repeats: int = 5, seed: int = 42):
    """shap 을 못 쓰는 상황의 대체 수단."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                               random_state=seed, scoring=scoring, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))
