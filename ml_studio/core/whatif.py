"""What-if 분석.

Y 의 과거값을 X 로 쓰지 않기로 했으므로, X 를 바꾸면 예측이 어떻게 움직이는지
그대로 읽을 수 있다. (Y lag 를 넣었다면 Y 를 바꿔야 X 가 바뀌는 순환이 생겨
이 화면의 해석이 무너진다.)

주의: 모델은 상관을 학습한 것이지 인과를 학습한 것이 아니다.
여기 나오는 값은 '모델이 그렇게 본다'는 뜻이고, 실제 설비 반응과는 다를 수 있다.
피처 간 물리적 종속(예: 유량을 올리면 차압도 오른다)은 아래 linked 옵션으로 함께 움직여야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SET = "set"        # 값을 지정
DELTA = "delta"    # 현재값 + a
PCT = "pct"        # 현재값 × (1 + a/100)


@dataclass
class Change:
    feature: str
    mode: str = DELTA
    value: float = 0.0


@dataclass
class ScenarioConfig:
    changes: list[Change] = field(default_factory=list)
    linked: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    # 예: {"flow": [("dp", 0.6)]}  flow 를 1 단위 올리면 dp 도 0.6 단위 따라 오른다


def apply_changes(X: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """시나리오를 적용한 X 사본을 만든다."""
    out = X.copy()
    for ch in cfg.changes:
        if ch.feature not in out.columns:
            continue
        col = out[ch.feature]
        if ch.mode == SET:
            new = pd.Series(ch.value, index=out.index, dtype="float64")
            shift = new - col
        elif ch.mode == PCT:
            new = col * (1.0 + ch.value / 100.0)
            shift = new - col
        else:
            new = col + ch.value
            shift = pd.Series(ch.value, index=out.index, dtype="float64")
        out[ch.feature] = new

        for dep, coef in cfg.linked.get(ch.feature, []):
            if dep in out.columns:
                out[dep] = out[dep] + shift * coef
    return out


def run_scenario(pipeline, X: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """기준 예측과 시나리오 예측을 나란히 돌려준다."""
    base = np.asarray(pipeline.predict(X), dtype=float)
    Xw = apply_changes(X, cfg)
    what = np.asarray(pipeline.predict(Xw), dtype=float)
    return pd.DataFrame(
        {"baseline": base, "scenario": what, "delta": what - base}, index=X.index
    )


def scenario_summary(res: pd.DataFrame) -> dict[str, float]:
    d = res["delta"]
    return {
        "기준 평균": float(res["baseline"].mean()),
        "시나리오 평균": float(res["scenario"].mean()),
        "평균 변화": float(d.mean()),
        "변화율(%)": float(100 * d.mean() / res["baseline"].mean())
        if abs(res["baseline"].mean()) > 1e-12 else float("nan"),
        "최대 상승": float(d.max()),
        "최대 하락": float(d.min()),
    }


def sweep(
    pipeline, X: pd.DataFrame, feature: str, values: np.ndarray, agg: str = "mean"
) -> pd.DataFrame:
    """한 피처를 값 범위로 훑으며 예측 평균을 낸다 (PDP)."""
    if feature not in X.columns:
        raise KeyError(f"'{feature}' 가 X 에 없습니다.")
    rows = []
    for v in values:
        Xi = X.copy()
        Xi[feature] = v
        p = np.asarray(pipeline.predict(Xi), dtype=float)
        rows.append({
            feature: float(v),
            "prediction": float(getattr(np, agg)(p)),
            "p10": float(np.percentile(p, 10)),
            "p90": float(np.percentile(p, 90)),
        })
    return pd.DataFrame(rows)


def ice_curves(
    pipeline, X: pd.DataFrame, feature: str, values: np.ndarray,
    n_lines: int = 40, seed: int = 42
) -> pd.DataFrame:
    """개별 시점별 반응 곡선 (ICE). 평균 뒤에 가려진 이질성을 본다."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(X), size=min(n_lines, len(X)), replace=False)
    Xs = X.iloc[np.sort(pos)]
    frames = []
    for v in values:
        Xi = Xs.copy()
        Xi[feature] = v
        frames.append(pd.DataFrame({
            "row": range(len(Xs)),
            "timestamp": Xs.index,
            feature: float(v),
            "prediction": np.asarray(pipeline.predict(Xi), dtype=float),
        }))
    return pd.concat(frames, ignore_index=True)


def suggest_range(X: pd.DataFrame, feature: str, n: int = 25, pad: float = 0.1) -> np.ndarray:
    """관측 범위를 조금 넓혀 훑을 값을 만든다. 지나친 외삽은 막는다."""
    s = pd.to_numeric(X[feature], errors="coerce").dropna()
    if s.empty:
        return np.array([0.0])
    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
    if hi <= lo:
        lo, hi = float(s.min()), float(s.max()) or 1.0
    span = hi - lo
    return np.linspace(lo - span * pad, hi + span * pad, n)


def extrapolation_flag(X: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """시나리오 값이 학습 데이터 범위를 벗어났는지 표시한다."""
    rows = []
    Xw = apply_changes(X, cfg)
    for ch in cfg.changes:
        if ch.feature not in X.columns:
            continue
        obs = pd.to_numeric(X[ch.feature], errors="coerce").dropna()
        new = pd.to_numeric(Xw[ch.feature], errors="coerce").dropna()
        if obs.empty or new.empty:
            continue
        lo, hi = float(obs.min()), float(obs.max())
        out_ratio = float(((new < lo) | (new > hi)).mean())
        rows.append({
            "feature": ch.feature,
            "학습 범위": f"{lo:.4g} ~ {hi:.4g}",
            "시나리오 범위": f"{new.min():.4g} ~ {new.max():.4g}",
            "범위 밖 비율": round(out_ratio, 4),
            "판정": "범위 내" if out_ratio == 0 else ("일부 외삽" if out_ratio < 0.2 else "대부분 외삽"),
        })
    return pd.DataFrame(rows)


def optimize_single(
    pipeline, X: pd.DataFrame, feature: str, target_value: float,
    n_grid: int = 60, direction: str = "closest"
) -> dict:
    """한 피처를 훑어 목표치에 가장 가까워지는 지점을 찾는다.

    최적화라기보다 격자 탐색이다. 결과는 후보 검토용이다.
    """
    values = suggest_range(X, feature, n=n_grid)
    curve = sweep(pipeline, X, feature, values)
    if direction == "min":
        best = curve.loc[curve["prediction"].idxmin()]
    elif direction == "max":
        best = curve.loc[curve["prediction"].idxmax()]
    else:
        best = curve.loc[(curve["prediction"] - target_value).abs().idxmin()]
    return {"curve": curve, "best_value": float(best[feature]),
            "expected": float(best["prediction"])}
