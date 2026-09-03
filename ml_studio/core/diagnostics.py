"""잔차 진단.

예측이 얼마나 맞았는지는 R2 한 숫자로 끝나지만, **어디서 어떻게 틀렸는지**는
잔차를 시간축에 놓고 봐야 보인다. 이 모듈이 보는 것은 세 가지다.

1. rolling — 잔차의 국소 평균·표준편차. 특정 구간에서만 편향이 생기는지
2. drift   — 구간을 잘라 통계가 이동하는지. 모델이 낡아가는 신호
3. outlier — 개별 이상점. 설비 이벤트와 대조할 지점

SPEC §20 의 "잔차 기반 이상감지"를 나중에 붙이려면 잔차 계산이 예측 화면에서
분리돼 있어야 한다. 그래서 plots 나 view 가 아니라 core 모듈로 둔다.
UI 를 import 하지 않으므로 Dataiku 로 그대로 옮길 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ResidualConfig", "residuals", "rolling_stats", "drift_table",
    "outliers", "autocorrelation", "summary", "drift_verdict",
]


@dataclass
class ResidualConfig:
    window: int = 96              # rolling 창 (행). 5분 간격이면 8시간
    n_segments: int = 6           # drift 를 볼 구간 수
    outlier_sigma: float = 3.0    # 이상점 판정 기준 (robust z)
    max_lag: int = 24             # 자기상관을 볼 최대 지연


def residuals(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """실측 - 예측. 양수면 모델이 과소예측한 것이다."""
    a, p = actual.align(predicted, join="inner")
    return (a - p).dropna().rename("residual")


def rolling_stats(res: pd.Series, cfg: ResidualConfig | None = None) -> pd.DataFrame:
    """잔차의 국소 평균·표준편차·절대평균.

    평균이 0 에서 멀어지는 구간은 그 구간에서만 계통 편향이 있다는 뜻이다.
    """
    cfg = cfg or ResidualConfig()
    w = max(int(cfg.window), 2)
    roll = res.rolling(window=w, min_periods=max(2, w // 4))
    return pd.DataFrame({
        "residual": res,
        "roll_mean": roll.mean(),
        "roll_std": roll.std(),
        "roll_mae": res.abs().rolling(window=w, min_periods=max(2, w // 4)).mean(),
    })


def drift_table(res: pd.Series, cfg: ResidualConfig | None = None) -> pd.DataFrame:
    """구간을 n 등분해 잔차 통계가 이동하는지 본다.

    첫 구간 대비 배율을 함께 낸다. 배율이 커지면 모델이 뒤로 갈수록 못 맞춘다는 뜻이고,
    보통 운전조건 변화나 센서 드리프트를 의심한다.
    """
    cfg = cfg or ResidualConfig()
    n = len(res)
    k = max(int(cfg.n_segments), 2)
    if n < k * 2:
        return pd.DataFrame()

    bounds = np.linspace(0, n, k + 1).astype(int)
    rows = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        seg = res.iloc[lo:hi]
        rows.append({
            "구간": i + 1,
            "시작": seg.index[0], "끝": seg.index[-1], "행수": len(seg),
            "mean": float(seg.mean()), "std": float(seg.std()),
            "MAE": float(seg.abs().mean()),
            "p95_abs": float(seg.abs().quantile(0.95)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    base_mae = out["MAE"].iloc[0]
    base_std = out["std"].iloc[0]
    out["MAE_배율"] = (out["MAE"] / base_mae).round(3) if base_mae > 0 else np.nan
    out["std_배율"] = (out["std"] / base_std).round(3) if base_std and base_std > 0 else np.nan
    return out


def drift_verdict(table: pd.DataFrame, threshold: float = 1.5) -> dict:
    """drift 표를 한 줄 판정으로 요약한다."""
    if table.empty or "MAE_배율" not in table.columns:
        return {"drift": False, "message": "구간이 부족해 판정하지 않았습니다."}
    ratios = table["MAE_배율"].dropna()
    if ratios.empty:
        return {"drift": False, "message": "판정 불가."}
    worst = float(ratios.max())
    where = int(table.loc[ratios.idxmax(), "구간"])
    last = float(ratios.iloc[-1])
    hit = worst >= threshold
    return {
        "drift": hit,
        "worst_ratio": round(worst, 3),
        "worst_segment": where,
        "last_ratio": round(last, 3),
        "message": (
            f"{where}구간에서 MAE 가 첫 구간의 {worst:.2f}배입니다. "
            "운전조건 변화나 센서 드리프트를 의심해 보세요."
            if hit else
            f"구간별 MAE 변동이 최대 {worst:.2f}배로 임계({threshold:.1f}배) 안입니다."
        ),
    }


def outliers(res: pd.Series, cfg: ResidualConfig | None = None) -> pd.DataFrame:
    """robust z-score 로 이상점을 뽑는다.

    평균·표준편차 대신 중앙값·MAD 를 쓴다. 이상점 자체가 평균을 끌어당겨
    자기를 정상으로 만들어 버리는 것을 막기 위해서다.
    """
    cfg = cfg or ResidualConfig()
    med = float(res.median())
    mad = float((res - med).abs().median())
    scale = mad * 1.4826 if mad > 0 else float(res.std())
    if not scale or not np.isfinite(scale):
        return pd.DataFrame(columns=["residual", "robust_z", "방향"])

    z = (res - med) / scale
    hit = z.abs() >= cfg.outlier_sigma
    out = pd.DataFrame({
        "residual": res[hit].round(4),
        "robust_z": z[hit].round(3),
        "방향": np.where(z[hit] > 0, "과소예측", "과대예측"),
    })
    return out.sort_values("robust_z", key=np.abs, ascending=False)


def autocorrelation(res: pd.Series, cfg: ResidualConfig | None = None) -> pd.DataFrame:
    """잔차 자기상관.

    잔차가 백색잡음이면 lag>=1 의 상관이 0 근처여야 한다. 크게 남아 있으면
    아직 모델이 못 뽑아낸 시간 구조가 있다는 뜻이다.
    """
    cfg = cfg or ResidualConfig()
    v = res.to_numpy(dtype=float)
    v = v - v.mean()
    denom = float(np.dot(v, v))
    if denom <= 0:
        return pd.DataFrame(columns=["lag", "acf"])
    rows = []
    for lag in range(1, min(int(cfg.max_lag), len(v) - 1) + 1):
        rows.append({"lag": lag, "acf": round(float(np.dot(v[lag:], v[:-lag]) / denom), 4)})
    return pd.DataFrame(rows)


def series_autocorr(y: pd.Series, lag: int = 1) -> float:
    """타겟 자기상관. 높으면 무작위 분할이 이웃 행을 학습에 넣어 점수를 띄운다."""
    v = pd.Series(y).astype(float).dropna().to_numpy()
    if len(v) <= lag + 1:
        return float("nan")
    v = v - v.mean()
    denom = float(np.dot(v, v))
    if denom <= 0:
        return float("nan")
    return float(np.dot(v[lag:], v[:-lag]) / denom)


def distribution_drift(X: pd.DataFrame, y: pd.Series, n_segments: int = 3) -> dict:
    """앞 구간과 뒤 구간의 분포 차이. 표준편차 단위로 잰다."""
    n = len(y)
    if n < n_segments * 10:
        return {"y_shift_sd": float("nan"), "x_shift_sd": float("nan")}
    k = n // n_segments
    head, tail = y.iloc[:k].astype(float), y.iloc[-k:].astype(float)
    sd = float(y.astype(float).std()) or 1.0
    y_shift = abs(float(tail.mean()) - float(head.mean())) / sd

    num = X.select_dtypes("number")
    shifts = []
    for c in num.columns:
        col = num[c].astype(float)
        s = float(col.std())
        if not s or not np.isfinite(s):
            continue
        shifts.append(abs(float(col.iloc[-k:].mean()) - float(col.iloc[:k].mean())) / s)
    return {
        "y_shift_sd": round(y_shift, 4),
        "x_shift_sd": round(float(np.mean(shifts)), 4) if shifts else float("nan"),
        "x_shift_max": round(float(np.max(shifts)), 4) if shifts else float("nan"),
    }


def split_gap_causes(
    gap: float,
    y: pd.Series,
    X: pd.DataFrame,
    threshold: float = 0.15,
    acf_threshold: float = 0.5,
    drift_threshold: float = 0.5,
) -> dict:
    """Random 과 Time 의 격차가 어디서 왔는지 후보를 제시한다.

    격차 자체는 증상이다. 원인은 보통 셋 중 하나다.
      자기상관 — 무작위 분할이 검증 행의 이웃을 학습에 넣었다
      drift    — 뒤 구간의 분포가 달라 Time split 이 더 어려운 문제를 푼다
      lag mismatch — 파생 lag 가 실제 반응지연과 안 맞아 미래 구간에서만 어긋난다
    """
    acf1 = series_autocorr(y, 1)
    drift = distribution_drift(X, y)
    causes = []

    if acf1 == acf1 and acf1 >= acf_threshold:
        causes.append({
            "원인 후보": "자기상관",
            "근거": f"타겟 lag1 자기상관 {acf1:.3f} (>= {acf_threshold})",
            "설명": "무작위 분할이 검증 행 바로 앞뒤를 학습에 넣습니다. "
                   "Random 점수가 높은 것은 실력이 아니라 이웃을 본 결과입니다.",
        })
    ys = drift.get("y_shift_sd", float("nan"))
    if ys == ys and ys >= drift_threshold:
        causes.append({
            "원인 후보": "분포 drift",
            "근거": f"타겟 평균이 앞뒤 구간 사이 {ys:.2f}σ 이동",
            "설명": "뒤 구간이 학습 구간과 다른 조건입니다. Time split 이 실제로 "
                   "더 어려운 문제를 풀고 있으며, 이쪽이 현실에 가깝습니다.",
        })
    xs = drift.get("x_shift_max", float("nan"))
    if xs == xs and xs >= drift_threshold * 2:
        causes.append({
            "원인 후보": "입력 drift",
            "근거": f"어떤 X 피처의 평균이 {xs:.2f}σ 이동",
            "설명": "센서 교정이나 운전조건 변경을 의심해 보세요. "
                   "해당 피처를 찾아 구간별 통계를 확인하는 게 좋습니다.",
        })
    if gap >= threshold and not causes:
        causes.append({
            "원인 후보": "lag mismatch",
            "근거": f"격차 {gap:.3f} 인데 자기상관·drift 로는 설명되지 않음",
            "설명": "파생 lag 가 실제 반응지연과 어긋났을 수 있습니다. "
                   "lag 를 물리 시간(분)으로 다시 지정해 보세요.",
        })

    return {
        "gap": round(float(gap), 4),
        "threshold": threshold,
        "significant": bool(gap >= threshold),
        "lag1_acf": round(acf1, 4) if acf1 == acf1 else None,
        **drift,
        "causes": causes,
    }


def summary(res: pd.Series, cfg: ResidualConfig | None = None) -> dict:
    """화면 상단 지표용 한 줄 요약."""
    cfg = cfg or ResidualConfig()
    acf = autocorrelation(res, cfg)
    lag1 = float(acf["acf"].iloc[0]) if not acf.empty else float("nan")
    return {
        "n": int(len(res)),
        "mean": float(res.mean()),
        "std": float(res.std()),
        "MAE": float(res.abs().mean()),
        "bias_ratio": float(res.mean() / res.abs().mean()) if res.abs().mean() > 0 else 0.0,
        "lag1_acf": lag1,
        "outliers": int(len(outliers(res, cfg))),
    }
