"""데이터 품질 진단.

전처리 전에 '이 컬럼을 학습에 쓸 수 있는가'를 판정한다.
제외 판단은 여기서 근거와 함께 만들고, 실제 제외는 사용자가 화면에서 확정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class QualityRule:
    max_missing_ratio: float = 0.30      # 결측 비율이 이보다 크면 제외 후보
    min_unique: int = 2                  # 고유값이 이보다 적으면 상수 취급
    max_dominant_ratio: float = 0.98     # 한 값이 이 비율 이상을 차지하면 준상수
    max_corr: float = 0.95               # 피처 간 상관이 이보다 높으면 중복 후보
    max_categorical_levels: int = 50     # 범주 수준이 너무 많으면 제외 후보


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 요약표. 화면에 그대로 띄울 수 있는 형태로 돌려준다."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        miss = float(s.isna().mean()) if n else 1.0
        nun = int(s.nunique(dropna=True))

        dominant = np.nan
        vc = s.value_counts(dropna=True)
        if not vc.empty and n:
            dominant = float(vc.iloc[0] / max(int(s.notna().sum()), 1))

        rec: dict = {
            "column": col,
            "dtype": str(s.dtype),
            "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else (
                "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "categorical"
            ),
            "missing_ratio": round(miss, 4),
            "n_unique": nun,
            "dominant_ratio": round(dominant, 4) if dominant == dominant else np.nan,
        }

        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce").dropna()
            if len(sn):
                q1, q3 = sn.quantile(0.25), sn.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    out = ((sn < q1 - 1.5 * iqr) | (sn > q3 + 1.5 * iqr)).mean()
                else:
                    out = 0.0
                rec.update(
                    mean=float(sn.mean()), std=float(sn.std()),
                    min=float(sn.min()), max=float(sn.max()),
                    outlier_ratio=round(float(out), 4),
                    flatline_ratio=round(float((sn.diff() == 0).mean()), 4),
                )
        rows.append(rec)

    out = pd.DataFrame(rows)
    order = ["column", "kind", "dtype", "missing_ratio", "n_unique", "dominant_ratio",
             "outlier_ratio", "flatline_ratio", "mean", "std", "min", "max"]
    return out[[c for c in order if c in out.columns]]


def find_correlated_pairs(df: pd.DataFrame, threshold: float = 0.95,
                          max_rows: int | None = None) -> pd.DataFrame:
    """상관이 임계값을 넘는 수치 피처 쌍. 둘 중 결측이 많은 쪽을 제외 후보로 제안.

    max_rows 를 주면 그 이상일 때 등간격 표본으로 상관을 잰다. 결측이 섞인
    DataFrame.corr() 는 쌍마다 따로 도는 느린 경로로 빠져서, 50만 행이면 화면이
    멈춘 것처럼 느껴진다. 여기 상관은 '중복 후보 제안'에 쓰는 값이고 0.95 수준의
    중복은 표본으로도 그대로 보이므로, 정확도보다 응답성을 택한다.
    (모델 학습에 쓰는 상관은 core/features.py 쪽이며 그쪽은 표본을 쓰지 않는다.)
    """
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "suggest_drop"])

    if max_rows and len(num) > max_rows:
        num = num.iloc[:: max(1, len(num) // max_rows)]

    corr = num.corr(numeric_only=True).abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack()
    pairs = pairs[pairs >= threshold].sort_values(ascending=False)

    miss = num.isna().mean()
    rows = []
    for (a, b), v in pairs.items():
        drop = a if miss.get(a, 0) >= miss.get(b, 0) else b
        rows.append({"feature_a": a, "feature_b": b, "corr": round(float(v), 4), "suggest_drop": drop})
    return pd.DataFrame(rows)


def suggest_drops(
    prof: pd.DataFrame,
    rule: QualityRule,
    protect: list[str] | None = None,
    corr_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """제외 후보와 사유. protect 에 담긴 컬럼(타겟 등)은 절대 후보에 넣지 않는다."""
    protect = set(protect or [])
    rows = []

    for _, r in prof.iterrows():
        col = r["column"]
        if col in protect:
            continue
        reasons = []
        if r["missing_ratio"] > rule.max_missing_ratio:
            reasons.append(f"결측 {r['missing_ratio']:.1%}")
        if r["n_unique"] < rule.min_unique:
            reasons.append("상수 컬럼")
        dom = r.get("dominant_ratio", np.nan)
        if dom == dom and dom > rule.max_dominant_ratio and r["n_unique"] >= rule.min_unique:
            reasons.append(f"단일값 편중 {dom:.1%}")
        fl = r.get("flatline_ratio", np.nan)
        if fl == fl and fl > 0.99:
            reasons.append("값 변화 없음(플랫라인)")
        if r["kind"] == "categorical" and r["n_unique"] > rule.max_categorical_levels:
            reasons.append(f"범주 {int(r['n_unique'])}종")
        if reasons:
            rows.append({"column": col, "reason": ", ".join(reasons), "source": "품질"})

    if corr_pairs is not None and not corr_pairs.empty:
        for _, r in corr_pairs.iterrows():
            col = r["suggest_drop"]
            if col in protect:
                continue
            other = r["feature_b"] if col == r["feature_a"] else r["feature_a"]
            rows.append({"column": col, "reason": f"{other} 와 상관 {r['corr']:.2f}", "source": "중복"})

    out = pd.DataFrame(rows, columns=["column", "reason", "source"])
    if out.empty:
        return out
    return (out.groupby("column", as_index=False)
               .agg({"reason": lambda x: " / ".join(dict.fromkeys(x)), "source": "first"}))


def gap_report(index: pd.DatetimeIndex, expected_freq: str | None) -> pd.DataFrame:
    """시간축 결손 구간. 예상 주기의 2배를 넘는 간격만 보고한다."""
    if expected_freq is None or len(index) < 3:
        return pd.DataFrame(columns=["start", "end", "duration"])
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_freq))
    except (ValueError, TypeError):
        step = pd.Series(index).diff().median()
    if pd.isna(step) or step == pd.Timedelta(0):
        return pd.DataFrame(columns=["start", "end", "duration"])

    d = pd.Series(index).diff()
    hit = d > step * 2
    rows = [
        {"start": index[i - 1], "end": index[i], "duration": str(d.iloc[i])}
        for i in np.flatnonzero(hit.to_numpy())
    ]
    return pd.DataFrame(rows, columns=["start", "end", "duration"])


def rule_to_dict(rule: QualityRule) -> dict:
    return asdict(rule)
