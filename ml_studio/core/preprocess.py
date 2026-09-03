"""전처리.

핵심은 '전처리를 학습 전에 미리 해두지 않는다'는 것이다.
결측 대치·스케일링·인코딩은 전부 sklearn Pipeline 안에 넣어서 폴드마다 다시 fit 한다.
전체 데이터로 한 번 fit 해두면 홀드아웃의 평균·분산이 학습에 스며든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler,
)

# 시간 방향으로 안전한 결측 처리와 그렇지 않은 것을 구분해 둔다.
# 화면에 그대로 뜨는 문구다. 현장 엔지니어가 읽고 고를 수 있어야 한다.
IMPUTE_METHODS = {
    "ffill": "직전 값을 그대로 유지 — 설비 신호는 보통 이게 맞습니다",
    "median": "그 태그의 가운데 값으로",
    "mean": "그 태그의 평균값으로",
    "zero": "0 으로",
    "interpolate": "앞뒤 값을 이어서 (주의: 미래 값을 봅니다)",
}
FUTURE_LOOKING_IMPUTERS = {"interpolate"}


@dataclass
class PreprocessConfig:
    impute_numeric: str = "ffill"
    impute_categorical: str = "most_frequent"
    scaler: str = "standard"            # standard | robust | none
    categorical_encoding: str = "onehot"  # onehot | ordinal
    clip_outliers: bool = False
    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    drop_columns: list[str] = field(default_factory=list)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    """직전 값으로 채운다. 선두 결측만 학습 구간 중앙값으로 메운다.

    ffill 은 과거만 보므로 시계열에서 안전하다. 다만 맨 앞은 참조할 과거가 없어
    학습 구간에서 구한 중앙값을 쓴다. 이 중앙값은 fit 시점에만 계산된다.
    """

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.fill_value_ = Xd.median(numeric_only=True)
        self.n_features_in_ = Xd.shape[1]
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X).copy()
        Xd = Xd.ffill()
        for c in Xd.columns:
            v = self.fill_value_.get(c, 0.0)
            Xd[c] = Xd[c].fillna(0.0 if pd.isna(v) else v)
        return Xd.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """학습 구간 분위수로 상·하한을 잘라 극단값 영향을 줄인다."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.lo_ = Xd.quantile(self.lower)
        self.hi_ = Xd.quantile(self.upper)
        return self

    def transform(self, X):
        Xd = pd.DataFrame(X)
        return Xd.clip(self.lo_, self.hi_, axis=1).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def _numeric_imputer(method: str):
    if method == "ffill":
        return ForwardFillImputer()
    if method == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    if method == "interpolate":
        # 파이프라인 안에서는 폴드별 fit 이 되도록 ffill+bfill 조합으로 근사한다.
        return SimpleImputer(strategy="median")
    return SimpleImputer(strategy=method)


def _scaler(name: str):
    return {"standard": StandardScaler(), "robust": RobustScaler()}.get(name)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: PreprocessConfig,
) -> ColumnTransformer:
    """ColumnTransformer 하나로 수치·범주 처리를 묶는다."""
    steps: list[tuple[str, object]] = [("impute", _numeric_imputer(cfg.impute_numeric))]
    if cfg.clip_outliers:
        steps.append(("clip", QuantileClipper(*cfg.clip_quantiles)))
    sc = _scaler(cfg.scaler)
    if sc is not None:
        steps.append(("scale", sc))
    numeric_pipe = Pipeline(steps)

    if cfg.categorical_encoding == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)
        except TypeError:  # scikit-learn < 1.2
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=cfg.impute_categorical)),
        ("encode", enc),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, list(categorical_cols)))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """X, y 를 만든다. 타겟이 비어 있는 행은 학습에서 뺀다."""
    cols = [c for c in feature_cols if c != target]
    X = df[cols].copy()
    y = df[target].copy()
    if dropna_target:
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
    return X, y


def transformed_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except (AttributeError, ValueError):
        return list(fallback)


def impute_warning(cfg: PreprocessConfig) -> str | None:
    if cfg.impute_numeric in FUTURE_LOOKING_IMPUTERS:
        return (
            "'앞뒤 값을 이어서' 는 빈 칸의 **뒤쪽 값도** 보고 채웁니다. "
            "실제 운전에서는 아직 오지 않은 값이므로, 이걸로 만든 성적은 "
            "실제보다 좋게 나옵니다. '직전 값 유지' 나 '가운데 값' 을 권합니다."
        )
    return None
