"""모델 목록.

설치돼 있는 라이브러리만 골라 담는다. XGBoost/LightGBM/CatBoost 가 없어도
sklearn 만으로 전체 흐름이 돌아간다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

TASK_REGRESSION = "regression"
TASK_CLASSIFICATION = "classification"


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    family: str                  # linear | tree | ensemble | boosting | kernel | neural
    supports_shap_tree: bool = False
    heavy: bool = False          # 대용량에서 느린 모델


def _try(import_path: str):
    try:
        module = __import__(import_path, fromlist=["*"])
        return module
    except Exception:  # noqa: BLE001 - 미설치 라이브러리를 조용히 건너뛴다
        return None


def detect_task(y: pd.Series, threshold: int = 15) -> str:
    """Y 성격으로 회귀·분류를 자동 판별한다."""
    s = y.dropna()
    if s.empty:
        return TASK_REGRESSION
    if not pd.api.types.is_numeric_dtype(s):
        return TASK_CLASSIFICATION
    nun = s.nunique()
    if nun <= 2:
        return TASK_CLASSIFICATION
    if nun <= threshold and np.allclose(s.to_numpy(dtype=float) % 1, 0):
        return TASK_CLASSIFICATION
    return TASK_REGRESSION


def get_model_zoo(
    task: str,
    seed: int = 42,
    include_heavy: bool = True,
    n_jobs_model: int = 1,
) -> dict[str, ModelSpec]:
    """이름 -> ModelSpec 사전."""
    zoo: dict[str, ModelSpec] = {}
    reg = task == TASK_REGRESSION

    def put(spec: ModelSpec) -> None:
        zoo[spec.name] = spec

    if reg:
        put(ModelSpec("Ridge", Ridge(alpha=1.0, random_state=seed), "linear"))
        put(ModelSpec("ElasticNet", ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=seed,
                                               max_iter=5000), "linear"))
    else:
        put(ModelSpec("LogisticRegression",
                      LogisticRegression(max_iter=2000, random_state=seed), "linear"))

    put(ModelSpec("DecisionTree",
                  (DecisionTreeRegressor if reg else DecisionTreeClassifier)(
                      max_depth=8, random_state=seed), "tree", True))
    put(ModelSpec("RandomForest",
                  (RandomForestRegressor if reg else RandomForestClassifier)(
                      n_estimators=400, min_samples_leaf=2, random_state=seed,
                      n_jobs=n_jobs_model), "ensemble", True))
    put(ModelSpec("ExtraTrees",
                  (ExtraTreesRegressor if reg else ExtraTreesClassifier)(
                      n_estimators=400, min_samples_leaf=2, random_state=seed,
                      n_jobs=n_jobs_model), "ensemble", True))
    put(ModelSpec("HistGradientBoosting",
                  (HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier)(
                      max_iter=400, learning_rate=0.05, random_state=seed), "boosting"))

    if (xgb := _try("xgboost")) is not None:
        cls = xgb.XGBRegressor if reg else xgb.XGBClassifier
        put(ModelSpec("XGBoost", cls(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=n_jobs_model, tree_method="hist",
            verbosity=0), "boosting", True))

    if (lgb := _try("lightgbm")) is not None:
        cls = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
        put(ModelSpec("LightGBM", cls(
            n_estimators=700, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=n_jobs_model, verbose=-1), "boosting", True))

    if (cat := _try("catboost")) is not None:
        cls = cat.CatBoostRegressor if reg else cat.CatBoostClassifier
        put(ModelSpec("CatBoost", cls(
            iterations=700, learning_rate=0.05, depth=6,
            random_seed=seed, verbose=0, allow_writing_files=False), "boosting", True))

    if include_heavy:
        put(ModelSpec("KNN", (KNeighborsRegressor if reg else KNeighborsClassifier)(
            n_neighbors=15, n_jobs=n_jobs_model), "kernel", heavy=True))
        put(ModelSpec("SVM", (SVR if reg else SVC)(C=10.0, gamma="scale"), "kernel", heavy=True))
        put(ModelSpec("MLP", (MLPRegressor if reg else MLPClassifier)(
            hidden_layer_sizes=(128, 64), max_iter=600, early_stopping=True,
            random_state=seed), "neural", heavy=True))

    return zoo


def default_selection(zoo: dict[str, ModelSpec], n_rows: int) -> list[str]:
    """데이터 크기에 맞춰 기본 체크 상태를 정한다."""
    names = list(zoo)
    if n_rows > 200_000:
        return [n for n in names if zoo[n].family in ("linear", "boosting")]
    if n_rows > 50_000:
        return [n for n in names if not zoo[n].heavy or zoo[n].family == "ensemble"]
    return names
