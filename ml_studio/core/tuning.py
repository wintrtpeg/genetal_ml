"""하이퍼파라미터 탐색 — nested CV 로 누수 없이.

왜 nested 인가
-------------
같은 구간에서 파라미터를 고르고 그 구간 점수를 성능이라 부르면, 앞서 홀드아웃을
모델 선택과 최종 보고에 겸용했던 것과 **똑같은 편향**이 생긴다. 조합을 많이 볼수록
더 낙관적이 된다.

그래서 두 겹으로 나눈다.

    바깥 폴드 k        [ 학습 ][ 검증 ]
                        │        └ 이 점수만 보고한다
                        └ 이 안에서 다시 나눠 파라미터를 고른다
    안쪽 폴드 (학습 안) [ 학습 ][ 검증 ]  ← 파라미터 선택 전용

바깥 폴드의 검증 구간은 파라미터 선택에 한 번도 쓰이지 않는다. 그래서 바깥 점수가
"이 절차 전체"의 성능 추정이 된다. 튜닝을 포함한 절차 전체를 평가하는 것이 핵심이다.

sklearn 의 GridSearchCV / RandomizedSearchCV 는 TimeSeriesSplit 을 그대로 받는다.
(Stacking 이 죽었던 이유인 cross_val_predict 를 쓰지 않기 때문이다.)

비용
----
탐색 조합 수 × 안쪽 폴드 수만큼 학습이 늘어난다. 기본은 RandomizedSearch 로
조합 수를 묶어 두고, 그리드는 모델별로 좁게 잡았다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

__all__ = ["TuneConfig", "param_grid", "make_search", "nested_scores", "tunable"]


@dataclass
class TuneConfig:
    enabled: bool = False
    n_iter: int = 12              # RandomizedSearch 조합 수
    inner_splits: int = 3         # 안쪽 폴드 수 (파라미터 선택용)
    seed: int = 42
    scoring: str | None = None    # None 이면 회귀 r2 / 분류 f1_macro
    models: list[str] = field(default_factory=list)   # 빈 목록이면 전부

    def applies_to(self, name: str) -> bool:
        return self.enabled and (not self.models or name in self.models)


# 모델별 탐색 범위. Pipeline 안의 추정기를 가리키므로 "est__" 접두사를 붙인다.
# 넓게 잡으면 비용만 늘고 과적합 위험이 커진다. 실무에서 효과가 큰 축만 남겼다.
_GRIDS: dict[str, dict] = {
    "Ridge": {"est__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "ElasticNet": {"est__alpha": [0.005, 0.01, 0.05, 0.1, 0.5],
                   "est__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "DecisionTree": {"est__max_depth": [4, 6, 8, 12, None],
                     "est__min_samples_leaf": [1, 2, 5, 10, 20]},
    "RandomForest": {"est__n_estimators": [200, 400, 800],
                     "est__max_depth": [8, 16, None],
                     "est__min_samples_leaf": [1, 2, 5],
                     "est__max_features": ["sqrt", 0.5, 1.0]},
    "ExtraTrees": {"est__n_estimators": [200, 400, 800],
                   "est__max_depth": [8, 16, None],
                   "est__min_samples_leaf": [1, 2, 5],
                   "est__max_features": ["sqrt", 0.5, 1.0]},
    "HistGradientBoosting": {"est__learning_rate": [0.02, 0.05, 0.1],
                             "est__max_iter": [200, 400, 800],
                             "est__max_leaf_nodes": [15, 31, 63],
                             "est__min_samples_leaf": [10, 20, 50],
                             "est__l2_regularization": [0.0, 0.1, 1.0]},
    "XGBoost": {"est__learning_rate": [0.02, 0.05, 0.1],
                "est__max_depth": [3, 6, 9],
                "est__n_estimators": [300, 600, 1000],
                "est__subsample": [0.7, 0.85, 1.0],
                "est__colsample_bytree": [0.6, 0.8, 1.0],
                "est__reg_lambda": [0.5, 1.0, 5.0]},
    "LightGBM": {"est__learning_rate": [0.02, 0.05, 0.1],
                 "est__num_leaves": [31, 63, 127],
                 "est__n_estimators": [300, 700, 1200],
                 "est__subsample": [0.7, 0.85, 1.0],
                 "est__colsample_bytree": [0.6, 0.8, 1.0],
                 "est__min_child_samples": [10, 20, 50]},
    "CatBoost": {"est__learning_rate": [0.02, 0.05, 0.1],
                 "est__depth": [4, 6, 8],
                 "est__iterations": [300, 700, 1200],
                 "est__l2_leaf_reg": [1.0, 3.0, 9.0]},
    "KNN": {"est__n_neighbors": [5, 10, 15, 25, 40],
            "est__weights": ["uniform", "distance"]},
    "SVM": {"est__C": [1.0, 10.0, 100.0], "est__gamma": ["scale", "auto"]},
    "MLP": {"est__hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
            "est__alpha": [1e-5, 1e-4, 1e-3],
            "est__learning_rate_init": [1e-3, 5e-3]},
    "LogisticRegression": {"est__C": [0.01, 0.1, 1.0, 10.0]},
}


def tunable(name: str) -> bool:
    return name in _GRIDS


def param_grid(name: str) -> dict:
    """모델 이름에 맞는 탐색 범위. 없으면 빈 사전 (탐색 건너뜀)."""
    return dict(_GRIDS.get(name, {}))


def make_search(
    pipe,
    name: str,
    cfg: TuneConfig,
    task: str = "regression",
    gap: int = 0,
    n_jobs: int = 1,
):
    """파이프라인을 RandomizedSearchCV 로 감싼다.

    cv 는 반드시 TimeSeriesSplit 이다. KFold 를 쓰면 미래 구간을 보고 파라미터를
    고르게 되어 튜닝 단계에서 누수가 생긴다.
    """
    grid = param_grid(name)
    if not grid:
        return None
    inner = TimeSeriesSplit(n_splits=max(int(cfg.inner_splits), 2), gap=gap)
    scoring = cfg.scoring or ("r2" if task == "regression" else "f1_macro")
    # 조합 수보다 많이 뽑으려 하면 sklearn 이 경고를 낸다. 격자 크기로 잘라 둔다.
    total = 1
    for v in grid.values():
        total *= max(len(v), 1)
    n_iter = max(min(int(cfg.n_iter), total), 1)
    return RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, cv=inner, scoring=scoring,
        random_state=cfg.seed, n_jobs=n_jobs, refit=True, error_score="raise",
    )


def nested_scores(
    make_pipe,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv,
    name: str,
    cfg: TuneConfig,
    score_fn,
    task: str = "regression",
) -> tuple[list[dict], list[dict], pd.Series]:
    """바깥 폴드마다 안쪽에서 파라미터를 고르고, 바깥 검증으로만 점수를 낸다.

    반환: (폴드별 점수, 폴드별 선택 파라미터, OOF 예측)

    바깥 검증 구간은 파라미터 선택에 한 번도 쓰이지 않으므로, 여기서 나온 점수가
    "튜닝을 포함한 절차 전체"의 편향 없는 추정이다.
    """
    from .validation import assert_temporal_order

    rows: list[dict] = []
    chosen: list[dict] = []
    oof = pd.Series(index=X.index, dtype="float64")
    gap = getattr(outer_cv, "gap", 0)

    for tr, va in outer_cv.split(X):
        assert_temporal_order(X.index, tr, va, gap=gap)
        search = make_search(make_pipe(), name, cfg, task=task, gap=gap)
        if search is None:
            return [], [], oof
        search.fit(X.iloc[tr], y.iloc[tr])          # 안쪽 탐색은 학습 구간 안에서만
        pred = search.best_estimator_.predict(X.iloc[va])
        rows.append(score_fn(y.iloc[va], pred))
        chosen.append({k.replace("est__", ""): v for k, v in search.best_params_.items()})
        if task == "regression":
            oof.iloc[va] = np.asarray(pred, dtype=float)

    return rows, chosen, oof


def stability(chosen: list[dict]) -> pd.DataFrame:
    """폴드마다 고른 파라미터가 얼마나 달라지는가.

    폴드마다 전혀 다른 값이 뽑히면 그 파라미터는 데이터가 결정해 주지 않는다는 뜻이고,
    튜닝 결과를 그대로 믿기 어렵다.
    """
    if not chosen:
        return pd.DataFrame()
    keys = sorted({k for d in chosen for k in d})
    rows = []
    for k in keys:
        vals = [d.get(k) for d in chosen]
        uniq = {str(v) for v in vals}
        rows.append({
            "파라미터": k,
            "폴드별 선택": ", ".join(str(v) for v in vals),
            "서로 다른 값": len(uniq),
            "안정성": "일관" if len(uniq) == 1 else ("보통" if len(uniq) <= 2 else "흔들림"),
        })
    return pd.DataFrame(rows)
