"""시계열 OOF 를 직접 만들어 스태킹·가중앙상블을 구성한다.

sklearn 의 StackingRegressor 를 쓰지 않는 이유
--------------------------------------------
StackingRegressor 는 내부에서 cross_val_predict 를 부른다. 이 함수는 모든 샘플이
정확히 한 번씩 validation 에 들어가는 분할만 허용한다. TimeSeriesSplit 은 선두
n/(k+1) 구간이 어느 fold 의 validation 에도 안 들어가므로 항상 예외가 난다.

    ValueError: cross_val_predict only works for partitions

그래서 OOF 를 직접 만든다. 대신 얻는 것이 두 가지 있다.

1. train.＿fit_one 이 이미 만들어 둔 _oof 를 재사용할 수 있다. base 모델을
   다시 학습하지 않으므로 앙상블이 사실상 공짜가 된다.
2. 같은 OOF 행렬로 가중앙상블(SPEC §15)까지 만들 수 있다. nnls 로 음수 없는
   가중치를 구하면 "어느 모델을 얼마나 믿는지"가 그대로 읽힌다.

누수 방지
--------
- OOF 의 각 행 t 는 t 이전 구간만으로 학습된 모델의 예측이다 (TimeSeriesSplit 은
  forward-only). 메타 학습기는 이 행렬만 본다.
- 선두 NaN 구간(어느 fold 의 validation 에도 안 들어간 구간)은 메타 학습에서 뺀다.
- 메타 학습기는 검증·Final Unseen 구간을 절대 보지 않는다.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .models import TASK_CLASSIFICATION, TASK_REGRESSION

__all__ = [
    "compute_oof", "oof_matrix", "OofStack", "WeightedBlend",
    "fit_ensembles", "adopt_ensemble",
]


# ─────────────────────────────────────────────────────────────
# OOF 수집
# ─────────────────────────────────────────────────────────────
def compute_oof(
    estimator,
    preprocessor,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cv,
) -> pd.Series:
    """한 모델의 시계열 OOF 예측을 만든다. 선두 미커버 구간은 NaN 으로 남는다."""
    from .validation import assert_temporal_order

    oof = pd.Series(index=X_tr.index, dtype="float64")
    pipe = Pipeline([("prep", clone(preprocessor)), ("est", clone(estimator))])
    for tr, va in cv.split(X_tr):
        assert_temporal_order(X_tr.index, tr, va, gap=getattr(cv, "gap", 0))
        fold = clone(pipe)
        fold.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        oof.iloc[va] = np.asarray(fold.predict(X_tr.iloc[va]), dtype=float)
    return oof


def oof_matrix(
    detail: dict[str, dict],
    base_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """train.train_all 이 남긴 _oof 를 모아 메타 학습용 행렬을 만든다.

    선두 NaN 행은 통째로 버린다. 어느 base 라도 NaN 이면 그 행은 못 쓴다.
    """
    cols: dict[str, pd.Series] = {}
    used: list[str] = []
    for n in base_names:
        rec = detail.get(n)
        if not rec or rec.get("status") != "ok":
            continue
        s = rec.get("_oof")
        if s is None or not isinstance(s, pd.Series) or s.notna().sum() == 0:
            continue
        cols[n] = s.astype("float64")
        used.append(n)
    if len(used) < 2:
        return pd.DataFrame(), used
    P = pd.DataFrame(cols)
    return P.dropna(axis=0, how="any"), used


# ─────────────────────────────────────────────────────────────
# 앙상블 추정기
# ─────────────────────────────────────────────────────────────
class _BaseBlend:
    """이미 학습된 base 파이프라인들을 감싸 predict 만 제공한다.

    base 를 다시 fit 하지 않는다. train_all 이 학습 구간 전체로 fit 해 둔
    _pipeline 을 그대로 쓴다.
    """

    def __init__(self, pipelines: dict[str, object], task: str = TASK_REGRESSION):
        self.pipelines = pipelines
        self.task = task
        self.member_names_ = list(pipelines)

    def _stack_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """base 예측을 메타 입력 행렬로 쌓는다. 컬럼 이름은 OOF 행렬과 같게 둔다."""
        return pd.DataFrame(
            {n: np.asarray(self.pipelines[n].predict(X), dtype=float)
             for n in self.member_names_},
            columns=self.member_names_,
            index=getattr(X, "index", None),
        )

    def fit(self, X=None, y=None):  # noqa: D102 - base 는 이미 학습돼 있다
        return self


class OofStack(_BaseBlend):
    """OOF 로 학습한 메타 모델이 base 예측을 결합한다."""

    def __init__(self, pipelines, meta, task: str = TASK_REGRESSION):
        super().__init__(pipelines, task)
        self.meta = meta
        self.meta_rows_: int = 0
        self.meta_span_: tuple | None = None

    def fit_meta(self, P: pd.DataFrame, y: pd.Series) -> "OofStack":
        P = P[self.member_names_]
        yy = y.loc[P.index]
        mask = yy.notna()
        if mask.sum() < 10:
            raise ValueError("메타 학습에 쓸 OOF 행이 부족합니다.")
        self.meta.fit(P.loc[mask], yy.loc[mask])
        self.meta_rows_ = int(mask.sum())
        self.meta_span_ = (P.index[mask][0], P.index[mask][-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta.predict(self._stack_predictions(X)), dtype=float)


class WeightedBlend(_BaseBlend):
    """음수 없는 가중치로 base 예측을 섞는다. 가중치 합은 1."""

    def __init__(self, pipelines, task: str = TASK_REGRESSION):
        super().__init__(pipelines, task)
        self.weights_: np.ndarray | None = None

    def fit_weights(self, P: pd.DataFrame, y: pd.Series) -> "WeightedBlend":
        from scipy.optimize import nnls

        P = P[self.member_names_]
        yy = y.loc[P.index]
        mask = yy.notna()
        A = P.loc[mask].to_numpy(dtype=float)
        b = yy.loc[mask].to_numpy(dtype=float)
        w, _ = nnls(A, b)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(A.shape[1], dtype=float)      # 전부 0 이면 단순 평균으로
        self.weights_ = w / w.sum()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._stack_predictions(X).to_numpy(dtype=float) @ self.weights_

    def weight_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model": self.member_names_,
            "weight": np.round(self.weights_, 4),
        }).sort_values("weight", ascending=False, ignore_index=True)


class MeanBlend(_BaseBlend):
    """단순 평균 (기존 Voting 과 같은 값). base 재학습 없이 계산한다."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._stack_predictions(X).to_numpy(dtype=float).mean(axis=1)


# ─────────────────────────────────────────────────────────────
# 조립
# ─────────────────────────────────────────────────────────────
def fit_ensembles(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    detail: dict[str, dict],
    base_names: list[str],
    task: str = TASK_REGRESSION,
    include_voting: bool = True,
    include_stacking: bool = True,
    include_weighted: bool = True,
    score_fn=None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """OOF 를 재사용해 앙상블 3종을 만들고 같은 잣대로 평가한다.

    base 모델을 다시 학습하지 않는다. train_all 결과(detail)를 그대로 쓴다.
    """
    if task == TASK_CLASSIFICATION:
        return pd.DataFrame(), {}      # 분류 OOF 는 아직 회귀 경로만 지원

    P, used = oof_matrix(detail, base_names)
    if P.empty or len(used) < 2:
        return pd.DataFrame(), {}

    pipelines = {n: detail[n]["_pipeline"] for n in used}
    y_tr = y.iloc[train_idx]
    X_ev, y_ev = X.iloc[eval_idx], y.iloc[eval_idx]

    specs: dict[str, object] = {}
    if include_voting:
        specs["Ensemble_Voting"] = MeanBlend(pipelines, task)
    if include_weighted:
        specs["Ensemble_Weighted"] = WeightedBlend(pipelines, task).fit_weights(P, y_tr)
    if include_stacking:
        from sklearn.linear_model import RidgeCV
        specs["Ensemble_Stacking"] = OofStack(
            pipelines, RidgeCV(alphas=np.logspace(-3, 3, 13)), task
        ).fit_meta(P, y_tr)

    results = []
    for name, est in specs.items():
        t0 = time.perf_counter()
        row: dict = {"model": name, "family": BLEND_FAMILY,
                     "members": ", ".join(used), "oof_rows": int(len(P))}
        try:
            p_ev = est.predict(X_ev)
            for k, v in score_fn(y_ev, p_ev).items():
                row[f"holdout_{k}"] = v
            p_tr = est.predict(X.iloc[train_idx])
            for k, v in score_fn(y_tr, p_tr).items():
                row[f"insample_{k}"] = v
            if isinstance(est, WeightedBlend):
                row["weights"] = "; ".join(
                    f"{n}={w:.3f}" for n, w in zip(used, est.weights_) if w > 1e-6)
            row.update(status="ok", fit_seconds=round(time.perf_counter() - t0, 2))
            row["_pipeline"] = est
            row["_holdout_pred"] = pd.Series(p_ev, index=X_ev.index)
        except Exception as e:  # noqa: BLE001
            row.update(status="failed", error=f"{type(e).__name__}: {e}",
                       fit_seconds=round(time.perf_counter() - t0, 2))
        results.append(row)

    det = {r["model"]: r for r in results}
    board = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in results])
    return board, det


BLEND_FAMILY = "blend"


def _is_blend(board: pd.DataFrame) -> pd.Series:
    """리더보드에서 '우리가 만든 결합 모델' 만 골라낸다.

    sklearn 분류에서 RandomForest·ExtraTrees 의 family 도 'ensemble' 이다.
    그 이름으로 판정하면 배깅 모델이 결합 모델로 분류되어, 단일 최고 모델과의
    비교 기준 자체가 틀어진다. family='blend' 로 구분하고, 예전에 저장된
    리더보드를 위해 이름 접두사도 함께 본다.
    """
    fam = board["family"].astype(str) if "family" in board.columns else pd.Series(
        "", index=board.index)
    name = board["model"].astype(str) if "model" in board.columns else pd.Series(
        "", index=board.index)
    return (fam == BLEND_FAMILY) | name.str.startswith("Ensemble_")


def adopt_ensemble(
    board: pd.DataFrame,
    metric: str,
    threshold: float = 0.03,
    prefix: str = "cv_",
) -> tuple[str | None, pd.DataFrame]:
    """SPEC §17 — 미미한 개선이면 복잡한 앙상블을 자동 선택하지 않는다.

    단일 최고 모델 대비 threshold(기본 3%) 이상 좋아진 앙상블만 채택한다.
    판정 근거를 표로 함께 돌려준다. 임계값은 화면에서 조절한다.
    """
    from .train import HIGHER_IS_BETTER

    col = f"{prefix}{metric}"
    ok = board[board.get("status", "ok") == "ok"] if "status" in board.columns else board
    if ok.empty or col not in ok.columns:
        return None, pd.DataFrame()

    higher = HIGHER_IS_BETTER.get(metric, True)
    is_blend = _is_blend(ok)
    singles = ok[~is_blend]
    ens = ok[is_blend]
    if singles.empty:
        return (str(ens.iloc[0]["model"]) if not ens.empty else None), pd.DataFrame()

    best_single = singles.sort_values(col, ascending=not higher, na_position="last").iloc[0]
    base = float(best_single[col])
    champion = str(best_single["model"])

    rows = [{"model": champion, "기준": round(base, 6), "개선율": 0.0,
             "판정": "단일 최고 (기준)"}]
    best_gain = 0.0
    for _, r in ens.iterrows():
        v = r.get(col)
        if pd.isna(v):
            continue
        # 높을수록 좋은 지표는 상대증가, 낮을수록 좋은 지표는 상대감소
        gain = ((float(v) - base) / abs(base)) if higher else ((base - float(v)) / abs(base))
        adopt = gain >= threshold
        rows.append({"model": str(r["model"]), "기준": round(float(v), 6),
                     "개선율": round(gain, 4),
                     "판정": "채택" if adopt else f"기각 (임계 {threshold:.0%} 미달)"})
        if adopt and gain > best_gain:
            best_gain, champion = gain, str(r["model"])

    return champion, pd.DataFrame(rows)
