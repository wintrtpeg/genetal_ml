"""여러 모델을 같은 조건으로 돌려 리더보드를 만들고 챔피언을 고른다.

평가는 두 단계다.
1. 학습 구간 안에서 rolling origin 교차검증 -> 모델 선택의 근거
2. 한 번도 쓰지 않은 홀드아웃 구간 -> 최종 성능 보고

모든 모델은 (전처리 + 추정기) 파이프라인으로 감싸서 폴드마다 전처리를 다시 fit 한다.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .models import TASK_CLASSIFICATION, TASK_REGRESSION, ModelSpec
from .validation import SplitConfig, assert_temporal_order, make_cv

REGRESSION_METRICS = ["R2", "RMSE", "MAE", "MAPE"]
CLASSIFICATION_METRICS = ["Accuracy", "F1", "ROC_AUC"]


@dataclass
class TrainConfig:
    task: str = TASK_REGRESSION
    split: SplitConfig = field(default_factory=SplitConfig)
    n_jobs: int = -1                 # 모델 간 병렬
    seed: int = 42
    champion_metric: str = "R2"
    cv_weight: float = 0.0           # 0이면 홀드아웃만 보고 선정
    # 폴드 내부 피처 선별 (D5 — 기본 ON, 토글로 끔)
    fold_selection: bool = True
    selection_top_k: int | None = None
    selection_corr: float = 0.98
    selection_min_variance: float = 1e-12
    # 앙상블 자동채택 임계값 (G-4 — 화면에서 조절)
    ensemble_threshold: float = 0.03
    # 하이퍼파라미터 탐색 (G-6 — 기본 OFF. 켜면 nested CV 로 돈다)
    tune: object | None = None


def regression_scores(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nz = np.abs(y_true) > 1e-9
        out["MAPE"] = (float(mean_absolute_percentage_error(y_true[nz], y_pred[nz]))
                       if nz.sum() else float("nan"))
    return out


def classification_scores(y_true, y_pred, y_proba=None) -> dict[str, float]:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    out["ROC_AUC"] = float("nan")
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif y_proba.ndim == 2:
                out["ROC_AUC"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except ValueError:
            pass
    return out


def score(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    if task == TASK_REGRESSION:
        return regression_scores(y_true, y_pred)
    return classification_scores(y_true, y_pred, y_proba)


HIGHER_IS_BETTER = {"R2": True, "RMSE": False, "MAE": False, "MAPE": False,
                    "Accuracy": True, "F1": True, "ROC_AUC": True}


def _predict(model, X, task: str):
    pred = model.predict(X)
    proba = None
    if task == TASK_CLASSIFICATION and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except (AttributeError, ValueError):
            proba = None
    return pred, proba


def _pipeline_steps(preprocessor, estimator, cfg: TrainConfig) -> list[tuple[str, object]]:
    """(전처리 -> [선별] -> 추정기). 선별을 Pipeline 안에 두는 것이 핵심이다.

    Pipeline 안에 있으면 sklearn 이 폴드마다 fit 을 다시 부른다. 밖에서 한 번만
    선별하면 fold-1 의 검증 구간이 이미 선별에 관측된 상태가 되어 CV 가 낙관 편향된다.
    """
    steps: list[tuple[str, object]] = [("prep", clone(preprocessor))]
    if cfg.fold_selection:
        from .features import FoldSelector
        steps.append(("select", FoldSelector(
            top_k=cfg.selection_top_k, corr_threshold=cfg.selection_corr,
            min_variance=cfg.selection_min_variance,
            task=cfg.task, enabled=True, seed=cfg.seed,
        )))
    steps.append(("est", clone(estimator)))
    return steps


def _tunable(name: str) -> bool:
    from .tuning import tunable
    return tunable(name)


def _fold_selected_names(fitted_pipe) -> set | None:
    """폴드에서 실제로 살아남은 피처 이름 집합. 선별이 꺼져 있으면 None."""
    sel = fitted_pipe.named_steps.get("select") if hasattr(fitted_pipe, "named_steps") else None
    if sel is None or not getattr(sel, "enabled", False):
        return None
    try:
        names = list(fitted_pipe.named_steps["prep"].get_feature_names_out())
    except (AttributeError, ValueError, KeyError):
        names = [str(i) for i in range(getattr(sel, "n_features_in_", 0))]
    return {names[i] for i in sel.selected_index_ if i < len(names)}


def _fit_one(
    name: str,
    spec: ModelSpec,
    preprocessor,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    cv,
    cfg: TrainConfig,
) -> dict:
    t0 = time.perf_counter()
    result: dict = {"model": name, "family": spec.family}

    try:
        pipe = Pipeline(_pipeline_steps(preprocessor, spec.estimator, cfg))

        # 1) 학습 구간 내 rolling origin 교차검증
        #    fold_selection 이 켜져 있으면 선별도 이 안에서 폴드마다 다시 fit 된다.
        cv_rows = []
        fold_sets: list[set] = []
        oof = pd.Series(index=X_tr.index, dtype="float64")
        tune = cfg.tune
        do_tune = tune is not None and tune.applies_to(name) and _tunable(name)

        for tr, va in cv.split(X_tr):
            assert_temporal_order(X_tr.index, tr, va, gap=getattr(cv, "gap", 0))
            if do_tune:
                # nested CV — 파라미터는 이 폴드의 학습 구간 안에서만 고른다.
                # 바깥 검증 구간(va)은 파라미터 선택에 쓰이지 않으므로 아래 점수가
                # '튜닝을 포함한 절차 전체'의 편향 없는 추정이 된다.
                from .tuning import make_search
                search = make_search(clone(pipe), name, tune, task=cfg.task,
                                     gap=getattr(cv, "gap", 0))
                search.fit(X_tr.iloc[tr], y_tr.iloc[tr])
                fold = search.best_estimator_
                result.setdefault("_fold_params", []).append(
                    {k.replace("est__", ""): v for k, v in search.best_params_.items()})
            else:
                fold = clone(pipe)
                fold.fit(X_tr.iloc[tr], y_tr.iloc[tr])
            p, pr = _predict(fold, X_tr.iloc[va], cfg.task)
            cv_rows.append(score(cfg.task, y_tr.iloc[va], p, pr))
            if cfg.task == TASK_REGRESSION:
                oof.iloc[va] = np.asarray(p, dtype=float)
            picked = _fold_selected_names(fold)
            if picked is not None:
                fold_sets.append(picked)

        if do_tune and result.get("_fold_params"):
            from .tuning import stability
            result["_param_stability"] = stability(result["_fold_params"])
            result["tuned"] = True

        if fold_sets:
            from .features import jaccard_stability
            jac = jaccard_stability(fold_sets)
            result["fold_features_mean"] = float(np.mean([len(s) for s in fold_sets]))
            if not jac.empty:
                result["fold_jaccard"] = float(jac["jaccard"].mean())
                result["_fold_jaccard_table"] = jac
            result["_fold_feature_sets"] = fold_sets

        if cv_rows:
            for k in cv_rows[0]:
                result[f"cv_{k}"] = float(np.nanmean([r[k] for r in cv_rows]))
                result[f"cv_{k}_std"] = float(np.nanstd([r[k] for r in cv_rows]))

        # 2) 학습 구간 전체로 다시 fit -> 홀드아웃 평가
        #    튜닝을 켰으면 여기서도 학습 구간 안에서만 파라미터를 다시 고른다.
        if do_tune:
            from .tuning import make_search
            final_search = make_search(clone(pipe), name, tune, task=cfg.task,
                                       gap=cfg.split.gap)
            final_search.fit(X_tr, y_tr)
            pipe = final_search.best_estimator_
            result["best_params"] = "; ".join(
                f"{k.replace('est__', '')}={v}"
                for k, v in sorted(final_search.best_params_.items()))
        else:
            pipe.fit(X_tr, y_tr)
        p_te, pr_te = _predict(pipe, X_te, cfg.task)
        for k, v in score(cfg.task, y_te, p_te, pr_te).items():
            result[f"holdout_{k}"] = v

        p_tr, _ = _predict(pipe, X_tr, cfg.task)
        for k, v in score(cfg.task, y_tr, p_tr, None).items():
            result[f"insample_{k}"] = v

        result["fit_seconds"] = round(time.perf_counter() - t0, 2)
        result["status"] = "ok"
        result["_pipeline"] = pipe
        result["_oof"] = oof
        result["_holdout_pred"] = pd.Series(np.asarray(p_te, dtype="float64"), index=X_te.index)
    except Exception as e:  # noqa: BLE001 - 한 모델이 죽어도 나머지는 계속 돈다
        result.update(status="failed", error=f"{type(e).__name__}: {e}",
                      fit_seconds=round(time.perf_counter() - t0, 2))
    return result


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    preprocessor,
    zoo: dict[str, ModelSpec],
    selected: list[str],
    cfg: TrainConfig,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """선택된 모델을 모두 학습한다. (리더보드, 상세결과) 반환."""
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    assert_temporal_order(X.index, train_idx, test_idx, gap=cfg.split.gap)
    cv = make_cv(cfg.split)

    names = [n for n in selected if n in zoo]
    if not names:
        raise ValueError("학습할 모델이 선택되지 않았습니다.")

    if progress is not None:
        results = []
        for i, n in enumerate(names, start=1):
            progress(i, len(names), n)
            results.append(_fit_one(n, zoo[n], preprocessor, X_tr, y_tr, X_te, y_te, cv, cfg))
    else:
        results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
            delayed(_fit_one)(n, zoo[n], preprocessor, X_tr, y_tr, X_te, y_te, cv, cfg)
            for n in names
        )

    detail = {r["model"]: r for r in results}
    board = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    return sort_leaderboard(board, cfg.champion_metric), detail


def sort_leaderboard(board: pd.DataFrame, metric: str, prefix: str = "holdout_") -> pd.DataFrame:
    """지표 기준으로 재정렬한다. 이미 rank 가 붙은 보드를 다시 넣어도 된다."""
    col = f"{prefix}{metric}"
    if board.empty or col not in board.columns:
        return board
    board = board.drop(columns=["rank"], errors="ignore")   # 재정렬 시 rank 중복 방지
    asc = not HIGHER_IS_BETTER.get(metric, True)
    ok = board[board["status"] == "ok"].sort_values(col, ascending=asc, na_position="last")
    bad = board[board["status"] != "ok"]
    out = pd.concat([ok, bad], ignore_index=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def pick_champion(board: pd.DataFrame, metric: str, prefix: str = "holdout_") -> str | None:
    """챔피언을 고른다.

    3분할에서 prefix='holdout_' 은 '검증 구간' 을 뜻한다. Final Unseen 은 여기에
    절대 들어오지 않는다 — evaluate_unseen 이 챔피언 확정 뒤 한 번만 접근한다.
    """
    ok = board[board["status"] == "ok"] if "status" in board.columns else board
    if ok.empty:
        return None
    col = f"{prefix}{metric}"
    if col not in ok.columns:
        return str(ok.iloc[0]["model"])
    asc = not HIGHER_IS_BETTER.get(metric, True)
    return str(ok.sort_values(col, ascending=asc, na_position="last").iloc[0]["model"])


class UnseenAccessError(RuntimeError):
    """Final Unseen 을 두 번 이상 열려고 했을 때 발생."""


class UnseenGuard:
    """Final Unseen 구간의 접근 횟수를 센다.

    SPEC §11 은 Final Unseen 을 알고리즘 선택에 쓰는 것을 금지한다. 사람이
    규율로 지키는 대신 코드가 세도록 했다. 두 번째 접근은 예외로 막는다.
    """

    def __init__(self, idx: np.ndarray):
        self.idx = np.asarray(idx)
        self.access_count = 0
        self.accessed_by: list[str] = []

    def open(self, who: str = "?") -> np.ndarray:
        if self.access_count >= 1:
            raise UnseenAccessError(
                f"Final Unseen 은 이미 {self.access_count}회 열렸습니다 "
                f"(최초: {self.accessed_by[0]}). 모델을 바꾸려면 분할부터 다시 하세요."
            )
        self.access_count += 1
        self.accessed_by.append(who)
        return self.idx


def evaluate_unseen(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    unseen_idx,
    cfg: TrainConfig,
    guard: UnseenGuard | None = None,
    who: str = "champion",
) -> dict[str, float]:
    """챔피언이 확정된 뒤 Final Unseen 을 단 한 번 평가한다.

    이 점수가 최종 일반화 성능 보고값이다. 홀드아웃(검증) 점수는 모델을 고르는
    데 이미 쓰였으므로 모델 수만큼 선택 편향이 들어가 있다. 두 값은 다르며,
    보통 unseen 쪽이 낮게 나오는 것이 정상이다.

    **guard 는 생략할 수 없다.** 예전에는 guard=None 이면 횟수 제한 없이 통과했다.
    화면에서 챔피언을 바꾸면 세션의 guard 가 None 이 되는 경로가 있었고, 그 상태로
    이 함수를 부르면 Final Unseen 을 몇 번이든 열 수 있었다. 규율이 아니라 코드가
    막기로 한 이상, "가드가 없으면 그냥 통과" 라는 뒷문이 있으면 안 된다.
    """
    idx_arr = np.asarray(unseen_idx) if unseen_idx is not None else np.array([], dtype=int)
    if len(idx_arr) == 0:
        return {}
    if guard is None:
        raise UnseenAccessError(
            "Final Unseen 접근에는 guard 가 필요합니다. 분할을 만들 때 함께 만든 "
            "UnseenGuard 를 넘기세요. (가드 없이 여는 경로는 열어 두지 않습니다)")
    idx = guard.open(who)
    if len(idx) == 0:
        return {}
    X_un, y_un = X.iloc[idx], y.iloc[idx]
    p, pr = _predict(pipeline, X_un, cfg.task)
    out = {f"unseen_{k}": v for k, v in score(cfg.task, y_un, p, pr).items()}
    out["unseen_rows"] = float(len(idx))
    return out


def selection_bias_report(
    board: pd.DataFrame, champion: str, unseen: dict[str, float], metric: str
) -> pd.DataFrame:
    """검증 점수와 Final Unseen 점수의 격차를 표로 남긴다. 낙관 편향의 크기다."""
    row = board[board["model"] == champion]
    v = float(row[f"holdout_{metric}"].iloc[0]) if len(row) and f"holdout_{metric}" in row else float("nan")
    u = unseen.get(f"unseen_{metric}", float("nan"))
    n_models = int((board["status"] == "ok").sum()) if "status" in board.columns else len(board)
    return pd.DataFrame([{
        "지표": metric,
        "검증(모델선택에 사용)": round(v, 6),
        "Final Unseen(최종 보고)": round(u, 6),
        "격차": round(v - u, 6) if v == v and u == u else float("nan"),
        "비교한 모델 수": n_models,
        "해석": "검증 점수는 이 모델들 중 1등을 고른 값이라 낙관적입니다. "
               "Final Unseen 쪽을 최종 성능으로 보고하세요.",
    }])


def build_ensembles(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    preprocessor,
    zoo: dict[str, ModelSpec],
    base_names: list[str],
    cfg: TrainConfig,
    include_voting: bool = True,
    include_stacking: bool = True,
    detail: dict[str, dict] | None = None,
    include_weighted: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """상위 모델들로 보팅·가중·스태킹을 만들고 같은 잣대로 평가한다.

    detail(train_all 결과)을 넘기면 거기 담긴 시계열 OOF 를 재사용한다.
    sklearn StackingRegressor 는 내부에서 cross_val_predict 를 부르는데 이 함수가
    TimeSeriesSplit 을 받지 못해 항상 ValueError 로 죽는다. core.ensemble 이
    OOF 를 직접 만들어 그 문제를 없애고, base 재학습도 건너뛴다.

    detail 이 없으면 OOF 를 여기서 계산한다 (base 를 폴드마다 학습하므로 느리다).
    """
    from . import ensemble as ens_mod

    reg = cfg.task == TASK_REGRESSION
    names = [n for n in base_names if n in zoo]
    if len(names) < 2:
        return pd.DataFrame(), {}

    if reg:
        det = dict(detail or {})
        missing = [n for n in names if n not in det or det[n].get("status") != "ok"]
        if missing:
            det.update(_bootstrap_oof(X, y, train_idx, preprocessor, zoo, missing, cfg))
        board, edet = ens_mod.fit_ensembles(
            X, y, np.asarray(train_idx), np.asarray(test_idx), det, names,
            task=cfg.task, include_voting=include_voting,
            include_stacking=include_stacking, include_weighted=include_weighted,
            score_fn=lambda a, b: score(cfg.task, a, b),
        )
        if not board.empty:
            return board, edet

    # 분류는 기존 sklearn 경로를 유지한다 (OOF proba 경로는 후속 항목).
    from sklearn.ensemble import VotingClassifier
    base = [(n, Pipeline([("prep", clone(preprocessor)), ("est", clone(zoo[n].estimator))]))
            for n in names]
    specs: dict[str, object] = {}
    if include_voting:
        specs["Ensemble_Voting"] = VotingClassifier(base, voting="soft", n_jobs=1)

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

    results = []
    for name, est in specs.items():
        t0 = time.perf_counter()
        row: dict = {"model": name, "family": "blend"}
        try:
            est.fit(X_tr, y_tr)
            p_te, pr_te = _predict(est, X_te, cfg.task)
            for k, v in score(cfg.task, y_te, p_te, pr_te).items():
                row[f"holdout_{k}"] = v
            p_tr, _ = _predict(est, X_tr, cfg.task)
            for k, v in score(cfg.task, y_tr, p_tr, None).items():
                row[f"insample_{k}"] = v
            row.update(status="ok", fit_seconds=round(time.perf_counter() - t0, 2),
                       members=", ".join(n for n, _ in base))
            row["_pipeline"] = est
            row["_holdout_pred"] = pd.Series(np.asarray(p_te, dtype="float64"), index=X_te.index)
        except Exception as e:  # noqa: BLE001
            row.update(status="failed", error=f"{type(e).__name__}: {e}",
                       fit_seconds=round(time.perf_counter() - t0, 2))
        results.append(row)

    detail = {r["model"]: r for r in results}
    board = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    return board, detail


def random_vs_time(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    zoo: dict[str, ModelSpec],
    names: list[str],
    cfg: TrainConfig,
    test_ratio: float = 0.2,
    threshold: float = 0.15,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """같은 모델을 Time split 과 Random split 으로 각각 평가해 격차를 본다.

    **진단 전용이다.** 여기서 나온 Random 점수는 챔피언 선정에도, 리더보드에도,
    리포트에도 들어가지 않는다 (G-3 결정). 시계열에서 무작위로 나누면 검증 행의
    바로 앞뒤가 학습에 들어가므로 그 점수는 미래 성능이 아니다.

    읽는 법: 격차가 크면 데이터에 시간 구조가 강하다는 뜻이고, 그 경우 Time
    쪽 숫자만 믿어야 한다. 격차가 거의 없으면 시간 의존이 약한 데이터다.
    """
    from . import diagnostics
    from .validation import random_split

    metric = cfg.champion_metric
    n = len(X)
    tr_t, te_t = time_holdout_for(n, test_ratio, cfg.split.gap)
    tr_r, te_r = random_split(n, test_ratio, seed=cfg.seed)

    rows = []
    picked = [m for m in names if m in zoo]
    for i, name in enumerate(picked, start=1):
        if progress is not None:
            progress(i, len(picked), name)
        row: dict = {"model": name, "family": zoo[name].family}
        for label, (tr, te) in (("time", (tr_t, te_t)), ("random", (tr_r, te_r))):
            try:
                pipe = Pipeline(_pipeline_steps(preprocessor, zoo[name].estimator, cfg))
                pipe.fit(X.iloc[tr], y.iloc[tr])
                p, pr = _predict(pipe, X.iloc[te], cfg.task)
                for k, v in score(cfg.task, y.iloc[te], p, pr).items():
                    row[f"{label}_{k}"] = v
            except Exception as e:  # noqa: BLE001
                row[f"{label}_error"] = f"{type(e).__name__}: {e}"
        t, r = row.get(f"time_{metric}"), row.get(f"random_{metric}")
        if t is not None and r is not None and pd.notna(t) and pd.notna(r):
            higher = HIGHER_IS_BETTER.get(metric, True)
            row["격차"] = round(float(r - t) if higher else float(t - r), 4)
        rows.append(row)

    table = pd.DataFrame(rows)
    gaps = table["격차"].dropna() if "격차" in table.columns else pd.Series(dtype=float)
    mean_gap = float(gaps.mean()) if not gaps.empty else float("nan")
    verdict = diagnostics.split_gap_causes(
        mean_gap if mean_gap == mean_gap else 0.0, y, X, threshold=threshold)

    return {
        "table": table,
        "metric": metric,
        "mean_gap": mean_gap,
        "verdict": verdict,
        "purpose": "diagnostic",   # 평가 경로가 이 결과를 쓰지 못하게 하는 표식
        "time_rows": int(len(te_t)),
        "random_rows": int(len(te_r)),
    }


def time_holdout_for(n: int, ratio: float, gap: int):
    """random_vs_time 이 쓰는 Time split. validation.time_holdout 을 그대로 부른다."""
    from .validation import time_holdout
    return time_holdout(n, ratio, gap)


def assert_not_diagnostic(result) -> None:
    """진단 전용 결과가 평가 경로로 새는 것을 막는다.

    G-3 결정: Random split 은 진단으로만 쓰고 챔피언 선정에서는 제외한다.
    사람이 지키게 두지 않고 코드가 막는다.
    """
    if isinstance(result, dict) and result.get("purpose") == "diagnostic":
        raise ValueError(
            "Random vs Time 진단 결과는 모델 선택·리더보드에 쓸 수 없습니다. "
            "무작위 분할 점수는 이웃 행을 학습에 넣은 값이라 미래 성능이 아닙니다."
        )


def rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    spec: ModelSpec,
    cfg: TrainConfig,
    n_folds: int = 5,
    test_size: int | None = None,
    expanding: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """시기를 굴리며 재학습·평가한다. (구간별 성능표, 이어붙인 예측) 반환.

    한 번의 홀드아웃 점수는 "그 시기가 어땠는가"에 크게 좌우된다. 여러 시기에서
    같은 절차를 반복해 보면 그 점수가 운이었는지 실력이었는지 갈린다.
    구간 간 표준편차가 크면 그 모델은 시기를 탄다는 뜻이다.
    """
    from .validation import rolling_windows

    windows = rolling_windows(len(X), n_folds=n_folds, test_size=test_size,
                              gap=cfg.split.gap, expanding=expanding)
    rows = []
    stitched = pd.Series(index=X.index, dtype="float64")

    for i, (tr, te) in enumerate(windows, start=1):
        if progress is not None:
            progress(i, len(windows), f"{X.index[te[0]]:%Y-%m-%d}")
        assert_temporal_order(X.index, tr, te, gap=cfg.split.gap)
        t0 = time.perf_counter()
        pipe = Pipeline(_pipeline_steps(preprocessor, spec.estimator, cfg))
        row = {
            "구간": i,
            "학습": f"{X.index[tr[0]]:%Y-%m-%d} ~ {X.index[tr[-1]]:%Y-%m-%d}",
            "평가시작": X.index[te[0]], "평가끝": X.index[te[-1]],
            "n_train": len(tr), "n_test": len(te),
        }
        try:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p, pr = _predict(pipe, X.iloc[te], cfg.task)
            row.update(score(cfg.task, y.iloc[te], p, pr))
            if cfg.task == TASK_REGRESSION:
                stitched.iloc[te] = np.asarray(p, dtype=float)
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        row["fit_seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)

    return pd.DataFrame(rows), stitched.dropna()


def backtest_summary(table: pd.DataFrame, metric: str = "R2") -> dict:
    """구간별 성능의 흔들림을 한 줄로 요약한다."""
    ok = table[table["status"] == "ok"] if "status" in table.columns else table
    if ok.empty or metric not in ok.columns:
        return {}
    v = ok[metric].astype(float)
    higher = HIGHER_IS_BETTER.get(metric, True)
    worst_i = int(v.idxmin() if higher else v.idxmax())
    return {
        "구간수": int(len(v)),
        "평균": round(float(v.mean()), 4),
        "표준편차": round(float(v.std()), 4),
        "최저": round(float(v.min()), 4),
        "최고": round(float(v.max()), 4),
        "최악구간": int(ok.loc[worst_i, "구간"]),
        "최악시작": ok.loc[worst_i, "평가시작"],
    }


def _bootstrap_oof(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    preprocessor,
    zoo: dict[str, ModelSpec],
    names: list[str],
    cfg: TrainConfig,
) -> dict[str, dict]:
    """detail 없이 앙상블을 부른 경우에만 쓰는 보조 경로. OOF 와 전체학습본을 만든다.

    OOF 도 _pipeline_steps 로 만든다. 메타 학습기가 보는 예측과 최종 예측이
    다른 파이프라인에서 나오면 스태킹 가중치가 틀어진다.
    """
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    cv = make_cv(cfg.split)
    out: dict[str, dict] = {}
    for n in names:
        proto = Pipeline(_pipeline_steps(preprocessor, zoo[n].estimator, cfg))
        oof = pd.Series(index=X_tr.index, dtype="float64")
        for tr, va in cv.split(X_tr):
            assert_temporal_order(X_tr.index, tr, va, gap=getattr(cv, "gap", 0))
            fold = clone(proto).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            oof.iloc[va] = np.asarray(fold.predict(X_tr.iloc[va]), dtype=float)
        out[n] = {"model": n, "status": "ok", "_oof": oof,
                  "_pipeline": clone(proto).fit(X_tr, y_tr)}
    return out


def predict_range(pipeline, X: pd.DataFrame, start=None, end=None) -> pd.Series:
    """선택 구간에 대해 예측값을 낸다."""
    sub = X
    if start is not None:
        sub = sub.loc[sub.index >= pd.Timestamp(start)]
    if end is not None:
        sub = sub.loc[sub.index <= pd.Timestamp(end)]
    if sub.empty:
        return pd.Series(dtype="float64")
    return pd.Series(np.asarray(pipeline.predict(sub), dtype="float64"),
                     index=sub.index, name="prediction")
