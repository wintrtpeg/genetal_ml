"""비지도학습 경로.

Y 없이 도는 분석이므로 지도학습과 흐름이 갈린다.
- 군집: 운전 모드 분류, 유사 구간 묶기
- 이상탐지: 시점별 이상 점수와 임계 초과 구간
- 차원축소: PCA 로 주요 변동 축 확인
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

CLUSTERING = "clustering"
ANOMALY = "anomaly"
REDUCTION = "reduction"


@dataclass
class UnsupervisedConfig:
    mode: str = CLUSTERING
    k_range: tuple[int, int] = (2, 8)
    contamination: float = 0.01
    n_components: int = 3
    seed: int = 42
    sample_limit: int = 50_000        # 실루엣 계산 표본 상한
    selected: list[str] = field(default_factory=list)


def clustering_zoo(k: int, seed: int) -> dict[str, object]:
    return {
        "KMeans": KMeans(n_clusters=k, n_init=10, random_state=seed),
        "GaussianMixture": GaussianMixture(n_components=k, random_state=seed),
        "Agglomerative": AgglomerativeClustering(n_clusters=k),
        "DBSCAN": DBSCAN(eps=0.8, min_samples=10),
    }


def anomaly_zoo(contamination: float, seed: int) -> dict[str, object]:
    return {
        "IsolationForest": IsolationForest(
            n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=35, contamination=contamination, novelty=False),
    }


def _cluster_metrics(Z: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> dict[str, float]:
    valid = labels >= 0
    uniq = np.unique(labels[valid])
    if uniq.size < 2 or valid.sum() < 10:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"), "n_clusters": int(uniq.size)}

    Zv, Lv = Z[valid], labels[valid]
    if len(Zv) > limit:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(Zv), size=limit, replace=False)
        Zs, Ls = Zv[pick], Lv[pick]
    else:
        Zs, Ls = Zv, Lv

    out = {"n_clusters": int(uniq.size), "noise_ratio": float((~valid).mean())}
    try:
        out["silhouette"] = float(silhouette_score(Zs, Ls))
        out["davies_bouldin"] = float(davies_bouldin_score(Zs, Ls))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Zs, Ls))
    except ValueError:
        out.update(silhouette=float("nan"), davies_bouldin=float("nan"),
                   calinski_harabasz=float("nan"))
    return out


def run_clustering(
    X: pd.DataFrame, preprocessor, cfg: UnsupervisedConfig
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """k 를 훑으면서 모델별 군집 품질을 비교한다."""
    pre = clone(preprocessor)
    Z = pre.fit_transform(X)
    rows, detail = [], {}

    for k in range(cfg.k_range[0], cfg.k_range[1] + 1):
        for name, model in clustering_zoo(k, cfg.seed).items():
            if name == "DBSCAN" and k != cfg.k_range[0]:
                continue  # DBSCAN 은 k 를 쓰지 않으므로 한 번만
            if cfg.selected and name not in cfg.selected:
                continue
            t0 = time.perf_counter()
            try:
                m = clone(model)
                labels = (m.fit_predict(Z) if hasattr(m, "fit_predict")
                          else m.fit(Z).predict(Z))
                labels = np.asarray(labels)
                row = {"model": name, "k": ("-" if name == "DBSCAN" else k), "status": "ok"}
                row.update(_cluster_metrics(Z, labels, cfg.sample_limit, cfg.seed))
                row["fit_seconds"] = round(time.perf_counter() - t0, 2)
                key = f"{name}_k{k}" if name != "DBSCAN" else name
                detail[key] = {"labels": pd.Series(labels, index=X.index, name="cluster"),
                               "model": m, "preprocessor": pre}
                row["key"] = key
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                rows.append({"model": name, "k": k, "status": "failed",
                             "error": f"{type(e).__name__}: {e}"})

    board = pd.DataFrame(rows)
    if not board.empty and "silhouette" in board.columns:
        board = board.sort_values("silhouette", ascending=False, na_position="last")
        board.insert(0, "rank", range(1, len(board) + 1))
    return board, detail


def run_anomaly(
    X: pd.DataFrame, preprocessor, cfg: UnsupervisedConfig
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """이상 점수를 만든다. 점수가 낮을수록 이상."""
    pre = clone(preprocessor)
    Z = pre.fit_transform(X)
    rows, detail = [], {}

    for name, model in anomaly_zoo(cfg.contamination, cfg.seed).items():
        if cfg.selected and name not in cfg.selected:
            continue
        t0 = time.perf_counter()
        try:
            m = clone(model)
            if name == "LocalOutlierFactor":
                flags = m.fit_predict(Z)
                scores = m.negative_outlier_factor_
            else:
                m.fit(Z)
                flags = m.predict(Z)
                scores = m.score_samples(Z)
            s = pd.Series(np.asarray(scores, dtype=float), index=X.index, name="anomaly_score")
            f = pd.Series(np.asarray(flags) == -1, index=X.index, name="is_anomaly")
            rows.append({
                "model": name, "status": "ok",
                "n_anomaly": int(f.sum()), "anomaly_ratio": round(float(f.mean()), 4),
                "score_min": float(s.min()), "score_median": float(s.median()),
                "fit_seconds": round(time.perf_counter() - t0, 2),
            })
            detail[name] = {"score": s, "flag": f, "model": m, "preprocessor": pre}
        except Exception as e:  # noqa: BLE001
            rows.append({"model": name, "status": "failed", "error": f"{type(e).__name__}: {e}"})

    return pd.DataFrame(rows), detail


def run_pca(X: pd.DataFrame, preprocessor, cfg: UnsupervisedConfig) -> dict:
    pre = clone(preprocessor)
    Z = pre.fit_transform(X)
    n = min(cfg.n_components, Z.shape[1])
    pca = PCA(n_components=n, random_state=cfg.seed).fit(Z)
    comps = pca.transform(Z)

    try:
        names = list(pre.get_feature_names_out())
    except (AttributeError, ValueError):
        names = [f"f{i}" for i in range(Z.shape[1])]

    scores = pd.DataFrame(comps, index=X.index,
                          columns=[f"PC{i+1}" for i in range(n)])
    loadings = pd.DataFrame(pca.components_.T, index=names,
                            columns=[f"PC{i+1}" for i in range(n)])
    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
        "model": pca,
    }


def cluster_profile(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """군집별 피처 평균. 각 군집이 어떤 운전 상태인지 읽는 용도."""
    num = X.select_dtypes("number")
    prof = num.groupby(labels.reindex(num.index)).mean().T
    prof.columns = [f"cluster_{c}" for c in prof.columns]
    prof.insert(0, "overall_mean", num.mean())
    return prof.reset_index(names="feature")
