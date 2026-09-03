"""파생변수 생성 (LLM 없이 룰 기반).

설계 원칙
1. 타겟에서 파생된 피처는 만들지 않는다. allow_target_derived=False 가 기본이고,
   생성 후에도 provenance(출처) 대장을 근거로 한 번 더 검사한다.
2. 시점 t 의 피처는 시점 t 까지의 정보만 쓴다. 중앙정렬 rolling, 역방향 shift,
   전체 구간 통계는 만들지 않는다.
3. 모든 피처는 어떤 원본 컬럼에서 어떻게 나왔는지 기록한다. 나중에 SHAP 결과를
   해석할 때 이 대장이 있어야 원인을 원본 태그까지 되짚을 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetLeakage(RuntimeError):
    """타겟에서 파생된 피처가 X 로 들어가려 할 때 발생."""


@dataclass
class FeatureConfig:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    # 물리 시간(분)으로도 지정할 수 있다. 샘플링 간격을 읽어 행 수로 환산해
    # lags / rolling_windows 에 합쳐진다. 5분 간격에서 60분 -> 12행.
    lag_minutes: list[float] = field(default_factory=list)
    rolling_minutes: list[float] = field(default_factory=list)
    rolling_windows: list[int] = field(default_factory=lambda: [6, 12, 24])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std"])
    ewm_spans: list[int] = field(default_factory=lambda: [12])
    diffs: list[int] = field(default_factory=lambda: [1])
    rate_of_change: bool = False          # 변화율 (pct_change)
    time_features: bool = True            # 시각·요일·월
    cyclical: bool = True                 # 시간의 주기성을 sin/cos 로
    interactions: list[tuple[str, str]] = field(default_factory=list)  # 비율·차이를 만들 쌍
    interaction_ops: list[str] = field(default_factory=lambda: ["ratio", "diff"])
    allow_target_derived: bool = False     # ★ 기본 차단
    max_features: int | None = None        # 생성 후 상한 (선별은 select_features 에서)


def _safe(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def step_minutes(index: pd.DatetimeIndex) -> float | None:
    """샘플링 간격을 분 단위로 돌려준다. 결측이 있어도 최빈 간격으로 잡는다."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return None
    mode = deltas.mode()
    step = (mode.iloc[0] if not mode.empty else deltas.median())
    secs = pd.Timedelta(step).total_seconds()
    return secs / 60.0 if secs > 0 else None


def minutes_to_rows(minutes, step_min: float | None) -> list[int]:
    """물리 시간(분)을 행 수로 환산한다. 1행 미만은 버린다.

    간격을 모르면 환산하지 않는다. 어림짐작으로 행 수를 만들면 lookback 이
    실제와 어긋나고, 그 값이 그대로 gap 계산에 들어가 누수 점검을 무력화한다.
    """
    if not minutes or not step_min or step_min <= 0:
        return []
    out = []
    for m in minutes:
        rows = int(round(float(m) / step_min))
        if rows >= 1:
            out.append(rows)
    return sorted(set(out))


def resolve_config(cfg: FeatureConfig, index: pd.DatetimeIndex) -> FeatureConfig:
    """분 단위 지정을 행 단위로 환산해 합친 사본을 돌려준다.

    원본 cfg 는 건드리지 않는다. 사용자가 화면에서 입력한 값을 그대로 보존해야
    재현 기록에 "60분" 으로 남길 수 있다.
    """
    step = step_minutes(index)
    lag_rows = minutes_to_rows(cfg.lag_minutes, step)
    roll_rows = minutes_to_rows(cfg.rolling_minutes, step)
    if not lag_rows and not roll_rows:
        return cfg
    out = replace(
        cfg,
        lags=sorted(set(list(cfg.lags) + lag_rows)),
        rolling_windows=sorted(set(list(cfg.rolling_windows) + roll_rows)),
    )
    return out


def describe_time_spec(cfg: FeatureConfig, index: pd.DatetimeIndex) -> pd.DataFrame:
    """분 지정이 몇 행으로 환산됐는지 화면에 보여줄 표."""
    step = step_minutes(index)
    rows = []
    for kind, mins in (("lag", cfg.lag_minutes), ("rolling", cfg.rolling_minutes)):
        for m in mins or []:
            r = minutes_to_rows([m], step)
            rows.append({
                "종류": kind, "지정(분)": m,
                "환산(행)": r[0] if r else "—",
                "실제(분)": round(r[0] * step, 2) if r and step else "—",
            })
    return pd.DataFrame(rows)


def generate(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    cfg: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """파생변수를 만들어 (확장된 DataFrame, provenance 대장) 을 돌려준다.

    df 는 DatetimeIndex 를 가진 정렬된 시계열이어야 한다.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DatetimeIndex 가 필요합니다. datasource.to_timeseries() 를 먼저 적용하세요.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("시간 오름차순으로 정렬된 데이터가 필요합니다.")

    # 분 단위 지정을 행 수로 환산해 합친다. 합집합이므로 두 번 불러도 결과가 같다.
    cfg = resolve_config(cfg, df.index)

    base = [c for c in feature_cols if c != target and c in df.columns]
    numeric = [c for c in base if pd.api.types.is_numeric_dtype(df[c])]

    # 2단계에서 제외한 컬럼은 여기서 되살아나면 안 된다.
    # df.copy() 를 쓰면 제외 컬럼이 그대로 남아 3단계에서 다시 X 후보가 되고,
    # 사용자의 제외 결정이 조용히 무시된다. 그래서 base + target 만 가져온다.
    out = df[base + ([target] if target in df.columns else [])].copy()
    prov: list[dict] = [
        {"feature": c, "origin": c, "transform": "raw", "lookback": 0} for c in base
    ]
    prov.append({"feature": target, "origin": target, "transform": "target", "lookback": 0})

    made: dict[str, pd.Series] = {}

    def add(name: str, series: pd.Series, origin: str, transform: str, lookback: int) -> None:
        made[name] = series
        prov.append({"feature": name, "origin": origin, "transform": transform, "lookback": lookback})

    for c in numeric:
        s = df[c]
        sc = _safe(c)

        for L in cfg.lags:
            add(f"{sc}__lag{L}", s.shift(L), c, f"lag({L})", L)

        for w in cfg.rolling_windows:
            roll = s.rolling(window=w, min_periods=max(2, w // 2))
            for stat in cfg.rolling_stats:
                if not hasattr(roll, stat):
                    continue
                add(f"{sc}__roll{w}_{stat}", getattr(roll, stat)(), c, f"rolling({w},{stat})", w)

        for span in cfg.ewm_spans:
            add(f"{sc}__ewm{span}", s.ewm(span=span, adjust=False).mean(), c, f"ewm({span})", span)

        for d in cfg.diffs:
            add(f"{sc}__diff{d}", s.diff(d), c, f"diff({d})", d)

        if cfg.rate_of_change:
            add(f"{sc}__pct1", s.pct_change().replace([np.inf, -np.inf], np.nan), c, "pct_change(1)", 1)

    for a, b in cfg.interactions:
        if a not in df.columns or b not in df.columns:
            continue
        sa_, sb = df[a], df[b]
        if "ratio" in cfg.interaction_ops:
            r = sa_ / sb.replace(0, np.nan)
            add(f"{_safe(a)}__over__{_safe(b)}", r.replace([np.inf, -np.inf], np.nan),
                f"{a}|{b}", "ratio", 0)
        if "diff" in cfg.interaction_ops:
            add(f"{_safe(a)}__minus__{_safe(b)}", sa_ - sb, f"{a}|{b}", "difference", 0)

    if cfg.time_features:
        idx = out.index
        add("tf__hour", pd.Series(idx.hour, index=idx, dtype="float64"), "__time__", "hour", 0)
        add("tf__dayofweek", pd.Series(idx.dayofweek, index=idx, dtype="float64"), "__time__", "dayofweek", 0)
        add("tf__month", pd.Series(idx.month, index=idx, dtype="float64"), "__time__", "month", 0)
        add("tf__is_weekend", pd.Series((idx.dayofweek >= 5).astype("float64"), index=idx),
            "__time__", "is_weekend", 0)
        if cfg.cyclical:
            for name, values, period in (
                ("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month, 12)
            ):
                ang = 2 * np.pi * np.asarray(values) / period
                add(f"tf__{name}_sin", pd.Series(np.sin(ang), index=idx), "__time__", f"{name}_sin", 0)
                add(f"tf__{name}_cos", pd.Series(np.cos(ang), index=idx), "__time__", f"{name}_cos", 0)

    if made:
        out = pd.concat([out, pd.DataFrame(made, index=out.index)], axis=1)

    provenance = pd.DataFrame(prov).drop_duplicates(subset=["feature"], keep="first")
    assert_no_target_derived(
        [c for c in out.columns if c != target], target, provenance,
        allow=cfg.allow_target_derived,
    )
    return out, provenance


def assert_no_target_derived(
    feature_names: list[str],
    target: str,
    provenance: pd.DataFrame,
    allow: bool = False,
) -> None:
    """X 후보 중 타겟에서 나온 것이 있으면 즉시 중단시킨다."""
    if allow:
        return
    lookup = provenance.set_index("feature")["origin"].to_dict()
    bad = []
    for f in feature_names:
        origin = lookup.get(f)
        if origin is None:
            continue
        if target in str(origin).split("|"):
            bad.append(f)
    if bad:
        raise TargetLeakage(
            "타겟에서 파생된 피처가 X 에 포함되어 있습니다: "
            + ", ".join(bad[:10])
            + (" 외" if len(bad) > 10 else "")
        )


def warmup_rows(cfg: FeatureConfig, index: pd.DatetimeIndex | None = None) -> int:
    """lag·rolling 때문에 앞쪽에서 버려야 하는 행 수.

    이 값이 gap 점검의 기준(max_lookback)이 되므로 분 단위 지정을 빠뜨리면
    누수 점검이 무력해진다. index 를 주면 환산까지 반영한다.
    """
    if index is not None:
        cfg = resolve_config(cfg, index)
    spans = list(cfg.lags) + list(cfg.rolling_windows) + list(cfg.ewm_spans) + list(cfg.diffs)
    return int(max(spans)) if spans else 0


def drop_warmup(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """선두 warm-up 구간을 잘라낸다. 중간 결측은 전처리 단계에서 다룬다."""
    k = warmup_rows(cfg, df.index if isinstance(df.index, pd.DatetimeIndex) else None)
    return df.iloc[k:] if k and k < len(df) else df


def _corr_duplicates(A: np.ndarray, keep: list[int], threshold: float) -> dict[int, int]:
    """상관이 threshold 이상인 뒤쪽 컬럼을 버린다. {버린 인덱스: 남긴 인덱스}."""
    dropped: dict[int, int] = {}
    if len(keep) < 2:
        return dropped
    sub = A[:, keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(sub, rowvar=False)
    C = np.abs(np.atleast_2d(C))
    for i in range(len(keep)):
        if keep[i] in dropped:
            continue
        for j in range(i + 1, len(keep)):
            if keep[j] in dropped:
                continue
            v = C[i, j]
            if v == v and v >= threshold:
                dropped[keep[j]] = keep[i]
    return dropped


def select_core(
    A: np.ndarray,
    y: np.ndarray,
    names: list[str],
    task: str = "regression",
    top_k: int | None = None,
    corr_threshold: float = 0.98,
    min_variance: float = 1e-12,
    seed: int = 0,
    compute_mi: bool = True,
) -> tuple[list[int], pd.DataFrame]:
    """분산 -> 상관중복 -> 상호정보 3단계 선별의 알맹이.

    DataFrame 경로(select_features)와 Pipeline 내부 경로(FoldSelector)가 같은
    함수를 쓰도록 분리했다. 두 경로의 결과가 갈리면 폴드 내부 선별을 신뢰할 수 없다.

    compute_mi=False 는 상호정보량 계산을 건너뛴다. MI 는 k-NN 기반이라 열 수와
    행 수 모두에 비례해 비싸다 — 50만 행 × 200열이면 폴드마다 수 분이 든다.
    top_k 가 없으면 MI 는 컷오프에 쓰이지 않고 사유 문자열에만 남으므로, 그 문자열을
    쓰지 않는 폴드 내부 경로에서는 순수한 낭비다. 화면에 MI 를 보여주는
    select_features 경로는 기본값 True 를 그대로 쓴다.

    반환: (남긴 컬럼 인덱스, 선별 사유 표)
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    A = np.asarray(A, dtype=float)
    n_col = A.shape[1]
    var = np.nanvar(A, axis=0)
    miss = np.isnan(A).mean(axis=0)

    status = ["selected"] * n_col
    reason = [""] * n_col

    keep = [i for i in range(n_col) if var[i] > min_variance]
    for i in range(n_col):
        if i not in keep:
            status[i] = "removed"
            reason[i] = f"분산 {var[i]:.3g} < {min_variance:g}"

    for gone, kept_i in _corr_duplicates(A, keep, corr_threshold).items():
        status[gone] = "removed"
        reason[gone] = f"|corr| >= {corr_threshold:g} with {names[kept_i]}"
    keep = [i for i in keep if status[i] == "selected"]

    mi = np.full(n_col, np.nan)
    ymask = ~pd.isna(y)
    if compute_mi and keep and int(ymask.sum()) > 10:
        filled = np.nan_to_num(A[:, keep][ymask], nan=0.0, posinf=0.0, neginf=0.0)
        fn = mutual_info_regression if task == "regression" else mutual_info_classif
        try:
            mi[keep] = fn(filled, np.asarray(y)[ymask], random_state=seed)
        except (ValueError, TypeError):
            mi[keep] = 0.0

    if top_k and len(keep) > top_k:
        order = sorted(keep, key=lambda i: (-(mi[i] if mi[i] == mi[i] else -np.inf), i))
        survivors = set(order[:top_k])
        for i in keep:
            if i not in survivors:
                rank = order.index(i) + 1
                status[i] = "removed"
                reason[i] = f"MI 순위 {rank}위 (상위 {top_k} 밖)"
        keep = [i for i in keep if status[i] == "selected"]

    if compute_mi:
        ranks = {i: r for r, i in enumerate(
            sorted(keep, key=lambda i: -(mi[i] if mi[i] == mi[i] else -np.inf)), start=1)}
        for i in keep:
            reason[i] = f"MI 상위 {ranks[i]}위"
    else:
        for i in keep:
            reason[i] = "분산·상관 통과 (MI 미계산)"

    report = pd.DataFrame({
        "feature": names,
        "variance": var,
        "missing_ratio": miss,
        "mutual_info": mi,
        "kept": [s == "selected" for s in status],
        "status": status,
        "reason": reason,
    })
    return keep, report


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "regression",
    top_k: int | None = None,
    corr_threshold: float = 0.98,
    min_variance: float = 1e-12,
) -> tuple[list[str], pd.DataFrame]:
    """생성된 피처를 추린다.

    학습 구간에서만 호출해야 한다. 홀드아웃을 보고 고르면 그 자체가 누수다.
    report 에는 피처별 선택·제외 사유(status/reason)가 함께 담긴다.
    """
    Xn = X.select_dtypes("number")
    names = list(Xn.columns)
    idx, report = select_core(
        Xn.to_numpy(dtype=float), y.to_numpy(), names,
        task=task, top_k=top_k, corr_threshold=corr_threshold, min_variance=min_variance,
    )
    return [names[i] for i in idx], report


def feature_report(
    report: pd.DataFrame,
    provenance: pd.DataFrame | None = None,
    X_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """선별 결과 · 출처 대장 · 품질 지표를 한 표로 합친다. 사람이 검토하는 화면용.

    **X_train 은 반드시 학습 구간만 넘겨야 한다.** 여기서 나온 표를 보고 사람이
    피처를 고르므로, 전체 구간 통계를 보여주면 사람이 누수 경로가 된다. 알고리즘이
    홀드아웃을 못 보게 막아 놓고 사람에게는 보여주는 것은 앞뒤가 맞지 않는다.
    """
    out = report.copy()

    if provenance is not None and not provenance.empty:
        cols = [c for c in ("feature", "origin", "transform", "lookback")
                if c in provenance.columns]
        out = out.merge(provenance[cols], on="feature", how="left")
    for c, default in (("origin", "—"), ("transform", "raw"), ("lookback", 0)):
        if c not in out.columns:
            out[c] = default
    out["origin"] = out["origin"].fillna("—")
    out["transform"] = out["transform"].fillna("raw")
    out["lookback"] = out["lookback"].fillna(0)

    if "mutual_info" in out.columns:
        rank = out["mutual_info"].rank(ascending=False, method="min")
        out["MI순위"] = rank.astype("Int64")

    if X_train is not None:
        num = X_train.select_dtypes("number")
        stats = pd.DataFrame({
            "feature": num.columns,
            "학습구간_평균": num.mean().to_numpy(),
            "학습구간_표준편차": num.std().to_numpy(),
            "결측률": num.isna().mean().to_numpy(),
        })
        out = out.merge(stats, on="feature", how="left")

    order = ["feature", "kept", "status", "reason", "mutual_info", "MI순위",
             "origin", "transform", "lookback", "variance", "missing_ratio",
             "학습구간_평균", "학습구간_표준편차"]
    rest = [c for c in out.columns if c not in order]
    return out[[c for c in order if c in out.columns] + rest]


def origin_rollup(report: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼별로 몇 개가 살아남았는지. 파생이 많을 때 이 단위로 보는 게 빠르다."""
    if report.empty or "origin" not in report.columns:
        return pd.DataFrame()
    g = report.groupby("origin", dropna=False)
    out = pd.DataFrame({
        "원본": g.size().index,
        "생성": g.size().to_numpy(),
        "선택": g["kept"].sum().to_numpy() if "kept" in report.columns else 0,
        "최고MI": g["mutual_info"].max().to_numpy() if "mutual_info" in report.columns else np.nan,
    })
    out["선택률"] = (out["선택"] / out["생성"]).round(3)
    return out.sort_values("최고MI", ascending=False, na_position="last", ignore_index=True)


def apply_manual_selection(
    report: pd.DataFrame,
    chosen: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """사람이 고른 목록을 반영하고, 자동 추천과 달라진 부분을 이력에 남긴다.

    감사 이력의 목적은 "왜 이 피처가 들어갔나"에 답하는 것이다. 사람이 바꾼 것도
    똑같이 답할 수 있어야 하므로 status 에 (수동) 표식을 남긴다.
    """
    out = report.copy()
    chosen_set = set(chosen)
    auto = set(out.loc[out.get("kept", False) == True, "feature"])  # noqa: E712

    added = sorted(chosen_set - auto)
    dropped = sorted(auto - chosen_set)

    if "status" not in out.columns:
        out["status"] = np.where(out["feature"].isin(auto), "selected", "removed")
    if "reason" not in out.columns:
        out["reason"] = ""

    for f in added:
        m = out["feature"] == f
        prev = out.loc[m, "reason"].iloc[0] if m.any() else ""
        out.loc[m, "status"] = "selected(수동추가)"
        out.loc[m, "reason"] = f"사용자가 직접 추가 (자동 판정: {prev or '제외'})"
    for f in dropped:
        m = out["feature"] == f
        prev = out.loc[m, "reason"].iloc[0] if m.any() else ""
        out.loc[m, "status"] = "removed(수동제외)"
        out.loc[m, "reason"] = f"사용자가 직접 제외 (자동 판정: {prev or '선택'})"

    out["kept"] = out["feature"].isin(chosen_set)
    return sorted(chosen_set), out


def selection_risks(
    chosen: list[str],
    report: pd.DataFrame,
    X_train: pd.DataFrame | None = None,
    corr_threshold: float = 0.98,
    min_mi: float = 1e-6,
    max_missing: float = 0.3,
) -> pd.DataFrame:
    """사람이 고른 목록에 위험한 조합이 있는지 본다. 막지는 않고 알리기만 한다.

    도메인 판단이 통계보다 옳은 경우가 많으므로 선택을 막지 않는다. 다만 무엇을
    감수하는 것인지는 알려 준다.
    """
    rows: list[dict] = []
    if not chosen:
        return pd.DataFrame(columns=["피처", "위험", "내용"])

    idx = report.set_index("feature") if "feature" in report.columns else report
    for f in chosen:
        if f not in idx.index:
            continue
        r = idx.loc[f]
        mi = r.get("mutual_info", np.nan)
        if mi == mi and float(mi) <= min_mi:
            rows.append({"피처": f, "위험": "정보량 0",
                         "내용": f"타겟과의 상호정보가 {float(mi):.2g} 입니다. 노이즈일 수 있습니다."})
        var = r.get("variance", np.nan)
        if var == var and float(var) <= 1e-12:
            rows.append({"피처": f, "위험": "분산 0",
                         "내용": "값이 변하지 않는 컬럼입니다. 모델이 쓸 수 없습니다."})
        miss = r.get("missing_ratio", np.nan)
        if miss == miss and float(miss) > max_missing:
            rows.append({"피처": f, "위험": "결측 과다",
                         "내용": f"학습 구간 결측률 {float(miss):.1%}. 대치값이 대부분을 차지합니다."})

    # 상관 중복을 둘 다 고른 경우
    if X_train is not None:
        num = X_train[[c for c in chosen if c in X_train.columns]].select_dtypes("number")
        if num.shape[1] >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = num.corr().abs()
            cols = list(C.columns)
            seen = set()
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    v = C.loc[a, b]
                    if v == v and v >= corr_threshold and (a, b) not in seen:
                        seen.add((a, b))
                        rows.append({
                            "피처": f"{a} · {b}", "위험": "중복",
                            "내용": f"학습 구간 상관 {float(v):.3f}. 둘 다 넣으면 "
                                   "기여도가 나뉘어 SHAP 해석이 흐려집니다.",
                        })
    return pd.DataFrame(rows, columns=["피처", "위험", "내용"])


class FoldSelector(BaseEstimator, TransformerMixin):
    """Pipeline 안에서 폴드마다 다시 선별한다.

    전처리와 같은 원칙이다. 선별을 폴드 밖에서 한 번만 하면 fold-1 의 검증
    구간 정보가 이미 선별 단계에서 관측된 상태가 되어 CV 점수가 낙관 편향된다.
    이 변환기를 (전처리 -> 선별 -> 추정기) 순서로 끼우면 sklearn 이 폴드마다
    fit 을 다시 부르므로 그 편향이 사라진다.

    enabled=False 면 아무것도 하지 않는다 (토글).
    """

    def __init__(self, top_k=None, corr_threshold=0.98, min_variance=1e-12,
                 task="regression", enabled=True, seed=0):
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.min_variance = min_variance
        self.task = task
        self.enabled = enabled
        self.seed = seed

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.n_features_in_ = A.shape[1]
        if not self.enabled or y is None:
            self.support_ = np.ones(A.shape[1], dtype=bool)
            return self
        names = [str(i) for i in range(A.shape[1])]
        idx, _ = select_core(
            A, np.asarray(y), names, task=self.task, top_k=self.top_k,
            corr_threshold=self.corr_threshold, min_variance=self.min_variance,
            seed=self.seed,
            # 사유 표를 버리는 경로다. top_k 컷오프에 안 쓰이면 MI 는 계산할 이유가 없다.
            compute_mi=bool(self.top_k),
        )
        support = np.zeros(A.shape[1], dtype=bool)
        support[idx] = True
        if not support.any():          # 전부 탈락하면 선별을 포기한다
            support[:] = True
        self.support_ = support
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray([f"f{i}" for i in range(self.n_features_in_)])
        return np.asarray(input_features)[self.support_]

    @property
    def selected_index_(self) -> list[int]:
        return list(np.flatnonzero(self.support_))


def jaccard_stability(fold_sets: list[set]) -> pd.DataFrame:
    """폴드 간 선별 피처 중복도. 낮으면 선별이 폴드마다 흔들린다는 뜻이다."""
    rows = []
    for i in range(len(fold_sets)):
        for j in range(i + 1, len(fold_sets)):
            a, b = fold_sets[i], fold_sets[j]
            union = len(a | b)
            rows.append({"fold_a": i + 1, "fold_b": j + 1,
                         "jaccard": round(len(a & b) / union, 4) if union else float("nan"),
                         "n_a": len(a), "n_b": len(b), "공통": len(a & b)})
    return pd.DataFrame(rows)
