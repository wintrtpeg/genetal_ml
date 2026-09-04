"""실행 결과를 runs/<타임스탬프>/ 아래에 남긴다.

접속 정보는 저장하지 않는다. 쿼리문은 재현을 위해 남기되, 계정·비밀번호는 뺀다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# 재현에 필요한 최소 패키지 목록. 없는 것은 조용히 건너뛴다.
TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "joblib",
    "xgboost", "lightgbm", "catboost", "shap", "plotly", "streamlit",
]


def package_versions() -> dict[str, str]:
    """설치된 패키지 버전. 몇 달 뒤 같은 결과가 안 나올 때 첫 번째로 볼 곳이다."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    out["python"] = ".".join(str(v) for v in __import__("sys").version_info[:3])
    return out


def dataset_fingerprint(df: pd.DataFrame, sample_rows: int = 5000) -> dict:
    """데이터셋 지문. 같은 파일인지 아닌지를 나중에 확인할 수 있게 한다.

    전체를 해싱하면 큰 데이터에서 느리므로, 형태·컬럼·시간범위와 균등 표본
    행의 해시를 합쳐 쓴다. 값이 하나라도 바뀌면 지문이 달라진다.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(f"{df.shape}".encode())
    h.update("|".join(map(str, df.columns)).encode())
    h.update("|".join(str(t) for t in df.dtypes).encode())

    if len(df):
        step = max(1, len(df) // max(sample_rows, 1))
        sample = df.iloc[::step]
        try:
            h.update(pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes())
        except (TypeError, ValueError):
            h.update(sample.to_csv().encode("utf-8", errors="replace"))

    out = {
        "sha256": h.hexdigest(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
    }
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        out["index_start"] = df.index[0].isoformat()
        out["index_end"] = df.index[-1].isoformat()
    return out


def split_bounds(index, split) -> dict:
    """train / validation / final unseen 의 경계 시각과 행 수."""
    out: dict = {}
    for name in ("train", "valid", "unseen"):
        idx = getattr(split, name, None)
        if idx is None or len(idx) == 0:
            continue
        out[name] = {
            "rows": int(len(idx)),
            "start": pd.Timestamp(index[int(min(idx))]).isoformat(),
            "end": pd.Timestamp(index[int(max(idx))]).isoformat(),
        }
    return out


def build_manifest(
    run_id: str,
    target: str | None = None,
    df: pd.DataFrame | None = None,
    split=None,
    index=None,
    seed: int | None = None,
    champion: str | None = None,
    configs: dict | None = None,
    selection_report: pd.DataFrame | None = None,
    unseen_scores: dict | None = None,
    source_desc: str = "",
) -> dict:
    """SPEC §26 재현성 기록. 이 파일만 있으면 같은 실행을 다시 만들 수 있어야 한다."""
    man: dict = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "champion": champion,
        "seed": seed,
        "source": source_desc,
        "packages": package_versions(),
    }
    if df is not None:
        man["dataset"] = dataset_fingerprint(df)
    if split is not None and index is not None:
        man["split_bounds"] = split_bounds(index, split)
    if configs:
        man["config"] = _jsonable(configs)
    if unseen_scores:
        man["final_unseen"] = _jsonable(unseen_scores)
    if selection_report is not None and not selection_report.empty:
        cols = [c for c in ("feature", "status", "reason") if c in selection_report.columns]
        if "status" in cols:
            rep = selection_report[cols]
            # status 는 "selected" / "removed" 외에 "selected(수동추가)" 처럼 사람이
            # 바꾼 표식이 붙는다. 접두사로 봐야 수동 변경분이 기록에서 빠지지 않는다.
            st = rep["status"].astype(str)
            man["features_selected"] = rep[st.str.startswith("selected")]["feature"].tolist()
            man["features_excluded"] = (
                rep[st.str.startswith("removed")][["feature", "reason"]].to_dict("records"))
            manual = rep[st.str.contains("수동", na=False)]
            if not manual.empty:
                man["manual_overrides"] = manual.to_dict("records")
    return man


def new_run_dir(tag: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    p = RUNS_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "size", 1) == 1:
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_run(
    run_dir: Path,
    pipeline=None,
    leaderboard: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    config: dict | None = None,
    provenance: pd.DataFrame | None = None,
    report_html: str | None = None,
    manifest: dict | None = None,
    selection_report: pd.DataFrame | None = None,
) -> dict[str, str]:
    """남길 수 있는 것만 남기고 저장된 경로를 돌려준다."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    if pipeline is not None:
        p = run_dir / "champion_model.joblib"
        joblib.dump(pipeline, p)
        saved["model"] = str(p)
    if leaderboard is not None:
        p = run_dir / "leaderboard.csv"
        leaderboard.to_csv(p, index=False, encoding="utf-8-sig")
        saved["leaderboard"] = str(p)
    if predictions is not None:
        p = run_dir / "predictions.csv"
        predictions.to_csv(p, encoding="utf-8-sig")
        saved["predictions"] = str(p)
    if provenance is not None:
        p = run_dir / "feature_provenance.csv"
        provenance.to_csv(p, index=False, encoding="utf-8-sig")
        saved["provenance"] = str(p)
    if config is not None:
        p = run_dir / "config.json"
        p.write_text(json.dumps(_jsonable(config), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        saved["config"] = str(p)
    if selection_report is not None:
        p = run_dir / "selection_report.csv"
        selection_report.to_csv(p, index=False, encoding="utf-8-sig")
        saved["selection_report"] = str(p)
    if manifest is not None:
        p = run_dir / "manifest.json"
        p.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        saved["manifest"] = str(p)
    if report_html is not None:
        p = run_dir / "report.html"
        p.write_text(report_html, encoding="utf-8")
        saved["report"] = str(p)
    return saved


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    out: dict = {"path": str(run_dir)}
    if (p := run_dir / "champion_model.joblib").exists():
        out["pipeline"] = joblib.load(p)
    if (p := run_dir / "leaderboard.csv").exists():
        out["leaderboard"] = pd.read_csv(p)
    if (p := run_dir / "predictions.csv").exists():
        out["predictions"] = pd.read_csv(p, index_col=0, parse_dates=True)
    if (p := run_dir / "config.json").exists():
        out["config"] = json.loads(p.read_text(encoding="utf-8"))
    if (p := run_dir / "manifest.json").exists():
        out["manifest"] = json.loads(p.read_text(encoding="utf-8"))
    if (p := run_dir / "selection_report.csv").exists():
        out["selection_report"] = pd.read_csv(p)
    return out


def challenge(
    champion_name: str,
    champion_scores: dict,
    challenger_name: str,
    challenger_scores: dict,
    metric: str = "R2",
    threshold: float = 0.02,
) -> dict:
    """Champion–Challenger 교체 판정 (SPEC §18).

    새 모델이 기존 모델보다 **의미 있게** 나아졌을 때만 교체한다. 운영에서 모델을
    바꾸는 데는 재검증·문서화 비용이 따르므로, 소수점 뒤 개선으로 갈아타면 손해다.

    두 점수는 반드시 **같은 구간**에서 나와야 한다. 서로 다른 unseen 에서 잰 값을
    비교하면 구간 난이도 차이를 성능 차이로 오독한다.
    """
    from .train import HIGHER_IS_BETTER

    key = metric if metric in champion_scores else f"unseen_{metric}"
    cur = champion_scores.get(key)
    new = challenger_scores.get(key)
    if cur is None or new is None:
        return {"decision": "판정 불가", "reason": f"{key} 점수가 양쪽에 모두 필요합니다."}

    cur, new = float(cur), float(new)
    higher = HIGHER_IS_BETTER.get(metric, True)
    gain = ((new - cur) / abs(cur)) if higher else ((cur - new) / abs(cur))
    swap = gain >= threshold

    return {
        "decision": "교체" if swap else "유지",
        "champion": champion_name,
        "challenger": challenger_name,
        "metric": metric,
        "champion_score": round(cur, 6),
        "challenger_score": round(new, 6),
        "개선율": round(gain, 4),
        "threshold": threshold,
        "reason": (
            f"{challenger_name} 이 {gain:.1%} 개선하여 임계 {threshold:.0%} 를 넘었습니다."
            if swap else
            f"개선율 {gain:.1%} 가 임계 {threshold:.0%} 에 못 미칩니다. "
            f"{champion_name} 을 유지합니다."
        ),
    }


def compare_manifests(a: dict, b: dict) -> pd.DataFrame:
    """두 run 이 정말 같은 조건이었는지 대조한다.

    "지난달 결과가 안 나온다"의 원인은 대개 데이터 지문이나 패키지 버전이다.
    """
    rows = []

    def add(field: str, va, vb) -> None:
        rows.append({"항목": field, "A": va, "B": vb,
                     "일치": "○" if va == vb else "✕"})

    add("dataset sha256", (a.get("dataset") or {}).get("sha256"),
        (b.get("dataset") or {}).get("sha256"))
    add("행수", (a.get("dataset") or {}).get("rows"), (b.get("dataset") or {}).get("rows"))
    add("target", a.get("target"), b.get("target"))
    add("seed", a.get("seed"), b.get("seed"))
    add("champion", a.get("champion"), b.get("champion"))

    pa, pb = a.get("packages", {}), b.get("packages", {})
    for name in sorted(set(pa) | set(pb)):
        add(f"pkg {name}", pa.get(name), pb.get(name))

    sa, sb = a.get("split_bounds", {}), b.get("split_bounds", {})
    for seg in ("train", "valid", "unseen"):
        add(f"{seg} 경계",
            f"{(sa.get(seg) or {}).get('start')} ~ {(sa.get(seg) or {}).get('end')}",
            f"{(sb.get(seg) or {}).get('start')} ~ {(sb.get(seg) or {}).get('end')}")
    return pd.DataFrame(rows)


def list_runs() -> pd.DataFrame:
    if not RUNS_DIR.exists():
        return pd.DataFrame(columns=["run", "생성", "모델", "리포트"])
    rows = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        rows.append({
            "run": d.name,
            "생성": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "모델": (d / "champion_model.joblib").exists(),
            "리포트": (d / "report.html").exists(),
        })
    return pd.DataFrame(rows)
