"""설치된 라이브러리로만 확인할 수 있는 것들을 한 번에 점검한다.

    python scripts/verify_env.py

왜 이 파일이 따로 있는가
------------------------
이 도구를 만든 환경에는 plotly · shap · streamlit · 부스팅 3종 · SQLAlchemy 가
설치돼 있지 않다 (사내망 패키지 설치 차단). 그래서 **그 라이브러리를 실제로
호출하는 코드는 한 번도 실행된 적이 없다.** 문법 검사와 소스 검사로 잡을 수 있는
것은 잡았지만, "이 버전에서 그 인자가 사라졌는가" 같은 것은 실제로 불러 봐야 안다.

7차에서 이미 한 번 당했다 — 테스트 러너의 fixture 탐지가 pytest 설치 환경에서만
실패했고, 그 분기는 만든 쪽에서 돌려 본 적이 없었다. 같은 일이 차트·SHAP·리포트
쪽에 남아 있을 수 있으므로, 실제 환경에서 한 번에 훑는 창구를 둔다.

실패하면 그 줄을 그대로 알려 주면 된다. 어떤 함수가 어떤 인자로 죽었는지까지 찍는다.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _enable_utf8() -> None:
    """윈도우 콘솔에서 한글·기호가 깨지거나 죽지 않게 한다.

    한글 윈도우의 기본 코덱은 cp949 이고, 여기에는 em dash(—, U+2014)가 없다.
    콘솔 창에 바로 찍을 때는 파이썬이 UTF-16 경로를 쓰므로 문제가 없지만,
    출력을 파일이나 파이프로 넘기는 순간 cp949 로 떨어져 UnicodeEncodeError 로
    죽는다. 결과를 로그로 남기려다 실행 자체가 실패하는 셈이라 미리 막는다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 지금 이 파일을 돌리고 있는 파이썬. 설치 안내를 이걸로 찍어야 사용자가
# **가상환경 안에** 설치하게 된다. 시스템 파이썬에 깔면 다시 돌려도 그대로 건너뜀이다.
VENV_PIP = sys.executable

PASS, FAIL, SKIP = [], [], []


def line(t: str) -> None:
    print(f"\n{'─' * 66}\n{t}\n{'─' * 66}")


def _is_missing_library(e: BaseException) -> bool:
    """'라이브러리가 없다' 와 '있는데 잘못 썼다' 를 갈라야 한다.

    앞쪽은 건너뛸 일이고, 뒤쪽은 고쳐야 할 결함이다. core 는 없는 라이브러리를
    ImportError 가 아니라 자기 예외로 바꿔 던지므로 (ShapUnavailable 등)
    그 이름까지 함께 본다.
    """
    if isinstance(e, ImportError) or type(e).__name__ in (
            "ShapUnavailable", "ModuleNotFoundError"):
        return True
    # SQLAlchemy 는 드라이버가 없으면 ImportError 가 아니라 NoSuchModuleError 를
    # 던진다. "pysqream 을 안 깔았다" 는 고칠 결함이 아니라 건너뛸 일이다.
    if type(e).__name__ == "NoSuchModuleError":
        return True
    return False


def _install_hint(name: str) -> str:
    """건너뛴 항목을 설치 명령으로 바꿔 준다.

    `.venv` 안에 깔아야 한다는 점이 중요하다. 그냥 `pip install` 하면 시스템
    파이썬에 들어가서, 다시 돌려도 여전히 건너뜀으로 남는다.
    """
    pip = f'"{VENV_PIP}" -m pip install' if VENV_PIP else "python -m pip install"
    table = {
        "SQream": "pysqream-sqlalchemy",
        "XGBoost": "xgboost", "LightGBM": "lightgbm", "CatBoost": "catboost",
        "SHAP": "shap", "shap": "shap",
        "차트": "plotly", "리포트": "plotly",
    }
    for key, pkg in table.items():
        if key in name:
            return f"{pip} {pkg}"
    return ""


def check(name: str, fn) -> None:
    """하나 돌려 보고 결과를 모은다. 하나가 죽어도 나머지는 계속 본다."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        if _is_missing_library(e):
            SKIP.append((name, str(e)))
            print(f"  건너뜀  {name}  ({e})")
        else:
            FAIL.append((name, f"{type(e).__name__}: {e}", traceback.format_exc()))
            print(f"  실패    {name}\n          {type(e).__name__}: {e}")
    else:
        PASS.append(name)
        print(f"  통과    {name}")


# ─────────────────────────────────────────────────────────────
def _versions() -> None:
    line("0. 설치된 패키지")
    mods = ["streamlit", "pandas", "numpy", "sklearn", "scipy", "plotly",
            "joblib", "yaml", "shap", "xgboost", "lightgbm", "catboost",
            "sqlalchemy", "pysqream", "pytest"]
    for m in mods:
        try:
            mod = __import__(m)
            v = getattr(mod, "__version__", "(버전 표기 없음)")
            print(f"  {m:<12} {v}")
        except Exception:  # noqa: BLE001
            print(f"  {m:<12} —  (없음)")


# ─────────────────────────────────────────────────────────────
def _charts() -> None:
    """차트 22종을 전부 그려 본다.

    plotly 는 인자가 틀려도 import 시점에는 조용하다. Figure 를 만들 때 터진다.
    화면에서 하나씩 눌러 확인하는 대신 여기서 한 번에 본다.
    """
    line("1. 차트 22종")
    from core import plots

    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    a = pd.Series(np.sin(np.linspace(0, 20, n)) * 10 + 50, index=idx)
    b = a + rng.normal(scale=0.5, size=n)
    resid = a - b

    stats = pd.DataFrame({"residual": resid.to_numpy(),
                          "roll_mean": resid.rolling(12).mean().to_numpy(),
                          "roll_std": resid.rolling(12).std().to_numpy()}, index=idx)
    outliers = pd.DataFrame({"residual": resid.to_numpy()[:5],
                             "robust_z": [3.1, -3.4, 4.0, 3.2, -3.9],
                             "방향": ["과대", "과소", "과대", "과대", "과소"]},
                            index=idx[:5])
    drift = pd.DataFrame({"구간": [1, 2, 3], "MAE": [0.4, 0.5, 0.45],
                          "std": [0.6, 0.8, 0.7]})
    bt = pd.DataFrame({"구간": [1, 2, 3], "평가시작": idx[[0, 100, 200]],
                       "R2": [0.9, 0.85, 0.88], "n_train": [100, 200, 300],
                       "n_test": [50, 50, 50], "status": ["ok"] * 3})
    acf = pd.DataFrame({"lag": range(1, 21), "acf": rng.normal(scale=0.1, size=20)})
    board = pd.DataFrame({"model": ["Ridge", "RandomForest", "ExtraTrees"],
                          "holdout_R2": [0.81, 0.93, 0.95],
                          "status": ["ok"] * 3, "family": ["linear", "ensemble", "ensemble"]})
    imp = pd.DataFrame({"feature": [f"f{i}" for i in range(8)],
                        "mean_abs_shap": np.linspace(1.0, 0.1, 8),
                        "contribution_pct": np.linspace(30, 2, 8)})
    # 컬럼 이름은 explain.dependence_data() 가 실제로 내는 것과 같아야 한다.
    # 아무 이름이나 넣으면 여기서만 KeyError 가 나고, 정작 제품은 멀쩡하다.
    dep = pd.DataFrame({"timestamp": idx,
                        "feature_value": rng.normal(size=n),
                        "shap_value": rng.normal(size=n),
                        "interaction_value": rng.normal(size=n)})
    shift = pd.DataFrame({"feature": [f"f{i}" for i in range(5)],
                          "P1": np.linspace(1, 0.2, 5), "P2": np.linspace(0.9, 0.1, 5)})
    values = pd.DataFrame(rng.normal(size=(n, 6)),
                          columns=[f"f{i}" for i in range(6)], index=idx)
    local = pd.DataFrame({"feature": [f"f{i}" for i in range(6)],
                          "shap_value": rng.normal(size=6),
                          "feature_value": rng.normal(size=6)})
    wres = pd.DataFrame({"baseline": a.to_numpy(), "scenario": b.to_numpy(),
                         "delta": (b - a).to_numpy()}, index=idx)
    curve = pd.DataFrame({"f0": np.linspace(0, 10, 25),
                          "prediction": np.linspace(1, 5, 25),
                          "p10": np.linspace(0.8, 4.6, 25),
                          "p90": np.linspace(1.2, 5.4, 25)})
    ice = pd.DataFrame({"row": np.repeat(np.arange(5), 25),
                        "f0": np.tile(np.linspace(0, 10, 25), 5),
                        "prediction": rng.normal(size=125)})
    score = pd.Series(rng.normal(size=n), index=idx)
    flag = pd.Series(score > 1.5, index=idx)
    labels = pd.Series(rng.integers(0, 3, size=n), index=idx)
    pts = pd.DataFrame({"PC1": rng.normal(size=n), "PC2": rng.normal(size=n)}, index=idx)
    holey = pd.DataFrame({f"c{i}": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n))
                          for i in range(5)}, index=idx)

    cases = [
        ("actual_vs_pred (경계선 포함)", lambda: plots.actual_vs_pred(a, b, train_end=idx[200])),
        ("actual_vs_pred (WebGL 임계 초과)",
         lambda: plots.actual_vs_pred(
             pd.Series(rng.normal(size=5000), index=pd.date_range("2025-01-01", periods=5000, freq="min")),
             pd.Series(rng.normal(size=5000), index=pd.date_range("2025-01-01", periods=5000, freq="min")))),
        ("residual_series", lambda: plots.residual_series(a, b)),
        ("residual_band", lambda: plots.residual_band(stats, outliers)),
        ("residual_drift", lambda: plots.residual_drift(drift)),
        ("backtest_series", lambda: plots.backtest_series(bt)),
        ("residual_acf", lambda: plots.residual_acf(acf, n)),
        ("scatter_actual_pred", lambda: plots.scatter_actual_pred(a, b)),
        ("leaderboard_bar", lambda: plots.leaderboard_bar(board, "R2")),
        ("shap_importance_bar", lambda: plots.shap_importance_bar(imp)),
        ("shap_dependence", lambda: plots.shap_dependence(dep, "f0")),
        ("shap_dependence (시점색)",
         lambda: plots.shap_dependence(dep, "f0", color_mode="time")),
        ("shap_dependence (단색)",
         lambda: plots.shap_dependence(dep, "f0", color_mode="none")),
        ("shap_dependence (구간비교)",
         lambda: plots.shap_dependence(
             dep.assign(period=["A"] * (n // 2) + ["B"] * (n - n // 2)),
             "f0", color_mode="period")),
        ("shap_period_shift", lambda: plots.shap_period_shift(shift, ["P1", "P2"])),
        ("shap_contribution_stream", lambda: plots.shap_contribution_stream(values)),
        ("shap_contribution_stream (리샘플)",
         lambda: plots.shap_contribution_stream(values, freq="1h")),
        ("local_waterfall", lambda: plots.local_waterfall(local, 50.0, 52.3)),
        ("whatif_compare", lambda: plots.whatif_compare(wres)),
        ("pdp_curve", lambda: plots.pdp_curve(curve, "f0")),
        ("pdp_curve (ICE)", lambda: plots.pdp_curve(curve, "f0", ice=ice)),
        ("anomaly_timeline", lambda: plots.anomaly_timeline(score, flag)),
        ("cluster_timeline", lambda: plots.cluster_timeline(labels)),
        ("scatter_2d", lambda: plots.scatter_2d(pts, "PC1", "PC2")),
        ("missing_heat", lambda: plots.missing_heat(holey)),
    ]
    for name, fn in cases:
        check(f"차트 · {name}", fn)


# ─────────────────────────────────────────────────────────────
def _shap() -> None:
    """SHAP 세 경로 — 트리 / 선형 / 그 외(커널·순열 대체).

    shap 은 버전마다 반환 모양이 달라진다 (values 가 ndarray 인지 list 인지,
    base_values 가 스칼라인지 배열인지). 세 경로를 다 지나가 봐야 안다.
    """
    line("2. SHAP 3경로")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    from core import explain, preprocess

    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=400, freq="5min")
    X = pd.DataFrame(rng.normal(size=(400, 6)),
                     columns=[f"f{i}" for i in range(6)], index=idx)
    y = pd.Series(X["f0"] * 2 + X["f1"] - X["f2"] * 0.5 + rng.normal(scale=0.1, size=400),
                  index=idx)

    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())

    from sklearn.pipeline import Pipeline

    def run(est, label):
        def _():
            pipe = Pipeline([("prep", pre), ("est", est)]).fit(X, y)
            res = explain.compute_shap(pipe, X, explain.ShapConfig(max_samples=200))
            imp = explain.importance(res)
            assert not imp.empty, "기여도 표가 비었습니다"
            lo, hi = explain.period_bounds(res)
            explain.slice_period(res, lo, hi)
            top = imp["feature"].iloc[0]
            explain.dependence_data(res, top)
            explain.local_explanation(res, X.index[10])
            print(f"          explainer={res['explainer']} · "
                  f"n={res['n_samples']} · base={res['base_value']:.4g}")
        check(f"SHAP · {label}", _)

    run(RandomForestRegressor(n_estimators=25, random_state=0), "트리 (TreeExplainer)")
    run(Ridge(), "선형 (LinearExplainer)")
    run(KNeighborsRegressor(n_neighbors=5), "그 외 (커널 또는 대체 경로)")

    def _fallback():
        pipe = Pipeline([("prep", pre), ("est", Ridge())]).fit(X, y)
        out = explain.permutation_importance_fallback(pipe, X, y)
        assert not out.empty
    check("SHAP · 순열 중요도 대체", _fallback)


# ─────────────────────────────────────────────────────────────
def _ensemble_shap() -> None:
    """앙상블 SHAP 분해를 **진짜 shap 으로** 돌려 본다.

    이 도구를 만든 환경에는 shap 이 없어서, 20차에 넣은 앙상블 분해는
    수식 성질과 합치는 계산만 가짜로 검증했다. **실제 라이브러리로 한 번도
    돌아본 적이 없는 경로**라 여기서 확인한다.

    핵심은 **가법성**이다 — SHAP 값의 합에 기준값을 더하면 모델 예측과 같아야
    한다. 분해가 틀렸다면 여기서 어긋난다. 이게 맞으면 근사가 아니라 정확한
    값이라는 주장이 실제로 성립하는 것이다.
    """
    line("2-b. 앙상블 SHAP 분해 (가법성)")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    from core import ensemble, explain

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)},
                     index=pd.date_range("2025-01-01", periods=n, freq="5min"))
    y = 3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.1, size=n)

    members = {}
    for name, est in (("rf", RandomForestRegressor(n_estimators=12, random_state=0)),
                      ("dt", DecisionTreeRegressor(max_depth=6, random_state=0))):
        pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
        pipe.fit(X, y)
        members[name] = pipe

    for label, blend in (("단순평균", ensemble.MeanBlend(members)),
                         ("가중블렌드", ensemble.WeightedBlend(members))):
        if isinstance(blend, ensemble.WeightedBlend):
            blend.weights_ = np.array([0.7, 0.3])

        def _(b=blend, nm=label):
            plan = explain.plan(b, len(X))
            if plan["method"] != "blend":
                raise RuntimeError(f"분해 경로로 안 갔습니다: {plan['method']}")
            res = explain.compute_shap(b, X, explain.ShapConfig(max_samples=120))
            got = res["values"].sum(axis=1) + res["base_value"]
            want = pd.Series(b.predict(X.loc[res["values"].index]),
                             index=res["values"].index)
            err = float((got - want).abs().max())
            scale = float(want.abs().mean()) or 1.0
            print(f"          {res['explainer']}")
            print(f"          가법성 최대오차 {err:.2e} (예측 평균크기 {scale:.2f})")
            if err > 1e-6 * max(scale, 1.0) * 100:
                raise RuntimeError(
                    f"SHAP 합 + 기준값 != 예측. 최대오차 {err:.3e} — 분해가 틀렸습니다.")
        check(f"앙상블 SHAP · {label}", _)


def _boosting() -> None:
    """부스팅 3종이 실제로 학습되는지. 설치만 되고 DLL 이 없는 경우가 있다."""
    line("3. 부스팅 3종")
    from core import models, preprocess, train, validation

    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-01-01", periods=600, freq="5min")
    X = pd.DataFrame(rng.normal(size=(600, 5)),
                     columns=[f"f{i}" for i in range(5)], index=idx)
    y = pd.Series(X["f0"] * 3 + rng.normal(scale=0.2, size=600), index=idx)

    zoo = models.get_model_zoo(models.TASK_REGRESSION, include_heavy=True)
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    split = validation.build_split(validation.SplitConfig(gap=0), X.index)
    cfg = train.TrainConfig(task=models.TASK_REGRESSION, n_jobs=1,
                            split=validation.SplitConfig(n_splits=2, gap=0))

    for name in ("XGBoost", "LightGBM", "CatBoost"):
        if name not in zoo:
            SKIP.append((f"부스팅 · {name}", "라이브러리 없음"))
            print(f"  건너뜀  부스팅 · {name}  (설치돼 있지 않습니다)")
            continue

        def _(nm=name):
            board, _d = train.train_all(X, y, split.train, split.valid, pre, zoo, [nm], cfg)
            row = board.iloc[0]
            if row["status"] != "ok":
                raise RuntimeError(row.get("error", "알 수 없는 실패"))
            print(f"          holdout_R2={row['holdout_R2']:.4f} · "
                  f"{row['fit_seconds']:.1f}s")
        check(f"부스팅 · {name}", _)


# ─────────────────────────────────────────────────────────────
def _sql() -> None:
    """SQLAlchemy 와 SQream 드라이버. 실제 접속은 하지 않는다 (계정이 필요하므로)."""
    line("4. SQL 경로 (접속은 하지 않음)")
    from core import datasource as d

    def _url_builds():
        """드라이버 없이도 확인할 수 있는 것 — URL 조립과 비밀번호 마스킹."""
        url = d.build_url("sqream", "host.example", 3108, "master", "svc", "p@ss w0rd")
        assert url.startswith("sqream://svc:")
        assert "@host.example:3108/master" in url
        assert "p%40ss+w0rd" in url, "비밀번호가 URL 인코딩되지 않았습니다"
        assert "p@ss w0rd" not in d.mask_url(url)
        print(f"          {d.mask_url(url)}")
    check("SQL · URL 조립 · 비밀번호 마스킹", _url_builds)

    def _engine_builds():
        # 드라이버(pysqream-sqlalchemy)가 없으면 여기서 NoSuchModuleError 가 난다.
        # 그건 고칠 결함이 아니라 "안 깔았다" 는 사실이므로 건너뜀으로 잡힌다.
        import sqlalchemy as sa  # noqa: F401
        url = d.build_url("sqream", "host.example", 3108, "master", "svc", "pw")
        src = d.SqlAlchemySource(url=url, query="SELECT 1",
                                 connect_args=d.DEFAULT_CONNECT_ARGS["sqream"])
        src._engine()
        print("          드라이버 로드 성공 (접속은 하지 않았습니다)")
    check("SQL · SQream 드라이버 로드", _engine_builds)

    def _guard():
        for bad in ("DROP TABLE t", "SELECT 1; DELETE FROM t", "UPDATE t SET a=1"):
            try:
                d.validate_select(bad)
            except d.QueryNotAllowed:
                continue
            raise AssertionError(f"막았어야 할 쿼리를 통과시켰습니다: {bad}")
        d.validate_select("WITH x AS (SELECT 1) SELECT * FROM x")
    check("SQL · 조회 전용 가드", _guard)


# ─────────────────────────────────────────────────────────────
def _report() -> None:
    """단독 HTML 리포트가 실제로 만들어지고 열리는 크기인지."""
    line("5. HTML 리포트")
    from core import report

    def _():
        html = report.build_report(
            title="점검용 리포트",
            meta={"데이터": "verify_env", "타겟": "y"},
            scores={"R2": 0.95, "RMSE": 1.2},
            sections=[{"title": "표 한 개",
                       "note": "여기에 설명이 들어갑니다.",
                       "tables": [pd.DataFrame({"a": [1, 2], "b": [3, 4]})]}],
        )
        out = ROOT / "runs" / "_verify_report.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.save_report(html, out)
        size = out.stat().st_size
        assert size > 1000, f"리포트가 너무 작습니다 ({size}바이트)"
        print(f"          {out} · {size / 1024:,.0f} KB")
    check("리포트 · HTML 생성", _)


# ─────────────────────────────────────────────────────────────
def _preflight_or_none() -> str:
    """묶음이 깨져 있으면 22개 항목을 돌리기 전에 그 사실부터 말한다.

    예전에는 "단계 전체 실패 — AttributeError" 를 두 번 찍고 넘어갔다.
    그건 SHAP·부스팅의 결함처럼 읽히지만 실제 원인은 numpy·scipy 조합이었다.
    """
    try:
        from scripts import envcheck
    except Exception:  # noqa: BLE001
        return ""
    return envcheck.probe()


def main() -> int:
    _enable_utf8()
    print("=" * 66)
    print("실행 환경 점검 — 이 PC 에서만 확인 가능한 것들")
    print("=" * 66)

    _versions()

    # **22개 항목을 돌리기 전에 묶음부터 본다.** 예전에는 numpy·scipy 조합이
    # 깨진 상태에서 "단계 전체 실패 — AttributeError" 를 두 번 찍고 넘어갔다.
    # 그건 SHAP·부스팅의 결함처럼 읽히지만 실제 원인은 전혀 다른 곳에 있었다.
    if (why := _preflight_or_none()):
        from scripts import envcheck
        line("중단 — 패키지 조합")
        print(envcheck.message(why))
        print("\n  이 상태로는 아래 항목을 돌려 봐야 전부 같은 이유로 실패합니다.")
        return 1
    for step in (_charts, _shap, _ensemble_shap, _boosting, _sql, _report):
        try:
            step()
        except Exception as e:  # noqa: BLE001
            if _is_missing_library(e):
                SKIP.append((step.__name__, str(e)))
                print(f"\n  단계 전체 건너뜀 — {e}")
            else:
                FAIL.append((step.__name__, f"{type(e).__name__}: {e}",
                             traceback.format_exc()))
                print(f"\n  단계 전체 실패 — {type(e).__name__}: {e}")

    line("결과")
    print(f"통과 {len(PASS)} · 실패 {len(FAIL)} · 건너뜀 {len(SKIP)}")

    if SKIP:
        print("\n건너뛴 항목 (설치 안 된 라이브러리 — 안 쓸 거면 그대로 두셔도 됩니다)")
        for name, why in SKIP:
            print(f"  · {name} — {why}")
            if (cmd := _install_hint(name)):
                # "무엇이 없다" 만 알려 주면 사용자가 패키지 이름을 다시 찾아야 한다.
                # 특히 SQream 드라이버는 이름이 pysqream-sqlalchemy 라 짐작이 안 된다.
                print(f"      설치하려면:  {cmd}")

    if FAIL:
        print("\n실패 항목 — 아래 블록을 그대로 복사해 알려 주세요")
        print("=" * 66)
        for name, msg, tb in FAIL:
            print(f"\n[{name}]\n{msg}\n")
            print("\n".join(tb.strip().splitlines()[-8:]))
        print("=" * 66)
        return 1

    if SKIP:
        print(f"\n실행된 {len(PASS)}건은 모두 통과했습니다. "
              f"다만 {len(SKIP)}건은 라이브러리가 없어 돌지 않았습니다 — "
              "그 경로는 아직 확인되지 않은 상태입니다.")
    else:
        print("\n전 항목 통과. 이 PC 에서 실행되는 경로에 알려진 문제가 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
