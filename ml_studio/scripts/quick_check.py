"""축소 end-to-end 점검.

뒷구간 일부 데이터에 가벼운 모델 3개만 돌려 파이프라인이 끝까지 이어지는지 본다.
저사양 PC 나 CI 에서 쓰는 용도다. 전체 모델 비교는 scripts/smoke_test.py 를 쓴다.

    python scripts/quick_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    datasource, explain, features, models, preprocess, profiling,
    train, validation, whatif,
)


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

LIGHT = ("Ridge", "RandomForest", "HistGradientBoosting", "ExtraTrees", "DecisionTree")


def line(t: str) -> None:
    print(f"\n{'─' * 66}\n{t}\n{'─' * 66}")


def main() -> int:
    _enable_utf8()
    t0 = time.time()
    target = "y_output"

    line("1. 적재")
    raw = datasource.CsvSource(ROOT / "data" / "demo_timeseries.csv").load()
    df = datasource.to_timeseries(raw, "timestamp").iloc[-3000:]
    print(f"{len(df):,}행 × {df.shape[1]}열 · 주기 {datasource.infer_freq(df.index)}")

    line("2. 품질")
    prof = profiling.profile(df)
    rule = profiling.QualityRule()
    drops = profiling.suggest_drops(prof, rule, protect=[target],
                                    corr_pairs=profiling.find_correlated_pairs(df, rule.max_corr))
    keep = [c for c in df.columns if c != target and c not in set(drops["column"])]
    print(f"유지 {len(keep)}개 · 제외 {len(drops)}개")

    line("3. 파생 (Y 파생 차단)")
    fcfg = features.FeatureConfig(lags=[1, 3], rolling_windows=[6], rolling_stats=["mean"],
                                  ewm_spans=[], diffs=[1], allow_target_derived=False)
    feat, prov = features.generate(df, target, keep, fcfg)
    feat = features.drop_warmup(feat, fcfg)
    print(f"X 후보 {feat.shape[1] - 1}개 · {len(feat):,}행")

    line("4. 분할·누수 점검")
    X_all, y_all = preprocess.prepare_xy(feat, target, [c for c in feat.columns if c != target])
    # gap 은 **파생 전체의 lookback** 이어야 한다. max(lags) 만 보면 rolling·ewm 창을
    # 빼먹는다. 여기서는 lags 최대가 3, rolling 창이 6 이라 gap 3 으로는
    # 학습 마지막 행과 홀드아웃 첫 행의 입력 창이 같은 원자료를 공유했다.
    # smoke_test.py 는 진작 고쳤는데 이 파일만 옛 계산이 남아 있었다.
    gap = features.warmup_rows(fcfg)
    split = validation.SplitConfig(holdout_ratio=0.2, n_splits=3, gap=gap)
    tr, te = validation.time_holdout(len(X_all), split.holdout_ratio, split.gap)
    sel, _ = features.select_features(X_all.iloc[tr], y_all.iloc[tr], top_k=20)
    X, y = X_all[sel], y_all
    check = validation.leakage_checklist(X.index, tr, te, sel, target, prov,
                                         split.gap, features.warmup_rows(fcfg))
    print(check.to_string(index=False))
    if (check["결과"] == "실패").any():
        raise SystemExit("누수 점검 실패")

    line("5. 학습 (가벼운 모델만)")
    task = models.detect_task(y)
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo(task, include_heavy=False)
    picked = [m for m in LIGHT if m in zoo] or list(zoo)[:3]
    tcfg = train.TrainConfig(task=task, split=split, n_jobs=1, champion_metric="R2")
    board, detail = train.train_all(X, y, tr, te, pre, zoo, picked, tcfg)
    show = [c for c in ["rank", "model", "cv_R2", "holdout_R2", "holdout_RMSE",
                        "holdout_MAE", "fit_seconds", "status"] if c in board.columns]
    print(board[show].to_string(index=False))

    champ = train.pick_champion(board, "R2")
    pipe = detail[champ]["_pipeline"]
    print(f"\n챔피언 → {champ}")

    line("6. 예측")
    pred = train.predict_range(pipe, X)
    print(f"{len(pred):,}개 시점 · 홀드아웃 R2 "
          f"{train.regression_scores(y.iloc[te], pred.iloc[te])['R2']:.4f}")

    line("7. SHAP 기간 기능")
    try:
        res = explain.compute_shap(pipe, X.iloc[tr], explain.ShapConfig(max_samples=400))
        lo, hi = explain.period_bounds(res)
        mid = lo + (hi - lo) / 2
        periods = [("A", lo, mid), ("B", mid, hi)]
        sub = explain.slice_period(res, mid, hi)
        dep = explain.dependence_by_periods(res, sel[0], periods)
        shift = explain.period_shift(res, periods)
        print(f"{res['explainer']} · 계산 {res['n_samples']}개 → 후반 구간 {sub['n_samples']}개")
        print(f"구간 겹침 {len(dep)}행 · 구간 {sorted(set(dep['period']))}")
        print(shift.head(5).to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    except explain.ShapUnavailable as e:
        print(f"건너뜀 — {e}")

    line("8. What-if")
    f0 = sel[0]
    cfg = whatif.ScenarioConfig(changes=[whatif.Change(feature=f0, mode="pct", value=10.0)])
    out = whatif.run_scenario(pipe, X.iloc[te], cfg)
    print(f"{f0} +10% → {whatif.scenario_summary(out)}")
    curve = whatif.sweep(pipe, X.iloc[te], f0, whatif.suggest_range(X, f0, n=8))
    print(f"sweep {len(curve)}점 · {curve['prediction'].min():.4g} ~ {curve['prediction'].max():.4g}")

    line(f"전 단계 통과 — {time.time() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
