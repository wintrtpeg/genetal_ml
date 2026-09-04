"""폴드 내부 선별의 실제 비용을 잰다.

'5배'는 상대값일 뿐이라 판단 근거가 못 된다. 학습 시간과 비교한 절대값이 필요하다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import features, models, preprocess, train, validation  # noqa: E402


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



def make(n_rows: int, n_feat: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="5min")
    X = pd.DataFrame(rng.normal(size=(n_rows, n_feat)),
                     index=idx, columns=[f"f{i:03d}" for i in range(n_feat)])
    y = pd.Series(X.iloc[:, :5].to_numpy() @ np.array([2.0, -1.5, 1.0, 0.7, -0.4])
                  + rng.normal(0, 0.5, n_rows), index=idx, name="y")
    return X, y


def main() -> int:
    _enable_utf8()
    print(f"{'행':>7} {'피처':>5} | {'선별 1회':>9} | {'4폴드 선별':>10} | {'추가시간':>8}")
    print("-" * 60)

    costs = {}
    for n_rows, n_feat in [(3_000, 60), (12_000, 200), (30_000, 200)]:
        X, y = make(n_rows, n_feat)
        tr, _ = validation.time_holdout(n_rows, 0.2, 0)
        Xt, yt = X.iloc[tr], y.iloc[tr]

        t0 = time.perf_counter()
        features.select_features(Xt, yt, top_k=40)
        once = time.perf_counter() - t0

        cv = validation.make_cv(validation.SplitConfig(n_splits=4))
        t0 = time.perf_counter()
        for f_tr, _ in cv.split(Xt):
            features.select_features(Xt.iloc[f_tr], yt.iloc[f_tr], top_k=40)
        folded = time.perf_counter() - t0

        costs[(n_rows, n_feat)] = (once, folded)
        print(f"{n_rows:>7,} {n_feat:>5} | {once:>8.2f}s | {folded:>9.2f}s | "
              f"{folded - once:>+7.2f}s")

    print()
    print("비교 기준 — 같은 데이터로 모델 5종 학습에 걸리는 시간")
    print("-" * 60)
    X, y = make(12_000, 200)
    tr, te = validation.time_holdout(12_000, 0.2, 0)
    sel, _ = features.select_features(X.iloc[tr], y.iloc[tr], top_k=40)
    Xs = X[sel]
    num, cat = preprocess.split_column_types(Xs)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo(models.TASK_REGRESSION, include_heavy=False)
    picked = [m for m in ("Ridge", "DecisionTree", "HistGradientBoosting") if m in zoo]
    cfg = train.TrainConfig(task=models.TASK_REGRESSION,
                            split=validation.SplitConfig(n_splits=4), n_jobs=1)
    t0 = time.perf_counter()
    board, _ = train.train_all(Xs, y, tr, te, pre, zoo, picked, cfg)
    fit = time.perf_counter() - t0
    print(f"  모델 {len(picked)}종 (1코어) : {fit:.1f}s")

    once, folded = costs[(12_000, 200)]
    extra = folded - once
    print()
    print(f"  12,000행 200피처 기준 폴드 내부 선별 추가비용: {extra:+.2f}s")
    print(f"  같은 조건 학습시간 대비: {100 * extra / fit:.1f}%")

    # ── 실제 경로 측정 — FoldSelector 를 Pipeline 에 끼운 전체 학습 시간 ──
    print()
    print("실측 — Pipeline 안의 FoldSelector 를 켜고 끈 전체 학습 시간 비교")
    print("-" * 60)
    import os
    n_jobs = -1
    print(f"  코어 {os.cpu_count()}개 · 모델 {len(picked)}종 · "
          f"선별 대상 {X.shape[1]}피처 → 상위 40")
    rows = []
    for label, on, top_k in (("선별 OFF (폴드 밖 1회)", False, None),
                             ("선별 ON  (폴드 내부)", True, 40)):
        cfg2 = train.TrainConfig(
            task=models.TASK_REGRESSION, split=validation.SplitConfig(n_splits=4),
            n_jobs=n_jobs, fold_selection=on, selection_top_k=top_k)
        src = Xs if not on else X          # ON 은 선별 전 전체 피처로 시작한다
        t0 = time.perf_counter()
        _, det = train.train_all(src, y, tr, te, pre if not on else
                                 preprocess.build_preprocessor(
                                     *preprocess.split_column_types(X),
                                     preprocess.PreprocessConfig()),
                                 zoo, picked, cfg2)
        el = time.perf_counter() - t0
        jac = [d["fold_jaccard"] for d in det.values() if "fold_jaccard" in d]
        rows.append((label, el, float(np.mean(jac)) if jac else float("nan")))
        print(f"  {label:<24} {el:>7.1f}s"
              + (f"  · 폴드간 Jaccard {np.mean(jac):.3f}" if jac else ""))

    off, on_ = rows[0][1], rows[1][1]
    print(f"\n  전체 학습시간 증가: {on_ - off:+.1f}s ({100 * (on_ - off) / off:+.1f}%)")
    print("  ※ ON 은 200피처에서 시작해 폴드마다 40개를 고르고, OFF 는 이미 선별된 40피처로 학습합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
