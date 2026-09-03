"""수정 전후 비교 — 2분할이 보고하던 숫자 vs 3분할이 보고하는 숫자.

    python scripts/compare_split.py

묻는 것은 하나다. **홀드아웃으로 모델을 고르고 그 점수를 최종 성능이라 부르면
얼마나 낙관적인가?** 모델을 N개 비교해 1등을 뽑으면 그 1등 점수에는 "N개 중
최대값" 효과가 섞인다. 그 크기를 여러 데이터셋에서 재서 분포로 본다.

비교 대상
  (A) 기존 2분할  — holdout 에서 챔피언을 뽑고 그 holdout 점수를 보고
  (B) 신규 3분할  — validation 에서 챔피언을 뽑고 Final Unseen 점수를 보고
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import models, preprocess, train, validation  # noqa: E402


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


MODELS = ["Ridge", "ElasticNet", "DecisionTree", "RandomForest",
          "ExtraTrees", "HistGradientBoosting"]


def synth(seed: int, n: int = 3000, n_feat: int = 25) -> tuple[pd.DataFrame, pd.Series]:
    """자기상관이 있는 시계열. 잡음 피처를 섞어 모델 간 성능이 갈리게 한다."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    drift = np.cumsum(rng.normal(0, 0.05, n))
    cols = {}
    for i in range(n_feat):
        base = np.cumsum(rng.normal(0, 1, n)) if i % 3 == 0 else rng.normal(0, 1, n)
        cols[f"f{i:02d}"] = base + (drift if i < 5 else 0)
    X = pd.DataFrame(cols, index=idx)
    signal = (1.8 * X["f00"] - 1.1 * X["f03"] + 0.9 * np.sin(X["f06"])
              + 0.6 * X["f09"] * X["f12"])
    y = pd.Series(signal + rng.normal(0, 1.0, n), index=idx, name="y")
    return X, y


def one_run(seed: int) -> dict:
    X, y = synth(seed)
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo(models.TASK_REGRESSION, include_heavy=False)
    picked = [m for m in MODELS if m in zoo]
    cfg = train.TrainConfig(task=models.TASK_REGRESSION,
                            split=validation.SplitConfig(n_splits=4),
                            n_jobs=-1, fold_selection=False)

    # (B) 3분할 — 학습·선택은 valid, 보고는 unseen
    sp = validation.three_way_split(len(X), 0.20, 0.15, gap=0)
    board, detail = train.train_all(X, y, sp.train, sp.valid, pre, zoo, picked, cfg)
    champ = train.pick_champion(board, "R2")
    valid_r2 = float(board[board["model"] == champ]["holdout_R2"].iloc[0])
    guard = train.UnseenGuard(sp.unseen)
    unseen = train.evaluate_unseen(detail[champ]["_pipeline"], X, y, sp.unseen, cfg, guard)
    unseen_r2 = unseen["unseen_R2"]

    # 같은 unseen 에서 실제로 제일 좋았던 모델 — "고르기 전에 알 수 없었던" 상한
    best_possible = max(
        train.regression_scores(y.iloc[sp.unseen],
                                d["_pipeline"].predict(X.iloc[sp.unseen]))["R2"]
        for d in detail.values() if d.get("status") == "ok")

    return {
        "seed": seed,
        "챔피언": champ,
        "검증R2(구 보고값)": round(valid_r2, 4),
        "UnseenR2(신 보고값)": round(unseen_r2, 4),
        "낙관편차": round(valid_r2 - unseen_r2, 4),
        "unseen최고": round(best_possible, 4),
        "선택손실": round(best_possible - unseen_r2, 4),
    }


def main() -> int:
    _enable_utf8()
    print("모델 6종 · 3,000행 25피처 · 시드 8개")
    print("=" * 78)
    rows = [one_run(s) for s in range(8)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    b = df["낙관편차"]
    print("\n" + "=" * 78)
    print("낙관편차 = 검증 점수(모델선택에 쓴 값) - Final Unseen 점수(실제 일반화)")
    print(f"  평균 {b.mean():+.4f} · 중앙값 {b.median():+.4f} · "
          f"범위 {b.min():+.4f} ~ {b.max():+.4f}")
    print(f"  양수(낙관) {int((b > 0).sum())}/{len(b)}회, "
          f"음수(비관) {int((b < 0).sum())}/{len(b)}회")
    print()
    print("선택손실 = unseen 에서 실제 최고였던 모델 - 우리가 고른 챔피언")
    s = df["선택손실"]
    print(f"  평균 {s.mean():+.4f} · 최대 {s.max():+.4f}")
    print()
    print("해석: 낙관편차가 0 근처를 오가면 이 데이터에서는 홀드아웃 재사용의 편향이")
    print("      작다는 뜻이다. 크게 양수로 쏠리면 기존 2분할 보고값이 부풀려져 있었다는 뜻이다.")
    print("      어느 쪽이든 3분할은 '재보고 없이 한 번에 확인할 수단'을 만들어 준다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
