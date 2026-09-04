"""P0 대상 결함 5건을 실제로 돌려 수정 전후를 확인한다.

    python scripts/audit_spec.py

각 검증은 "고쳐졌다"를 코드 읽기가 아니라 실행 결과로 보여준다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import ensemble, features, models, preprocess, train, validation  # noqa: E402


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


_enable_utf8()


rng = np.random.default_rng(0)
n = 800
idx = pd.date_range("2025-01-01", periods=n, freq="5min")
x1 = np.cumsum(rng.normal(0, 1, n)) + 50
x2 = rng.normal(10, 2, n)
x3 = rng.normal(0, 1, n)
y = 0.5 * x1 + 2 * x2 + rng.normal(0, 0.5, n)
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=idx)
Y = pd.Series(y, index=idx, name="y")

num, cat = preprocess.split_column_types(X)
pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
zoo = models.get_model_zoo(models.TASK_REGRESSION, include_heavy=False)
PASS, FAIL = "  [OK]  ", "  [!!]  "


def head(k: str, t: str) -> None:
    print("\n" + "=" * 72)
    print(f"[{k}] {t}")
    print("=" * 72)


# ── 결함 1 ──────────────────────────────────────────────────
head("결함 1", "스태킹이 TimeSeriesSplit 에서 실행되는가")
split = validation.SplitConfig(holdout_ratio=0.2, n_splits=4, gap=0)
sp = validation.build_split(split, X.index)
cfg = train.TrainConfig(task=models.TASK_REGRESSION, split=split, n_jobs=1,
                        fold_selection=False)
bases = ["Ridge", "DecisionTree", "RandomForest"]
board, detail = train.train_all(X, Y, sp.train, sp.valid, pre, zoo, bases, cfg)
eb, ed = train.build_ensembles(X, Y, sp.train, sp.valid, pre, zoo, bases, cfg, detail=detail)
for _, r in eb.iterrows():
    mark = PASS if r["status"] == "ok" else FAIL
    print(f"{mark}{r['model']:<22} status={r['status']}  "
          f"holdout_R2={r.get('holdout_R2', float('nan')):.4f}  oof_rows={r.get('oof_rows')}")
    if r["status"] != "ok":
        print(f"         └─ {r.get('error')}")
print("       (이전: Ensemble_Stacking → ValueError: cross_val_predict only works for partitions)")

# ── 결함 2 ──────────────────────────────────────────────────
head("결함 2", "선별에 쓴 구간이 평가 구간으로 새는 것을 점검표가 잡는가")
tr_sel, _ = validation.time_holdout(n, 0.2, 0)
tr_new, te_new = validation.time_holdout(n, 0.35, 0)
overlap = np.intersect1d(tr_sel, te_new)
print(f"  3단계 선별 train = 0 ~ {tr_sel[-1]} ({len(tr_sel)}행)")
print(f"  4단계 재설정 후 holdout = {te_new[0]} ~ {te_new[-1]}")
print(f"  → 선별에 쓴 구간 중 새 홀드아웃에 들어간 행: {overlap.size}개\n")
chk = validation.leakage_checklist(X.index, tr_new, te_new, ["x1"], "y", None, 0, 0,
                                   selection_idx=tr_sel)
print("  " + chk.to_string(index=False).replace("\n", "\n  "))
row = chk[chk["항목"] == "선별 구간 격리"]
print(("\n" + PASS + "침범을 '실패'로 잡아냅니다.")
      if len(row) and row["결과"].iloc[0] == "실패"
      else "\n" + FAIL + "여전히 못 잡습니다.")
print("       (이전: 5개 항목 전부 '통과' 로 표시됨)")

# ── 결함 3 ──────────────────────────────────────────────────
head("결함 3", "홀드아웃이 모델 선택과 최종 보고를 겸하는가")
sp3 = validation.three_way_split(n, 0.2, 0.15, gap=0)
print(sp3.describe(X.index).to_string(index=False))
b3, d3 = train.train_all(X, Y, sp3.train, sp3.valid, pre, zoo,
                         ["Ridge", "DecisionTree", "RandomForest", "ExtraTrees"], cfg)
champ = train.pick_champion(b3, "R2")
guard = train.UnseenGuard(sp3.unseen)
un = train.evaluate_unseen(d3[champ]["_pipeline"], X, Y, sp3.unseen, cfg, guard, who=champ)
bias = train.selection_bias_report(b3, champ, un, "R2")
print(f"\n  챔피언 {champ} — 검증 구간에서 {len(b3)}개 모델 중 1등")
print("  " + bias[["검증(모델선택에 사용)", "Final Unseen(최종 보고)", "격차",
                   "비교한 모델 수"]].to_string(index=False).replace("\n", "\n  "))
try:
    guard.open("두번째")
    print(FAIL + "Unseen 재접근이 막히지 않았습니다.")
except train.UnseenAccessError:
    print(PASS + f"Unseen 접근 {guard.access_count}회로 고정. 재접근은 예외로 막힙니다.")

# ── 결함 4 ──────────────────────────────────────────────────
head("결함 4", "gap 확보 점검이 실제로 검사하는가")
for gap, lb in ((6, 12), (12, 12), (0, 0)):
    tr, te = validation.time_holdout(n, 0.2, gap)
    c = validation.leakage_checklist(X.index, tr, te, ["x1"], "y", None, gap, lb)
    r = c[c["항목"] == "gap 확보"].iloc[0]
    want = "실패" if (lb and gap < lb) else "통과"
    mark = PASS if r["결과"] == want else FAIL
    print(f"{mark}gap={gap:<3} lookback={lb:<3} → {r['결과']} (기대 {want})")
print("       (이전: gap >= 0 이 항상 참이라 세 경우 모두 '통과')")

# ── 결함 5 (P0-12) ──────────────────────────────────────────
head("항목 12", "피처 선별이 CV 폴드 내부에서 다시 도는가")
cfg_on = train.TrainConfig(task=models.TASK_REGRESSION, split=split, n_jobs=1,
                           fold_selection=True, selection_top_k=2)
_, d_on = train.train_all(X, Y, sp.train, sp.valid, pre, zoo, ["Ridge"], cfg_on)
sets = d_on["Ridge"].get("_fold_feature_sets", [])
print(f"  폴드 수 {split.n_splits - 1 if False else len(sets)} · 폴드별 선택 피처")
for i, s in enumerate(sets, 1):
    print(f"    fold{i}: {sorted(s)}")
jac = d_on["Ridge"].get("fold_jaccard")
print(f"{PASS}폴드마다 선별이 다시 돌았습니다. 폴드간 Jaccard {jac:.4f}"
      if sets else f"{FAIL}폴드 내부 선별이 돌지 않았습니다.")

# ── 신규 결함 B ─────────────────────────────────────────────
head("신규 B", "리더보드 재정렬이 두 번 이상 가능한가")
try:
    once = train.sort_leaderboard(b3, "R2")
    twice = train.sort_leaderboard(once, "R2")
    print(f"{PASS}재정렬 성공. rank {list(twice['rank'])[:5]}")
except ValueError as e:
    print(f"{FAIL}{type(e).__name__}: {e}")
print("       (이전: ValueError: cannot insert rank, already exists — smoke_test 가 8단계 전에 죽음)")

print("\n" + "=" * 72)
