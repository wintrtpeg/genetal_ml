"""시간 순서를 지키는 분할과 누수 검증.

지도학습 경로에서 이 모듈을 우회하는 길은 없다.
- 홀드아웃은 항상 시간축 마지막 구간이고, 학습·튜닝·피처선별 어디에서도 보지 않는다.
- 교차검증은 rolling origin (TimeSeriesSplit) 만 쓴다. 셔플은 제공하지 않는다.
- gap 을 두어 파생변수의 lookback 이 검증 구간을 넘겨보지 못하게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class LeakageError(RuntimeError):
    """시간 순서가 깨졌을 때 발생."""


@dataclass
class SplitConfig:
    holdout_ratio: float = 0.2      # 검증(모델선택) 구간 비율
    n_splits: int = 5               # rolling origin 폴드 수
    gap: int = 0                    # 학습-검증 사이에 비우는 행 수
    max_train_size: int | None = None
    unseen_ratio: float = 0.0       # Final Unseen 비율. 0 이면 기존 2분할 (하위호환)
    valid_cut: object | None = None   # 날짜 지정 시 검증 시작 시각
    unseen_cut: object | None = None  # 날짜 지정 시 Final Unseen 시작 시각

    @property
    def three_way(self) -> bool:
        return self.unseen_ratio > 0 or self.unseen_cut is not None


@dataclass(frozen=True)
class Split:
    """[Train][gap][Validation][gap][Final Unseen] 구간의 행 인덱스.

    unseen 이 비어 있으면 기존 2분할과 동일하다.
    """

    train: np.ndarray
    valid: np.ndarray
    unseen: np.ndarray

    @property
    def three_way(self) -> bool:
        return len(self.unseen) > 0

    def as_holdout(self) -> tuple[np.ndarray, np.ndarray]:
        """기존 (train_idx, test_idx) 2분할 튜플로 돌려준다."""
        return self.train, self.valid

    def describe(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for name, idx in (("Train", self.train), ("Validation", self.valid),
                          ("Final Unseen", self.unseen)):
            if len(idx) == 0:
                continue
            rows.append({
                "구간": name, "행수": len(idx),
                "시작": index[int(np.min(idx))], "끝": index[int(np.max(idx))],
                "비율": round(len(idx) / len(index), 4),
            })
        return pd.DataFrame(rows)


def time_holdout(
    n: int, holdout_ratio: float = 0.2, gap: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """앞 구간을 학습, 뒤 구간을 홀드아웃으로 나눈다."""
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio 는 0 과 1 사이여야 합니다.")
    n_test = max(int(round(n * holdout_ratio)), 1)
    n_train = n - n_test - gap
    if n_train <= 0:
        raise ValueError("홀드아웃 비율 또는 gap 이 너무 큽니다. 학습 구간이 남지 않습니다.")
    return np.arange(n_train), np.arange(n - n_test, n)


def holdout_by_date(index: pd.DatetimeIndex, cut: pd.Timestamp, gap: int = 0):
    """날짜를 직접 지정해서 나눈다."""
    pos = int(np.searchsorted(index.values, np.datetime64(cut)))
    if pos <= 0 or pos >= len(index):
        raise ValueError("기준 시점이 데이터 범위를 벗어났습니다.")
    return np.arange(max(pos - gap, 0)), np.arange(pos, len(index))


def three_way_split(
    n: int, valid_ratio: float = 0.2, unseen_ratio: float = 0.15, gap: int = 0
) -> Split:
    """비율로 3분할한다.

    구간은 시간순으로 [Train][gap][Validation][gap][Final Unseen] 이다.
    Final Unseen 은 챔피언이 확정된 뒤 단 한 번만 접근한다.

    unseen_ratio=0 이면 time_holdout 과 정확히 같은 경계를 돌려준다 (하위호환).
    """
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio 는 0 과 1 사이여야 합니다.")
    if not 0 <= unseen_ratio < 1:
        raise ValueError("unseen_ratio 는 0 이상 1 미만이어야 합니다.")
    if valid_ratio + unseen_ratio >= 1:
        raise ValueError("valid_ratio + unseen_ratio 가 1 이상입니다. 학습 구간이 남지 않습니다.")
    if gap < 0:
        raise ValueError("gap 은 0 이상이어야 합니다.")

    n_unseen = max(int(round(n * unseen_ratio)), 1) if unseen_ratio > 0 else 0
    n_valid = max(int(round(n * valid_ratio)), 1)

    unseen_start = n - n_unseen
    valid_end = unseen_start - (gap if n_unseen else 0)
    valid_start = valid_end - n_valid
    train_end = valid_start - gap

    if train_end <= 0:
        raise ValueError(
            "비율 또는 gap 이 너무 큽니다. 학습 구간이 남지 않습니다 "
            f"(n={n}, valid={n_valid}, unseen={n_unseen}, gap={gap})."
        )
    return Split(
        train=np.arange(train_end),
        valid=np.arange(valid_start, valid_end),
        unseen=np.arange(unseen_start, n) if n_unseen else np.arange(0, dtype=int),
    )


def three_way_split_by_date(
    index: pd.DatetimeIndex,
    valid_cut,
    unseen_cut=None,
    gap: int = 0,
) -> Split:
    """날짜로 3분할한다. 설비 운전조건 변경 시점을 경계로 잡을 때 쓴다.

    valid_cut  — 검증 구간이 시작하는 시각
    unseen_cut — Final Unseen 이 시작하는 시각. None 이면 2분할.
    """
    n = len(index)
    v = int(np.searchsorted(index.values, np.datetime64(pd.Timestamp(valid_cut))))
    if v <= 0 or v >= n:
        raise ValueError("검증 시작 시점이 데이터 범위를 벗어났습니다.")

    if unseen_cut is None:
        u = n
    else:
        u = int(np.searchsorted(index.values, np.datetime64(pd.Timestamp(unseen_cut))))
        if u <= v or u >= n:
            raise ValueError("Final Unseen 시작 시점은 검증 시작보다 뒤, 데이터 끝보다 앞이어야 합니다.")

    valid_end = u - (gap if unseen_cut is not None else 0)
    train_end = v - gap
    if train_end <= 0 or valid_end <= v:
        raise ValueError("gap 이 너무 커서 구간이 비었습니다.")

    return Split(
        train=np.arange(train_end),
        valid=np.arange(v, valid_end),
        unseen=np.arange(u, n) if unseen_cut is not None else np.arange(0, dtype=int),
    )


def build_split(cfg: SplitConfig, index: pd.DatetimeIndex) -> Split:
    """SplitConfig 를 보고 비율 경로와 날짜 경로 중 하나를 고른다."""
    if cfg.valid_cut is not None:
        return three_way_split_by_date(index, cfg.valid_cut, cfg.unseen_cut, cfg.gap)
    return three_way_split(len(index), cfg.holdout_ratio, cfg.unseen_ratio, cfg.gap)


DIAGNOSTIC_ONLY = "diagnostic"


def random_split(
    n: int, test_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """무작위 분할 — **진단 전용**이다.

    시계열에서 무작위로 나누면 검증 행의 바로 앞뒤 행이 학습에 들어간다.
    자기상관이 있는 신호에서는 그것만으로도 점수가 크게 올라가므로, 이 점수는
    미래 성능이 아니다. 그래서 챔피언 선정·리더보드·리포트 어디에도 넣지 않고
    Time split 과의 **격차를 읽는 용도**로만 쓴다 (SPEC §10, §12).

    호출부가 실수로 이걸 평가 경로에 쓰지 못하도록 train 쪽에서 한 번 더 막는다.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio 는 0 과 1 사이여야 합니다.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(int(round(n * test_ratio)), 1)
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def rolling_windows(
    n: int,
    n_folds: int = 5,
    test_size: int | None = None,
    gap: int = 0,
    expanding: bool = True,
    min_train: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling backtest 구간을 만든다.

    CV 와 다른 점은 목적이다. CV 는 모델을 고르려고 점수를 평균 내고, backtest 는
    **시기마다 성능이 어떻게 달라지는지** 보려고 점수를 나열한다. 한 시기만 유난히
    나쁘면 그 구간에 무슨 일이 있었는지 찾아야 한다.

    expanding=True  학습 구간이 계속 커진다 (누적)
    expanding=False 학습 구간 길이를 고정하고 창을 민다 (sliding)
    """
    if n_folds < 1:
        raise ValueError("n_folds 는 1 이상이어야 합니다.")
    test = int(test_size) if test_size else max(n // (n_folds + 1), 1)
    if test < 1:
        raise ValueError("test_size 가 너무 작습니다.")

    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds, 0, -1):
        te_hi = n - (k - 1) * test
        te_lo = te_hi - test
        tr_hi = te_lo - gap
        if te_lo <= 0 or tr_hi <= 0:
            continue
        tr_lo = 0 if expanding else max(tr_hi - (min_train or test * 3), 0)
        if tr_hi - tr_lo < (min_train or 1):
            continue
        windows.append((np.arange(tr_lo, tr_hi), np.arange(te_lo, te_hi)))
    if not windows:
        raise ValueError("backtest 구간을 만들지 못했습니다. 폴드 수나 구간 길이를 줄이세요.")
    return windows


def assert_disjoint(split: Split) -> None:
    """세 구간이 겹치지 않고 시간순인지 단정한다."""
    for a, b in (("train", "valid"), ("valid", "unseen"), ("train", "unseen")):
        ia, ib = getattr(split, a), getattr(split, b)
        if len(ia) == 0 or len(ib) == 0:
            continue
        if np.intersect1d(ia, ib).size:
            raise LeakageError(f"{a} 와 {b} 구간이 겹칩니다.")
        if np.max(ia) >= np.min(ib):
            raise LeakageError(f"{a} 가 {b} 보다 뒤에 있습니다. 시간 순서가 깨졌습니다.")


def make_cv(cfg: SplitConfig) -> TimeSeriesSplit:
    return TimeSeriesSplit(
        n_splits=cfg.n_splits, gap=cfg.gap, max_train_size=cfg.max_train_size
    )


def assert_temporal_order(index: pd.DatetimeIndex, train_idx, test_idx, gap: int = 0) -> None:
    """학습 구간의 마지막 시점이 검증 구간의 첫 시점보다 앞서는지 확인."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise LeakageError("빈 분할이 생성되었습니다.")
    tr_end = index[np.max(train_idx)]
    te_start = index[np.min(test_idx)]
    if tr_end >= te_start:
        raise LeakageError(
            f"학습 구간이 검증 구간을 침범했습니다. 학습 끝 {tr_end}, 검증 시작 {te_start}"
        )
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(test_idx))
    if overlap.size:
        raise LeakageError(f"학습·검증 구간이 {overlap.size}개 행에서 겹칩니다.")


def audit_splits(index: pd.DatetimeIndex, cv: TimeSeriesSplit, n: int) -> pd.DataFrame:
    """모든 폴드를 실제로 돌려보고 경계 시점을 표로 남긴다."""
    rows = []
    dummy = np.zeros((n, 1))
    for k, (tr, te) in enumerate(cv.split(dummy), start=1):
        assert_temporal_order(index, tr, te, gap=cv.gap)
        rows.append({
            "fold": k,
            "train_start": index[tr[0]], "train_end": index[tr[-1]], "n_train": len(tr),
            "valid_start": index[te[0]], "valid_end": index[te[-1]], "n_valid": len(te),
        })
    return pd.DataFrame(rows)


def leakage_checklist(
    index: pd.DatetimeIndex,
    train_idx,
    test_idx,
    feature_names: list[str],
    target: str,
    provenance: pd.DataFrame | None,
    gap: int,
    max_lookback: int,
    selection_idx=None,
    unseen_idx=None,
) -> pd.DataFrame:
    """화면에 그대로 띄우는 누수 점검표. 하나라도 실패하면 학습을 막는다.

    selection_idx — 피처 선별에 실제로 쓴 행 인덱스. 넘기면 평가구간 침범을 검사한다.
    unseen_idx    — Final Unseen 구간. 넘기면 학습·검증과의 격리를 검사한다.
    """
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"항목": name, "결과": "통과" if ok else "실패", "내용": detail})

    ok_order = True
    detail = ""
    try:
        assert_temporal_order(index, train_idx, test_idx, gap)
        detail = f"학습 끝 {index[np.max(train_idx)]} < 홀드아웃 시작 {index[np.min(test_idx)]}"
    except LeakageError as e:
        ok_order, detail = False, str(e)
    add("시간 순서", ok_order, detail)

    add("정렬 상태", bool(index.is_monotonic_increasing), "시간 오름차순 여부")
    # duplicated() 는 전 구간에 해시를 돌린다. .any() 와 .sum() 을 따로 부르면
    # 50만 행짜리 해싱을 두 번 하게 된다. 한 번만 만들어 쓴다.
    dup = int(index.duplicated().sum())
    add("중복 시점", dup == 0, f"중복 {dup}건")

    ok_target = True
    tdetail = "타겟 파생 피처 없음"
    if provenance is not None and not provenance.empty:
        lookup = provenance.set_index("feature")["origin"].to_dict()
        bad = [f for f in feature_names if target in str(lookup.get(f, "")).split("|")]
        ok_target = not bad
        if bad:
            tdetail = "타겟 파생: " + ", ".join(bad[:5])
    add("Y 파생 차단", ok_target, tdetail)

    # gap 은 파생변수의 lookback 창이 평가구간 창과 겹치지 않을 만큼 넓어야 한다.
    # gap < max_lookback 이면 학습 마지막 행과 검증 첫 행의 입력 창이 같은 원자료를 공유한다.
    ok_gap = max_lookback == 0 or gap >= max_lookback
    add("gap 확보", ok_gap,
        f"파생 최대 lookback {max_lookback}행, 설정 gap {gap}행"
        + ("" if ok_gap else f" → gap 을 {max_lookback}행 이상으로 올리세요"))

    # 선별에 쓴 구간이 평가 구간을 침범했는지. 시간 순서만으로는 잡히지 않는다.
    if selection_idx is not None:
        sel = np.asarray(selection_idx)
        evaluated = np.asarray(test_idx)
        if unseen_idx is not None and len(np.asarray(unseen_idx)):
            # 검증과 unseen 은 서로 겹치지 않는 구간이라 union1d 의 정렬·중복제거가
            # 필요 없다. 이어 붙이기만 하면 되고, 그래야 assume_unique 도 성립한다.
            evaluated = np.concatenate([evaluated, np.asarray(unseen_idx)])
        bleed = np.intersect1d(sel, evaluated, assume_unique=True)
        detail_sel = (
            f"선별 {len(sel):,}행 · 평가 {len(evaluated):,}행 · 침범 0행"
            if bleed.size == 0 else
            f"선별에 쓴 {bleed.size:,}행이 평가 구간에 들어갔습니다 "
            f"({index[int(bleed.min())]} ~ {index[int(bleed.max())]}). "
            "3단계 선별을 다시 실행하세요."
        )
        add("선별 구간 격리", bleed.size == 0, detail_sel)

    # Final Unseen 은 학습·검증 어디에도 섞이면 안 된다.
    if unseen_idx is not None and len(np.asarray(unseen_idx)):
        uns = np.asarray(unseen_idx)
        tr, te = np.asarray(train_idx), np.asarray(test_idx)
        used = np.concatenate([tr, te])          # 학습·검증도 서로 겹치지 않는다
        clash = np.intersect1d(uns, used, assume_unique=True)
        after = np.min(uns) > max(int(tr.max()), int(te.max()))
        add("Final Unseen 격리", clash.size == 0 and after,
            f"unseen {len(uns):,}행, 학습·검증과 겹침 {clash.size}행, "
            f"시작 {index[int(np.min(uns))]}")

    return pd.DataFrame(checks)
