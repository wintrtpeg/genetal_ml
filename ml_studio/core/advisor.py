"""데이터를 먼저 보고 설정을 추천한다.

왜 필요한가
-----------
전처리 방식·lag 창 같은 설정을 **아무 근거 없이 고르라고 하면 안 된다.**
현장 엔지니어가 "결측 대치를 ffill 로 할까 median 으로 할까" 를 판단하려면
결측이 어떤 모양으로 나 있는지를 먼저 알아야 하는데, 그건 데이터를 봐야 안다.
그리고 그건 도구가 대신 볼 수 있다.

그래서 이 모듈은 세 가지를 함께 돌려준다.

  1. **추천값** — 이 데이터라면 이게 맞다
  2. **사유** — 왜 그렇게 봤는지 한 문장
  3. **근거** — 그 판단을 뒷받침하는 표

세 번째가 중요하다. 사유만 있으면 믿거나 말거나가 되지만, 근거 표가 있으면
"우리 설비는 저 태그가 원래 저렇다" 는 현장 판단으로 뒤집을 수 있다.
**추천은 추천일 뿐 강제가 아니다** — 이 도구 전체의 원칙과 같다.

물리적 한계
-----------
lag 추천은 통계만 보면 엉뚱한 값이 나올 수 있다. 상관은 우연히도 높아지므로,
"이 공정에서 반응이 4시간 뒤에 온다는 건 물리적으로 말이 안 된다" 는 지식이
있으면 그걸 상한으로 걸어야 한다. `PhysicalLimits` 가 그 역할이다.
한계를 넘는 후보는 추천에서도, 선택지에서도 빠진다.

streamlit 을 모른다 — 화면은 결과만 받아서 그린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
@dataclass
class Advice:
    """추천 하나. 값·사유·근거를 함께 들고 다닌다."""

    value: Any
    reason: str
    detail: pd.DataFrame | None = None
    confidence: str = "보통"          # 높음 / 보통 / 낮음
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:       # `if advice:` 를 값 기준으로
        return bool(self.value)


@dataclass
class PhysicalLimits:
    """공정 지식으로 거는 상한. 통계가 뭐라 하든 여기를 넘지 않는다.

    max_lag_minutes     — 원인이 결과에 나타나기까지 걸릴 수 있는 최대 시간
    max_rolling_minutes — 의미 있게 묶어 볼 수 있는 최대 구간
    """

    max_lag_minutes: float | None = None
    max_rolling_minutes: float | None = None

    def lag_rows(self, step_min: float | None) -> int | None:
        return _to_rows(self.max_lag_minutes, step_min)

    def rolling_rows(self, step_min: float | None) -> int | None:
        return _to_rows(self.max_rolling_minutes, step_min)


def _to_rows(minutes: float | None, step_min: float | None) -> int | None:
    if not minutes or not step_min or step_min <= 0:
        return None
    return max(1, int(round(minutes / step_min)))


def describe_rows(rows: int, step_min: float | None) -> str:
    """행 수를 사람이 아는 시간으로. 화면과 사유 문장이 같은 표기를 쓰게 한다."""
    if not step_min:
        return f"{rows}행"
    m = rows * step_min
    if m < 60:
        return f"{rows}행({m:g}분)"
    if m < 1440:
        h = m / 60
        return f"{rows}행({h:g}시간)" if h != int(h) else f"{rows}행({int(h)}시간)"
    d = m / 1440
    return f"{rows}행({d:g}일)" if d != int(d) else f"{rows}행({int(d)}일)"


# ─────────────────────────────────────────────────────────────
# 결측 — 어떤 모양으로 비어 있는가
# ─────────────────────────────────────────────────────────────
def missing_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """컬럼별 결측 비율과 **연속 결측 길이**를 잰다.

    비율만으로는 대치 방법을 못 정한다. 같은 5% 라도
      · 2~3행씩 짧게 끊긴 것   → 통신 순단. 직전 값 유지가 맞다
      · 한 번에 500행 비어 있는 것 → 설비 정지. 직전 값을 500행 끌면 거짓말이 된다
    """
    num = df[cols] if cols else df.select_dtypes("number")
    num = num.select_dtypes("number")
    EMPTY = ["컬럼", "결측률", "결측 구간 수", "최장 연속", "평균 연속"]
    if num.empty or not len(num.columns):
        # 글자 컬럼만 있는 표에서도 화면은 떠야 한다. 빈 DataFrame 에
        # sort_values 를 걸면 KeyError 로 죽어서 추천 전체가 날아갔다.
        return pd.DataFrame(columns=EMPTY)
    rows = []
    for c in num.columns:
        s = num[c]
        na = s.isna()
        n_missing = int(na.sum())
        if n_missing == 0:
            rows.append({"컬럼": c, "결측률": 0.0, "결측 구간 수": 0,
                         "최장 연속": 0, "평균 연속": 0.0})
            continue
        # 연속 결측 구간의 길이 분포
        grp = (na != na.shift()).cumsum()[na]
        runs = na[na].groupby(grp).size()
        rows.append({
            "컬럼": c,
            "결측률": float(na.mean()),
            "결측 구간 수": int(len(runs)),
            "최장 연속": int(runs.max()),
            "평균 연속": float(runs.mean()),
        })
    return pd.DataFrame(rows).sort_values("결측률", ascending=False).reset_index(drop=True)


def recommend_impute(df: pd.DataFrame, cols: list[str] | None = None,
                     step_min: float | None = None) -> Advice:
    prof = missing_profile(df, cols)
    if prof.empty or prof["결측률"].max() == 0:
        return Advice("ffill", "결측이 없습니다. 어느 방법을 골라도 결과가 같습니다.",
                      prof, "높음")

    worst = prof.iloc[0]
    hit = prof[prof["결측률"] > 0]

    # **최악의 태그 하나로 전체를 판단하면 안 된다.** 센서 하나가 60% 비어 있다고
    # 멀쩡한 나머지 태그까지 중앙값으로 메우면 손해다. 그 하나는 2단계에서
    # 빼는 게 맞는 대응이고, 대치 방식은 '보통의 태그' 를 기준으로 정한다.
    typical_run = float(hit["평균 연속"].median())
    typical_rate = float(hit["결측률"].median())
    n_long = int((hit["평균 연속"] >= 6).sum())
    long_label = describe_rows(int(hit["최장 연속"].max()), step_min)

    notes = []
    outliers = hit[hit["결측률"] > 0.4]
    if len(outliers):
        notes.append(
            f"결측률이 40%를 넘는 태그가 {len(outliers)}개 있습니다 "
            f"({', '.join(outliers['컬럼'].head(3))}). 대치 방식으로 해결할 문제가 "
            "아니라 2단계에서 빼는 쪽을 먼저 검토하세요.")

    # 결측이 있는 태그의 **절반 이상**이 긴 공백이면 그때 median 이 맞다
    if n_long >= max(1, len(hit) / 2) and typical_run >= 6:
        return Advice(
            "median",
            f"결측이 있는 {len(hit)}개 태그 중 {n_long}개에서 공백이 길게 이어집니다 "
            f"(대표값 {typical_run:.0f}행 연속, 최장 {long_label}). "
            "설비 정지·센서 탈락처럼 긴 공백에서는 직전 값을 그대로 끌면 "
            "없던 평탄 구간을 만들어 냅니다. 중앙값이 안전합니다.",
            prof, "높음",
            notes + ["긴 결측이 정기보수 구간이라면 그 기간을 아예 빼는 편이 낫습니다."])

    return Advice(
        "ffill",
        f"결측이 있는 태그 {len(hit)}개에서 공백이 대체로 짧게 끊깁니다 "
        f"(대표값 {typical_run:.1f}행 연속 · 결측률 {typical_rate:.1%}). "
        "통신 순단 형태이므로 직전 값 유지가 물리적으로 맞습니다 — "
        "실제 값은 그동안에도 유지되고 있었을 테니까요.",
        prof, "높음" if not len(outliers) else "보통", notes)


# ─────────────────────────────────────────────────────────────
# 스케일 — 태그 간 크기 차이와 이상값
# ─────────────────────────────────────────────────────────────
def scale_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """컬럼별 크기와 이상값 정도. 스케일러 선택의 근거가 된다."""
    num = (df[cols] if cols else df).select_dtypes("number")
    rows = []
    for c in num.columns:
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if s.empty:
            continue
        med = float(s.median())
        mad = float((s - med).abs().median())
        # robust z — 표준편차 대신 MAD 를 쓴다. 이상값 자체에 안 끌려간다
        if mad > 0:
            z = (s - med).abs() / (1.4826 * mad)
        else:
            # **MAD 가 0 이어도 이상값이 없는 건 아니다.** 값이 고정된 센서에
            # 스파이크 하나가 튀면 절반 이상이 같은 값이라 MAD 는 0 이 되고,
            # 그러면 z 도 0 이 되어 그 스파이크가 스스로를 감춘다.
            # 이때는 표준편차로 물러선다 (없는 것보다 낫다).
            sd = float(s.std())
            z = ((s - med).abs() / sd) if sd > 0 else pd.Series(0.0, index=s.index)
        rng = float(s.max() - s.min())
        rows.append({
            "컬럼": c, "평균": float(s.mean()), "표준편차": float(s.std()),
            "범위": rng, "최대 robust z": float(z.max()) if len(z) else 0.0,
            "z>5 개수": int((z > 5).sum()),
        })
    return pd.DataFrame(rows)


def recommend_scaler(df: pd.DataFrame, cols: list[str] | None = None) -> Advice:
    prof = scale_profile(df, cols)
    if prof.empty:
        return Advice("standard", "수치 컬럼이 없어 판단할 근거가 없습니다.", prof, "낮음")

    # **거의 상수인 태그를 비율에 넣으면 안 된다.** 표준편차가 1e-6 인 flat_tag 가
    # 끼면 "1,800만 배 차이" 같은 무의미한 숫자가 나오고, 사유 문장이 우스워진다.
    # 그런 태그는 2단계에서 제외 후보로 이미 잡히므로 여기서는 뺀다.
    # 절대 임계로는 못 가린다 — 값이 1.000001 인 태그의 표준편차는 1e-6 이라
    # 임계를 통과하고, 그 결과 "1,800만 배 차이" 같은 무의미한 문장이 나온다.
    # 자기 크기 대비로 얼마나 흔들리는지(변동계수)를 봐야 한다.
    cv = prof["표준편차"] / prof["평균"].abs().replace(0, np.nan)
    alive = (cv.fillna(0) > 1e-4) & (prof["표준편차"] > 0)
    live, flat = prof[alive], prof[~alive]
    scales = live["표준편차"].dropna()
    ratio = float(scales.max() / scales.min()) if len(scales) >= 2 else 1.0
    heavy = prof[prof["최대 robust z"] > 10]

    if len(heavy) >= max(1, len(prof) // 5):
        names = ", ".join(heavy["컬럼"].head(3))
        return Advice(
            "robust",
            f"{len(heavy)}개 태그에 이상값이 뚜렷합니다 "
            f"(최대 robust z {heavy['최대 robust z'].max():.0f} — {names}). "
            "평균·표준편차 기준으로 맞추면 그 이상값이 눈금 자체를 끌어당깁니다. "
            "중앙값·IQR 기준인 Robust 가 적합합니다.",
            prof, "높음")

    notes = []
    if len(flat):
        notes.append(f"거의 움직이지 않는 태그 {len(flat)}개는 비교에서 뺐습니다 "
                     f"({', '.join(flat['컬럼'].head(3))}). 2단계에서 제외 후보로 "
                     "잡혔을 것입니다.")

    if ratio >= 50:
        big = live.loc[live["표준편차"].idxmax(), "컬럼"]
        small = live.loc[live["표준편차"].idxmin(), "컬럼"]
        return Advice(
            "standard",
            f"태그 간 값의 흔들리는 폭이 최대 {ratio:,.0f}배 차이납니다 "
            f"({big} 이 가장 크고 {small} 이 가장 작습니다). "
            "이대로 넣으면 선형·거리 기반 모델이 큰 숫자 쪽을 더 중요하게 봅니다. "
            "Standard 로 맞추는 것이 맞습니다.",
            prof, "높음", notes)

    return Advice(
        "standard",
        f"태그 간 크기 차이가 {ratio:,.1f}배로 크지 않고 이상값도 두드러지지 "
        "않습니다. Standard 로 두면 무난하고, 트리 계열만 쓰실 거면 '없음' 도 "
        "무방합니다.",
        prof, "보통", notes)


# ─────────────────────────────────────────────────────────────
# 범주 인코딩
# ─────────────────────────────────────────────────────────────
_ORDER_HINTS = ("등급", "grade", "level", "레벨", "단계", "step", "class", "rank", "순위")


def categorical_profile(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    sub = df[cols] if cols else df
    obj = sub.select_dtypes(exclude=["number", "datetime", "datetimetz"])
    rows = []
    for c in obj.columns:
        s = obj[c].astype(str)
        vals = list(pd.unique(s.dropna()))[:6]
        rows.append({"컬럼": c, "고유값 수": int(s.nunique()),
                     "예시": ", ".join(map(str, vals))})
    return pd.DataFrame(rows)


def _looks_ordered(name: str, values: list[str]) -> bool:
    """이름이나 값에 순서가 드러나는가 — 1등급/2등급, A/B/C, LOW/MID/HIGH."""
    if any(h in str(name).lower() for h in _ORDER_HINTS):
        return True
    vs = [str(v).strip().upper() for v in values]
    if not vs:
        return False
    if all(v.replace(".", "").isdigit() for v in vs):
        return True
    if set(vs) <= {"LOW", "MID", "MIDDLE", "HIGH", "저", "중", "고"}:
        return True
    if len(vs) <= 8 and all(len(v) == 1 and v.isalpha() for v in vs):
        return True
    return False


def recommend_encoding(df: pd.DataFrame, cols: list[str] | None = None) -> Advice:
    prof = categorical_profile(df, cols)
    if prof.empty:
        return Advice("onehot", "글자로 된 컬럼이 없어 이 설정은 쓰이지 않습니다.",
                      prof, "높음")

    sub = df[cols] if cols else df
    ordered, unordered, wide = [], [], []
    for _, r in prof.iterrows():
        c = r["컬럼"]
        vals = list(pd.unique(sub[c].astype(str).dropna()))[:12]
        if int(r["고유값 수"]) > 20:
            wide.append(c)
        elif _looks_ordered(c, vals):
            ordered.append(c)
        else:
            unordered.append(c)

    notes = []
    if wide:
        notes.append(f"고유값이 20개를 넘는 컬럼이 있습니다 ({', '.join(wide[:3])}). "
                     "상태값이 아니라 ID·설비번호일 수 있으니 2단계에서 확인하세요.")

    if ordered and not unordered:
        return Advice(
            "ordinal",
            f"{', '.join(ordered[:3])} 의 값에 크기 순서가 보입니다 "
            "(등급·레벨 형태). 순서를 살리는 Ordinal 이 적합합니다.",
            prof, "보통", notes)

    if unordered:
        ex = prof[prof["컬럼"] == unordered[0]]["예시"].iloc[0]
        return Advice(
            "onehot",
            f"'{unordered[0]}' 의 값이 순서 없는 상태값으로 보입니다 ({ex}). "
            "번호를 매기면 모델이 없는 크기 관계를 학습하므로 One-Hot 이 맞습니다.",
            prof, "높음", notes)

    return Advice("onehot", "판단할 근거가 약해 기본값(One-Hot)을 둡니다.",
                  prof, "낮음", notes)


# ─────────────────────────────────────────────────────────────
# 극단값 clip
# ─────────────────────────────────────────────────────────────
def recommend_clip(df: pd.DataFrame, cols: list[str] | None = None) -> Advice:
    prof = scale_profile(df, cols)
    if prof.empty:
        return Advice(False, "수치 컬럼이 없습니다.", prof, "낮음")

    hit = prof[prof["z>5 개수"] > 0].sort_values("최대 robust z", ascending=False)
    if hit.empty:
        return Advice(False,
                      "robust z 5를 넘는 값이 없습니다. 계측 오류로 보이는 극단값이 "
                      "없으므로 그대로 두어도 됩니다.", prof, "높음")

    top = hit.iloc[0]
    total = int(hit["z>5 개수"].sum())
    return Advice(
        True,
        f"{len(hit)}개 태그에서 극단값이 발견됐습니다 — "
        f"'{top['컬럼']}' 에 robust z {top['최대 robust z']:.0f} 인 값이 있고 "
        f"전체 {total:,}개 지점이 z>5 입니다. 계측 오류일 가능성이 높고, "
        "그대로 두면 모델이 그 값에 끌려갑니다.",
        hit.reset_index(drop=True), "높음",
        ["실제 운전에서 나올 수 있는 값이라면 끄십시오 — "
         "정상적인 급변을 잘라내면 그 현상을 못 배웁니다."])


# ─────────────────────────────────────────────────────────────
# lag — 상호상관으로 반응 지연을 잰다
# ─────────────────────────────────────────────────────────────
def lag_scan(
    df: pd.DataFrame,
    target: str,
    cols: list[str] | None = None,
    max_lag: int = 96,
    step_min: float | None = None,
    max_cols: int = 40,
    sample: int = 20000,
) -> pd.DataFrame:
    """각 태그를 얼마나 지연시켰을 때 타깃과 가장 잘 맞는지 찾는다.

    공정에서 밸브를 열면 유량이 먼저 바뀌고 온도가 나중에 따라온다. 그 '나중' 이
    몇 분인지는 데이터가 알고 있다. 태그마다 lag 를 0..max_lag 로 밀어 가며
    타깃과의 상관을 재고, 가장 큰 지점을 그 태그의 반응 지연으로 본다.

    **주의** — 상관은 인과가 아니다. 여기서 나온 값은 '후보' 이고, 물리적으로
    말이 되는지는 사람이 봐야 한다. 그래서 PhysicalLimits 로 상한을 건다.
    """
    if target not in df.columns:
        raise KeyError(f"타깃 '{target}' 이 없습니다.")

    num = df.select_dtypes("number")
    use = [c for c in (cols or num.columns) if c in num.columns and c != target]
    if not use:
        return pd.DataFrame(columns=["태그", "최적 lag", "지연 시간", "상관",
                                     "0 lag 상관", "오차 감소율"])

    work = num[[target] + use]
    if len(work) > sample:                     # 등간격 표본 — 지연 구조는 유지된다
        work = work.iloc[:: max(1, len(work) // sample)]
    y = work[target]

    # 열이 아주 많으면 0 lag 상관이 큰 것부터 본다. 전부 훑으면 느리다.
    if len(use) > max_cols:
        base = work[use].corrwith(y).abs().sort_values(ascending=False)
        use = list(base.head(max_cols).index)

    lags = _lag_grid(max_lag)
    rows = []
    for c in use:
        x = work[c]
        best_lag, best_r = 0, 0.0
        at_zero = 0.0
        for L in lags:
            r = x.shift(L).corr(y)
            if r is None or r != r:
                continue
            if L == 0:
                at_zero = abs(float(r))
            if abs(float(r)) > abs(best_r):
                best_lag, best_r = L, float(r)
        # **절대 개선치로 재면 안 된다.** 상관이 0.993 → 0.999 로 가는 것은
        # 절대값으로는 +0.006 이라 하찮아 보이지만, 설명 못 하던 부분의 88%가
        # 사라진 것이다. 실제로 이 기준 때문에 진짜 1시간 지연을 놓쳤다.
        # 남은 오차가 얼마나 줄었는지(설명 못 한 분산의 감소율)로 잰다.
        left0 = max(1e-12, 1.0 - at_zero ** 2)
        leftL = max(0.0, 1.0 - best_r ** 2)
        gain_ratio = (left0 - leftL) / left0
        rows.append({
            "태그": c,
            "최적 lag": best_lag,
            "지연 시간": describe_rows(best_lag, step_min) if best_lag else "즉시",
            "상관": round(abs(best_r), 4),
            "0 lag 상관": round(at_zero, 4),
            "오차 감소율": round(float(max(0.0, gain_ratio)), 4),
        })
    out = pd.DataFrame(rows).sort_values("상관", ascending=False).reset_index(drop=True)
    return out


def _lag_grid(max_lag: int) -> list[int]:
    """0..max_lag 를 촘촘하게 시작해 성기게 끝난다. 전수 조사는 느리고 불필요하다."""
    grid = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144, 192, 288}
    return sorted(x for x in grid if x <= max_lag)


def recommend_lags(
    df: pd.DataFrame,
    target: str,
    cols: list[str] | None = None,
    step_min: float | None = None,
    limits: PhysicalLimits | None = None,
    n_lags: int = 5,
) -> Advice:
    """상호상관에서 나온 지연들을 lag 목록으로 정리한다."""
    limits = limits or PhysicalLimits()
    cap = limits.lag_rows(step_min) or 96
    scan = lag_scan(df, target, cols, max_lag=cap, step_min=step_min)

    if scan.empty:
        return Advice([1, 2, 3, 6, 12],
                      "지연을 잴 수치 태그가 없어 일반적인 기본값을 둡니다.",
                      scan, "낮음")

    # 두 가지를 함께 요구한다.
    #   1) 그 태그가 타깃과 애초에 관련이 있어야 하고 (상관 0.2 이상)
    #   2) **지연시켰을 때 실제로 나아져야 한다** (0 lag 대비 개선 0.02 이상)
    #
    # 하나만 보면 엉뚱한 값이 나온다. 실제로 상관 0.10 짜리 노이즈 태그가
    # 8시간 지연에서 우연히 0.1026 이 되는 것만으로 "8시간 lag 추천" 이 나왔다.
    # 상관이 낮으면 지연 곡선 전체가 잡음이라 최댓값 위치에 의미가 없다.
    RELATED, GAIN = 0.2, 0.10        # 상관 0.2 이상 · 오차 10% 이상 감소
    related = scan[scan["상관"] >= RELATED]
    delayed = related[(related["최적 lag"] > 0) & (related["오차 감소율"] >= GAIN)]

    fallback = [l for l in (1, 2, 3, 6, 12) if l <= cap] or [1]
    notes = []
    if limits.max_lag_minutes:
        notes.append(f"물리적 한계 {limits.max_lag_minutes:g}분(={cap}행)을 걸어 두셨습니다. "
                     "그 너머는 후보에서 제외했습니다.")

    if related.empty:
        return Advice(
            fallback,
            f"타깃과 상관 {RELATED} 이상인 태그가 없습니다 "
            f"(최대 {scan['상관'].max():.3f}). 지연 구조를 신뢰할 근거가 없어 "
            "짧은 lag 위주로 넓게 깔았습니다.",
            scan, "낮음",
            notes + ["관계가 비선형이면 상관으로는 안 잡힙니다. "
                     "3단계 MI 순위를 함께 보세요."])

    if delayed.empty:
        lead = related.iloc[0]
        return Advice(
            fallback,
            f"관련 있는 태그 {len(related)}개 모두 **지연시켜도 나아지지 않습니다** "
            f"(가장 큰 '{lead['태그']}' 도 지연 없을 때가 최대, 상관 "
            f"{lead['상관']:.3f}). 즉각 반응하는 공정으로 보입니다. "
            "그래도 짧은 lag 는 노이즈를 걸러 주므로 기본 후보를 남깁니다.",
            scan, "높음", notes)

    # 여러 태그의 최적 지연이 비슷한 값에 몰리면 그게 진짜 공정 지연이다
    picked = sorted({int(v) for v in delayed["최적 lag"] if 0 < int(v) <= cap})[:n_lags]
    if 1 not in picked and cap >= 1:
        picked = [1] + picked                 # 짧은 lag 는 거의 항상 값을 한다
    picked = sorted(set(picked))[:n_lags + 1] or fallback

    lead = delayed.iloc[0]
    gain = float(lead["오차 감소율"])
    near_cap = delayed[delayed["최적 lag"] >= cap]
    if len(near_cap):
        notes.append(f"{len(near_cap)}개 태그는 상한 근처에서 상관이 최대였습니다. "
                     "실제 지연이 더 길 수 있으니 한계치를 확인해 보세요.")

    return Advice(
        picked,
        f"'{lead['태그']}' 는 {describe_rows(int(lead['최적 lag']), step_min)} "
        f"지연했을 때 상관이 최대입니다 ({lead['0 lag 상관']:.3f} → "
        f"{lead['상관']:.3f}, 설명 못 하던 오차의 {gain:.0%}가 사라집니다). "
        f"지연이 실제로 도움이 되는 {len(delayed)}개 태그의 "
        f"최적값을 모아 {', '.join(describe_rows(l, step_min) for l in picked)} 를 "
        "추천합니다.",
        scan, "높음" if gain >= 0.3 else "보통", notes)


# ─────────────────────────────────────────────────────────────
# rolling — 타깃 자기상관이 꺼지는 지점
# ─────────────────────────────────────────────────────────────
def autocorr_profile(s: pd.Series, max_lag: int = 288) -> pd.DataFrame:
    """타깃이 자기 과거와 얼마나 닮았는지. 이동평균 창의 근거가 된다."""
    y = pd.to_numeric(s, errors="coerce").dropna()
    rows = []
    for L in _lag_grid(min(max_lag, max(1, len(y) // 4))):
        if L == 0:
            continue
        r = y.autocorr(L)
        if r is not None and r == r:
            rows.append({"lag": L, "자기상관": round(float(r), 4)})
    return pd.DataFrame(rows)


def recommend_rolling(
    df: pd.DataFrame,
    target: str,
    step_min: float | None = None,
    limits: PhysicalLimits | None = None,
) -> Advice:
    """자기상관이 0.5 아래로 떨어지는 지점을 기준으로 창 크기를 고른다.

    그보다 훨씬 긴 창으로 평균을 내면 지금 상태와 무관한 과거까지 섞인다.
    """
    limits = limits or PhysicalLimits()
    cap = limits.rolling_rows(step_min) or 288

    if target not in df.columns:
        return Advice([6, 12, 24], "타깃을 찾지 못해 기본값을 둡니다.", None, "낮음")

    acf = autocorr_profile(df[target], max_lag=cap)
    if acf.empty:
        return Advice([l for l in (6, 12, 24) if l <= cap] or [3],
                      "자기상관을 계산할 수 없어 기본값을 둡니다.", acf, "낮음")

    below = acf[acf["자기상관"] < 0.5]
    half = int(below["lag"].iloc[0]) if len(below) else int(acf["lag"].iloc[-1])
    half = max(2, min(half, cap))

    picks = sorted({max(2, half // 4), max(3, half // 2), half})
    picks = [p for p in picks if p <= cap][:3] or [min(6, cap)]

    notes = []
    if limits.max_rolling_minutes:
        notes.append(f"물리적 한계 {limits.max_rolling_minutes:g}분(={cap}행) 안에서 골랐습니다.")
    if not len(below):
        notes.append("자기상관이 끝까지 0.5 위입니다 — 아주 천천히 움직이는 값입니다. "
                     "더 긴 창이 유효할 수 있으니 한계치를 늘려 보세요.")

    return Advice(
        picks,
        f"타깃의 자기상관이 {describe_rows(half, step_min)} 에서 0.5 아래로 "
        f"떨어집니다. 그보다 긴 창으로 평균을 내면 지금과 무관한 과거가 섞이므로, "
        f"그 절반·전후로 {', '.join(describe_rows(p, step_min) for p in picks)} 를 "
        "추천합니다.",
        acf, "보통" if len(below) else "낮음", notes)


# ─────────────────────────────────────────────────────────────
# 선별 상한 — 행 수 대비 피처 수
# ─────────────────────────────────────────────────────────────
def recommend_top_k(n_rows: int, n_features: int) -> Advice:
    """표본이 적은데 피처가 많으면 과적합한다. 경험칙으로 상한을 제안한다."""
    if n_features <= 0:
        return Advice(20, "피처가 없습니다.", None, "낮음")

    # 피처 하나당 최소 20행 정도는 있어야 안정적이다 (경험칙)
    by_rows = max(5, min(200, n_rows // 20))
    k = int(min(by_rows, n_features, 80))
    ratio = n_rows / max(n_features, 1)

    if ratio < 20:
        conf, extra = "높음", ("피처 하나당 학습 행이 "
                             f"{ratio:.0f}행뿐입니다. 과적합 위험이 커서 많이 줄였습니다.")
    else:
        conf, extra = "보통", (f"피처 하나당 학습 행이 {ratio:.0f}행으로 여유가 있습니다.")

    return Advice(
        k,
        f"학습 구간 {n_rows:,}행 · 후보 {n_features:,}개 기준으로 {k}개를 추천합니다. "
        "피처 하나당 20행 이상을 확보하는 경험칙을 씁니다. " + extra,
        None, conf)


# ─────────────────────────────────────────────────────────────
# 한꺼번에
# ─────────────────────────────────────────────────────────────
def recommend_preprocess(df: pd.DataFrame, cols: list[str] | None = None,
                         step_min: float | None = None) -> dict[str, Advice]:
    """전처리 네 가지를 한 번에. 화면은 이걸 받아 기본값으로 쓴다."""
    return {
        "impute": recommend_impute(df, cols, step_min),
        "scaler": recommend_scaler(df, cols),
        "encoding": recommend_encoding(df, cols),
        "clip": recommend_clip(df, cols),
    }


def summary_table(advices: dict[str, Advice], labels: dict[str, str] | None = None,
                  value_labels: dict[str, dict] | None = None) -> pd.DataFrame:
    """추천 묶음을 화면에 띄울 표 하나로. 값·사유·확신도를 나란히 둔다."""
    labels = labels or {}
    value_labels = value_labels or {}
    rows = []
    for key, a in advices.items():
        v = a.value
        vl = value_labels.get(key, {})
        # lag·rolling 추천은 리스트다 — dict.get 에 그대로 넣으면 unhashable 로
        # 죽는다. 화면이 이 표를 그대로 그리므로 여기서 죽으면 전체가 안 뜬다.
        if isinstance(v, bool):
            shown = vl.get(v, "켬" if v else "끔")
        elif isinstance(v, (list, tuple, set)):
            shown = ", ".join(str(vl.get(x, x)) for x in v)
        else:
            try:
                shown = vl.get(v, v)
            except TypeError:
                shown = v
        if isinstance(shown, bool):
            shown = "켬" if shown else "끔"
        elif isinstance(shown, (list, tuple, set)):
            shown = ", ".join(map(str, shown))
        rows.append({"항목": labels.get(key, key), "추천": str(shown),
                     "확신": a.confidence, "근거": a.reason})
    return pd.DataFrame(rows)
