"""추천 엔진 회귀 테스트.

이 파일이 지키는 것은 "추천이 나온다" 가 아니라 **"틀린 추천이 안 나온다"** 다.
사용자는 추천을 그대로 받아들일 가능성이 높으므로, 조용히 나쁜 값을 권하는 것이
추천이 아예 없는 것보다 나쁘다. 그래서 아래 세 개는 실제로 겪은 오추천을
그대로 재현해 놓은 것이다 — 고친 로직이 되돌아가면 여기서 잡힌다.

  1. 결측 대치를 **최악의 태그 하나**로 판단해 멀쩡한 태그까지 median 으로 밀었다
  2. 거의 상수인 태그가 스케일 비율에 끼어 "1,800만 배 차이" 라는 사유가 나왔다
  3. lag 를 **절대 상관 개선치**로 재서, 노이즈의 8시간을 추천하고
     진짜 1시간 지연은 놓쳤다

3번이 특히 위험하다. 사유 문장이 그럴듯해서 사람이 검증하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import advisor  # noqa: E402


STEP = 5.0        # 5분 간격 — 12행 = 1시간


def _clock(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="5min")


# ── 공통 계약 ────────────────────────────────────────────────
def _demo(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """정상적인 공정 데이터 한 벌. 특별한 병이 없는 기준선."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 50 + 8 * np.sin(t / 60) + rng.normal(0, 0.6, n)
    return pd.DataFrame({
        "temp": base,
        "pressure": 3 + 0.02 * base + rng.normal(0, 0.05, n),
        "flow": 900 + 12 * base + rng.normal(0, 20, n),
        "y": 1.4 * base + rng.normal(0, 1.0, n),
    }, index=_clock(n))


def test_every_advice_carries_a_reason():
    """값만 주고 사유가 없으면 사용자가 검증할 수 없다 — 그럼 추천이 아니라 강요다."""
    df = _demo()
    advices = advisor.recommend_preprocess(df, step_min=STEP)
    advices["lag"] = advisor.recommend_lags(df, "y", step_min=STEP)
    advices["rolling"] = advisor.recommend_rolling(df, "y", step_min=STEP)
    advices["top_k"] = advisor.recommend_top_k(2000, 120)

    for key, a in advices.items():
        assert isinstance(a, advisor.Advice), key
        assert a.reason and len(a.reason) > 20, f"{key} 의 사유가 비었습니다: {a.reason!r}"
        assert a.confidence in ("높음", "보통", "낮음"), f"{key}: {a.confidence}"


def test_summary_table_renders_every_value_type():
    """bool·list·str 이 섞여 있어도 표 한 장으로 나와야 한다 (화면이 이걸 그대로 쓴다)."""
    advices = {
        "impute": advisor.Advice("ffill", "짧게 끊깁니다" * 3),
        "clip": advisor.Advice(True, "극단값이 있습니다" * 3),
        "lag": advisor.Advice([1, 12], "지연이 보입니다" * 3),
    }
    tbl = advisor.summary_table(
        advices,
        labels={"impute": "결측 처리", "clip": "극단값", "lag": "지연"},
        value_labels={"impute": {"ffill": "직전 값 유지"}})
    assert list(tbl["항목"]) == ["결측 처리", "극단값", "지연"]
    assert tbl.loc[0, "추천"] == "직전 값 유지"
    assert tbl.loc[1, "추천"] == "켬"           # bool 이 True/False 로 새면 안 된다
    assert tbl.loc[2, "추천"] == "1, 12"


# ── 1. 결측 — 최악의 태그가 전체를 대표하면 안 된다 ──────────
def test_short_gaps_recommend_ffill():
    """2~3행씩 끊긴 통신 순단 — 직전 값 유지가 물리적으로 맞다."""
    df = _demo()
    for c in ("temp", "pressure", "flow"):
        idx = df.index[np.arange(100, 1900, 97)]
        df.loc[idx, c] = np.nan
        df.loc[df.index[df.index.get_indexer(idx) + 1], c] = np.nan

    a = advisor.recommend_impute(df, step_min=STEP)
    assert a.value == "ffill", a.reason
    assert a.confidence == "높음"


def test_long_outage_recommends_median():
    """설비 정지처럼 길게 비면 직전 값을 끌 수 없다 — 없던 평탄 구간이 생긴다."""
    df = _demo()
    for c in ("temp", "pressure", "flow"):
        df.iloc[300:600, df.columns.get_loc(c)] = np.nan

    a = advisor.recommend_impute(df, step_min=STEP)
    assert a.value == "median", a.reason
    assert "긴" in a.reason or "이어집니다" in a.reason


def test_one_broken_sensor_does_not_flip_the_whole_recommendation():
    """실제로 겪은 오추천 — 센서 하나가 60% 비었다고 멀쩡한 태그까지 median 으로 밀었다.

    그 하나는 2단계에서 빼는 게 맞는 대응이지, 나머지 태그의 대치 방식을
    바꿀 이유가 아니다. 대치는 '보통의 태그' 를 기준으로 정해야 한다.
    """
    df = _demo()
    # 멀쩡한 태그 셋 — 짧게 끊김
    for c in ("temp", "pressure", "flow"):
        idx = np.arange(100, 1900, 91)
        df.iloc[idx, df.columns.get_loc(c)] = np.nan
    # 고장난 센서 하나 — 60% 가 통으로 빔
    df["dead_sensor"] = df["temp"].to_numpy()
    df.iloc[:1200, df.columns.get_loc("dead_sensor")] = np.nan

    a = advisor.recommend_impute(df, step_min=STEP)
    assert a.value == "ffill", f"고장 센서 하나에 끌려갔습니다: {a.reason}"
    # 대신 그 센서는 따로 짚어 줘야 한다
    assert any("40%" in n and "dead_sensor" in n for n in a.notes), a.notes


def test_missing_profile_measures_run_length_not_just_rate():
    """비율만으로는 방법을 못 정한다 — 같은 5%라도 모양이 다르다."""
    n = 1000
    short = pd.Series(1.0, index=range(n))
    short.iloc[np.arange(0, n, 20)] = np.nan          # 5%, 1행씩 50번
    long_ = pd.Series(1.0, index=range(n))
    long_.iloc[100:150] = np.nan                       # 5%, 50행 1번
    prof = advisor.missing_profile(pd.DataFrame({"short": short, "long": long_}))
    prof = prof.set_index("컬럼")

    assert prof.loc["short", "결측률"] == pytest.approx(prof.loc["long", "결측률"], abs=0.01)
    assert prof.loc["short", "최장 연속"] == 1
    assert prof.loc["long", "최장 연속"] == 50
    assert prof.loc["short", "결측 구간 수"] > prof.loc["long", "결측 구간 수"]


def test_no_missing_says_so_plainly():
    a = advisor.recommend_impute(_demo(), step_min=STEP)
    assert "결측이 없습니다" in a.reason
    assert a.confidence == "높음"


# ── 2. 스케일 — 거의 상수인 태그가 비율을 오염시키면 안 된다 ─
def test_flat_tag_does_not_produce_an_absurd_ratio():
    """실제로 겪은 오추천 — '18,091,036배 차이' 라는 사유가 화면에 떴다.

    값이 1.000001 인 태그의 표준편차는 1e-6 이라 **절대 임계로는 못 거른다.**
    자기 크기 대비 얼마나 흔들리는지(변동계수)를 봐야 한다.
    """
    df = _demo()
    rng = np.random.default_rng(1)
    df["flat_tag"] = 1.0 + rng.normal(0, 1e-6, len(df))     # 사실상 상수

    a = advisor.recommend_scaler(df)
    ratio = float(a.reason.split("최대 ")[1].split("배")[0].replace(",", "")) \
        if "최대 " in a.reason and "배" in a.reason else 0.0
    assert ratio < 10000, f"거의 상수인 태그가 비율에 끼었습니다: {a.reason}"
    assert any("flat_tag" in n for n in a.notes), a.notes


def test_wide_scale_spread_recommends_standard():
    df = _demo()          # flow(σ≈100) vs pressure(σ≈0.2) — 500배쯤
    a = advisor.recommend_scaler(df)
    assert a.value == "standard"
    assert "배 차이" in a.reason


def test_heavy_outliers_recommend_robust():
    """이상값이 눈금 자체를 끌어당기면 평균·표준편차 기준을 쓰면 안 된다."""
    df = _demo()
    for c in ("temp", "pressure", "flow", "y"):
        df.iloc[[10, 500, 1500], df.columns.get_loc(c)] *= 400

    a = advisor.recommend_scaler(df)
    assert a.value == "robust", a.reason
    assert "이상값" in a.reason


def test_outlier_score_does_not_let_a_spike_hide_itself():
    """이상값 점수는 이상값 자체에 끌려가면 안 된다.

    보통은 MAD 를 쓰면 해결되지만, **값이 고정된 센서에 스파이크 하나**가 튀면
    MAD 가 0 이 되어 점수 자체가 0 으로 죽는다 — 스파이크가 스스로를 감춘다.
    그 경우 표준편차로 물러서서라도 잡아내야 recommend_scaler 의 z>10 선에 걸린다.
    """
    rng = np.random.default_rng(11)
    normal = pd.Series(np.r_[10 + rng.normal(0, 0.1, 999), 1e6])      # 흔들리는 센서
    stuck = pd.Series([10.0] * 999 + [1e6])                            # 고정된 센서
    for name, s in (("흔들림", normal), ("고정", stuck)):
        z = advisor.scale_profile(pd.DataFrame({"x": s})).loc[0, "최대 robust z"]
        assert z > 10, f"{name} 센서의 스파이크가 감춰졌습니다 (z={z})"


# ── 3. lag — 진짜 지연은 찾고 노이즈는 거른다 ────────────────
def _known_lag(n: int = 3000, lag: int = 12, seed: int = 3) -> pd.DataFrame:
    """y 가 flow 를 정확히 `lag` 만큼 늦게 따라오는 데이터. 정답을 아는 시험지."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    flow = 100 + 30 * np.sin(t / 80) + rng.normal(0, 1.0, n)
    y = np.roll(flow, lag) * 1.5 + 20 + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "flow": flow,
        "noise": rng.normal(0, 1, n),
        "y": y,
    }, index=_clock(n)).iloc[lag + 5:]


def test_recovers_a_known_delay():
    """정답이 12행인 데이터에서 12를 찾아야 한다. 못 찾으면 이 기능은 무용지물이다."""
    a = advisor.recommend_lags(_known_lag(lag=12), "y", step_min=STEP)
    assert 12 in a.value, f"12행 지연을 놓쳤습니다: {a.value} / {a.reason}"
    assert a.confidence == "높음"
    assert "1시간" in a.reason        # 행 수가 아니라 사람이 아는 시간으로


def test_gain_is_measured_as_error_reduction_not_absolute_correlation():
    """실제로 겪은 오추천의 뿌리 — 0.993 → 0.999 는 절대값으로 +0.006 이라
    하찮아 보이지만 설명 못 하던 오차의 88%가 사라진 것이다.
    절대 개선치로 재면 진짜 지연을 놓친다."""
    scan = advisor.lag_scan(_known_lag(lag=12), "y", step_min=STEP)
    row = scan[scan["태그"] == "flow"].iloc[0]
    absolute = float(row["상관"]) - float(row["0 lag 상관"])
    assert absolute < 0.10, "이 시험지는 절대 개선치가 작아야 의미가 있습니다"
    assert float(row["오차 감소율"]) > 0.5, (
        f"오차 감소율이 지연을 못 잡아냅니다: {row.to_dict()}")


def test_noise_tag_never_earns_a_lag_recommendation():
    """실제로 겪은 오추천 — 상관 0.10 짜리 노이즈가 8시간에서 우연히 0.1026 이
    되는 것만으로 '8시간 lag' 가 추천됐다. 상관이 낮으면 지연 곡선 전체가
    잡음이라 최댓값 위치에 의미가 없다."""
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({
        "noise_a": rng.normal(0, 1, n),
        "noise_b": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    }, index=_clock(n))

    a = advisor.recommend_lags(df, "y", step_min=STEP)
    assert a.confidence == "낮음", a.reason
    assert set(a.value) <= {1, 2, 3, 6, 12}, f"노이즈에서 지연을 만들어 냈습니다: {a.value}"


def test_instant_process_says_no_delay_helps():
    """즉각 반응하는 공정에서 '지연이 있다' 고 우기면 안 된다."""
    a = advisor.recommend_lags(_demo(), "y", step_min=STEP)
    assert "나아지지 않습니다" in a.reason or "즉각" in a.reason, a.reason
    assert a.confidence == "높음"          # '모르겠다' 가 아니라 '없다' 는 확실한 결론


def test_lag_scan_reports_time_not_just_rows():
    scan = advisor.lag_scan(_known_lag(lag=12), "y", step_min=STEP)
    row = scan[scan["태그"] == "flow"].iloc[0]
    assert row["지연 시간"] == "12행(1시간)"


# ── 물리적 한계 — 통계가 뭐라 하든 넘지 않는다 ───────────────
def test_physical_limit_caps_the_recommendation():
    """공정 지식이 통계를 이긴다. 30분 한계면 12행(1시간) 지연은 추천될 수 없다."""
    df = _known_lag(lag=12)
    limits = advisor.PhysicalLimits(max_lag_minutes=30)     # 30분 = 6행
    a = advisor.recommend_lags(df, "y", step_min=STEP, limits=limits)

    assert max(a.value) <= 6, f"한계 6행을 넘었습니다: {a.value}"
    assert any("물리적 한계" in n for n in a.notes), a.notes


def test_physical_limit_warns_when_the_best_lag_sits_at_the_cap():
    """상한에 딱 붙어 최대가 나오면 '더 길 수도 있다' 고 알려 줘야 한다.
    안 그러면 사용자가 자기가 건 한계 때문에 잘린 줄 모른다."""
    df = _known_lag(lag=24)
    limits = advisor.PhysicalLimits(max_lag_minutes=120)    # 120분 = 24행, 정확히 정답
    a = advisor.recommend_lags(df, "y", step_min=STEP, limits=limits)
    assert any("상한 근처" in n for n in a.notes), a.notes


def test_limits_convert_minutes_to_rows_by_sampling_interval():
    """같은 60분이라도 1분 간격이면 60행, 10분 간격이면 6행이다."""
    lim = advisor.PhysicalLimits(max_lag_minutes=60, max_rolling_minutes=180)
    assert lim.lag_rows(1) == 60
    assert lim.lag_rows(10) == 6
    assert lim.rolling_rows(5) == 36
    assert lim.lag_rows(None) is None          # 간격을 모르면 환산하지 않는다
    assert advisor.PhysicalLimits().lag_rows(5) is None


def test_rolling_limit_is_respected():
    df = _demo(4000)
    limits = advisor.PhysicalLimits(max_rolling_minutes=60)   # 12행
    a = advisor.recommend_rolling(df, "y", step_min=STEP, limits=limits)
    assert max(a.value) <= 12, a.value
    assert any("물리적 한계" in n for n in a.notes), a.notes


# ── rolling / top_k / encoding / clip ────────────────────────
def test_rolling_follows_autocorrelation_decay():
    """천천히 움직이는 값은 긴 창을, 빨리 움직이는 값은 짧은 창을 받아야 한다."""
    n, rng = 4000, np.random.default_rng(5)
    slow = pd.Series(np.sin(np.arange(n) / 400) * 10 + rng.normal(0, 0.2, n))
    fast = pd.Series(rng.normal(0, 1, n)).rolling(2).mean().bfill()
    idx = _clock(n)

    a_slow = advisor.recommend_rolling(pd.DataFrame({"y": slow.to_numpy()}, index=idx),
                                       "y", step_min=STEP)
    a_fast = advisor.recommend_rolling(pd.DataFrame({"y": fast.to_numpy()}, index=idx),
                                       "y", step_min=STEP)
    assert max(a_slow.value) > max(a_fast.value), (a_slow.value, a_fast.value)


def test_top_k_shrinks_when_rows_are_scarce():
    """행이 적은데 피처가 많으면 과적합한다."""
    tight = advisor.recommend_top_k(n_rows=600, n_features=200)
    roomy = advisor.recommend_top_k(n_rows=200_000, n_features=200)
    assert tight.value < roomy.value
    assert tight.value <= 200 and roomy.value <= 200
    assert "과적합" in tight.reason


def test_top_k_never_exceeds_available_features():
    a = advisor.recommend_top_k(n_rows=500_000, n_features=7)
    assert a.value <= 7


def test_ordered_categories_get_ordinal():
    df = pd.DataFrame({"등급": ["1", "2", "3"] * 40, "y": range(120)})
    a = advisor.recommend_encoding(df)
    assert a.value == "ordinal", a.reason


def test_unordered_states_get_onehot():
    """상태값에 번호를 매기면 모델이 없는 크기 관계를 배운다."""
    df = pd.DataFrame({"라인": ["A동", "B동", "C동"] * 40, "y": range(120)})
    a = advisor.recommend_encoding(df)
    assert a.value == "onehot", a.reason


def test_high_cardinality_column_is_flagged_not_silently_encoded():
    """고유값 500개짜리를 One-Hot 하면 열이 500개 생긴다 — 먼저 알려 줘야 한다."""
    df = pd.DataFrame({"설비ID": [f"EQ{i:04d}" for i in range(500)],
                       "y": range(500)})
    a = advisor.recommend_encoding(df)
    assert any("20개를 넘는" in n for n in a.notes), a.notes


def test_clip_off_when_data_is_clean():
    a = advisor.recommend_clip(_demo())
    assert a.value is False, a.reason


def test_clip_on_with_spikes_and_a_warning_about_real_events():
    """극단값을 자르는 건 정상적인 급변까지 지울 수 있다 — 그 경고가 붙어야 한다."""
    df = _demo()
    df.iloc[[7, 800], df.columns.get_loc("temp")] = 1e5
    a = advisor.recommend_clip(df)
    assert a.value is True, a.reason
    assert any("실제 운전" in n for n in a.notes), a.notes


# ── 견고성 — 이상한 입력에도 안 죽는다 ───────────────────────
def test_all_recommenders_survive_degenerate_input():
    """빈 표·상수열·전부 결측 — 화면이 죽으면 사용자는 원인을 못 찾는다."""
    cases = {
        "빈 표": pd.DataFrame({"y": pd.Series(dtype=float)}),
        "한 행": pd.DataFrame({"x": [1.0], "y": [2.0]}),
        "전부 결측": pd.DataFrame({"x": [np.nan] * 50, "y": [np.nan] * 50}),
        "상수열": pd.DataFrame({"x": [5.0] * 50, "y": [1.0] * 50}),
        "글자만": pd.DataFrame({"x": list("abcde") * 10, "y": list("vwxyz") * 10}),
    }
    for name, df in cases.items():
        advisor.recommend_preprocess(df)                       # 죽지 않으면 통과
        advisor.recommend_lags(df, "y")
        advisor.recommend_rolling(df, "y")


def test_missing_target_does_not_crash_rolling():
    a = advisor.recommend_rolling(_demo(), "없는컬럼", step_min=STEP)
    assert a.confidence == "낮음"
    assert a.value


def test_lag_scan_raises_on_missing_target():
    """rolling 과 달리 lag_scan 은 조용히 넘어가면 안 된다 — 호출자의 버그다."""
    with pytest.raises(KeyError):
        advisor.lag_scan(_demo(), "없는컬럼")


def test_describe_rows_reads_like_a_human_wrote_it():
    assert advisor.describe_rows(6, 5) == "6행(30분)"
    assert advisor.describe_rows(12, 5) == "12행(1시간)"
    assert advisor.describe_rows(288, 5) == "288행(1일)"
    assert advisor.describe_rows(7, None) == "7행"


def test_advice_truthiness_follows_the_value():
    """`if advice:` 를 썼을 때 clip=False 가 참이 되면 화면이 반대로 그려진다."""
    assert not advisor.Advice(False, "x")
    assert not advisor.Advice([], "x")
    assert advisor.Advice(True, "x")
    assert advisor.Advice([1, 2], "x")


def test_advisor_does_not_import_streamlit():
    """core 는 화면을 몰라야 한다 (Dataiku 이식 전제)."""
    src = (ROOT / "core" / "advisor.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src
    assert "from streamlit" not in src
