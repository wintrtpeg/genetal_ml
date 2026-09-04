"""가상 시계열 데이터 생성.

실데이터 없이 전체 흐름을 검증하기 위한 것이다.
Y 는 아래 관계로 만들어지므로, 도구가 이 구조를 되찾아내는지로 정확도를 판정할 수 있다.

  - flow        : 선형 기여 (양)
  - temp        : 임계 65 이상에서만 급격히 기여 (비선형)
  - pressure    : 3스텝 지연 후 기여 (lag 파생변수가 있어야 잡힌다)
  - valve       : flow 와 곱해져 기여 (상호작용)
  - noise_tag   : 기여 없음 (SHAP 에서 하위로 밀려야 정상)
  - flat_tag    : 거의 상수 (품질 필터가 걸러야 정상)
  - sparse_tag  : 결측 60% (품질 필터가 걸러야 정상)

사용: python scripts/make_demo_data.py --days 45 --freq 5min
"""

from __future__ import annotations

import argparse
import sys
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


def make(days: int = 45, freq: str = "5min", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=int(days * 24 * 60 / 5), freq=freq)
    n = len(idx)

    hour = idx.hour + idx.minute / 60
    dow = idx.dayofweek
    day_cycle = np.sin(2 * np.pi * hour / 24)
    week_cycle = np.sin(2 * np.pi * dow / 7)

    flow = 120 + 25 * day_cycle + 8 * week_cycle + rng.normal(0, 4, n)
    flow += np.where((hour > 8) & (hour < 18), 12, 0)

    temp = 58 + 9 * day_cycle + rng.normal(0, 1.8, n) + np.linspace(0, 4, n)

    pressure = 3.2 + 0.012 * flow + rng.normal(0, 0.09, n)

    valve = np.clip(55 + 18 * np.sin(2 * np.pi * np.arange(n) / (n / 6))
                    + rng.normal(0, 3, n), 10, 100)

    conc = np.cumsum(rng.normal(0, 0.05, n)) + 12
    conc = conc - np.linspace(0, conc[-1] - 12, n)

    noise_tag = rng.normal(50, 10, n)
    flat_tag = np.full(n, 1.0) + rng.normal(0, 1e-6, n)
    sparse_tag = rng.normal(30, 5, n)

    mode = np.where(valve > 70, "HIGH", np.where(valve > 40, "NORMAL", "LOW"))

    pressure_lag = pd.Series(pressure).shift(3).bfill().to_numpy()
    temp_effect = np.where(temp > 65, 3.5 * (temp - 65) ** 1.4, 0.0)

    y = (
        20.0
        + 0.35 * flow
        + temp_effect
        + 6.0 * pressure_lag
        + 0.018 * flow * (valve / 100.0)
        - 0.9 * conc
        + rng.normal(0, 2.0, n)
    )

    df = pd.DataFrame({
        "timestamp": idx,
        "flow": flow.round(3),
        "temp": temp.round(3),
        "pressure": pressure.round(4),
        "valve": valve.round(2),
        "conc": conc.round(4),
        "noise_tag": noise_tag.round(3),
        "flat_tag": flat_tag.round(6),
        "sparse_tag": sparse_tag.round(3),
        "op_mode": mode,
        "y_output": y.round(3),
    })

    # 현실감: 결측과 짧은 계측 중단 구간
    df.loc[rng.random(n) < 0.60, "sparse_tag"] = np.nan
    df.loc[rng.random(n) < 0.01, "temp"] = np.nan
    gap = rng.integers(0, n - 300)
    df.loc[gap:gap + 200, ["flow", "pressure"]] = np.nan
    return df


def main() -> None:
    _enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = make(a.days, a.freq, a.seed)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "data" / "demo_timeseries.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df):,}행 저장: {out}")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 세로형(long) 견본도 같이 만든다. PI · IP.21 같은 히스토리언은 이 모양이
    # 기본이라, 세로형 처리가 실제로 되는지 눌러 볼 데이터가 필요하다.
    num = [c for c in df.columns
           if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    long = df.melt(id_vars="timestamp", value_vars=num,
                   var_name="tag_name", value_name="value")
    long = long.rename(columns={"timestamp": "tag_time"})
    long = long.sort_values(["tag_time", "tag_name"]).reset_index(drop=True)
    long_out = out.with_name("demo_timeseries_long.csv")
    long.to_csv(long_out, index=False, encoding="utf-8-sig")
    print(f"{len(long):,}행 저장: {long_out}  (세로형 견본 · 태그 {len(num)}개)")


if __name__ == "__main__":
    main()
