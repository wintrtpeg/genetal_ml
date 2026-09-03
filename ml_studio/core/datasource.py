"""데이터 소스 계층.

방법1) 로컬 CSV 업로드            -> CsvSource
방법2) SQL 쿼리 기반 데이터마트   -> SqlAlchemySource
(추후) 사내 API + RAW SQL         -> ApiSource  (인터페이스만 확보)

모든 소스는 load() -> pd.DataFrame 하나만 구현하면 되므로,
나중에 접속 방식이 바뀌어도 상위 파이프라인은 손대지 않는다.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import pandas as pd


# ─────────────────────────────────────────────────────────────
# 쿼리 안전장치
# ─────────────────────────────────────────────────────────────
class QueryNotAllowed(ValueError):
    """조회 목적이 아닌 쿼리를 막을 때 발생."""


_COMMENT_LINE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)

# 오탐을 줄이기 위해 '키워드 + 뒤따르는 절'까지 함께 확인한다.
# (예: create_dt 같은 컬럼명은 걸리지 않는다)
_FORBIDDEN = [
    re.compile(r"\binsert\s+into\b", re.I),
    re.compile(r"\bupdate\s+\S+\s+set\b", re.I),
    re.compile(r"\bdelete\s+from\b", re.I),
    re.compile(r"\bdrop\s+(table|view|index|schema|database)\b", re.I),
    re.compile(r"\btruncate\s+table\b", re.I),
    re.compile(r"\balter\s+(table|view|session|system)\b", re.I),
    re.compile(r"\bcreate\s+(table|view|index|or\s+replace)\b", re.I),
    re.compile(r"\b(grant|revoke)\s+\S+", re.I),
    re.compile(r"\binto\s+outfile\b", re.I),
]


def strip_sql_comments(sql: str) -> str:
    sql = _COMMENT_BLOCK.sub(" ", sql)
    sql = _COMMENT_LINE.sub(" ", sql)
    return sql


def validate_select(sql: str) -> str:
    """SELECT / WITH 로 시작하는 단일 조회 쿼리만 통과시킨다.

    통과한 쿼리를 (끝 세미콜론 제거한 상태로) 돌려준다.
    """
    if sql is None or not sql.strip():
        raise QueryNotAllowed("쿼리가 비어 있습니다.")

    body = strip_sql_comments(sql).strip().rstrip(";").strip()
    if not body:
        raise QueryNotAllowed("주석을 제외하면 실행할 내용이 없습니다.")

    if ";" in body:
        raise QueryNotAllowed("한 번에 하나의 쿼리만 실행합니다. 세미콜론으로 구문을 나누지 마세요.")

    if not re.match(r"^(select|with)\b", body, re.I):
        raise QueryNotAllowed("SELECT 또는 WITH 로 시작하는 조회 쿼리만 사용할 수 있습니다.")

    for pat in _FORBIDDEN:
        hit = pat.search(body)
        if hit:
            raise QueryNotAllowed(f"조회 전용 도구입니다. 데이터를 변경하는 구문을 찾았습니다: {hit.group(0)}")

    return sql.strip().rstrip(";").strip()


# ─────────────────────────────────────────────────────────────
# 접속 URL 조립
# ─────────────────────────────────────────────────────────────
DIALECTS: dict[str, str] = {
    "SQream": "sqream",
    "PostgreSQL": "postgresql+psycopg2",
    "MySQL / MariaDB": "mysql+pymysql",
    "Oracle": "oracle+oracledb",
    "SQL Server": "mssql+pyodbc",
    "Trino / Presto": "trino",
    "Hive": "hive",
    "SQLite (파일)": "sqlite",
}

# 화면에서 포트 칸을 미리 채워 준다. 매번 찾아보지 않아도 되게.
DEFAULT_PORTS: dict[str, int] = {
    "sqream": 3108,
    "postgresql+psycopg2": 5432,
    "mysql+pymysql": 3306,
    "oracle+oracledb": 1521,
    "mssql+pyodbc": 1433,
    "trino": 8080,
    "hive": 10000,
}

# 드라이버마다 따로 설치해야 하는 파이썬 패키지.
DRIVER_PACKAGES: dict[str, str] = {
    "sqream": "pysqream-sqlalchemy",
    "postgresql+psycopg2": "psycopg2-binary",
    "mysql+pymysql": "pymysql",
    "oracle+oracledb": "oracledb",
    "mssql+pyodbc": "pyodbc",
    "trino": "trino[sqlalchemy]",
    "hive": "pyhive[hive]",
    "sqlite": "(내장)",
}

# 드라이버가 URL 이 아니라 connect_args 로 받아야 하는 옵션의 기본값.
# SQream 은 다중 클러스터 접속 여부를 여기서 받는다.
DEFAULT_CONNECT_ARGS: dict[str, dict] = {
    "sqream": {"clustered": True},
}


def build_url(
    driver: str,
    host: str = "",
    port: str | int | None = None,
    database: str = "",
    user: str = "",
    password: str = "",
    query_params: dict[str, str] | None = None,
) -> str:
    """SQLAlchemy 접속 URL 문자열을 만든다. 비밀번호는 URL 인코딩한다."""
    if driver.startswith("sqlite"):
        return f"sqlite:///{database}"

    cred = ""
    if user:
        cred = quote_plus(user)
        if password:
            cred += f":{quote_plus(password)}"
        cred += "@"

    netloc = host or ""
    if port:
        netloc += f":{port}"

    url = f"{driver}://{cred}{netloc}/{database}"
    if query_params:
        url += "?" + "&".join(f"{k}={quote_plus(str(v))}" for k, v in query_params.items())
    return url


def mask_url(url: str) -> str:
    """로그·화면 표시용. 비밀번호를 가린다."""
    return re.sub(r"(://[^:/@]+):([^@]*)@", r"\1:***@", url or "")


# ─────────────────────────────────────────────────────────────
# 소스 구현
# ─────────────────────────────────────────────────────────────
class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


@dataclass
class CsvSource(DataSource):
    path_or_buffer: Any
    sep: str = ","
    encoding: str = "utf-8"
    nrows: int | None = None

    def load(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path_or_buffer,
            sep=self.sep,
            encoding=self.encoding,
            nrows=self.nrows,
            low_memory=False,
        )


@dataclass
class SqlAlchemySource(DataSource):
    """사내 데이터마트 조회.

        import sqlalchemy as sa
        engine = sa.create_engine(<사내주소 / 계정 / 비밀번호>)
        query  = "SELECT ... "        <- 대시보드에서 사용자가 직접 작성
        df     = pd.read_sql(query, engine)
    """

    url: str
    query: str
    params: dict[str, Any] | None = None
    chunksize: int | None = None
    connect_args: dict[str, Any] = field(default_factory=dict)
    max_rows: int | None = None

    def _engine(self):
        import sqlalchemy as sa

        return sa.create_engine(self.url, connect_args=self.connect_args, pool_pre_ping=True)

    def test_connection(self) -> str:
        """접속만 확인한다. 성공하면 DB가 돌려준 인사말을 문자열로 반환."""
        import sqlalchemy as sa

        engine = self._engine()
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return f"접속 성공 — {mask_url(self.url)}"

    def load(self) -> pd.DataFrame:
        sql = validate_select(self.query)
        engine = self._engine()

        if self.chunksize:
            frames, total = [], 0
            for chunk in pd.read_sql(sql, engine, params=self.params, chunksize=self.chunksize):
                frames.append(chunk)
                total += len(chunk)
                if self.max_rows and total >= self.max_rows:
                    break
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            df = pd.read_sql(sql, engine, params=self.params)

        if self.max_rows:
            df = df.head(self.max_rows)
        return df

    def preview(self, n: int = 200) -> pd.DataFrame:
        """전량을 끌어오지 않고 앞부분만 본다. 방언에 상관없이 동작한다."""
        sql = validate_select(self.query)
        engine = self._engine()
        for chunk in pd.read_sql(sql, engine, params=self.params, chunksize=n):
            return chunk
        return pd.DataFrame()


@dataclass
class ApiSource(DataSource):
    """추후 사내 API(RAW SQL 전달 방식) 전환용 자리.

    Python에서 데이터마트를 직접 조회하는 경로가 닫히면 이 클래스만 채우면 되고,
    상위 파이프라인은 그대로 둔다.
    """

    endpoint: str
    api_key: str
    query: str
    timeout: int = 300

    def load(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            "API 경로는 아직 연결되지 않았습니다. 발급받은 엔드포인트 규격이 확정되면 "
            "core/datasource.py 의 ApiSource.load() 만 채우면 나머지는 그대로 동작합니다."
        )


# ─────────────────────────────────────────────────────────────
# 표 모양 — wide / long
# ─────────────────────────────────────────────────────────────
# 설비 데이터는 두 가지 모양으로 나온다.
#
#   wide   한 행이 한 시점, 태그마다 한 컬럼
#          tag_time            flow   temp   press
#          2025-01-01 00:00    52.3   65.1   3.2
#
#   long   한 행이 '한 시점의 한 태그'
#          tag_time            tag_name   value
#          2025-01-01 00:00    flow       52.3
#          2025-01-01 00:00    temp       65.1
#
# PI · IP.21 같은 히스토리언은 **long 이 기본**이다. 태그를 나중에 추가해도
# 테이블 구조를 안 바꿔도 되기 때문이다. 그래서 사내 데이터마트에서 뽑으면
# long 으로 나오는 경우가 아주 흔하다.
#
# 모델은 wide 를 전제로 한다 (한 줄 = 한 시점의 설비 상태). 그래서 long 이면
# 먼저 돌려 세워야 한다. 그 판정과 변환을 여기서 한다 — 화면은 결과만 쓴다.

TAG_COL_HINTS = ("tag", "태그", "item", "point", "name", "이름", "변수",
                 "sensor", "센서", "signal", "code", "코드", "지점")
VALUE_COL_HINTS = ("value", "val", "값", "측정", "meas", "reading", "data")


def _looks_like_tag_column(s: pd.Series, n_rows: int) -> bool:
    """태그 이름이 담긴 컬럼인가.

    값이 글자이고, 종류가 적당히 적으면서, 여러 번 반복되면 태그 목록이다.
    반복되지 않으면 그건 메모나 ID 이지 태그가 아니다.
    """
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    nun = int(s.nunique(dropna=True))
    if nun < 2 or nun > 5000:
        return False
    # 태그 하나당 최소 몇 번은 나와야 시계열이다
    return n_rows / max(nun, 1) >= 3


def detect_layout(df: pd.DataFrame, time_col: str | None = None) -> dict:
    """이 표가 wide 인지 long 인지 추측한다.

    반환: {"layout": "wide"|"long", "confidence": 0~1, "time_col", "tag_col",
           "value_col", "reasons": [...], "n_tags": int}

    **가장 강한 단서는 시각의 중복**이다. wide 는 시각마다 한 줄이라 시각이
    유일하고, long 은 태그 수만큼 같은 시각이 반복된다.
    """
    out: dict = {"layout": "wide", "confidence": 0.0, "time_col": time_col,
                 "tag_col": None, "value_col": None, "reasons": [], "n_tags": 0}
    if df is None or df.empty or df.shape[1] < 3:
        out["reasons"].append("컬럼이 3개 미만이라 long 으로 볼 수 없습니다.")
        return out

    tcol = time_col or guess_time_column(df)
    out["time_col"] = tcol
    if tcol is None or tcol not in df.columns:
        out["reasons"].append("시간 컬럼을 찾지 못했습니다.")
        return out

    ts = pd.to_datetime(df[tcol], errors="coerce")
    valid = ts.notna()
    if int(valid.sum()) < 3:
        out["reasons"].append("시간으로 읽히는 값이 거의 없습니다.")
        return out

    n = int(valid.sum())
    uniq_t = int(ts[valid].nunique())
    dup_ratio = 1.0 - uniq_t / n            # 1 에 가까울수록 같은 시각이 많이 반복

    others = [c for c in df.columns if c != tcol]
    tag_cands = [c for c in others if _looks_like_tag_column(df[c], n)]
    num_cands = [c for c in others
                 if pd.api.types.is_numeric_dtype(df[c]) and c not in tag_cands]

    score = 0.0
    if dup_ratio > 0.3:
        score += 0.5
        out["reasons"].append(
            f"같은 시각이 여러 줄에 걸쳐 나옵니다 (중복 {dup_ratio:.0%}). "
            "시각마다 한 줄인 wide 라면 이런 일이 없습니다.")
    if tag_cands:
        score += 0.3
        out["reasons"].append(f"태그 이름으로 보이는 컬럼이 있습니다: {tag_cands[0]}")
    if len(others) <= 4:
        score += 0.1
        out["reasons"].append(f"컬럼이 {len(df.columns)}개뿐입니다 (long 은 보통 3~5개).")
    if num_cands:
        score += 0.1

    if not tag_cands or not num_cands:
        out["confidence"] = 0.0
        out["reasons"].append("태그 컬럼과 값 컬럼을 둘 다 찾지 못해 wide 로 봅니다.")
        return out

    # 이름으로 한 번 더 고른다 — tag_name · value 같은 흔한 이름을 우선한다
    def _prefer(cands: list[str], hints) -> str:
        for c in cands:
            if any(h in str(c).lower() for h in hints):
                return c
        return cands[0]

    tag_col = _prefer(tag_cands, TAG_COL_HINTS)
    value_col = _prefer(num_cands, VALUE_COL_HINTS)

    out.update(tag_col=tag_col, value_col=value_col,
               n_tags=int(df[tag_col].nunique(dropna=True)),
               confidence=round(min(score, 1.0), 2))
    if score >= 0.6:
        out["layout"] = "long"
    else:
        out["reasons"].append("long 이라고 보기엔 근거가 약해 wide 로 봅니다.")
    return out


def guess_time_column(df: pd.DataFrame) -> str | None:
    """시간 컬럼 이름을 추측한다. 화면과 판정이 같은 규칙을 쓰도록 여기에 둔다."""
    hints = ("time", "date", "dt", "stamp", "일시", "시각", "일자", "시간")
    for c in df.columns:
        if any(h in str(c).lower() for h in hints):
            return str(c)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)
    # 문자열이라도 시간으로 읽히면 후보
    for c in df.columns:
        if df[c].dtype == object:
            head = df[c].dropna().head(50)
            if len(head) and pd.to_datetime(head, errors="coerce").notna().mean() > 0.8:
                return str(c)
    return None


class LayoutError(ValueError):
    """long → wide 변환을 할 수 없을 때."""


def long_to_wide(
    df: pd.DataFrame,
    time_col: str,
    tag_col: str,
    value_col: str,
    agg: str = "mean",
    tags: list[str] | None = None,
) -> pd.DataFrame:
    """long 표를 wide 로 돌려 세운다. 시간 컬럼은 그대로 컬럼으로 남긴다.

    같은 (시각, 태그) 가 두 번 이상 나오면 agg 로 합친다 — 히스토리언에서
    같은 초에 두 번 찍히는 일이 종종 있다. 그냥 pivot 하면 거기서 죽는다.

    반환한 표는 to_timeseries() 에 그대로 넣을 수 있다.
    """
    for c in (time_col, tag_col, value_col):
        if c not in df.columns:
            raise LayoutError(f"컬럼 '{c}' 이 표에 없습니다.")
    if tag_col == value_col or time_col in (tag_col, value_col):
        raise LayoutError("시간·태그·값 컬럼은 서로 달라야 합니다.")

    work = df[[time_col, tag_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.loc[work[time_col].notna()]
    if work.empty:
        raise LayoutError("시간으로 읽히는 줄이 없습니다.")

    work[tag_col] = work[tag_col].astype(str).str.strip()
    if tags:
        work = work.loc[work[tag_col].isin(tags)]
        if work.empty:
            raise LayoutError("고르신 태그에 해당하는 줄이 없습니다.")

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    dropped = int(work[value_col].isna().sum())
    work = work.loc[work[value_col].notna()]
    if work.empty:
        raise LayoutError(f"'{value_col}' 에 숫자로 읽히는 값이 없습니다.")

    dup = int(work.duplicated([time_col, tag_col]).sum())
    wide = work.pivot_table(index=time_col, columns=tag_col, values=value_col,
                            aggfunc=agg)
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.sort_index().reset_index()
    wide.columns.name = None

    wide.attrs["long_to_wide"] = {
        "태그 수": int(len(wide.columns) - 1),
        "시점 수": int(len(wide)),
        "숫자가 아니라 버린 줄": dropped,
        "같은 시각·태그 중복": dup,
        "중복 합치는 방법": agg,
    }
    return wide


def tag_inventory(df: pd.DataFrame, time_col: str, tag_col: str,
                  value_col: str) -> pd.DataFrame:
    """long 표에 어떤 태그가 몇 줄씩 있는지. 돌려 세우기 전에 고르라고 보여준다."""
    work = df[[time_col, tag_col, value_col]].copy()
    work[tag_col] = work[tag_col].astype(str).str.strip()
    num = pd.to_numeric(work[value_col], errors="coerce")
    g = work.assign(_v=num).groupby(tag_col)
    out = pd.DataFrame({
        "태그": g.size().index,
        "줄 수": g.size().to_numpy(),
        "숫자로 읽힌 비율": g["_v"].apply(lambda s: float(s.notna().mean())).to_numpy(),
        "평균": g["_v"].mean().to_numpy(),
        "최소": g["_v"].min().to_numpy(),
        "최대": g["_v"].max().to_numpy(),
    })
    return out.sort_values("줄 수", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 시계열 정규화
# ─────────────────────────────────────────────────────────────
def to_timeseries(
    df: pd.DataFrame,
    time_col: str,
    resample: str | None = None,
    agg: str = "mean",
    tz: str | None = None,
) -> pd.DataFrame:
    """시간 컬럼을 DatetimeIndex 로 바꾸고 정렬·중복제거한다."""
    out = df.copy()
    if time_col not in out.columns:
        raise KeyError(f"시간 컬럼 '{time_col}' 을 찾을 수 없습니다.")

    ts = pd.to_datetime(out[time_col], errors="coerce")
    bad = int(ts.isna().sum())
    out = out.loc[ts.notna()].copy()
    out.index = pd.DatetimeIndex(ts.loc[ts.notna()], name=time_col)
    out = out.drop(columns=[time_col])

    if tz:
        out.index = out.index.tz_localize(tz) if out.index.tz is None else out.index.tz_convert(tz)

    out = out[~out.index.duplicated(keep="last")].sort_index()

    if resample:
        num = out.select_dtypes("number")
        obj = out.drop(columns=num.columns)
        parts = [getattr(num.resample(resample), agg)()]
        if not obj.empty:
            parts.append(obj.resample(resample).last())
        out = pd.concat(parts, axis=1)[list(num.columns) + list(obj.columns)]

    out.attrs["dropped_bad_timestamps"] = bad
    return out


def infer_freq(index: pd.DatetimeIndex) -> str | None:
    """샘플링 주기를 추정한다. 결측 구간이 있어도 최빈 간격으로 잡는다."""
    if len(index) < 3:
        return None
    try:
        f = pd.infer_freq(index)
        if f:
            return f
    except (ValueError, TypeError):
        pass
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return None
    mode = deltas.mode()
    if mode.empty:
        return None
    return str(mode.iloc[0])
