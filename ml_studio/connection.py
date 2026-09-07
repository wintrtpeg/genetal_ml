"""데이터마트 접속 설정 — **이 파일만 고치시면 됩니다.**

여기에 값을 채워 두면 화면(1단계 → SQL 쿼리)에서 접속 항목이 사라지고
**SQL 입력창만** 남습니다. 호스트·계정·비밀번호는 화면 어디에도 뜨지 않습니다.

비워 두면 예전처럼 화면에서 직접 입력하는 방식으로 돕니다. 그래서 이 파일을
건드리지 않아도 도구는 그대로 동작합니다.

고치는 법
---------
아래 `HOST` 부터 `PASSWORD` 까지 다섯 줄에 평소 쓰시던 SQL 툴(DBeaver·DataGrip
등)의 접속 설정에 적힌 값을 그대로 넣으세요. 따옴표 안에만 채우면 됩니다.

    HOST     = "dm.internal.example.com"
    PORT     = 3108
    DATABASE = "master"
    USER     = "svc_ml"
    PASSWORD = "..."

저장하고 `run.bat` 을 다시 실행하면 반영됩니다.

주의 — 비밀번호가 적힌 채로 git 에 올리지 마세요
-----------------------------------------------
이 파일은 **비워진 채로** 저장소에 들어 있습니다. 값을 채운 뒤 그대로
커밋하면 비밀번호가 저장소에 남습니다. 회사 PC 안에서만 채워 쓰시고,
저장소에 올릴 일이 있으면 다시 비우고 올리세요.

비밀번호를 파일에 적기 곤란하면 환경변수로도 됩니다. 환경변수가 있으면
아래 값보다 **환경변수가 우선**합니다.

    set ML_STUDIO_DB_PASSWORD=...
    run.bat

같은 방식으로 ML_STUDIO_DB_HOST · _PORT · _DATABASE · _USER · _DRIVER 를
쓸 수 있습니다.
"""

from __future__ import annotations

import os

# ── 여기부터 고치세요 ────────────────────────────────────────────────────

# 데이터베이스 종류. SQream 이면 그대로 두세요.
# 다른 DBMS 는 core/datasource.py 의 DIALECTS 에 있는 값을 쓰면 됩니다.
#   postgresql+psycopg2 · mysql+pymysql · oracle+oracledb · mssql+pyodbc · trino
DRIVER = "sqream"

HOST = ""            # 예: "dm.internal.example.com"  또는 "192.168.0.10"
PORT = 3108          # SQream 기본 3108
DATABASE = ""        # 예: "master"
USER = ""            # 예: "svc_ml"
PASSWORD = ""        # 예: "..."

# 화면 맨 위에 표시할 이름. 어디에 붙는 중인지 알아보기 위한 것이라
# 호스트 주소는 넣지 마세요.
LABEL = "데이터마트"

# URL 뒤에 붙는 추가 파라미터. 담당자가 알려준 것이 있을 때만.
#   예: {"service": "sqream"}
PARAMS: dict[str, str] = {}

# 드라이버에 직접 넘기는 옵션. SQream 은 로드밸런서를 거치면 clustered 가
# True 여야 합니다. 비워 두면 DBMS 별 기본값이 자동으로 들어갑니다.
CONNECT_ARGS: dict = {}

# ── 여기까지 ────────────────────────────────────────────────────────────
# 아래는 건드리지 않으셔도 됩니다.


def _env(name: str, fallback):
    """환경변수가 있으면 그것을 쓴다. 파일에 비밀번호를 적기 곤란한 경우용."""
    got = os.environ.get(f"ML_STUDIO_DB_{name}")
    return got if got not in (None, "") else fallback


def settings() -> dict:
    """지금 적용되는 접속 설정. 환경변수가 파일 값보다 우선한다."""
    return {
        "driver": str(_env("DRIVER", DRIVER) or "").strip(),
        "host": str(_env("HOST", HOST) or "").strip(),
        "port": _env("PORT", PORT),
        "database": str(_env("DATABASE", DATABASE) or "").strip(),
        "user": str(_env("USER", USER) or "").strip(),
        "password": str(_env("PASSWORD", PASSWORD) or ""),
        "label": LABEL,
        "params": dict(PARAMS),
        "connect_args": dict(CONNECT_ARGS),
    }


def is_configured() -> bool:
    """접속 정보가 채워져 있는가.

    호스트와 DB 이름이 있으면 채워진 것으로 본다. 계정 없이 붙는 DB 도 있어서
    계정·비밀번호는 조건에 넣지 않는다.
    """
    s = settings()
    if s["driver"].startswith("sqlite"):
        return bool(s["database"])
    return bool(s["host"] and s["database"])


def missing() -> list[str]:
    """무엇이 비어 있는지. 화면에서 안내할 때 쓴다."""
    s = settings()
    if s["driver"].startswith("sqlite"):
        return [] if s["database"] else ["DATABASE"]
    gaps = []
    if not s["host"]:
        gaps.append("HOST")
    if not s["database"]:
        gaps.append("DATABASE")
    return gaps
