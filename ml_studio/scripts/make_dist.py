"""회사 PC 로 가져갈 배포 zip 을 만든다.

    python scripts/make_dist.py

폴더 옆에 `dist/ml_studio_YYYYMMDD.zip` 이 생깁니다. 그 파일 하나만 옮겨서
압축을 풀고 `run.bat` 을 누르면 됩니다.

무엇을 빼는가
-------------
`.venv` 는 **절대 넣지 않습니다.** 2GB 가 넘고, 무엇보다 PC 가 바뀌면
그대로는 못 씁니다 (경로가 박혀 있고 OS·파이썬 버전도 다릅니다). 회사 PC 에서
`run.bat` 이 그 PC 에 맞는 가상환경을 새로 만듭니다.

`runs/` (지난 실행 결과), `__pycache__`, `diagnostic_report.txt` 도 뺍니다.
받는 쪽에 필요 없고, `runs/` 에는 분석한 데이터의 흔적이 남아 있습니다.

접속 정보는 기본적으로 **비워서** 나갑니다
-------------------------------------------
`connection.py` 에 데이터마트 계정을 적어 두었다면, 그대로 zip 에 넣어 남에게
보내는 순간 비밀번호가 함께 갑니다. 그래서 기본값은 **비우고 넣기** 입니다.
회사 PC 에서 압축을 푼 뒤 그쪽에서 한 번 적으시면 됩니다.

내 PC 로만 옮기는 것이라 그대로 가져가고 싶으면:

    python scripts/make_dist.py --with-connection

이때는 경고를 한 번 띄우고 넣습니다.
"""

from __future__ import annotations

import re
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# 넣지 않을 것들. 디렉터리 이름은 경로 어디에 나와도 통째로 뺀다.
SKIP_DIRS = {".venv", "venv", "runs", "__pycache__", ".git", ".pytest_cache",
             ".idea", ".vscode", "dist", ".mypy_cache", ".ruff_cache"}
SKIP_FILES = {"diagnostic_report.txt", ".DS_Store", "Thumbs.db"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log"}

# 비밀번호가 들어갈 수 있는 줄. 배포본에서는 따옴표 안을 비운다.
SECRET_FIELDS = ("HOST", "DATABASE", "USER", "PASSWORD")


def _enable_utf8() -> None:
    """한글 윈도우 콘솔(cp949)에서 출력이 죽지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.name in SKIP_FILES or path.suffix in SKIP_SUFFIX:
        return False
    return True


def _blank_connection(text: str) -> tuple[str, list[str]]:
    """connection.py 의 접속 항목을 비운다. 무엇을 비웠는지 함께 돌려준다."""
    cleared = []
    out = text
    for field in SECRET_FIELDS:
        # 모듈 최상단의 `FIELD = "..."` 한 줄만 바꾼다 (docstring 예시는 건드리지 않음)
        pattern = rf'^({field}\s*=\s*)(["\']).*?\2'
        def _sub(m):
            # 원래 비어 있던 줄은 '비웠다' 고 세지 않는다 — 세면 아무것도 안
            # 했는데 뭔가 한 것처럼 보고된다.
            if m.group(0) != f'{m.group(1)}{m.group(2)}{m.group(2)}':
                cleared.append(field)
            return f'{m.group(1)}{m.group(2)}{m.group(2)}'
        out = re.sub(pattern, _sub, out, count=1, flags=re.M)
    return out, cleared


def _connection_has_values() -> list[str]:
    """connection.py 에 실제로 채워진 항목. 없으면 빈 목록."""
    sys.path.insert(0, str(ROOT))
    try:
        import connection                      # noqa: PLC0415
    except Exception:                          # noqa: BLE001
        return []
    filled = []
    for field in SECRET_FIELDS:
        if str(getattr(connection, field, "") or "").strip():
            filled.append(field)
    return filled


def main() -> int:
    _enable_utf8()
    keep_conn = "--with-connection" in sys.argv

    filled = _connection_has_values()
    print("=" * 60)
    print("  배포 zip 만들기")
    print("=" * 60)

    if filled and keep_conn:
        print()
        print("  ! connection.py 의 접속 정보를 **그대로 넣습니다** — "
              + ", ".join(filled))
        print("    이 zip 을 남에게 보내면 계정·비밀번호가 함께 갑니다.")
    elif filled:
        print()
        print("  connection.py 에 적힌 접속 정보(" + ", ".join(filled) + ")는 "
              "비워서 넣습니다.")
        print("  회사 PC 에서 압축을 푼 뒤 그쪽 connection.py 에 적으시면 됩니다.")
        print("  그대로 가져가려면:  python scripts/make_dist.py --with-connection")

    DIST.mkdir(exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    out = DIST / f"ml_studio_{stamp}.zip"
    if out.exists():
        out.unlink()

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _wanted(p))
    if not files:
        print("[X] 넣을 파일을 찾지 못했습니다.")
        return 1

    n_blanked = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            rel = f.relative_to(ROOT)
            arc = Path("ml_studio") / rel        # 풀면 ml_studio/ 폴더가 생기게
            if rel.as_posix() == "connection.py" and not keep_conn:
                text, cleared = _blank_connection(f.read_text(encoding="utf-8"))
                z.writestr(str(arc), text)
                n_blanked = len(cleared)
            else:
                z.write(f, str(arc))

    size_mb = out.stat().st_size / (1024 * 1024)
    print()
    print(f"  파일 {len(files)}개 · {size_mb:.1f} MB")
    if n_blanked:
        print(f"  connection.py 의 {n_blanked}개 항목을 비웠습니다.")
    print(f"  {out}")
    print()
    print("-" * 60)
    print(" 회사 PC 에서")
    print("   1. 이 zip 을 옮겨 압축을 풉니다")
    print("   2. 폴더 안의 run.bat 을 더블클릭합니다 (첫 실행 3~10분)")
    print("   3. 데이터마트를 쓰시면 connection.py 를 열어 접속 정보를 적고")
    print("      run.bat 을 다시 실행합니다")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
