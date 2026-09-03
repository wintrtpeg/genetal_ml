"""첫 실행 도우미 — 가상환경 생성부터 화면 실행까지.

    python scripts/setup.py

run.bat / run.sh 가 이 파일을 부른다. 실제 일은 전부 여기서 한다.

왜 배치파일이 아니라 파이썬인가
-----------------------------
cmd.exe 는 `chcp 65001` 로 코드페이지를 바꾼 뒤 같은 배치파일 안에 멀티바이트
문자가 있으면 파일 읽는 위치를 잃어버린다. 줄 중간이 잘려 나가 엉뚱한 조각이
명령어로 실행된다. 그래서 배치파일은 ASCII 로만 두고, 안내와 분기는 전부
이쪽으로 옮겼다. 덤으로 윈도우·맥·리눅스가 같은 코드를 쓴다.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # `python scripts/setup.py` 로 실행되면 sys.path[0] 은 scripts/ 다.
    # scripts.launch 를 읽으려면 프로젝트 루트가 들어가 있어야 한다.
    sys.path.insert(0, str(ROOT))
VENV = ROOT / ".venv"
MIN_PY = (3, 10)
TESTED_PY = (3, 12)     # 여기까지 검증했다. 이보다 높으면 wheel 이 없을 수 있다.


# ── 출력 ────────────────────────────────────────────────────
def _enable_utf8() -> None:
    """윈도우 콘솔에서 한글이 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n[{n}/{total}] {msg}")


def die(msg: str, *hints: str) -> None:
    say(f"\n[X] {msg}")
    for h in hints:
        say(f"    {h}")
    say("\n" + "-" * 60)
    # 콘솔에서 오류 줄을 찾아 옮겨 적는 것이 제일 번거롭다. 파일 하나로 만든다.
    say(" 원인 파악에 필요한 것을 파일 하나로 모으려면:")
    say("     report.bat            (윈도우 — 더블클릭)")
    say("     ./report.sh           (macOS / Linux)")
    say(" 끝나면 'diagnostic_report.txt' 가 이 폴더에 생깁니다. 그 파일만 보내주세요.")
    say("-" * 60)
    sys.exit(1)


# ── 실행 ────────────────────────────────────────────────────
def run(cmd: list[str], *, must_succeed: bool = True, quiet: bool = False) -> int:
    """하위 프로세스 실행. must_succeed=False 면 실패해도 넘어간다.

    자식 프로세스는 이 파일의 _enable_utf8() 를 물려받지 않는다. 그래서 출력
    인코딩을 환경변수로 따로 넘긴다 — 한글 윈도우에서 출력을 파일로 넘기면
    cp949 로 떨어지고, em dash(—) 하나 때문에 테스트 전체가 UnicodeEncodeError
    로 죽는다. 결과를 로그로 남기려다 실행이 실패하는 셈이다.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8:replace", "PYTHONUTF8": "1"}
    kw = {"cwd": ROOT, "env": env}
    if quiet:
        kw["stdout"] = subprocess.DEVNULL
    code = subprocess.call(cmd, **kw)
    if code != 0 and must_succeed:
        raise SystemExit(code)
    return code


def venv_python() -> Path:
    return VENV / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")


# ── 어떤 파이썬으로 가상환경을 만들 것인가 ──────────────────
def _installed_pythons() -> list[tuple[tuple[int, int], str]]:
    """이 PC 에 깔린 파이썬 목록. [( (3,12), 실행경로 ), ...]"""
    found: dict[tuple[int, int], str] = {}

    if os.name == "nt":
        # py 런처가 알고 있는 것들. -0p 는 버전과 경로를 함께 준다.
        try:
            out = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                                 timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():
            m = re.search(r"(\d+)\.(\d+)", line)
            p = re.search(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", line, re.I)
            if m and p:
                found.setdefault((int(m.group(1)), int(m.group(2))), p.group(1))
    else:
        for minor in range(13, 9, -1):
            exe = shutil.which(f"python3.{minor}")
            if exe:
                found.setdefault((3, minor), exe)

    # 지금 돌고 있는 인터프리터도 후보에 넣는다
    found.setdefault(sys.version_info[:2], sys.executable)
    return sorted(found.items(), reverse=True)


def pick_interpreter(want: str | None) -> tuple[str, str]:
    """가상환경을 만들 파이썬을 고른다. (실행경로, 고른 이유) 반환.

    `py -3` 는 **설치된 것 중 가장 높은 버전**을 준다. 그래서 3.12 가 깔려 있어도
    갓 나온 3.14 가 잡히고, 아직 wheel 이 없는 패키지에서 막힌다. 검증 범위
    (3.10~3.12) 안에서 가장 높은 것을 우선으로 고른다.
    """
    cands = _installed_pythons()

    if want:                                   # 사용자가 --python 3.12 로 지정
        try:
            target = tuple(int(x) for x in want.split(".")[:2])
        except ValueError:
            die(f"버전 형식이 잘못됐습니다: {want}", "예: --python 3.12")
        for ver, exe in cands:
            if ver == target:
                return exe, f"지정하신 {want}"
        die(f"파이썬 {want} 을 찾지 못했습니다.",
            "설치된 버전: " + ", ".join(f"{a}.{b}" for (a, b), _ in cands))

    in_range = [(v, e) for v, e in cands if MIN_PY <= v <= TESTED_PY]
    if in_range:
        ver, exe = in_range[0]
        return exe, f"검증 범위에서 가장 높은 {ver[0]}.{ver[1]}"

    usable = [(v, e) for v, e in cands if v >= MIN_PY]
    if not usable:
        die(f"파이썬 {MIN_PY[0]}.{MIN_PY[1]} 이상을 찾지 못했습니다.",
            "python.org 에서 3.12 를 설치해 주세요.")
    ver, exe = usable[-1]                       # 검증범위 밖이면 가장 낮은 것
    return exe, f"검증 범위(~{TESTED_PY[0]}.{TESTED_PY[1]}) 안에 없어 {ver[0]}.{ver[1]} 사용"


def _version_of(exe: str) -> str:
    try:
        return subprocess.run(
            [exe, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"


STAMP = VENV / ".setup_ok"

# 없어도 도구는 돌지만, 있으면 쓸 수 있게 되는 것들. 하나씩 넣어서 실패한 것만
# 빼고 나머지는 살린다 — 한 줄로 몰아 설치하면 하나가 실패할 때 전부 안 깔린다.
#
# pysqream-sqlalchemy 는 SQream 데이터마트 접속 드라이버다. 예전에는 "쓰는 DBMS
# 것만 골라 깔라" 며 빼 뒀는데, 정작 이 도구를 쓰는 곳이 SQream 이라 매번
# "건너뜀 1" 로 남았다. 자동으로 넣되 실패는 감수한다.
EXTRAS = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2", "shap>=0.44",
          "pysqream-sqlalchemy"]


# 부가 패키지가 절대 건드리면 안 되는 것들. 여기가 어긋나면 도구 전체가 죽는다.
CORE_LOCK = ["numpy", "scipy", "scikit-learn", "pandas"]
CONSTRAINTS = VENV / "core-constraints.txt"


def _freeze(vpy: str) -> dict[str, str]:
    """지금 깔려 있는 버전을 읽는다. {패키지: 버전}."""
    try:
        out = subprocess.run([vpy, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=180).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    got = {}
    for line in out.splitlines():
        if "==" in line:
            name, _, ver = line.partition("==")
            got[name.strip().lower()] = ver.strip()
    return got


def _write_constraints(vpy: str) -> bool:
    """핵심 패키지 버전을 못으로 박아 둔다.

    **이게 없어서 실제로 도구가 통째로 죽었다.** 부가 패키지를 하나씩 설치하면
    하나가 실패해도 나머지는 살릴 수 있지만, 대신 **뒤에 깔리는 것이 앞의 것을
    끌어내릴 수 있다.** 실제로 shap 이 딸고 오는 numba 가 numpy 를 2.x → 1.26 으로
    내렸고, 이미 깔려 있던 scipy 가 그 numpy 에서 `np.long` 을 찾다가 죽었다.
    sklearn 이 scipy 를 import 하므로 결국 전 단계가 무너졌다.
    설치는 '성공' 으로 끝났고 아무도 알려주지 않았다.

    그래서 핵심 4종을 제약 파일에 고정하고, 부가 설치는 그 안에서만 하게 한다.
    제약을 못 맞추는 부가 패키지는 **설치되지 않는다** — 그게 맞는 결과다.
    없어도 되는 것 때문에 있어야 하는 것이 깨지면 안 된다.
    """
    have = _freeze(vpy)
    lines = [f"{p}=={have[p]}" for p in CORE_LOCK if p in have]
    if not lines:
        return False
    try:
        CONSTRAINTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    say("      기준 버전 고정: " + " · ".join(lines))
    return True


def _stack_is_sane(vpy: str) -> tuple[bool, str]:
    """핵심 묶음이 실제로 import 되는가. 버전 번호가 아니라 동작으로 확인한다."""
    probe = ("import numpy, scipy.sparse, sklearn, pandas;"
             "print(numpy.__version__, scipy.__version__, sklearn.__version__)")
    try:
        r = subprocess.run([vpy, "-c", probe], capture_output=True, text=True,
                           timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if r.returncode == 0:
        return True, r.stdout.strip()
    tail = [l for l in (r.stderr or "").strip().splitlines() if l.strip()]
    return False, tail[-1] if tail else "알 수 없는 실패"


def _repair(vpy: str) -> int:
    """깨진 조합을 되돌린다. 핵심을 다시 해석시키는 것이 핵심이다."""
    step(1, 3, "핵심 패키지 버전 재해석")
    run([vpy, "-m", "pip", "install", "--upgrade", "--prefer-binary",
         "-r", "requirements-core.txt"], must_succeed=False)

    step(2, 3, "묶음 점검")
    ok, msg = _stack_is_sane(vpy)
    if ok:
        say(f"      정상 — numpy·scipy·sklearn {msg}")
    else:
        say(f"      아직 깨져 있습니다 — {msg}")

    step(3, 3, "의존성 정합성")
    run([vpy, "-m", "pip", "check"], must_succeed=False)
    if ok:
        _write_constraints(vpy)
        say("\n복구됐습니다. run.bat 을 다시 실행하세요.")
        return 0
    say("\n자동 복구로 안 됩니다. .venv 폴더를 지우고 run.bat 을 다시 실행해 주세요.")
    say("(설치가 처음부터 다시 돌아 3~10분 걸립니다)")
    return 1


def _warn_if_sqlalchemy_downgraded(vpy: str) -> None:
    """SQLAlchemy 가 2.0 아래로 내려갔는지 본다.

    드라이버 패키지가 옛 SQLAlchemy 를 요구하면 pip 이 말없이 내려버린다.
    설치는 '성공' 으로 끝나고 SQL 경로만 조용히 다르게 도는데, 그 조용함이
    나중에 원인 못 찾는 오류가 된다. 여기서 한 번 짚고 넘어간다.
    """
    try:
        out = subprocess.run(
            [vpy, "-c", "import sqlalchemy; print(sqlalchemy.__version__)"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return
    ver = (out.stdout or "").strip()
    if not ver:
        return
    major = ver.split(".")[0]
    if major.isdigit() and int(major) < 2:
        # 정상이다 — SQream 드라이버가 1.4 를 요구한다. 이 도구가 쓰는 API 넷은
        # 1.4·2.0 에서 동작이 같아서 문제되지 않는다. 다만 왜 내려갔는지는
        # 남겨 둔다. 나중에 "왜 2.0 이 아니지" 를 다시 조사하지 않도록.
        say(f"      SQLAlchemy {ver} (SQream 드라이버가 요구하는 버전 — 정상입니다)")
    else:
        say(f"      SQLAlchemy {ver}")


def _requirements_fingerprint() -> str:
    """무엇을 설치하기로 했는지의 지문.

    **이게 없으면 조용한 함정이 생긴다.** 새 버전을 기존 폴더에 덮어썼는데
    설치 목록이 바뀌었다면, 도장만 보고 빠른 경로로 가서 새로 추가된 패키지가
    영영 안 깔린다. 사용자는 "받았는데 왜 그대로지" 가 된다.
    목록이 조금이라도 달라지면 전체 설치를 다시 한다.
    """
    parts = ["|".join(EXTRAS)]
    for f in ("requirements-core.txt", "requirements-extra.txt"):
        p = ROOT / f
        parts.append(p.read_text(encoding="utf-8") if p.exists() else "")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _ready() -> bool:
    """설치·점검을 이미 끝냈는가.

    두 번째 실행부터는 설치와 테스트를 다시 돌 이유가 없다. 예전에는 매번
    6단계를 전부 돌아서 화면 한 번 보려고 몇 분을 기다려야 했다.
    도장은 **설치가 끝난 뒤에** 찍으므로, 중간에 끊긴 실행은 다시 처음부터 간다.
    """
    if not (STAMP.exists() and venv_python().exists()):
        return False
    try:
        if STAMP.read_text(encoding="utf-8").strip() != _requirements_fingerprint():
            return False              # 설치 목록이 바뀌었다 — 다시 깐다
    except OSError:
        return False
    try:
        # 핵심 패키지가 실제로 임포트되는지까지 본다. 도장만 믿으면
        # 사용자가 .venv 를 건드린 뒤에 엉뚱한 오류로 죽는다.
        return subprocess.run(
            [str(venv_python()), "-c",
             "import streamlit, pandas, numpy, scipy.sparse, sklearn, plotly"],
            capture_output=True, timeout=90).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> int:
    _enable_utf8()

    if "--repair" in sys.argv:
        say("=" * 60)
        say("  패키지 조합 복구")
        say("=" * 60)
        if not venv_python().exists():
            die("가상환경이 없습니다.", "run.bat 을 먼저 실행해 주세요.")
        return _repair(str(venv_python()))

    full = "--full" in sys.argv or "--setup" in sys.argv
    if not full and _ready():
        # 빠른 경로 — 설치·테스트를 건너뛰고 바로 화면을 띄운다.
        say("=" * 60)
        say("  시계열 ML 스튜디오")
        say("=" * 60)
        say()
        say("    설치는 이미 끝나 있습니다. 바로 화면을 띄웁니다.")
        say("    패키지를 다시 설치하거나 점검하려면:  run.bat --full")
        from scripts.launch import launch
        return launch(str(venv_python()))

    say("=" * 60)
    say("  시계열 ML 스튜디오 — 첫 실행")
    say("=" * 60)

    # 1. 쓸 파이썬 고르기
    # 인자 해석은 **전부 여기서** 한다. 배치파일에 분기를 넣었더니 cmd.exe 가
    # ( ... ) 블록을 통째로 먼저 해석하면서 빈 변수 때문에 ") 은(는) 예상되지
    # 않습니다" 로 죽고 창이 닫혔다. 배치파일은 %* 로 넘기기만 한다.
    want = None
    for i, a in enumerate(sys.argv):
        if a == "--python" and i + 1 < len(sys.argv):
            want = sys.argv[i + 1]
        elif a.startswith("--python="):
            want = a.split("=", 1)[1]
        elif i > 0 and re.fullmatch(r"\d+\.\d+", a):
            want = a          # `run.bat 3.12` 처럼 버전만 준 경우

    exe, why = pick_interpreter(want)
    say(f"    파이썬 {_version_of(exe)} — {why}")
    if exe != sys.executable:
        say(f"      ({exe})")
    say("      다른 버전을 쓰려면:  run.bat 3.12   (또는 --python 3.12)")

    total = 6

    # 2. 가상환경
    if venv_python().exists():
        cur = _version_of(str(venv_python()))
        step(1, total, f"가상환경 확인 — 이미 있습니다 (파이썬 {cur})")
        if want and not cur.startswith(want):
            die(f"기존 가상환경은 파이썬 {cur} 로 만들어졌습니다.",
                f"{want} 로 다시 만들려면 이 폴더의 .venv 를 지우고 다시 실행하세요.")
    else:
        step(1, total, "가상환경 생성")
        try:
            run([exe, "-m", "venv", str(VENV)])
        except SystemExit:
            die("가상환경 생성에 실패했습니다.",
                "회사 PC 보안정책으로 막히는 경우가 있습니다.",
                "가상환경 없이 진행하려면:",
                f"  {exe} -m pip install --user -r requirements-core.txt",
                f"  {exe} -m streamlit run app/main.py")
    vpy = str(venv_python())
    v = tuple(int(x) for x in (_version_of(vpy).split(".") + ["0", "0"])[:2])

    # 3. 핵심 패키지 — 반드시 성공해야 한다
    step(2, total, "핵심 패키지 설치 — 처음에는 3~10분 걸립니다")
    say("      진행 줄이 한동안 안 움직여도 정상입니다. 큰 wheel 을 받는 중이거나")
    say("      의존성을 맞춰 보는 중입니다. 10분을 넘기면 그때 끊으세요.")
    run([vpy, "-m", "pip", "install", "--upgrade", "pip"],
        must_succeed=False, quiet=True)
    try:
        # --prefer-binary : 소스 빌드를 피한다. 윈도우에서 컴파일이 걸리면
        #                   출력이 멈춘 것처럼 보이는 채로 수십 분이 간다.
        # --progress-bar on : 무슨 일이 일어나는지 보이게 한다.
        run([vpy, "-m", "pip", "install", "--prefer-binary",
             "--progress-bar", "on", "-r", "requirements-core.txt"])
    except SystemExit:
        hints = ["사내 프록시 환경이면 아래를 먼저 설정하고 다시 실행해 보세요:",
                 "  set HTTP_PROXY=http://프록시주소:포트",
                 "  set HTTPS_PROXY=http://프록시주소:포트"]
        if v > TESTED_PY:
            hints = [f"파이썬 {v[0]}.{v[1]} 용 wheel 이 아직 없을 가능성이 큽니다.",
                     "python.org 에서 3.12 를 설치한 뒤,",
                     "이 폴더의 .venv 폴더를 지우고 다시 실행해 주세요."] + hints
        die("핵심 패키지 설치에 실패했습니다.", *hints)

    # 설치는 끝났는데 묶음이 깨져 있을 수 있다 — 예전 실행이 남긴 .venv 에
    # 이미 numpy 가 내려가 있는 경우다. **이 상태로 버전을 고정하면 깨진 조합을
    # 못으로 박아 버린다.** 고정하기 전에 한 번 되돌린다.
    ok, msg = _stack_is_sane(vpy)
    if not ok:
        say()
        say(f"      ! 기존 환경의 numpy·scipy 조합이 깨져 있습니다 — {msg}")
        say("      버전을 다시 해석합니다.")
        run([vpy, "-m", "pip", "install", "--upgrade", "--prefer-binary",
             "-r", "requirements-core.txt"], must_succeed=False)
        ok, msg = _stack_is_sane(vpy)
        if not ok:
            run([vpy, "-m", "pip", "check"], must_succeed=False)
            die(f"핵심 묶음을 되살리지 못했습니다 — {msg}",
                "이 폴더의 .venv 폴더를 지우고 run.bat 을 다시 실행해 주세요.",
                "(설치가 처음부터 다시 돌아 3~10분 걸립니다)")
        say(f"      되돌렸습니다 — numpy·scipy·sklearn {msg}")

    # 4. 부스팅 · SHAP — 없어도 동작한다
    step(3, total, "부스팅 / SHAP 설치 — 실패해도 계속 진행합니다")
    # 한 줄로 몰아 설치하면 하나만 실패해도 나머지가 통째로 안 깔린다.
    # 특히 catboost 는 용량이 크고 lightgbm 은 윈도우에서 재배포 패키지를 요구한다.
    # 하나씩 넣어 성공한 것만이라도 남긴다.
    # 핵심 버전을 못으로 박고 그 안에서만 설치한다. 제약을 못 맞추는 부가
    # 패키지는 안 깔린다 — 없어도 되는 것 때문에 있어야 하는 것이 깨지면 안 된다.
    pinned = _write_constraints(vpy)
    limit = ["-c", str(CONSTRAINTS)] if pinned else []

    failed = []
    for pkg in EXTRAS:
        name = pkg.split(">")[0]
        say(f"      · {name}")
        if run([vpy, "-m", "pip", "install", "--prefer-binary", *limit, pkg],
               must_succeed=False, quiet=True) != 0:
            failed.append(name)

    # SQream 드라이버가 SQLAlchemy 를 1.x 로 끌어내리는 일이 있다. 그러면 설치는
    # "성공" 인데 SQL 경로가 조용히 다르게 동작한다. 조용한 게 제일 나쁘다.
    _warn_if_sqlalchemy_downgraded(vpy)

    # **여기서 반드시 막아야 한다.** 예전에는 부가 설치가 numpy 를 끌어내려
    # scipy·sklearn 을 깨뜨렸는데도 "설치 완료" 로 넘어갔다. 그 뒤 테스트 24건이
    # 죽고, 사용자는 600줄짜리 진단 리포트를 받아 보고서야 원인을 알았다.
    # 깨진 채로 다음 단계에 가느니 여기서 멈추고 복구 명령을 주는 게 낫다.
    ok, msg = _stack_is_sane(vpy)
    if not ok:
        say()
        say(f"      ! 핵심 묶음이 깨졌습니다 — {msg}")
        run([vpy, "-m", "pip", "check"], must_succeed=False)
        die("부가 패키지가 numpy·scipy 조합을 깨뜨렸습니다.",
            "아래 한 줄이면 되돌아갑니다:",
            "",
            f"  {Path(vpy).name if os.name != 'nt' else vpy} 로 실행:",
            "  run.bat --repair          (윈도우)",
            "  ./run.sh --repair         (macOS / Linux)",
            "",
            "그래도 안 되면 이 폴더의 .venv 를 지우고 run.bat 을 다시 실행하세요.")
    say(f"      묶음 정상 — numpy·scipy·sklearn {msg}")
    if failed:
        say(f"      ! 설치 실패: {', '.join(failed)}")
        say("        나머지는 그대로 쓰입니다. 설치된 모델만 목록에 잡히고,")
        say("        SHAP 이 없으면 해석 화면이 순열 중요도로 대체됩니다.")
    else:
        say("      전부 설치됨.")

    # 5. 누수 회귀 테스트 — 여기가 통과해야 결과를 믿을 수 있다
    step(4, total, "누수 회귀 테스트 — 1~3분")
    if run([vpy, str(ROOT / "tests" / "run_tests.py")], must_succeed=False) != 0:
        die("회귀 테스트가 실패했습니다.",
            "누수 방지 장치를 검증하는 테스트라, 여기가 깨지면 결과를 믿을 수 없습니다.")

    # 6. 실행 환경 점검 — 여기서만 확인되는 것들
    #    이 도구를 만든 환경에는 plotly · shap · 부스팅 · SQLAlchemy 가 없어서
    #    그 코드는 한 번도 실행된 적이 없다. 설치가 끝난 지금 한 번 훑는다.
    #    실패해도 진행은 막지 않는다 — 대부분은 특정 화면 하나의 문제이고,
    #    나머지 화면은 정상 동작하기 때문이다.
    step(5, total, "실행 환경 점검 — 차트 · SHAP · 부스팅 · SQL · 리포트")
    if run([vpy, str(ROOT / "scripts" / "verify_env.py")], must_succeed=False) != 0:
        say("      ! 일부 항목이 실패했습니다. 나머지 화면은 그대로 쓰셔도 됩니다.")
        say("        원인 파악용 파일을 만들려면 이 창을 닫고 report.bat 을 실행하세요.")
        say("        (diagnostic_report.txt 하나만 보내주시면 됩니다)")

    demo = ROOT / "data" / "demo_timeseries.csv"
    if not demo.exists():
        run([vpy, str(ROOT / "scripts" / "make_demo_data.py")], must_succeed=False)

    # 여기까지 왔으면 설치는 끝났다. 도장을 찍어 두면 다음 실행부터는
    # 위 단계를 전부 건너뛰고 곧장 화면으로 간다.
    try:
        STAMP.write_text(_requirements_fingerprint(), encoding="utf-8")
    except OSError:
        pass                       # 못 써도 동작에는 지장 없다 (매번 전체 실행)

    # 7. 화면
    step(6, total, "화면 실행")
    say("      다음부터 run.bat 을 실행하면 이 단계로 바로 옵니다.")
    from scripts.launch import launch
    return launch(vpy)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\n중단했습니다.")
        sys.exit(130)
