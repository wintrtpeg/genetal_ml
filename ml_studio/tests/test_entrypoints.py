"""실행 진입점(`.bat` · `.sh`)을 검증한다.

**왜 이 파일이 생겼나 — 실제 사고**

`run.bat` 에 인자 처리 분기를 넣었다가 사용자가 실행을 못 했다. cmd 창에
`) 은(는) 예상되지 않습니다` 만 뜨고 바로 닫혔다. 그때 회귀 테스트는
**288건 전부 통과**였다. 배치파일을 검증하는 테스트가 하나도 없었기 때문이다.

커버리지가 0인 파일을 고치면서 커버리지를 먼저 만들지 않은 것이 원인이다.
파이썬은 288건이 지키는데 정작 **사용자가 제일 먼저 누르는 파일**은 아무도
안 봤다. 여기서 그걸 메운다.

핵심은 `_cmd_block_parse` 다 — cmd.exe 가 `( ... )` 블록을 실행 전에 통째로
해석하면서 변수를 먼저 펼치는 동작을 흉내 낸다. 인자 없이 더블클릭하면 변수가
비므로, **변수를 전부 빈 문자열로 펼친 뒤에도 문법이 성립하는지**를 본다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATS = sorted(ROOT.glob("*.bat"))
SHS = sorted(ROOT.glob("*.sh"))


# ── cmd.exe 흉내 ─────────────────────────────────────────────
_VAR = re.compile(r"%(?:~?\d|\*|[A-Za-z_][A-Za-z0-9_]*(?::[^%]*)?)%?")


def _expand_empty(line: str) -> str:
    """변수를 전부 빈 값으로 펼친다 — 인자 없이 더블클릭한 상황."""
    return _VAR.sub("", line)


def _cmd_block_parse(text: str) -> list[str]:
    """cmd 가 블록을 먼저 해석하다 깨지는 지점을 찾는다. 문제를 목록으로 돌려준다."""
    problems: list[str] = []
    depth = 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        low = line.lower()
        if not line or low.startswith(("rem", "::")):
            continue

        # 블록 안에서는 실행 전에 변수가 펼쳐진다. 그 상태로 문법을 본다.
        probe = _expand_empty(line) if depth > 0 or line.endswith("(") else line

        if not low.startswith("echo"):
            # if 비교문의 양쪽이 남아 있는가. `if =="-"` 같은 꼴이면 깨진다.
            m = re.match(r"^if\s+(?:not\s+)?(.+?)\s*==\s*(.+?)(\s*\(.*)?$",
                         probe, re.I)
            if m and (not m.group(1).strip() or not m.group(2).strip()):
                problems.append(f"{n}: 비교 대상이 사라집니다 — {line}")
            # 따옴표가 홀수면 그 자리에서 파서가 어긋난다
            if probe.count('"') % 2:
                problems.append(f"{n}: 따옴표가 짝이 안 맞습니다 — {line}")
            # 블록 안의 중첩 if / else 는 한 줄에 두 블록이 걸려 깨진다
            if depth > 0 and re.match(r"^if\b", low):
                problems.append(f"{n}: 블록 안 중첩 if — {line}")
            if re.search(r"\belse\b", line, re.I):
                problems.append(f"{n}: else 절 — {line}")

        body = "" if low.startswith("echo") else line
        depth += body.count("(") - body.count(")")
        if depth < 0:
            problems.append(f"{n}: 닫는 괄호가 더 많습니다 — {line}")
            depth = 0
    if depth:
        problems.append(f"끝까지 안 닫힌 괄호 {depth}개")
    return problems


# ── 본 검증 ──────────────────────────────────────────────────
def test_batch_files_survive_a_bare_double_click():
    """인자 없이 더블클릭해도 문법이 성립해야 한다. 실제로 여기서 터졌다."""
    assert BATS, "배치파일을 하나도 못 찾았습니다"
    bad = []
    for bat in BATS:
        for p in _cmd_block_parse(bat.read_text(encoding="utf-8")):
            bad.append(f"{bat.name}:{p}")
    assert not bad, ("cmd 에서 깨집니다 (인자 없이 실행한 상황):\n  "
                     + "\n  ".join(bad))


def test_the_checker_catches_the_bug_that_actually_shipped():
    """검사기 자체를 검사한다.

    예전에 sprintf 검사기가 `%.2%` 를 통과시킨 적이 있다. 검사기가 조용히
    아무것도 안 잡으면 없느니만 못하다. 실제로 사용자를 막았던 그 코드를
    그대로 넣어 본다.
    """
    shipped_broken = (
        '@echo off\n'
        'set "ARG="\n'
        'set "A1=%~1"\n'
        'if defined A1 (\n'
        '  if "%A1:~0,1%"=="-" (set "ARG=%A1%") else (set "ARG=--python %A1%")\n'
        ')\n'
    )
    found = _cmd_block_parse(shipped_broken)
    assert found, "실제로 터진 코드를 검사기가 통과시킵니다"
    assert any("중첩 if" in f or "else" in f for f in found), found


def test_goto_targets_exist():
    """없는 라벨로 goto 하면 그 자리에서 죽고 창이 닫힌다."""
    bad = []
    for bat in BATS:
        src = bat.read_text(encoding="utf-8")
        labels = {m.group(1).lower()
                  for m in re.finditer(r"^\s*:([A-Za-z_][\w]*)", src, re.M)}
        labels |= {"eof"}                       # goto :eof 는 내장
        for m in re.finditer(r"\bgoto\s+:?([A-Za-z_][\w]*)", src, re.I):
            if m.group(1).lower() not in labels:
                bad.append(f"{bat.name}: goto {m.group(1)} — 라벨이 없습니다")
    assert not bad, "\n".join(bad)


def test_batch_files_are_ascii_only():
    """한글을 넣으면 cp949/UTF-8 코드페이지 사이에서 파일 읽는 위치가 어긋난다.

    줄 중간이 잘려 엉뚱한 조각이 명령으로 실행된다. 그래서 안내문은 전부
    scripts/setup.py 쪽에 둔다.
    """
    bad = []
    for bat in BATS:
        raw = bat.read_bytes()
        if any(b > 0x7F for b in raw):
            n = next(i for i, line in enumerate(raw.splitlines(), 1)
                     if any(b > 0x7F for b in line))
            bad.append(f"{bat.name}:{n} 에 ASCII 밖 문자")
    assert not bad, "\n".join(bad)


def test_batch_files_call_a_script_that_exists():
    """경로를 잘못 적으면 '지정된 경로를 찾을 수 없습니다' 로 끝난다."""
    bad = []
    for bat in BATS:
        for m in re.finditer(r'"(scripts\\[\w.]+\.py)"',
                             bat.read_text(encoding="utf-8")):
            target = ROOT / m.group(1).replace("\\", "/")
            if not target.exists():
                bad.append(f"{bat.name}: {m.group(1)} 가 없습니다")
    assert not bad, "\n".join(bad)


def test_batch_files_keep_the_window_open():
    """실패했을 때 창이 그냥 닫히면 사용자는 원인을 볼 수 없다.

    이번 사고에서 사용자가 본 것은 오류 한 줄과 닫힌 창이 전부였다.
    """
    for bat in BATS:
        src = bat.read_text(encoding="utf-8").lower()
        assert "pause" in src or "cmd /k" in src, f"{bat.name}: 창이 바로 닫힙니다"


def test_shell_scripts_parse():
    """bash -n 으로 문법만 검사한다 (실행은 하지 않는다).

    윈도우에는 bash 가 없다. 예전 판은 그걸 안 봐서 회사 PC 에서
    `FileNotFoundError: 지정된 파일을 찾을 수 없습니다` 로 **실패**했다 —
    검사할 도구가 없는 것은 제품 결함이 아니라 건너뛸 일이다.
    """
    import shutil as _sh

    assert SHS, "쉘 스크립트를 하나도 못 찾았습니다"
    if not _sh.which("bash"):
        pytest.skip("이 PC 에 bash 가 없습니다 (윈도우에서는 정상)")
    bad = []
    for sh in SHS:
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{sh.name}: {r.stderr.strip()}")
    assert not bad, "\n".join(bad)


def test_shell_scripts_pass_arguments_through():
    """`./run.sh --full` 이 동작해야 한다."""
    src = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert '"$@"' in src or "--python" in src, "인자를 전달하지 않습니다"


def test_run_bat_passes_arguments_to_python():
    """인자 해석은 파이썬이 한다 — 배치파일에서 뜯어보면 이번 사고가 재발한다."""
    src = (ROOT / "run.bat").read_text(encoding="utf-8")
    assert "%*" in src, "인자를 통과시키지 않습니다"
    assert "%~1" not in src, "배치파일이 인자를 해석하고 있습니다"


def test_setup_still_accepts_a_bare_version():
    """`run.bat 3.12` 는 계속 동작해야 한다. 해석 위치만 옮긴 것이다."""
    from scripts import setup

    keep = sys.argv
    try:
        for argv, expect in ((["setup.py", "3.12"], "3.12"),
                             (["setup.py", "--python", "3.11"], "3.11"),
                             (["setup.py", "--python=3.10"], "3.10"),
                             (["setup.py", "--full"], None),
                             (["setup.py"], None)):
            sys.argv = argv
            want = None
            for i, a in enumerate(sys.argv):
                if a == "--python" and i + 1 < len(sys.argv):
                    want = sys.argv[i + 1]
                elif a.startswith("--python="):
                    want = a.split("=", 1)[1]
                elif i > 0 and re.fullmatch(r"\d+\.\d+", a):
                    want = a
            assert want == expect, f"{argv} → {want} (기대 {expect})"
    finally:
        sys.argv = keep
    assert setup is not None


# ── 설치 목록이 갈라지지 않는가 ──────────────────────────────
def _pkgs(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-r"):
            continue
        out.add(re.split(r"[<>=!\[]", line)[0].strip().lower())
    return out


def test_requirements_files_do_not_drift():
    """같은 패키지가 두 파일에 적히면 버전이 갈라진다.

    pysqream 을 core 에서 extra 로 옮기면서 실제로 두 군데에 남을 뻔했다.
    """
    core = _pkgs(ROOT / "requirements-core.txt")
    extra = _pkgs(ROOT / "requirements-extra.txt")
    both = core & extra
    assert not both, f"두 파일에 겹쳐 적힌 패키지: {sorted(both)}"

    allin = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "requirements-core.txt" in allin and "requirements-extra.txt" in allin, \
        "전체 설치 파일이 둘을 다 참조하지 않습니다"


def test_setup_extras_match_the_extra_requirements_file():
    """setup.py 가 깔 목록과 requirements-extra.txt 가 어긋나면,
    run.bat 로 깐 사람과 pip -r 로 깐 사람이 다른 환경을 갖게 된다."""
    from scripts import setup

    listed = {re.split(r"[<>=!\[]", p)[0].strip().lower() for p in setup.EXTRAS}
    infile = _pkgs(ROOT / "requirements-extra.txt")
    assert listed == infile, (f"setup.py EXTRAS={sorted(listed)} vs "
                              f"requirements-extra.txt={sorted(infile)}")


# ── 패키지 조합 붕괴 — 실제로 회사 PC 를 멈춘 것 ─────────────
def test_envcheck_reports_a_broken_stack_with_a_repair_command():
    """묶음이 깨졌을 때 **무엇을 하면 되는지**까지 말해야 한다.

    실제 사고 — numpy 1.26 과 scipy 1.18 이 함께 깔렸다. scipy 는 numpy 2.x 를
    전제로 `np.long` 을 참조하는데 1.26 에는 없다. sklearn 이 scipy 를
    import 하므로 도구 전체가 죽었고, 화면에는 **원인 하나가 아니라 증상 24개**가
    떴다. 사용자는 600줄 리포트를 만들어 보내고서야 원인을 알았다.
    """
    from scripts import envcheck

    msg = envcheck.message("scipy.sparse — AttributeError: "
                           "module 'numpy' has no attribute 'long'")
    assert "코드 문제가 아닙니다" in msg, "제품 결함처럼 읽힙니다"
    assert "run.bat --repair" in msg, "윈도우 복구 명령이 없습니다"
    assert "run.sh --repair" in msg, "macOS/Linux 복구 명령이 없습니다"
    assert "numpy" in msg and "scipy" in msg
    assert envcheck.probe() == "", "이 환경은 정상인데 깨졌다고 합니다"


def test_runner_stops_with_one_cause_not_twenty_four_symptoms():
    """묶음이 깨졌을 때 러너가 **증상 대신 원인**을 찍고 멈춰야 한다.

    실제로 회사 PC 에서 24건이 전부 같은 AttributeError 로 실패했다.
    사용자가 본 것은 원인 하나가 아니라 증상 스물넷이었다.

    소스를 훑는 대신 진짜로 돌려 본다 — scipy.sparse 임포트를 그때 그 예외로
    막아 놓고 러너를 실행한다.
    """
    poison = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name.startswith('scipy.sparse'):\n"
        "        raise AttributeError(\"module 'numpy' has no attribute 'long'\")\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "sys.path.insert(0, %r)\n"
        "sys.argv = ['run_tests.py']\n"
        "import tests.run_tests as R\n"
        "sys.exit(R.main())\n" % str(ROOT))
    # 시간 제한이 검증의 일부다 — preflight 가 없으면 전체 모듈을 다 돌다가
    # 몇 분을 쓴다. "빨리 멈춘다" 까지가 이 기능의 요건이다.
    try:
        r = subprocess.run([sys.executable, "-c", poison], capture_output=True,
                           text=True, cwd=str(ROOT), timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "깨진 상태에서 곧바로 멈추지 않고 전체 테스트를 돌고 있습니다 — "
            "preflight 가 동작하지 않습니다") from None
    assert r.returncode == 1, f"멈추지 않았습니다 (종료코드 {r.returncode})"
    out = r.stdout
    assert "코드 문제가 아닙니다" in out, f"원인을 설명하지 않습니다:\n{out[:600]}"
    assert "--repair" in out, "복구 방법을 알려주지 않습니다"
    assert out.count("no attribute 'long'") <= 3, (
        "증상을 여러 번 반복해 찍습니다 — 원인 하나만 찍어야 합니다")


def test_all_three_entrypoints_use_the_same_diagnosis():
    """회귀 테스트·환경 점검·진단 리포트가 **같은 말**을 해야 한다.

    각자 다른 문장과 다른 복구 명령을 주면 사용자가 어느 쪽을 믿을지 모른다.
    """
    for f in ("tests/run_tests.py", "scripts/verify_env.py",
              "scripts/collect_report.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "envcheck" in src, f"{f} 가 공용 진단을 안 씁니다"


def test_setup_pins_the_core_stack_before_installing_extras():
    """부가 패키지가 핵심 버전을 끌어내리지 못하게 막아야 한다.

    **이 제약이 없어서 실제로 도구가 죽었다.** shap 이 딸고 오는 numba 가
    numpy 를 내렸고, 이미 깔려 있던 scipy 가 그 numpy 에서 죽었다.
    설치는 '성공' 으로 끝났고 아무도 알려주지 않았다.
    """
    from scripts import setup

    src = (ROOT / "scripts" / "setup.py").read_text(encoding="utf-8")
    assert "numpy" in setup.CORE_LOCK and "scipy" in setup.CORE_LOCK
    assert "scikit-learn" in setup.CORE_LOCK, "sklearn 이 빠지면 반쪽입니다"
    # 부가 설치 명령에 제약 파일이 실제로 붙는가
    i = src.index("for pkg in EXTRAS:")
    block = src[i:i + 600]
    assert '"-c"' in src[:i] or "-c" in block, "제약 파일을 안 씁니다"
    assert "*limit" in block, "부가 설치에 제약이 적용되지 않습니다"


def test_setup_refuses_to_pin_an_already_broken_stack():
    """깨진 상태에서 버전을 고정하면 **깨진 조합을 못으로 박는다.**

    사용자의 .venv 에는 이미 numpy 1.26 이 들어 있었다. 그 상태로 고정했다면
    다시 설치해도 영원히 1.26 이었을 것이다.

    소스에 그 문자열이 있는지가 아니라 **main() 안에서 호출 순서**를 본다 —
    문자열 검사로는 블록을 통째로 지워도 다른 곳의 같은 이름에 걸려 통과한다.
    """
    import ast

    tree = ast.parse((ROOT / "scripts" / "setup.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    order = [n.func.id for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_stack_is_sane" in order, "main() 이 묶음을 한 번도 확인하지 않습니다"
    assert "_write_constraints" in order, "main() 이 버전을 고정하지 않습니다"
    # ast.walk 는 순서를 보장하지 않으므로 줄 번호로 본다
    sane_at = min(n.lineno for n in ast.walk(main)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "_stack_is_sane")
    pin_at = min(n.lineno for n in ast.walk(main)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_write_constraints")
    assert sane_at < pin_at, ("버전을 고정하기 전에 묶음을 확인하지 않습니다 — "
                              "깨진 조합을 그대로 못박게 됩니다")


def test_setup_has_a_repair_path():
    """복구 명령이 실제로 존재해야 한다 — 안내문만 있고 기능이 없으면 안 된다."""
    from scripts import setup

    assert hasattr(setup, "_repair")
    src = (ROOT / "scripts" / "setup.py").read_text(encoding="utf-8")
    assert '"--repair" in sys.argv' in src, "--repair 를 받지 않습니다"


def test_tests_do_not_assume_posix_tools_exist():
    """윈도우에 없는 명령을 그냥 부르면 **제품이 아니라 테스트가 실패한다.**

    실제로 `bash -n` 이 회사 PC 에서 FileNotFoundError 로 실패했다.
    검사할 도구가 없는 것은 건너뛸 일이지 결함이 아니다.
    """
    posix_only = ("bash", "sh", "grep", "sed", "awk", "which", "ls", "chmod")
    bad = []
    for f in sorted((ROOT / "tests").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r'subprocess\.(?:run|call|Popen)\(\s*\[\s*"(\w+)"', src):
            tool = m.group(1)
            if tool in posix_only and "which(" not in src:
                bad.append(f"{f.name}: {tool} 이 있는지 확인하지 않고 부릅니다")
    assert not bad, "\n".join(bad)


def test_core_sticks_to_sqlalchemy_apis_that_work_on_1_4_and_2_0():
    """SQream 드라이버가 SQLAlchemy 를 1.4 로 끌어내린다.

    회사 PC 에서 실제로 그렇게 깔렸다(1.4.46). 드라이버를 포기할 수는 없으므로
    **우리가 1.4 에서도 도는 API 만 쓰는 쪽**을 택했다. 그 약속을 여기서 지킨다.
    2.0 전용 API 를 쓰기 시작하면 SQream 을 쓰는 PC 에서만 조용히 깨진다.
    """
    src = (ROOT / "core" / "datasource.py").read_text(encoding="utf-8")
    only_2x = {
        "sa.Connection": "2.0 전용 타입",
        "create_pool_from_url": "2.0 에서 추가",
        "sa.orm.DeclarativeBase": "2.0 에서 추가",
        "insertmanyvalues": "2.0 에서 추가",
    }
    bad = [f"{k} — {why}" for k, why in only_2x.items() if k in src]
    assert not bad, "1.4 에서 안 도는 API 를 씁니다:\n  " + "\n  ".join(bad)

    core = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    assert "SQLAlchemy>=1.4" in core, ("선언과 실제가 어긋납니다 — SQream 을 깔면 "
                                       "1.4 가 되는데 2.0 이상을 요구하고 있습니다")


def test_no_product_code_uses_so_reuseaddr():
    """`SO_REUSEADDR` 은 **OS 마다 뜻이 다르다.** 제품 코드에서 쓰지 않는다.

    리눅스 : "TIME_WAIT 로 남은 주소를 재사용하겠다"
    윈도우 : "**다른 소켓이 이미 잡고 있어도 같이 잡겠다**" 에 가깝다

    그래서 `free_port()` 가 streamlit 이 이미 듣고 있는 8501 에 bind 성공해
    "비었다" 고 돌려줬다. 두 번째 창을 띄우면 서버가 못 올라온다.
    회사 PC 회귀 테스트에서 이 한 건이 잡혔다.

    **이 규칙을 동작 테스트로는 못 지킨다** — 리눅스에서는 틀린 코드도
    올바르게 동작해서, 옵션을 되돌려도 테스트가 통과한다(실제로 확인했다).
    재현할 수 없는 플랫폼 차이는 **규칙으로 박아 두는 것**이 유일한 방법이다.
    """
    import ast

    # 주석·docstring 에서 "쓰지 말라" 고 설명하는 것은 당연히 괜찮다.
    # **실제로 그 상수를 참조하는 코드**만 잡아야 하므로 AST 로 본다.
    bad = []
    for folder in ("scripts", "app", "core"):
        for f in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "SO_REUSEADDR":
                    bad.append(f"{folder}/{f.name}:{node.lineno}")
    assert not bad, ("SO_REUSEADDR 은 윈도우에서 뜻이 반대입니다 — "
                     "포트 탐색에 쓰면 안 됩니다:\n  " + "\n  ".join(bad))


def test_free_port_checks_for_a_listener_not_just_bind():
    """bind 성공만으로는 '빈 포트' 를 판정할 수 없다 (위 이유).

    **듣고 있는 사람이 없는지**를 먼저 봐야 두 OS 에서 같은 답이 나온다.
    """
    import ast

    tree = ast.parse((ROOT / "scripts" / "launch.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "free_port")
    calls = [c.func.id for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert "_alive" in calls, ("free_port 가 bind 성공만 보고 판정합니다 — "
                               "윈도우에서 쓰이는 포트를 빈 것으로 봅니다")
