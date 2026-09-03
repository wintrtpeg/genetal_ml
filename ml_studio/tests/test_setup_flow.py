"""설치 흐름을 처음부터 끝까지 흉내 내 본다 — 네트워크 없이.

**왜 필요한가 — 실제 사고 두 건이 전부 여기서 났다**

  1. 부가 패키지가 numpy 를 끌어내려 scipy·sklearn 이 죽었다.
     설치는 "성공" 으로 끝났고, 사용자는 테스트 24건 실패로 알게 됐다.
  2. 그 전에는 배치파일이 아예 안 켜졌다.

두 건 모두 **설치 흐름에 테스트가 하나도 없었다**는 공통점이 있다. 파이썬
로직은 300건이 지키는데, 정작 사용자가 처음 겪는 3~10분짜리 과정은 아무도
안 봤다. 실제 pip 을 부를 수는 없으니(네트워크·시간), **pip 에 무엇을 어떤
순서로 시키는지**를 본다. 거기가 잘못돼서 두 번 다 터졌다.

여기서 확인하는 것
  · 핵심 → 고정 → 부가 순서가 지켜지는가
  · **부가 설치 명령마다 제약 파일이 붙는가** (1번 사고의 직접 원인)
  · 깨진 상태에서 버전을 고정하지 않는가 (그러면 깨진 조합을 못박는다)
  · 묶음이 깨졌는데도 다음 단계로 넘어가지 않는가
  · 도장은 전부 끝난 뒤에만 찍히는가
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Recorder:
    """pip 에 내린 명령을 그대로 모은다."""

    def __init__(self, sane_after: int = 0, fail_extras: set[str] | None = None,
                 broken_calls: set[int] | None = None):
        self.cmds: list[list[str]] = []
        self.sane_calls = 0
        self.sane_after = sane_after      # 이 횟수 이후부터 '정상' 이라고 답한다
        # 특정 확인 시점만 깨진 것으로 답한다. 실제 사고가 이 모양이었다 —
        # 핵심 설치 직후에는 멀쩡했는데 **부가 설치가 numpy 를 내려서** 깨졌다.
        self.broken_calls = broken_calls or set()
        self.fail_extras = fail_extras or set()
        self.launched = False

    def run(self, cmd, *, must_succeed=True, quiet=False):
        self.cmds.append(list(cmd))
        for name in self.fail_extras:
            if any(name in str(c) for c in cmd):
                return 1
        return 0

    def pip_lines(self) -> list[str]:
        return [" ".join(c) for c in self.cmds if "-m" in c and "pip" in c]


def _harness(monkeypatch, rec: Recorder, *, venv_exists=True, stamp_ok=False):
    """setup 모듈의 바깥세상(pip·파이썬·파일)을 전부 대역으로 바꾼다."""
    from scripts import setup

    fake_py = ROOT / ".venv" / "bin" / "python"

    monkeypatch.setattr(setup, "run", rec.run)
    monkeypatch.setattr(setup, "venv_python", lambda: fake_py)
    monkeypatch.setattr(setup, "_version_of", lambda e: "3.12.3")
    monkeypatch.setattr(setup, "pick_interpreter", lambda w: ("py3.12", "테스트"))
    monkeypatch.setattr(Path, "exists", lambda self: venv_exists
                        if self == fake_py else True)

    def sane(vpy):
        rec.sane_calls += 1
        if rec.sane_calls in rec.broken_calls or rec.sane_calls <= rec.sane_after:
            return False, "module 'numpy' has no attribute 'long'"
        return True, "2.1.0 1.14.0 1.5.0"
    monkeypatch.setattr(setup, "_stack_is_sane", sane)
    monkeypatch.setattr(setup, "_freeze", lambda v: {
        "numpy": "2.1.0", "scipy": "1.14.0",
        "scikit-learn": "1.5.0", "pandas": "2.2.0"})

    written = {}
    monkeypatch.setattr(setup, "CONSTRAINTS",
                        types.SimpleNamespace(
                            write_text=lambda t, **k: written.__setitem__("c", t),
                            __str__=lambda s: "CONSTRAINTS.txt"))
    monkeypatch.setattr(setup, "STAMP",
                        types.SimpleNamespace(
                            exists=lambda: stamp_ok,
                            read_text=lambda **k: "",
                            write_text=lambda t, **k: written.__setitem__("s", t)))

    launch_mod = types.ModuleType("scripts.launch")

    def launch(vpy=None, port=None, open_browser=True):
        rec.launched = True
        return 0
    launch_mod.launch = launch
    monkeypatch.setitem(sys.modules, "scripts.launch", launch_mod)
    return setup, written


def _run(monkeypatch, rec, argv=("setup.py", "--full"), **kw):
    setup, written = _harness(monkeypatch, rec, **kw)
    monkeypatch.setattr(sys, "argv", list(argv))
    try:
        code = setup.main()
    except SystemExit as e:
        code = int(e.code) if e.code is not None else 0
    return code, written


# ── 정상 설치 ────────────────────────────────────────────────
def test_happy_path_installs_core_then_pins_then_extras(monkeypatch):
    """순서가 전부다. 고정을 부가 설치 뒤에 하면 아무 의미가 없다."""
    rec = Recorder()
    code, written = _run(monkeypatch, rec)

    assert code == 0, "정상 경로인데 실패했습니다"
    assert rec.launched, "설치가 끝났는데 화면을 안 띄웠습니다"

    lines = rec.pip_lines()
    core = next(i for i, l in enumerate(lines) if "requirements-core.txt" in l)
    extras = [i for i, l in enumerate(lines) if "xgboost" in l or "shap" in l]
    assert extras, "부가 패키지를 설치하지 않았습니다"
    assert core < min(extras), "핵심보다 부가를 먼저 깔고 있습니다"
    assert "c" in written, "기준 버전을 고정하지 않았습니다"
    assert "numpy==2.1.0" in written["c"], written.get("c")
    assert "scipy==1.14.0" in written["c"], written.get("c")


def test_every_extra_install_carries_the_constraint_file(monkeypatch):
    """**1번 사고의 직접 원인.** 제약 없이 부가를 깔면 numpy 가 내려간다."""
    rec = Recorder()
    _run(monkeypatch, rec)

    bare = []
    for cmd in rec.cmds:
        joined = " ".join(str(c) for c in cmd)
        if "install" not in joined:
            continue
        if any(p in joined for p in ("xgboost", "lightgbm", "catboost",
                                     "shap", "pysqream")):
            if "-c" not in cmd:
                bare.append(joined)
    assert not bare, ("제약 없이 설치하는 부가 패키지가 있습니다 — "
                      "여기서 numpy 가 내려갑니다:\n  " + "\n  ".join(bare))


def test_stamp_is_written_only_after_everything_passes(monkeypatch):
    """중간에 끊긴 설치에 도장을 찍으면, 다음 실행이 빠른 경로로 새 버린다."""
    rec = Recorder()
    _, written = _run(monkeypatch, rec)
    assert "s" in written, "정상 설치인데 도장을 안 찍었습니다"

    rec2 = Recorder(sane_after=99)          # 끝까지 깨져 있는 환경
    code, written2 = _run(monkeypatch, rec2)
    assert code != 0, "묶음이 깨졌는데 성공으로 끝냈습니다"
    assert "s" not in written2, "깨진 상태에 도장을 찍었습니다"


# ── 깨진 환경 ────────────────────────────────────────────────
def test_a_broken_venv_is_repaired_before_pinning(monkeypatch):
    """사용자의 .venv 에는 이미 numpy 1.26 이 있었다.

    그 상태로 고정했다면 **깨진 조합을 영원히 못박았을** 것이다.
    """
    rec = Recorder(sane_after=1)            # 첫 확인만 실패 → 되돌린 뒤 성공
    code, written = _run(monkeypatch, rec)

    assert code == 0, "되돌릴 수 있는 상황인데 죽었습니다"
    lines = rec.pip_lines()
    fixes = [i for i, l in enumerate(lines)
             if "--upgrade" in l and "requirements-core.txt" in l]
    assert fixes, "깨진 것을 확인하고도 되돌리지 않았습니다"
    assert "numpy==2.1.0" in written.get("c", ""), \
        "되돌린 뒤의 버전이 아니라 깨진 버전을 고정했습니다"


def test_it_stops_instead_of_continuing_on_a_broken_stack(monkeypatch):
    """깨진 채로 테스트·화면 단계로 가면 원인 모를 실패가 쏟아진다."""
    rec = Recorder(sane_after=99)
    code, _ = _run(monkeypatch, rec)

    assert code != 0
    assert not rec.launched, "묶음이 깨졌는데 화면을 띄웠습니다"
    lines = " ".join(rec.pip_lines())
    assert "run_tests.py" not in " ".join(" ".join(c) for c in rec.cmds), \
        "깨진 상태로 회귀 테스트를 돌렸습니다"


def test_it_catches_a_stack_broken_by_the_extras(monkeypatch):
    """**실제 사고의 순서 그대로.**

    핵심 설치 직후에는 멀쩡했다. 그 다음 부가 패키지를 깔면서 numpy 가 내려갔고,
    그때부터 scipy·sklearn 이 죽었다. 그런데 설치는 "성공" 으로 끝났고,
    사용자는 회귀 테스트 24건 실패로 알게 됐다.

    첫 관문(핵심 직후)만으로는 이걸 못 잡는다 — 그때는 멀쩡했으니까.
    **부가 설치 뒤에도 한 번 더 봐야 한다.**
    """
    rec = Recorder(broken_calls={2})       # 1회차 정상 → 부가 설치 → 2회차 깨짐
    code, written = _run(monkeypatch, rec)

    assert code != 0, ("부가 설치가 묶음을 깨뜨렸는데 성공으로 끝냈습니다 — "
                       "이게 실제로 일어난 일입니다")
    assert not rec.launched, "깨진 채로 화면을 띄웠습니다"
    assert "s" not in written, "깨진 상태에 도장을 찍었습니다"
    assert rec.sane_calls >= 2, ("부가 설치 뒤에 묶음을 다시 확인하지 않습니다 — "
                                 "첫 확인만으로는 이 사고를 못 잡습니다")


def test_a_failing_extra_does_not_stop_the_install(monkeypatch):
    """부스팅 하나가 안 깔려도 나머지는 살아야 한다 — 원래 의도."""
    rec = Recorder(fail_extras={"catboost"})
    code, _ = _run(monkeypatch, rec)

    assert code == 0, "부가 패키지 하나 때문에 전체가 멈췄습니다"
    assert rec.launched
    got = " ".join(rec.pip_lines())
    assert "shap" in got, "catboost 실패 뒤 나머지를 건너뛰었습니다"


# ── 복구 명령 ────────────────────────────────────────────────
def test_repair_reresolves_core_and_rechecks(monkeypatch):
    rec = Recorder()
    code, written = _run(monkeypatch, rec, argv=("setup.py", "--repair"))

    assert code == 0
    lines = rec.pip_lines()
    assert any("--upgrade" in l and "requirements-core.txt" in l for l in lines), \
        "핵심을 다시 해석시키지 않습니다"
    assert any("check" in l for l in lines), "pip check 를 돌리지 않습니다"
    assert not rec.launched, "복구는 화면을 띄우지 않아야 합니다"


def test_repair_reports_failure_when_it_cannot_fix(monkeypatch):
    """못 고쳤으면 고쳤다고 하면 안 된다."""
    rec = Recorder(sane_after=99)
    code, _ = _run(monkeypatch, rec, argv=("setup.py", "--repair"))
    assert code != 0, "복구에 실패했는데 성공으로 보고했습니다"


def test_repair_without_a_venv_says_so(monkeypatch):
    rec = Recorder()
    code, _ = _run(monkeypatch, rec, argv=("setup.py", "--repair"),
                   venv_exists=False)
    assert code != 0
