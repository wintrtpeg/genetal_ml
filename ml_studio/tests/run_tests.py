"""테스트를 돌린다. pytest 가 없어도 돈다.

    python tests/run_tests.py            내장 러너 (기본)
    python tests/run_tests.py --pytest   설치된 pytest 로 위임
    pytest tests/ -v                     pytest 를 직접 써도 된다

기본은 **항상 내장 대역**을 쓴다. fixture / raises / approx / parametrize /
monkeypatch 를 흉내 낸 최소 구현이라 pytest 설치 여부와 무관하게 결과가 같다.
사내 PC 에 패키지 설치가 막혀 있어도 누수 검증만은 돌아가야 해서 둔 장치이고,
어느 환경에서든 같은 경로로 도는 편이 신뢰할 만하다.

주의 — 예전에는 "pytest 가 있으면 그걸 쓴다" 였는데, 진짜 pytest 의
@pytest.fixture 는 함수에 __is_fixture__ 를 붙이지 않아서 이 파일의 fixture
탐지가 통째로 실패했다. fixture 를 쓰는 테스트 전부가 TypeError 로 죽었고,
정작 pytest 가 깔린 환경(= 실제 사용 환경)에서만 그랬다. 그래서 기본 경로를
하나로 고정하고, _fixtures() 는 진짜 pytest 의 표식도 함께 알아보게 했다.
"""

from __future__ import annotations

import importlib
import sys
import traceback
import types
from pathlib import Path


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


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ["tests.test_leakage", "tests.test_shap_period", "tests.test_split_leakage",
           "tests.test_p1_p2", "tests.test_p3", "tests.test_review_gate",
           "tests.test_auto_pipeline", "tests.test_ui_perf",
           "tests.test_runtime_guards", "tests.test_view_render",
           "tests.test_advisor", "tests.test_entrypoints", "tests.test_housekeeping",
           "tests.test_setup_flow"]


# ── pytest 최소 대역 ─────────────────────────────────────────────────────
def _install_stub() -> None:
    m = types.ModuleType("pytest")

    def fixture(fn=None, **_kw):
        def wrap(f):
            f.__is_fixture__ = True
            return f
        return wrap(fn) if fn is not None else wrap

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc = exc
            self.match = match
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"{self.exc.__name__} 이 발생하지 않았습니다.")
            if not issubclass(et, self.exc):
                return False
            self.value = ev
            if self.match is not None:
                import re
                if not re.search(self.match, str(ev)):
                    raise AssertionError(
                        f"예외 메시지가 '{self.match}' 와 맞지 않습니다: {ev}")
            return True

    class _Approx:
        def __init__(self, val, abs=None, rel=None):  # noqa: A002
            self.val, self.abs, self.rel = val, abs, rel

        def __eq__(self, other):
            tol = self.abs if self.abs is not None else max(
                abs(self.val) * (self.rel if self.rel is not None else 1e-6), 1e-12)
            return abs(other - self.val) <= tol

        def __repr__(self):
            return f"approx({self.val})"

    def parametrize(argnames, argvalues):
        names = [a.strip() for a in argnames.split(",")]

        def wrap(fn):
            fn.__params__ = (names, list(argvalues))
            return fn
        return wrap

    def skip(reason: str = "", **_kw):
        # 진짜 pytest 의 skip 은 예외를 던져 그 자리에서 테스트를 끝낸다.
        # 예전 대역은 None 을 돌려줘서 skip 뒤의 코드가 그대로 실행됐다 —
        # 없는 라이브러리를 건드리는 테스트가 조용히 통과하거나 엉뚱하게 죽었다.
        raise Skipped(reason)

    def importorskip(name: str, *_a, **_kw):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            raise Skipped(f"{name} 이(가) 설치돼 있지 않습니다") from None

    m.fixture = fixture
    m.raises = _Raises
    m.approx = _Approx
    m.mark = types.SimpleNamespace(parametrize=parametrize)
    m.skip = skip
    m.importorskip = importorskip
    sys.modules["pytest"] = m


class Skipped(Exception):
    """내장 대역의 skip 신호. 진짜 pytest 의 Skipped 와 같은 자리를 맡는다."""


class MonkeyPatch:
    """pytest 의 monkeypatch 최소 대역. setattr 만 지원하고 끝나면 되돌린다."""

    def __init__(self):
        self._undo: list[tuple] = []

    def setattr(self, target, name, value):  # noqa: A003
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def setitem(self, dic, key, value):
        sentinel = object()
        self._undo.append((dic, key, dic.get(key, sentinel)))
        dic[key] = value

    def undo(self) -> None:
        for target, name, old in reversed(self._undo):
            if isinstance(target, dict):
                if old is None or type(old).__name__ == "object":
                    target.pop(name, None)
                else:
                    target[name] = old
            else:
                setattr(target, name, old)
        self._undo.clear()


# ── 실행 ────────────────────────────────────────────────────────────────
def _fixtures(mod) -> dict:
    """fixture 로 선언된 함수를 모은다.

    표식이 세 가지다. 내장 대역은 __is_fixture__ 를 붙이고, pytest 8.4 미만은
    함수에 _pytestfixturefunction 을 붙이며, 8.4 이상은 원본 함수를 감싼
    객체(_fixture_function)를 돌려준다. 셋 다 알아봐야 어느 환경에서든 돈다.
    """
    out: dict = {}
    for name, obj in vars(mod).items():
        if getattr(obj, "__is_fixture__", False):
            out[name] = obj
        elif hasattr(obj, "_pytestfixturefunction"):          # pytest < 8.4
            out[name] = obj
        elif hasattr(obj, "_fixture_function"):               # pytest >= 8.4
            out[name] = obj._fixture_function
    return out


def _cases(mod):
    fx = _fixtures(mod)
    for name, fn in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        params = getattr(fn, "__params__", None)
        if params:
            names, values = params
            for v in values:
                v = v if isinstance(v, tuple) else (v,)
                label = f"{name}[{', '.join(str(x)[:34] for x in v)}]"
                yield label, fn, dict(zip(names, v)), fx
        else:
            yield name, fn, {}, fx



def _preflight() -> str:
    """핵심 묶음이 실제로 import 되는지 먼저 본다.

    numpy 와 scipy 버전이 어긋나면 scipy 가 `np.long` 을 찾다 죽고, sklearn 이
    scipy 를 import 하므로 테스트 모듈 대부분이 로드조차 안 된다. 실제로 회사
    PC 에서 **24건이 전부 같은 이유**로 실패했다 — 화면에 뜬 건 원인 하나가
    아니라 증상 24개였다. 증상을 24번 찍는 대신 원인을 한 번 찍는다.
    """
    try:
        from scripts import envcheck
    except Exception:                                     # noqa: BLE001
        return ""                                          # 검사기가 없으면 그냥 진행
    return envcheck.probe()


def _report_broken_stack(why: str) -> int:
    from scripts import envcheck

    print("=" * 66)
    print("  " + envcheck.message(why).splitlines()[0])
    print("=" * 66)
    print("\n".join(envcheck.message(why).splitlines()[1:]))
    print("\n  테스트는 여기서 멈춥니다 — 이 상태로 돌려 봐야 전부 같은 이유로 "
          "실패합니다.")
    return 1


def main() -> int:
    _enable_utf8()
    # --pytest 를 준 경우에만 진짜 pytest 에 위임한다.
    if "--pytest" in sys.argv:
        try:
            import pytest
        except ModuleNotFoundError:
            print("pytest 가 설치돼 있지 않습니다. 내장 러너로 진행합니다.\n")
        else:
            return pytest.main(["-q", str(ROOT / "tests")])

    # 무엇보다 먼저 — 묶음이 깨져 있으면 증상 24개 대신 원인 하나를 찍는다.
    if (why := _preflight()):
        return _report_broken_stack(why)

    # 기본 경로 — 항상 내장 대역. pytest 설치 여부와 무관하게 같은 결과가 나온다.
    _install_stub()
    print("내장 러너로 실행합니다 (pytest 로 돌리려면 --pytest)\n")

    passed = failed = skipped = 0
    skip_notes: list[str] = []
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            print(f"{modname}  로드 실패 — {type(e).__name__}: {e}")
            failed += 1
            continue

        print(f"── {modname} " + "─" * max(0, 56 - len(modname)))
        for label, fn, kwargs, fx in _cases(mod):
            mp = None
            try:
                code = fn.__code__
                for arg in code.co_varnames[:code.co_argcount]:
                    if arg in kwargs:
                        continue
                    if arg == "monkeypatch":
                        mp = MonkeyPatch()
                        kwargs[arg] = mp
                    elif arg in fx:
                        kwargs[arg] = fx[arg]()
                fn(**kwargs)
                print(f"  PASS  {label}")
                passed += 1
            except Skipped as e:
                print(f"  SKIP  {label}  ({e})")
                skip_notes.append(f"{label} — {e}")
                skipped += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL  {label}\n        {type(e).__name__}: {e}")
                if "-v" in sys.argv:
                    traceback.print_exc()
                failed += 1
            finally:
                if mp is not None:
                    mp.undo()
        print()

    line = f"{passed} 통과, {failed} 실패"
    if skipped:
        line += f", {skipped} 건너뜀"
    print(line)
    if skip_notes:
        # 건너뛴 것을 조용히 넘기면 "여기서는 안 돌아본 코드" 가 생긴다.
        # 패키지가 다 깔린 환경에서는 이 목록이 비어야 정상이다.
        print("\n건너뛴 항목 — 이 환경에 없는 패키지 때문입니다:")
        for n in skip_notes:
            print(f"  · {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
