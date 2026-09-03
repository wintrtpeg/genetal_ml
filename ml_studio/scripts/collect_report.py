"""진단 리포트를 파일 하나로 모은다.

    python scripts/collect_report.py       (또는 report.bat 더블클릭)

무엇이 어디서 왜 실패했는지를 한 파일에 담는다. 콘솔에서 오류 줄을 찾아
옮겨 적을 필요 없이 **그 파일 하나만 첨부하면 된다.**

설계 원칙 — **무슨 일이 있어도 파일이 남아야 한다**
--------------------------------------------------
첫 판은 전부 모은 뒤 마지막에 한 번 썼다. 그래서 중간에 죽으면 파일이 아예
안 생겼고, cmd 창은 닫혀 버려서 사용자에게 남는 정보가 하나도 없었다.
실제로 그렇게 당했다.

지금은 두 가지로 막는다.

1. **첫 줄부터 파일에 바로 쓴다.** 단계가 끝날 때마다 flush 한다.
   중간에 죽어도 거기까지가 파일에 남는다.
2. **각 단계를 하위 프로세스로 돌린다.** 파이썬이 통째로 죽는 종류의 사고
   (C 확장 크래시, os._exit, 메모리 부족)가 나도 이쪽 프로세스는 살아서
   "여기서 죽었다" 를 기록한다. 같은 프로세스에서 exec 하면 같이 죽는다.
   인코딩은 PYTHONIOENCODING 으로 자식에게 넘긴다.

담는 것
  0. 환경  — OS · 파이썬 · 패키지 버전 · pip check
  1. 회귀 테스트 — tests/run_tests.py 전문
  2. 실행 환경 점검 — scripts/verify_env.py 전문
  3. 축소 end-to-end — scripts/quick_check.py (--quick 으로 끔)

담지 않는 것 — 데이터 값, 접속 정보, 비밀번호.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _enable_utf8() -> None:
    """윈도우 콘솔에서 한글·기호가 깨지거나 죽지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


PACKAGES = ["streamlit", "pandas", "numpy", "sklearn", "scipy", "plotly",
            "joblib", "yaml", "shap", "xgboost", "lightgbm", "catboost",
            "sqlalchemy", "pysqream", "pytest"]

# 한 단계가 이보다 오래 걸리면 끊고 다음으로 간다. 하나가 매달려서 리포트
# 전체를 못 만드는 것이 제일 나쁘다.
STEP_TIMEOUT = 1800


class Writer:
    """파일에 바로바로 쓴다. 중간에 죽어도 거기까지는 남는다."""

    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self.f.write(text + "\n")
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except OSError:
            pass

    def section(self, title: str) -> None:
        self.write(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:  # noqa: BLE001
            pass


def _environment(w: Writer) -> None:
    w.section("0. 환경")
    w.write(f"OS          : {platform.platform()}")
    w.write(f"파이썬       : {sys.version.split()[0]}")
    w.write(f"              {sys.executable}")
    w.write(f"인코딩       : stdout={sys.stdout.encoding} "
            f"filesystem={sys.getfilesystemencoding()}")
    try:
        import locale
        w.write(f"로케일       : {locale.getpreferredencoding(False)}")
    except Exception:  # noqa: BLE001
        pass
    venv = ROOT / ".venv"
    w.write(f"가상환경     : {'있음' if venv.exists() else '없음'}  ({venv})")

    w.write("")
    w.write("설치된 패키지")
    for m in PACKAGES:
        try:
            mod = __import__(m)
            w.write(f"  {m:<12} {getattr(mod, '__version__', '(버전 표기 없음)')}")
        except Exception as e:  # noqa: BLE001
            # "없음" 과 "있는데 깨졌음" 은 완전히 다른 문제다. 설치가 중간에
            # 끊기면 임포트가 ImportError 가 아닌 예외로 죽는 일이 있다.
            kind = "(없음)" if isinstance(e, ImportError) else f"({type(e).__name__}: {e})"
            w.write(f"  {m:<12} —  {kind}")

    w.write("")
    w.write("pip check (의존성 정합성)")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "check"],
                           capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace")
        body = (r.stdout + r.stderr).strip() or "(출력 없음)"
        for line in body.splitlines()[:25]:
            w.write(f"  {line}")
        if r.returncode:
            w.write("  → 의존성이 어긋나 있습니다. 설치가 중간에 끊겼을 수 있습니다.")
    except Exception as e:  # noqa: BLE001
        w.write(f"  (확인 실패: {type(e).__name__}: {e})")

    w.write("")
    w.write("pip list (상위 40개)")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "list"],
                           capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace")
        for line in (r.stdout or "").splitlines()[:40]:
            w.write(f"  {line}")
    except Exception as e:  # noqa: BLE001
        w.write(f"  (확인 실패: {type(e).__name__}: {e})")


def _run_step(w: Writer, num: str, label: str, rel: str,
              argv: list[str] | None = None) -> bool:
    """하위 프로세스로 돌리고 출력을 그대로 파일에 넣는다. 성공하면 True.

    하위 프로세스로 돌리는 이유 — 파이썬이 통째로 죽는 사고가 나도 이쪽은
    살아서 "여기서 죽었다" 를 기록할 수 있다.
    """
    path = ROOT / rel
    w.section(f"{num}. {label}  ({rel})")
    print(f"[{num}] {label} …", flush=True)

    if not path.exists():
        w.write(f"(파일이 없습니다: {rel})")
        print("      파일 없음")
        return True

    env = {**os.environ, "PYTHONIOENCODING": "utf-8:replace", "PYTHONUTF8": "1"}
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(path), *(argv or [])],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=STEP_TIMEOUT)
        body = (r.stdout or "") + (r.stderr or "")
        code = r.returncode
    except subprocess.TimeoutExpired as e:
        body = ((e.stdout or "") if isinstance(e.stdout, str) else "") + \
               f"\n[{STEP_TIMEOUT}초를 넘겨 중단했습니다 — 어딘가에서 매달렸습니다]"
        code = -9
    except Exception as e:  # noqa: BLE001
        import traceback
        body = f"[이 단계를 실행하지 못했습니다]\n{traceback.format_exc()}"
        code = -1

    w.write(body.rstrip())
    took = time.time() - t0
    w.write(f"\n[{label} 종료코드 {code} · {took:.1f}초]")
    print(f"      종료코드 {code}  ({took:.0f}초)")
    return code == 0


def _stack_problem() -> str:
    """핵심 묶음이 깨졌는지. 깨졌으면 사용자가 그대로 읽을 안내문을 돌려준다."""
    try:
        from scripts import envcheck
    except Exception:  # noqa: BLE001
        return ""
    why = envcheck.probe()
    return envcheck.message(why) if why else ""


def main() -> int:
    _enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="축소 end-to-end 점검을 건너뜁니다 (빨라집니다)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "diagnostic_report.txt"
    w = Writer(out)

    print("=" * 70)
    print("  진단 리포트를 만듭니다. 3~15분 걸립니다.")
    print(f"  결과 파일 : {out}")
    print("  중간에 죽어도 거기까지는 파일에 남습니다.")
    print("=" * 70)
    print()

    failures: list[str] = []
    broken = ""
    try:
        w.write("시계열 ML 스튜디오 — 진단 리포트")
        w.write(f"생성 시각 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w.write(f"폴더      : {ROOT}")
        w.write("")
        w.write("※ 이 줄 아래가 끝까지 안 채워져 있으면, 마지막으로 적힌 단계에서")
        w.write("   프로그램이 죽은 것입니다. 그 상태 그대로 보내주시면 됩니다.")

        _environment(w)

        # **묶음이 깨졌으면 그 사실을 리포트 맨 앞에 박는다.**
        # 예전 리포트는 요약에 "실패 3건 — 회귀 테스트, 환경 점검, end-to-end"
        # 라고만 적혔다. 셋 다 같은 원인(numpy·scipy 조합)의 증상인데 원인은
        # 600줄 아래 traceback 안에 있었다. 원인을 위로 끌어올린다.
        broken = _stack_problem()
        if broken:
            w.section("!! 먼저 볼 것 — 패키지 조합")
            w.write(broken)
            w.write("")
            w.write("아래 절들의 실패는 대부분 이 하나에서 나온 증상입니다.")
            print("[!] 패키지 조합이 깨져 있습니다 — 리포트 맨 앞에 적었습니다.")

        if not _run_step(w, "1", "회귀 테스트", "tests/run_tests.py"):
            failures.append("회귀 테스트")
        if not _run_step(w, "2", "실행 환경 점검", "scripts/verify_env.py"):
            failures.append("실행 환경 점검")
        if args.quick:
            w.section("3. 축소 end-to-end")
            w.write("(--quick 으로 건너뜀)")
            print("[3] 축소 end-to-end — 건너뜀")
        elif not _run_step(w, "3", "축소 end-to-end", "scripts/quick_check.py"):
            failures.append("축소 end-to-end")

        w.section("요약")
        if broken:
            w.write("원인 — 설치된 패키지 조합이 깨져 있습니다 (코드 문제가 아닙니다).")
            w.write("맨 위 '먼저 볼 것' 절에 복구 명령이 있습니다.")
            if failures:
                w.write(f"이 때문에 {len(failures)}건이 실패했습니다 "
                        f"— {', '.join(failures)}")
        elif failures:
            w.write(f"실패 {len(failures)}건 — {', '.join(failures)}")
            w.write("각 절에서 '실패' · 'FAIL' · 'Traceback' 을 찾아보세요.")
        else:
            w.write("전 항목 통과")
    except BaseException as e:  # noqa: BLE001
        # Ctrl+C 나 그보다 심한 것도 여기서 받아 파일에 남긴다.
        import traceback
        try:
            w.write("\n" + "=" * 70)
            w.write("[리포트 수집기 자체가 여기서 멈췄습니다]")
            w.write(f"{type(e).__name__}: {e}")
            w.write(traceback.format_exc())
        except Exception:  # noqa: BLE001
            pass
        failures.append("리포트 수집기")
    finally:
        w.close()

    # 리포트를 만드느라 남긴 찌꺼기를 치운다. 실행 결과(runs)는 손대지 않는다 —
    # 진단하러 왔다가 사용자 산출물을 지우면 안 된다.
    try:
        from core import housekeeping as hk
        res = hk.apply(hk.junk_plan())
        if res.removed:
            print(f"  정리 — {res.summary()}")
    except Exception:  # noqa: BLE001
        pass

    size = out.stat().st_size if out.exists() else 0
    print()
    print("=" * 70)
    if failures:
        print(f"  실패 {len(failures)}건 — {', '.join(failures)}")
    else:
        print("  전 항목 통과")
    print(f"  리포트 : {out}")
    print(f"           ({size / 1024:,.0f} KB)")
    print()
    print("  이 파일 하나만 대화창에 첨부해 주세요.")
    print("  (데이터 값·접속정보는 들어가지 않습니다)")
    print("=" * 70)
    return 0            # 리포트를 만드는 것 자체는 성공이므로 항상 0


if __name__ == "__main__":
    raise SystemExit(main())
