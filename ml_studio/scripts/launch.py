"""화면을 띄운다. 포트가 실제로 열린 뒤에 브라우저를 연다.

왜 직접 여는가
--------------
streamlit 자체도 브라우저를 열 수 있지만(`server.headless=false`), 두 가지가
걸린다.

  · **서버가 준비되기 전에** 연다. 첫 실행은 임포트에만 수 초가 걸려서
    브라우저가 먼저 뜨고 "연결할 수 없음" 이 보인다. 사용자는 실패로 읽는다.
  · 사내 PC 의 `.streamlit/config.toml` 이나 STREAMLIT_SERVER_HEADLESS 환경변수가
    이미 headless 로 잡혀 있으면 **아무것도 안 열린다.** 그런 PC 에서는
    "URL 을 복사해서 붙여넣으세요" 로 되돌아간다.

그래서 streamlit 은 headless 로 띄우고, 포트가 실제로 응답할 때 우리가 연다.
탭이 두 개 뜨는 일도 없다.

포트도 직접 고른다 — 8501 이 이미 쓰이고 있으면 streamlit 은 다음 포트로
넘어가는데, 그러면 우리가 열어야 할 주소를 모른다.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "main.py"

WAIT_SECONDS = 90          # 첫 실행은 임포트가 길다. 넉넉하게 기다린다.
POLL = 0.4


def _enable_utf8() -> None:
    """윈도우 콘솔에서 한글·기호가 깨지거나 죽지 않게 한다.

    한글 윈도우의 기본 코덱은 cp949 이고 em dash(—)가 없다. 콘솔에 바로 찍을
    때는 문제가 없지만 출력을 파일로 넘기는 순간 UnicodeEncodeError 로 죽는다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _alive(port: int) -> bool:
    """그 포트에서 누가 듣고 있는가."""
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port(start: int = 8501, tries: int = 40) -> int:
    """비어 있는 포트를 찾는다. 이미 띄워 둔 창이 있어도 충돌하지 않는다.

    **SO_REUSEADDR 을 쓰면 안 된다 — 윈도우에서 의미가 반대다.**
    리눅스에서는 "TIME_WAIT 로 남은 포트를 재사용하겠다" 는 뜻이지만,
    윈도우에서는 "**다른 소켓이 이미 잡고 있어도 같이 잡겠다**" 에 가깝다.
    그래서 streamlit 이 8501 에서 돌고 있는데도 bind 가 성공해 버렸고,
    "빈 포트" 라며 8501 을 돌려줬다. 두 번째 창을 띄우면 서버가 못 올라오거나
    엉뚱한 창이 열린다. 회사 PC 회귀 테스트에서 이 한 건이 잡혔다.

    그래서 두 가지를 함께 본다 — **듣고 있는 사람이 없고**(connect 실패),
    **bind 도 되는**(옵션 없이) 포트만 빈 것으로 친다.
    """
    for p in range(start, start + tries):
        if _alive(p):
            continue                       # 누가 이미 서비스 중이다
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))   # SO_REUSEADDR 없이 — 엄격하게
                return p
            except OSError:
                continue
    return start


def _open_when_ready(url: str, port: int, proc: subprocess.Popen) -> None:
    """서버가 응답하면 그때 연다. 죽었으면 열지 않는다."""
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            return                      # 서버가 먼저 끝났다 — 열 이유가 없다
        if _alive(port):
            time.sleep(0.6)             # 첫 렌더까지 한 박자
            try:
                webbrowser.open(url)
            except Exception:           # noqa: BLE001
                pass                    # 못 열어도 URL 은 창에 찍혀 있다
            return
        time.sleep(POLL)


def tidy(when: str = "") -> None:
    """남은 파일을 정리한다. 드라이브를 무한정 먹지 않게.

    **시작할 때와 끝날 때 두 번 부른다.**

      시작 — 지난번에 남긴 것을 치운다. 창을 그냥 X 로 닫으면 종료 정리가
             안 돌 수 있으므로, **이쪽이 확실한 쪽**이다.
      종료 — 이번에 쌓인 것을 바로 치운다. 다음 실행까지 기다리지 않는다.

    빠르고 조용해야 한다 — 화면 띄우는 길을 막으면 안 된다. 그래서 실패해도
    그냥 넘어가고, 뭔가 지웠을 때만 한 줄 찍는다.

    삭제는 **영구 삭제**다 (휴지통으로 가지 않는다). 그래서 정책이 보수적이다 —
    최신 실행은 남기고, 보관 지정한 것은 절대 안 지운다. 자세한 설정과 수동
    정리는 화면의 '설정 > 저장공간' 에 있다.
    """
    if "--no-tidy" in sys.argv:
        return                       # 문제 추적할 때 정리를 꺼 두는 통로
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from core import housekeeping as hk

        res = hk.sweep(hk.RetentionPolicy())
        if res.removed:
            tag = f"[{when}] " if when else ""
            print(f"    {tag}정리 — {res.summary()}")
        if (notice := res.notice()):
            print(f"    {notice}")
    except Exception:  # noqa: BLE001
        pass           # 정리는 부가 기능이다. 실패해도 실행은 계속한다


def launch(python_exe: str | None = None, port: int | None = None,
           open_browser: bool = True) -> int:
    _enable_utf8()
    tidy("시작")
    exe = python_exe or sys.executable
    port = port or free_port()
    url = f"http://localhost:{port}"

    env = dict(os.environ)
    # 우리가 직접 열 것이므로 streamlit 은 열지 않게 한다. 사내 PC 의 기존
    # 설정이 뭐로 잡혀 있든 여기서 덮어써야 동작이 일정해진다.
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env.setdefault("PYTHONUTF8", "1")

    print()
    print(f"    화면 주소: {url}")
    print("    브라우저가 곧 자동으로 열립니다. 안 열리면 위 주소를 직접 여세요.")
    print("    끝낼 때는 이 창에서 Ctrl+C 를 누르세요.")
    print()

    cmd = [exe, "-m", "streamlit", "run", str(APP),
           "--server.port", str(port), "--server.headless", "true"]
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT))
    except FileNotFoundError:
        print("[X] streamlit 을 찾지 못했습니다. run.bat 을 다시 실행해 주세요.")
        return 1

    if open_browser:
        threading.Thread(target=_open_when_ready, args=(url, port, proc),
                         daemon=True).start()
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)          # 파일을 다 놓을 때까지 기다린다
        except BaseException:              # noqa: BLE001
            # **Exception 만 잡으면 안 된다.** Ctrl+C 를 두 번 누르면 여기서
            # KeyboardInterrupt 가 또 올라오고, 그건 Exception 이 아니라
            # BaseException 이라 그대로 빠져나가 콘솔에 트레이스백이 찍힌다.
            # 정리(finally)는 어차피 돌지만, 사용자에게는 '고장난 종료' 로 보인다.
            pass
        return 130
    finally:
        # 이번 실행에서 쌓인 것을 바로 치운다. Ctrl+C 로 끊었을 때도 돈다.
        # 창을 X 로 닫으면 여기가 안 돌 수 있는데, 그건 다음 실행의 '시작'
        # 정리가 받아 준다 — 그래서 양쪽에 다 걸어 둔다.
        tidy("종료")


def main() -> int:
    _enable_utf8()
    exe = None
    port = None
    no_open = "--no-browser" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--python" and i + 1 < len(sys.argv):
            exe = sys.argv[i + 1]
        elif a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    return launch(exe, port, open_browser=not no_open)


if __name__ == "__main__":
    sys.exit(main())
