"""핵심 패키지 묶음이 성립하는지 본다. 진입점 세 곳이 같은 말을 하게 한다.

**왜 따로 있나 — 실제 사고**

회사 PC 에서 numpy 1.26 과 scipy 1.18 이 함께 깔렸다. scipy 1.18 은 numpy 2.x 를
전제로 `np.long` 을 참조하는데 numpy 1.26 에는 그게 없다. scipy 가 죽으면
sklearn 이 죽고, 그러면 이 도구는 아무것도 못 한다.

화면에 뜬 것은 원인 하나가 아니라 **증상 24개**였다 —
`AttributeError: module 'numpy' has no attribute 'long'` 이 테스트마다 반복됐다.
사용자는 600줄짜리 진단 리포트를 만들어 보내고서야 원인을 알았다.

증상을 스물네 번 찍는 대신 **원인을 한 번** 찍는다. 그리고 세 진입점
(회귀 테스트 · 환경 점검 · 진단 리포트)이 **같은 문장과 같은 복구 명령**을
쓰게 한다 — 각자 다른 말을 하면 사용자가 어느 쪽을 믿어야 할지 모른다.
"""

from __future__ import annotations

# 이 묶음이 성립해야 나머지가 의미를 갖는다. 순서대로 import 해 본다.
CORE = ("numpy", "pandas", "scipy.sparse", "sklearn")

REPAIR = """  고치는 법 — 이 폴더에서 한 줄:

      run.bat --repair          (윈도우)
      ./run.sh --repair         (macOS / Linux)

  그래도 안 되면 이 폴더의 .venv 를 지우고 run.bat 을 다시 실행하세요."""


def probe() -> str:
    """정상이면 빈 문자열, 깨졌으면 이유 한 줄."""
    import importlib

    for name in CORE:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            return f"{name} — {type(e).__name__}: {e}"
    return ""


def versions() -> list[str]:
    """지금 깔려 있는 버전. 원인 판단에 이 두 줄이면 대개 충분하다."""
    import importlib

    out = []
    for name in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            out.append(f"{name} {importlib.import_module(name).__version__}")
        except Exception:  # noqa: BLE001
            out.append(f"{name} —")
    return out


def message(why: str) -> str:
    """사용자가 그대로 읽고 행동할 수 있는 안내문."""
    return (
        "설치된 패키지 조합이 깨져 있습니다 — 코드 문제가 아닙니다.\n\n"
        f"  {why}\n\n"
        f"  {' · '.join(versions())}\n\n"
        "  numpy 와 scipy 의 버전이 서로 맞지 않습니다. 부가 패키지(shap 이 딸고\n"
        "  오는 numba 등)를 설치하면서 numpy 가 내려갔을 때 생깁니다.\n\n"
        + REPAIR)
