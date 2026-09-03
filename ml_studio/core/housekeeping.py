"""남은 파일을 정리한다. 드라이브를 무한정 먹지 않게.

무엇이 쌓이나
-------------
가장 큰 것은 `runs/<실행>/champion_model.joblib` 이다. RandomForest·앙상블이면
**한 번에 수백 MB** 가 나온다. 실행할 때마다 하나씩 쌓이므로, 며칠 쓰면
기가 단위가 된다. 나머지(csv·json·html)는 합쳐도 대개 수 MB다.

지우는 것을 두 부류로 완전히 갈라 놓는다
----------------------------------------
**1. 쓰레기** — 아무도 원하지 않는다. 조건 없이 지운다.
   `__pycache__`, 점검용 임시 리포트, 중간에 죽어 아무것도 안 남은 실행 폴더.

**2. 사용자 산출물(runs)** — 이건 다르다. 사용자가 나중에 열어 볼 수 있는
   결과물이고, **말없이 지우면 그건 데이터 손실이다.** 그래서
   · 정책을 눈에 보이게 두고 (최신 N개 · 총 용량 예산)
   · 고정(KEEP)해 둔 실행은 **정책과 무관하게 절대 안 지운다**
   · 무엇을 얼마나 지웠는지 항상 보고한다

안전장치
--------
`runs/` 밖으로는 한 발도 나가지 않는다. 경로를 resolve 해서 RUNS_DIR 안에
있는지 확인하고, 아니면 예외를 던진다. 실수로 상위 폴더를 지우는 일은
"조심하면 된다" 가 아니라 코드로 막아야 하는 종류다.

streamlit 을 모른다 — 화면은 결과만 받아서 그린다.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

# 이 표식이 있는 실행 폴더는 정책과 무관하게 남는다.
KEEP_MARK = "KEEP"

# 실행 폴더가 '쓸모 있다' 고 보는 최소 조건. 이 중 하나도 없으면 중간에 죽어
# 껍데기만 남은 것이므로 쓰레기로 본다.
MEANINGFUL = ("manifest.json", "champion_model.joblib", "report.html",
              "leaderboard.csv")

# 프로젝트 루트에 남는 로그·리포트. **바로 지우면 안 되는 것들이다** —
# diagnostic_report.txt 는 사용자가 보내려고 만든 파일이라, 만들자마자 정리가
# 지워버리면 황당한 일이 된다. 그래서 나이를 본다.
AGED_FILES = {
    "report_console.log": 3,        # (파일명, 며칠 지나면 지울지)
    "diagnostic_report.txt": 7,
}

# 훑지 않는다. **`.venv` 안에는 파일이 수만 개**라 그냥 rglob 하면 회사 PC 에서
# 한 번에 수십 초가 걸린다. 정리는 실행할 때마다 도는 기능이라 그 비용을
# 감당할 수 없다 — 실제로 회귀 테스트가 6분에서 30분으로 늘었다.
SKIP_DIRS = {".venv", ".git", "node_modules", "site-packages", ".mypy_cache"}


class UnsafePath(RuntimeError):
    """runs/ 밖을 지우려 했다. 절대 일어나면 안 되는 일이다."""


@dataclass
class RetentionPolicy:
    """얼마나 남길 것인가. 화면에서 사용자가 바꾼다.

    keep_runs   — 최신 몇 개를 남길지. 0 이면 개수 제한 없음
    max_total_mb— runs/ 전체 용량 예산. 0 이면 용량 제한 없음
    keep_days   — 이 일수보다 오래된 것은 정리 대상. 0 이면 기간 무시
    """

    keep_runs: int = 10
    max_total_mb: float = 2000.0
    keep_days: int = 0

    def describe(self) -> str:
        parts = []
        if self.keep_runs:
            parts.append(f"최신 {self.keep_runs}개")
        if self.max_total_mb:
            parts.append(f"총 {self.max_total_mb:,.0f}MB 이내")
        if self.keep_days:
            parts.append(f"{self.keep_days}일 이내")
        return " · ".join(parts) if parts else "제한 없음 (계속 쌓입니다)"


@dataclass
class Plan:
    """무엇을 지울지. **아직 안 지웠다.**"""

    junk: list[Path] = field(default_factory=list)        # 쓰레기
    runs: list[Path] = field(default_factory=list)        # 정책에 걸린 실행
    junk_bytes: int = 0
    runs_bytes: int = 0
    kept: list[str] = field(default_factory=list)         # 고정돼서 남긴 것
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.junk_bytes + self.runs_bytes

    def __bool__(self) -> bool:
        return bool(self.junk or self.runs)


@dataclass
class Result:
    freed_bytes: int = 0
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    # 첫 실행에서 '지울 뻔했지만 미룬' 것들. 사용자에게 알리기만 한다.
    deferred: list[str] = field(default_factory=list)
    deferred_bytes: int = 0

    def summary(self) -> str:
        if not self.removed and not self.failed:
            return "정리할 것이 없었습니다."
        s = f"{len(self.removed)}개 정리 · {mb(self.freed_bytes)} 확보"
        if self.failed:
            s += f" (실패 {len(self.failed)}개 — 다른 프로그램이 쓰는 중일 수 있습니다)"
        return s

    def notice(self) -> str:
        """첫 실행 안내. 무엇을 미뤘고 어떻게 하면 되는지."""
        if not self.deferred:
            return ""
        names = ", ".join(self.deferred[:3])
        more = f" 외 {len(self.deferred) - 3}개" if len(self.deferred) > 3 else ""
        return (f"[알림] 저장공간 자동 정리가 켜졌습니다. 기준에 걸리는 실행 "
                f"{len(self.deferred)}개({mb(self.deferred_bytes)})가 있습니다 "
                f"— {names}{more}.\n"
                "        이번에는 지우지 않았습니다. 남길 것이 있으면 화면의 "
                "'설정 > 저장공간' 에서 보관 지정해 두세요.\n"
                "        다음 실행부터 기준대로 **영구 삭제**됩니다.")


# ─────────────────────────────────────────────────────────────
def mb(n: int | float) -> str:
    n = float(n)
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.2f}GB"


def dir_size(p: Path) -> int:
    """폴더가 실제로 차지하는 바이트. 접근 못 하는 파일은 건너뛴다."""
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _show(p: Path) -> str:
    """보고용 경로. **표시 때문에 삭제가 실패하면 안 된다.**

    윈도우에서 %TEMP% 가 8.3 단축이름(JPIL~1.HWA)으로 잡혀 있으면 resolve 한
    경로와 ROOT 의 표기가 달라 relative_to 가 ValueError 를 던진다. 그건
    '어떻게 보여줄까' 의 문제일 뿐인데, 그 예외가 정리 전체를 중단시켰다.
    """
    try:
        return str(Path(p).relative_to(ROOT))
    except (ValueError, OSError):
        return str(p)


def _inside_runs(p: Path) -> Path:
    """runs/ 안인지 확인하고 정규화한 경로를 돌려준다.

    **경로 검사를 지우기 직전에 한다.** 목록을 만들 때만 확인하고 넘어가면,
    그 사이에 목록이 바뀌었을 때 막을 방법이 없다.
    """
    rp = Path(p).resolve()
    runs = RUNS_DIR.resolve()
    if rp == runs or runs not in rp.parents:
        raise UnsafePath(f"runs/ 밖은 지우지 않습니다: {rp}")
    return rp


def is_pinned(run: Path) -> bool:
    return (Path(run) / KEEP_MARK).exists()


def pin(run: Path, note: str = "") -> None:
    """이 실행을 정리 대상에서 뺀다."""
    p = _inside_runs(run)
    (p / KEEP_MARK).write_text(
        note or f"보관 지정 {datetime.now():%Y-%m-%d %H:%M}\n", encoding="utf-8")


def unpin(run: Path) -> None:
    p = _inside_runs(run)
    (p / KEEP_MARK).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
def scan() -> pd.DataFrame:
    """실행 폴더별 용량·나이·고정 여부. 화면이 이 표를 그대로 그린다."""
    cols = ["실행", "용량", "bytes", "생성", "일수", "보관", "모델", "리포트"]
    if not RUNS_DIR.exists():
        return pd.DataFrame(columns=cols)

    now = datetime.now()
    rows = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            made = datetime.fromtimestamp(d.stat().st_mtime)
        except OSError:
            continue
        size = dir_size(d)
        rows.append({
            "실행": d.name,
            "용량": mb(size),
            "bytes": size,
            "생성": made.strftime("%Y-%m-%d %H:%M"),
            "일수": (now - made).days,
            "보관": is_pinned(d),
            "모델": (d / "champion_model.joblib").exists(),
            "리포트": (d / "report.html").exists(),
        })
    return pd.DataFrame(rows, columns=cols)


def find_junk() -> tuple[list[Path], int]:
    """아무도 원하지 않는 것들. 정책과 무관하게 지워도 된다."""
    out: list[Path] = []

    # **rglob 대신 직접 걸어 내려가며 가지치기를 한다.**
    # rglob 은 결과를 걸러 낼 뿐 .venv 안을 다 들어갔다 나온다. 우리는 아예
    # 들어가지 않아야 한다.
    stack = [ROOT]
    while stack:
        cur = stack.pop()
        try:
            for child in cur.iterdir():
                if not child.is_dir():
                    continue
                if child.name in SKIP_DIRS:
                    continue                    # 여기부터는 내려가지 않는다
                if child.name == "__pycache__":
                    out.append(child)
                    continue                    # 통째로 지울 것이라 더 안 본다
                stack.append(child)
        except OSError:
            continue

    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            # 점검이 남긴 임시 리포트
            if d.is_file() and d.name.startswith("_verify"):
                out.append(d)
            # 중간에 죽어 아무 산출물도 없는 실행 폴더
            elif d.is_dir() and not is_pinned(d):
                if not any((d / f).exists() for f in MEANINGFUL):
                    out.append(d)

    now = datetime.now().timestamp()
    for name, days in AGED_FILES.items():
        p = ROOT / name
        try:
            if p.exists() and (now - p.stat().st_mtime) > days * 86400:
                out.append(p)
        except OSError:
            continue

    total = sum(dir_size(p) if p.is_dir() else
                (p.stat().st_size if p.exists() else 0) for p in out)
    return out, total


def junk_plan() -> Plan:
    """찌꺼기만. **실행 결과는 절대 건드리지 않는다.**

    진단 리포트를 만드는 길처럼 "청소하러 온 게 아닌" 자리에서 쓴다.
    원인 보러 왔다가 사용자 산출물을 지우면 안 된다.
    """
    p = Plan()
    p.junk, p.junk_bytes = find_junk()
    return p


def plan(policy: RetentionPolicy | None = None,
         protect: tuple[str, ...] = ()) -> Plan:
    """무엇을 지울지 정한다. **여기서는 지우지 않는다.**

    protect — 지금 화면이 쓰고 있는 실행 이름. 쓰는 중인 것을 지우면
              사용자가 보던 결과가 사라진다.
    """
    policy = policy or RetentionPolicy()
    p = Plan()

    p.junk, p.junk_bytes = find_junk()
    junk_names = {x.name for x in p.junk}

    table = scan()
    if table.empty:
        return p

    now_protected = set(protect)
    alive = table[~table["실행"].isin(junk_names)]

    # **기본은 "전부 남긴다" 이고, 기준에 걸리는 것만 뺀다.**
    #
    # 처음에는 반대로 짰다 — 남길 것을 고르고 나머지를 지우는 방식. 그랬더니
    # 기준을 전부 0(제한 없음)으로 두면 아무것도 'keep' 에 안 들어가서
    # **실행 전체가 삭제 대상**이 됐다. "제한 없음" 이 "전부 삭제" 로 동작한
    # 것이다. 지우는 코드에서 기본값이 파괴적인 쪽이면 안 된다.
    doomed: dict[str, str] = {}

    rows = []
    for _, r in alive.iterrows():
        if r["보관"]:
            p.kept.append(f"{r['실행']} (보관 지정)")
        elif r["실행"] in now_protected:
            p.kept.append(f"{r['실행']} (지금 사용 중)")
        else:
            rows.append(r)
    rows.sort(key=lambda r: r["생성"], reverse=True)      # 최신 우선

    # 1) 개수
    if policy.keep_runs:
        for r in rows[policy.keep_runs:]:
            doomed[r["실행"]] = f"최신 {policy.keep_runs}개 밖"

    # 2) 기간
    if policy.keep_days:
        for r in rows:
            if int(r["일수"]) > policy.keep_days:
                doomed.setdefault(
                    r["실행"],
                    f"{policy.keep_days}일 초과 ({int(r['일수'])}일 지남)")

    # 3) 용량 — 살아남은 것만 최신부터 담다가 넘치면 거기서 끊는다.
    #    보관 지정한 것도 자리는 차지한다 (지우지는 못하지만 공간은 먹는다).
    if policy.max_total_mb:
        budget = policy.max_total_mb * 1024 ** 2
        used = sum(int(r["bytes"]) for _, r in alive.iterrows() if r["보관"])
        for r in rows:
            if r["실행"] in doomed:
                continue
            size = int(r["bytes"])
            if used + size > budget:
                doomed[r["실행"]] = f"용량 예산 {policy.max_total_mb:,.0f}MB 초과"
            else:
                used += size

    for r in rows:
        name = r["실행"]
        if name in doomed:
            p.reasons[name] = doomed[name]
            p.runs.append(RUNS_DIR / name)
            p.runs_bytes += int(r["bytes"])

    return p


def apply(p: Plan) -> Result:
    """정한 대로 지운다. 경로 검사는 **지우기 직전에 다시** 한다."""
    res = Result()

    for path in p.junk:
        try:
            if not path.exists():
                continue
            size = dir_size(path) if path.is_dir() else path.stat().st_size
            if path.is_dir():
                # __pycache__ 는 runs/ 밖에도 있으므로 여기서만 예외를 둔다.
                if path.name != "__pycache__":
                    _inside_runs(path)
                shutil.rmtree(path, ignore_errors=False)
            else:
                if path.parent.resolve() == ROOT.resolve():
                    path.unlink()          # 프로젝트 루트의 로그 파일
                else:
                    _inside_runs(path)
                    path.unlink()
            res.freed_bytes += size
            res.removed.append(_show(path))
        except (OSError, UnsafePath, ValueError) as e:
            res.failed.append(f"{path.name}: {e}")

    for path in p.runs:
        try:
            safe = _inside_runs(path)
            if is_pinned(safe):
                continue                   # 그 사이에 보관 지정됐다면 건드리지 않는다
            size = dir_size(safe)
            shutil.rmtree(safe)
            res.freed_bytes += size
            res.removed.append(_show(safe))
        except (OSError, UnsafePath, ValueError) as e:
            res.failed.append(f"{path.name}: {e}")

    return res


ARM_MARK = ".tidy_armed"

# 이 프로세스가 시작한 시각. **표식이 이번 실행 중에 찍힌 것인지 가르는 데 쓴다.**
#
# 처음 만든 판은 유예가 3초였다 — 시작 정리가 알리면서 표식을 찍고, 같은 실행의
# 종료 정리가 그 표식을 보고 바로 지워 버렸다. 알림을 읽고 보관 지정을 할 틈이
# 없었으니 유예가 아예 없는 것과 같다. 표식이 **이번 실행보다 먼저** 있었을
# 때만 삭제를 허용한다.
_STARTED_AT = time.time()


def is_armed() -> bool:
    """실행 결과를 지워도 되는가 — 지난 실행에서 이미 알린 적이 있는가."""
    mark = RUNS_DIR / ARM_MARK
    try:
        if not mark.exists():
            return False
        return mark.stat().st_mtime < _STARTED_AT
    except OSError:
        return False


def arm() -> None:
    # runs/ 가 아직 없으면 지킬 것도 없다. 표식을 남기려고 폴더를 만들지 않는다 —
    # 빈 폴더를 만들어 두면 "아무것도 없는데 왜 생겼지" 가 되고, 용량 표시도
    # 0 이 아니게 된다.
    if not RUNS_DIR.exists():
        return
    note = f"정리 기준을 처음 알린 시각 {datetime.now():%Y-%m-%d %H:%M}\n"
    try:
        (RUNS_DIR / ARM_MARK).write_text(note, encoding="utf-8")
    except OSError:
        pass


def sweep(policy: RetentionPolicy | None = None,
          protect: tuple[str, ...] = ()) -> Result:
    """정하고 지운다. 실행 시작·종료 때 부르는 것.

    **처음 한 번은 실행 결과를 지우지 않고 알리기만 한다.**

    이 기능이 붙기 전에 만든 결과물이 이미 폴더에 있을 수 있다. 사용자는
    기준이 생긴 줄도 모르는데, 새 버전을 처음 켜자마자 오래된 실행이 말없이
    **영구 삭제**되면 그건 배신이다. 한 번 미룬다고 디스크가 넘치지 않는다.

    찌꺼기(캐시·빈 폴더)는 첫 실행에도 그냥 치운다 — 잃을 것이 없다.
    """
    p = plan(policy, protect)

    if not is_armed():
        pending = list(p.runs)
        pending_bytes = p.runs_bytes
        p.runs, p.runs_bytes = [], 0          # 이번에는 실행 결과를 건드리지 않는다
        res = apply(p)
        arm()
        if pending:
            res.deferred = [d.name for d in pending]
            res.deferred_bytes = pending_bytes
        return res

    return apply(p)


def usage() -> dict:
    """지금 얼마나 쓰고 있나. 화면 상단에 한 줄로 띄운다."""
    runs = dir_size(RUNS_DIR) if RUNS_DIR.exists() else 0
    junk = find_junk()[1]
    free = None
    try:
        free = shutil.disk_usage(ROOT).free
    except OSError:
        pass
    return {"runs_bytes": runs, "junk_bytes": junk, "free_bytes": free,
            "runs": mb(runs), "junk": mb(junk),
            "free": mb(free) if free is not None else "?"}
