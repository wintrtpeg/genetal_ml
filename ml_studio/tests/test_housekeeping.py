"""남은 파일 정리 — 지우는 코드라서 특히 조심해서 본다.

**지우는 기능은 틀리면 되돌릴 수 없다.** 다른 결함은 다시 돌리면 되지만
이건 사용자의 결과물이 사라진다. 그래서 여기 테스트는 "잘 지우는가" 보다
**"지우면 안 되는 것을 안 지키는가"** 에 더 무게를 둔다.

지키는 선 세 가지
  1. `runs/` 밖으로는 한 발도 안 나간다 (심볼릭 링크·`..` 포함)
  2. 보관(KEEP) 지정한 실행은 **어떤 기준에도** 안 걸린다
  3. 지금 쓰고 있는 실행은 안 지운다
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import housekeeping as hk  # noqa: E402


# ── 준비 ─────────────────────────────────────────────────────
def _make_run(runs: Path, name: str, *, model_mb: float = 0.0,
              pinned: bool = False, age_days: int = 0) -> Path:
    import os
    import time

    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    if model_mb:
        (d / "champion_model.joblib").write_bytes(b"0" * int(model_mb * 1024 * 1024))
    if pinned:
        (d / hk.KEEP_MARK).write_text("keep", encoding="utf-8")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(d, (old, old))
    return d


@pytest.fixture
def runs(tmp_path=None, monkeypatch=None):
    """진짜 파일시스템에 임시 runs/ 를 만들고 모듈이 그쪽을 보게 한다."""
    import tempfile

    # **resolve() 를 빼먹으면 안 된다.** 윈도우에서 %TEMP% 가 8.3 단축이름
    # (JPIL~1.HWA)으로 잡혀 있으면 mkdtemp 는 단축 표기를 주는데, 제품 안에서
    # 경로를 resolve 하면 긴 이름(jpil.hwang)이 되어 서로 다른 문자열이 된다.
    # 실제 ROOT 는 임포트 시점에 resolve 되므로, 테스트도 같은 모양이어야
    # 제품과 같은 조건이 된다. 이걸 빠뜨려서 회사 PC 에서만 4건이 실패했다.
    tmp = Path(tempfile.mkdtemp()).resolve()
    r = tmp / "runs"
    r.mkdir()
    hk.ROOT, hk.RUNS_DIR = tmp, r
    return r


def _restore():
    hk.ROOT = Path(__file__).resolve().parent.parent
    hk.RUNS_DIR = hk.ROOT / "runs"


# ── 1. 절대 넘으면 안 되는 선 ────────────────────────────────
def test_refuses_to_touch_anything_outside_runs(runs):
    """**이게 제일 중요하다.** 실수로 상위 폴더를 지우는 일은
    '조심하면 된다' 가 아니라 코드로 막아야 한다."""
    try:
        outside = [
            runs.parent,                       # runs 의 부모
            runs.parent / "중요한폴더",
            runs / ".." / "탈출",              # .. 로 빠져나가기
            Path.home(),
            hk.RUNS_DIR,                       # runs 자기 자신도 안 된다
        ]
        for p in outside:
            with pytest.raises(hk.UnsafePath):
                hk._inside_runs(p)
        # 안쪽은 통과해야 한다
        ok = _make_run(runs, "run_ok")
        assert hk._inside_runs(ok) == ok.resolve()
    finally:
        _restore()


def test_apply_rechecks_the_path_right_before_deleting(runs):
    """목록을 만든 뒤 지우기 직전에 **다시** 확인해야 한다.

    만들 때만 확인하고 넘어가면, 그 사이 목록이 바뀌었을 때 막을 방법이 없다.
    """
    try:
        victim = runs.parent / "지우면안됨"
        victim.mkdir()
        (victim / "소중한파일.txt").write_text("x", encoding="utf-8")

        plan = hk.Plan(runs=[victim])          # 목록을 손으로 오염시킨다
        res = hk.apply(plan)

        assert victim.exists(), "runs 밖 폴더를 지웠습니다"
        assert (victim / "소중한파일.txt").exists()
        assert res.failed, "막았으면서 실패로 보고하지 않았습니다"
    finally:
        _restore()


# ── 2. 보관 지정은 절대 안 지운다 ────────────────────────────
def test_pinned_runs_survive_every_policy(runs):
    try:
        _make_run(runs, "pinned_old", model_mb=5, pinned=True, age_days=999)
        for i in range(5):
            _make_run(runs, f"run_{i}", model_mb=1)

        harsh = hk.RetentionPolicy(keep_runs=1, max_total_mb=1, keep_days=1)
        p = hk.plan(harsh)
        names = [d.name for d in p.runs]
        assert "pinned_old" not in names, f"보관 지정을 지우려 합니다: {names}"
        assert names, "그 외에는 정리 대상이 있어야 합니다"

        hk.apply(p)
        assert (runs / "pinned_old").exists()
    finally:
        _restore()


def test_pin_and_unpin_round_trip(runs):
    try:
        d = _make_run(runs, "run_a")
        assert not hk.is_pinned(d)
        hk.pin(d)
        assert hk.is_pinned(d)
        hk.unpin(d)
        assert not hk.is_pinned(d)
    finally:
        _restore()


def test_a_run_pinned_after_planning_is_still_spared(runs):
    """계획을 세운 뒤 사용자가 보관을 눌렀다면 그 뜻을 따라야 한다."""
    try:
        for i in range(4):
            _make_run(runs, f"run_{i}")
        p = hk.plan(hk.RetentionPolicy(keep_runs=1, max_total_mb=0))
        assert p.runs
        hk.pin(p.runs[0])                      # 계획 뒤 보관 지정
        target = p.runs[0]
        hk.apply(p)
        assert target.exists(), "계획 뒤 보관 지정한 실행을 지웠습니다"
    finally:
        _restore()


# ── 3. 쓰는 중인 것은 안 지운다 ──────────────────────────────
def test_the_run_in_use_is_protected(runs):
    try:
        for i in range(5):
            _make_run(runs, f"run_{i}")
        p = hk.plan(hk.RetentionPolicy(keep_runs=1, max_total_mb=0),
                    protect=("run_0",))
        assert "run_0" not in [d.name for d in p.runs]
        assert any("사용 중" in k for k in p.kept), p.kept
    finally:
        _restore()


# ── 정책이 실제로 동작하는가 ─────────────────────────────────
def test_keeps_the_newest_n(runs):
    try:
        import os
        import time
        for i in range(8):
            d = _make_run(runs, f"run_{i}")
            t = time.time() - (8 - i) * 3600      # run_7 이 가장 최신
            os.utime(d, (t, t))
        p = hk.plan(hk.RetentionPolicy(keep_runs=3, max_total_mb=0))
        gone = {d.name for d in p.runs}
        assert "run_7" not in gone and "run_6" not in gone and "run_5" not in gone
        assert "run_0" in gone
        assert len(gone) == 5, gone
    finally:
        _restore()


def test_budget_drops_oldest_until_it_fits(runs):
    try:
        import os
        import time
        for i in range(5):
            d = _make_run(runs, f"run_{i}", model_mb=3)
            t = time.time() - (5 - i) * 3600
            os.utime(d, (t, t))
        # 5개 × 3MB = 15MB. 예산 7MB 면 최신 두 개만 남아야 한다.
        p = hk.plan(hk.RetentionPolicy(keep_runs=0, max_total_mb=7))
        assert len(p.runs) == 3, [d.name for d in p.runs]
        assert any("용량 예산" in v for v in p.reasons.values()), p.reasons
    finally:
        _restore()


def test_age_limit_applies(runs):
    try:
        _make_run(runs, "fresh", age_days=1)
        _make_run(runs, "stale", age_days=90)
        p = hk.plan(hk.RetentionPolicy(keep_runs=10, max_total_mb=0, keep_days=30))
        assert [d.name for d in p.runs] == ["stale"], [d.name for d in p.runs]
    finally:
        _restore()


def test_no_policy_means_nothing_is_removed(runs):
    """전부 0 이면 '제한 없음' 이다. 그때 지우면 배신이다."""
    try:
        for i in range(20):
            _make_run(runs, f"run_{i}", model_mb=1)
        p = hk.plan(hk.RetentionPolicy(keep_runs=0, max_total_mb=0, keep_days=0))
        assert not p.runs, [d.name for d in p.runs]
    finally:
        _restore()


# ── 찌꺼기 ───────────────────────────────────────────────────
def test_a_crashed_run_with_nothing_in_it_is_junk(runs):
    """중간에 죽어 껍데기만 남은 폴더는 아무도 원하지 않는다."""
    try:
        (runs / "empty_crash").mkdir()
        _make_run(runs, "real_run")
        junk, _ = hk.find_junk()
        names = [p.name for p in junk]
        assert "empty_crash" in names
        assert "real_run" not in names
    finally:
        _restore()


def test_a_crashed_run_that_was_pinned_is_not_junk(runs):
    """비어 있어도 사용자가 남기라고 했으면 남긴다."""
    try:
        d = runs / "empty_but_pinned"
        d.mkdir()
        (d / hk.KEEP_MARK).write_text("keep", encoding="utf-8")
        assert "empty_but_pinned" not in [p.name for p in hk.find_junk()[0]]
    finally:
        _restore()


def test_verify_scratch_report_is_junk(runs):
    try:
        (runs / "_verify_report.html").write_text("x", encoding="utf-8")
        assert "_verify_report.html" in [p.name for p in hk.find_junk()[0]]
    finally:
        _restore()


def test_apply_actually_frees_space_and_reports_it(runs):
    try:
        _make_run(runs, "big", model_mb=4)
        _make_run(runs, "keep_me", model_mb=1)
        p = hk.plan(hk.RetentionPolicy(keep_runs=1, max_total_mb=0))
        before = hk.dir_size(runs)
        res = hk.apply(p)
        after = hk.dir_size(runs)
        assert after < before
        assert res.freed_bytes > 0
        assert "확보" in res.summary()
    finally:
        _restore()


# ── 구조 ─────────────────────────────────────────────────────
def test_housekeeping_does_not_import_streamlit():
    src = (ROOT / "core" / "housekeeping.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src


def test_sweep_survives_a_missing_runs_folder():
    """runs/ 가 아직 없는 첫 실행에서도 죽으면 안 된다."""
    import tempfile
    tmp = Path(tempfile.mkdtemp()).resolve()
    hk.ROOT, hk.RUNS_DIR = tmp, tmp / "runs"
    try:
        assert hk.scan().empty
        hk.sweep()                     # 죽지 않으면 통과
        assert hk.usage()["runs_bytes"] == 0
    finally:
        _restore()


def test_mb_reads_like_a_human_wrote_it():
    assert hk.mb(512) == "512B"
    assert hk.mb(2048) == "2KB"
    assert hk.mb(5 * 1024 ** 2) == "5.0MB"
    assert hk.mb(3 * 1024 ** 3) == "3.00GB"


# ── 자동 실행 시점 — 시작과 종료 양쪽 ────────────────────────
def test_launch_tidies_at_start_and_at_exit(monkeypatch):
    """**양쪽에 다 걸려 있어야 한다.**

    종료 정리만 두면 창을 X 로 닫았을 때 안 돌고, 시작 정리만 두면 이번
    실행에서 쌓인 것이 다음 실행까지 그대로 남는다.
    """
    import types

    from scripts import launch

    calls = []
    monkeypatch.setattr(launch, "tidy", lambda when="": calls.append(when))
    monkeypatch.setattr(launch, "free_port", lambda *a, **k: 9999)
    monkeypatch.setattr(launch, "subprocess",
                        types.SimpleNamespace(
                            Popen=lambda *a, **k: types.SimpleNamespace(
                                wait=lambda *a, **k: 0, poll=lambda: 0,
                                terminate=lambda: None)))
    launch.launch("python", 9999, open_browser=False)

    assert "시작" in calls, f"시작 정리가 없습니다: {calls}"
    assert "종료" in calls, f"종료 정리가 없습니다: {calls}"
    assert calls.index("시작") < calls.index("종료")


def test_exit_tidy_runs_even_when_the_app_crashes(monkeypatch):
    """앱이 죽어도 정리는 돌아야 한다 — 죽었을 때가 오히려 찌꺼기가 많다."""
    import types

    from scripts import launch

    calls = []
    monkeypatch.setattr(launch, "tidy", lambda when="": calls.append(when))
    monkeypatch.setattr(launch, "free_port", lambda *a, **k: 9999)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(launch, "subprocess",
                        types.SimpleNamespace(
                            Popen=lambda *a, **k: types.SimpleNamespace(
                                wait=boom, poll=lambda: None,
                                terminate=lambda: None)))
    launch.launch("python", 9999, open_browser=False)
    assert "종료" in calls, f"중단됐는데 정리가 안 돌았습니다: {calls}"


def test_tidy_can_be_turned_off(monkeypatch):
    """문제를 쫓을 때 정리를 꺼 둘 통로가 있어야 한다."""
    import sys as _sys

    from scripts import launch

    hit = []
    monkeypatch.setattr(_sys, "argv", ["launch.py", "--no-tidy"])
    monkeypatch.setattr(launch, "ROOT", launch.ROOT)
    try:
        import core.housekeeping as _hk
        monkeypatch.setattr(_hk, "sweep", lambda *a, **k: hit.append(1))
    except Exception:  # noqa: BLE001
        pass
    launch.tidy("시작")
    assert not hit, "--no-tidy 인데 정리가 돌았습니다"


# ── 방금 만든 파일을 지우면 안 된다 ──────────────────────────
def test_a_fresh_diagnostic_report_is_never_deleted(runs):
    """사용자가 보내려고 방금 만든 파일을 정리가 지워버리면 황당한 일이다."""
    try:
        fresh = hk.ROOT / "diagnostic_report.txt"
        fresh.write_text("방금 만든 리포트", encoding="utf-8")
        assert "diagnostic_report.txt" not in [p.name for p in hk.find_junk()[0]]
        assert fresh.exists()
    finally:
        _restore()


def test_an_old_diagnostic_report_is_cleaned_up(runs):
    """다 보낸 지 오래된 리포트는 남겨 둘 이유가 없다."""
    import os
    import time

    try:
        old = hk.ROOT / "diagnostic_report.txt"
        old.write_text("옛날 리포트", encoding="utf-8")
        past = time.time() - 30 * 86400
        os.utime(old, (past, past))
        assert "diagnostic_report.txt" in [p.name for p in hk.find_junk()[0]]
    finally:
        _restore()


def test_junk_plan_never_touches_run_results(runs):
    """진단하러 온 자리에서 사용자 산출물을 지우면 안 된다."""
    try:
        for i in range(30):
            _make_run(runs, f"run_{i}", model_mb=1)
        (runs / "빈폴더").mkdir()
        p = hk.junk_plan()
        assert p.runs == [], "junk_plan 이 실행 결과를 건드립니다"
        assert p.junk, "찌꺼기는 잡아야 합니다"
        hk.apply(p)
        assert len(list(runs.glob("run_*"))) == 30, "실행이 지워졌습니다"
    finally:
        _restore()


def test_deletion_is_permanent_not_a_recycle_bin(runs):
    """'영구삭제' 라고 했으면 실제로 사라져야 한다.

    휴지통으로 보내면 드라이브 용량은 그대로다 — 목적을 못 이룬다.
    """
    try:
        d = _make_run(runs, "gone", model_mb=1)
        target = d / "champion_model.joblib"
        assert target.exists()
        hk.apply(hk.Plan(runs=[d]))
        assert not d.exists()
        assert not target.exists()
    finally:
        _restore()


# ── 윈도우에서만 드러난 것들 ─────────────────────────────────
def test_reporting_a_path_never_breaks_the_cleanup(runs):
    """**표시 때문에 삭제가 실패하면 안 된다.**

    회사 PC 에서 `relative_to` 가 ValueError 를 던져 정리 전체가 중단됐다.
    %TEMP% 가 8.3 단축이름으로 잡혀 있어 resolve 한 경로와 ROOT 의 표기가
    달랐기 때문이다. '어떻게 보여줄까' 의 문제가 '지우지 못함' 이 되면 안 된다.
    """
    try:
        # ROOT 와 아무 관계 없는 경로를 넣어도 표시가 죽지 않아야 한다
        assert hk._show(Path("/전혀/다른/곳/파일.txt"))
        assert hk._show(runs / "run_x")

        d = _make_run(runs, "run_x", model_mb=1)
        hk.ROOT = Path("/완전히/다른/뿌리")      # 표기를 일부러 어긋나게 한다
        res = hk.apply(hk.Plan(runs=[d]))
        assert not d.exists(), "표시가 어긋났다고 삭제를 못 했습니다"
        assert res.removed, "지웠는데 보고가 없습니다"
    finally:
        _restore()


def test_junk_scan_does_not_walk_the_virtualenv(monkeypatch):
    """`.venv` 안에는 파일이 수만 개다. 훑으면 정리가 수십 초를 먹는다.

    정리는 실행할 때마다 도는 기능이라 그 비용을 감당할 수 없다 —
    실제로 회사 PC 에서 회귀 테스트가 6분에서 30분으로 늘었다.

    **시간으로 재지 않는다.** rglob 판도 결과에서는 `.venv` 를 걸러 내므로
    출력이 같고, 리눅스에서는 빨라서 시간 기준을 통과해 버린다(실제로
    변형을 넣었더니 안 잡혔다). 대신 **어디를 들여다봤는지**를 직접 센다 —
    그게 이 기능이 지켜야 할 성질 그 자체다.
    """
    import os
    import tempfile

    # **os 수준에서 잡아야 한다.** Path.iterdir 만 감시하면 rglob 을 못 본다 —
    # rglob 은 내부적으로 os.scandir 를 쓰기 때문이다. 실제로 iterdir 만
    # 감시했더니 변형(rglob 되돌리기)이 안 잡혔다.
    visited: list[str] = []
    real_listdir, real_scandir = os.listdir, os.scandir

    def spy_listdir(path="."):
        visited.append(str(path))
        return real_listdir(path)

    def spy_scandir(path="."):
        visited.append(str(path))
        return real_scandir(path)

    tmp = Path(tempfile.mkdtemp()).resolve()
    sp = tmp / ".venv" / "Lib" / "site-packages"
    for i in range(5):
        (sp / f"pkg_{i}" / "__pycache__").mkdir(parents=True)
    (tmp / "core" / "__pycache__").mkdir(parents=True)
    (tmp / "runs").mkdir()

    hk.ROOT, hk.RUNS_DIR = tmp, tmp / "runs"
    monkeypatch.setattr(os, "listdir", spy_listdir)
    monkeypatch.setattr(os, "scandir", spy_scandir)
    try:
        junk, _ = hk.find_junk()
    finally:
        monkeypatch.undo()
        _restore()

    inside = [v for v in visited if ".venv" in v]
    assert not inside, ("가상환경 안을 들여다봤습니다 — 회사 PC 에서는 파일이 "
                        f"수만 개입니다:\n  " + "\n  ".join(inside[:5]))
    names = [str(p) for p in junk]
    assert any("core" in n for n in names), "정작 프로젝트 캐시는 놓쳤습니다"
    assert not any(".venv" in n for n in names)


# ── 첫 실행은 지우지 않고 알린다 ─────────────────────────────
def test_the_very_first_run_reports_instead_of_deleting(runs):
    """**이 기능이 붙기 전에 만든 결과물이 이미 있을 수 있다.**

    사용자는 기준이 생긴 줄도 모르는데 새 버전을 처음 켜자마자 오래된 실행이
    말없이 영구 삭제되면 그건 배신이다. 한 번 미룬다고 디스크가 넘치지 않는다.
    """
    try:
        for i in range(15):
            _make_run(runs, f"run_{i:02d}", model_mb=1)
        (runs / "빈껍데기").mkdir()

        assert not hk.is_armed()
        res = hk.sweep(hk.RetentionPolicy(keep_runs=3, max_total_mb=0))

        assert len(list(runs.glob("run_*"))) == 15, "첫 실행에서 결과를 지웠습니다"
        assert not (runs / "빈껍데기").exists(), "찌꺼기는 첫 실행에도 치워야 합니다"
        assert res.deferred, "지울 뻔한 것을 알리지 않았습니다"
        assert "영구 삭제" in res.notice(), res.notice()
        assert "보관 지정" in res.notice(), res.notice()
    finally:
        _restore()


def test_the_second_run_actually_enforces_the_policy(runs, monkeypatch):
    """알린 뒤에는 기준대로 지운다 — 안 그러면 기능이 없는 것과 같다.

    '다음 실행' 은 새 프로세스다. 표식보다 늦게 시작한 프로세스를 흉내 낸다.
    """
    try:
        for i in range(15):
            _make_run(runs, f"run_{i:02d}", model_mb=1)
        hk.sweep(hk.RetentionPolicy(keep_runs=3, max_total_mb=0))   # 1회차: 알림
        assert (hk.RUNS_DIR / hk.ARM_MARK).exists()

        monkeypatch.setattr(hk, "_STARTED_AT", hk._STARTED_AT + 10_000)
        assert hk.is_armed()
        res = hk.sweep(hk.RetentionPolicy(keep_runs=3, max_total_mb=0))  # 2회차
        assert len(list(runs.glob("run_*"))) == 3, "두 번째에도 안 지웁니다"
        assert res.removed
        assert not res.deferred
    finally:
        _restore()


def test_first_run_with_nothing_to_defer_is_silent(runs):
    """지울 것이 없으면 굳이 알릴 이유가 없다."""
    try:
        _make_run(runs, "only_one")
        res = hk.sweep(hk.RetentionPolicy(keep_runs=10, max_total_mb=0))
        assert res.notice() == "", res.notice()
        assert (hk.RUNS_DIR / hk.ARM_MARK).exists(), "표식을 안 남겼습니다"
    finally:
        _restore()


def test_the_arm_marker_is_not_mistaken_for_a_run(runs):
    """표식 파일이 실행 목록에 끼면 안 된다."""
    try:
        _make_run(runs, "run_a")
        hk.arm()
        assert "arm" not in " ".join(hk.scan()["실행"])
        assert hk.ARM_MARK not in [p.name for p in hk.find_junk()[0]]
    finally:
        _restore()


def test_the_grace_period_survives_the_same_run(runs, monkeypatch):
    """**알린 실행에서 바로 지우면 유예가 아니다.**

    처음 만든 판이 그랬다 — 시작 정리가 알리며 표식을 찍고, 같은 실행의 종료
    정리가 그 표식을 보고 지워 버렸다. 유예가 3초였다. 사용자가 알림을 읽고
    보관 지정을 할 틈이 없으니 없는 것과 같다.
    """
    try:
        for i in range(10):
            _make_run(runs, f"run_{i:02d}", model_mb=1)
        policy = hk.RetentionPolicy(keep_runs=2, max_total_mb=0)

        first = hk.sweep(policy)                     # 시작 정리 — 알림
        assert first.deferred
        again = hk.sweep(policy)                     # 같은 실행의 종료 정리
        assert not again.removed, "알린 실행에서 바로 지웠습니다 — 유예가 없습니다"
        assert len(list(runs.glob("run_*"))) == 10

        # 다음 실행(= 프로세스 시작 시각이 표식보다 나중)에서는 지운다
        monkeypatch.setattr(hk, "_STARTED_AT", hk._STARTED_AT + 10_000)
        later = hk.sweep(policy)
        assert later.removed, "다음 실행에서도 안 지웁니다"
        assert len(list(runs.glob("run_*"))) == 2
    finally:
        _restore()
