"""화면 성능·이동 회귀 테스트.

여기 있는 것들은 "느려짐"이 조용히 돌아오는 걸 막는 장치다. 성능 문제는 기능이
깨지지 않으므로 테스트 없이는 다음 수정에서 그대로 되살아난다.

실제로 겪은 증상 세 가지
  1. select_slider 에 12,948개 옵션을 넘겨 화면이 멈춤
  2. 체크박스 하나마다 st.rerun() 으로 전체 재계산
  3. 12,000점 시계열을 SVG 로 그려 스크롤이 버벅임
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import plots  # noqa: E402

VIEWS = ROOT / "app" / "views"


def _src(name: str) -> str:
    return (VIEWS / name).read_text(encoding="utf-8")


# ── 1. 대용량 위젯 ──────────────────────────────────────────
def test_no_widget_takes_full_index_as_options():
    """행 수만큼 옵션을 만드는 위젯이 없어야 한다.

    select_slider(options=list(idx)) 는 12,000행이면 옵션 12,000개를 브라우저로
    보낸다. 화면이 멈춘 직접적 원인이었다.
    """
    bad = []
    for f in sorted(VIEWS.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        for pat in ("options=list(idx)", "options=list(X.index)",
                    "options=list(df.index)", "options=list(S.df.index)"):
            if pat in src:
                bad.append(f"{f.name}: {pat}")
    assert not bad, f"행 수만큼 옵션을 만드는 위젯: {bad}"


# ── 2. 재실행 폭풍 ──────────────────────────────────────────
def test_review_gate_does_not_rerun_on_every_checkbox():
    """data_editor 반환값을 처리한 직후 st.rerun() 이 있으면 안 된다.

    위젯 조작 자체가 이미 재실행을 일으킨다. 거기서 또 rerun 을 부르면 화면이
    두 번 그려지고, 그때마다 상관행렬을 다시 계산해 클릭이 씹힌다.
    """
    src = _src("features_view.py")
    i = src.index("edited = st.data_editor(")
    after = src[i:src.index("# ── 위험 경고 ──", i)]
    # 주석은 뺀다. "왜 rerun 을 안 쓰는지" 설명하는 주석까지 잡으면 안 된다.
    code = "\n".join(ln.split("#", 1)[0] for ln in after.splitlines())
    assert "st.rerun()" not in code, \
        "체크박스 반영 직후에 st.rerun() 이 남아 있습니다"


def test_bulk_select_bumps_editor_generation():
    """일괄 선택은 편집표를 새로 그려야 반영된다.

    data_editor 는 자기 편집 상태를 key 로 붙들고 있어서, 바깥에서 값만 바꾸면
    화면이 안 바뀐다. key 에 세대 번호를 넣어 강제로 다시 그리게 한다.
    """
    src = _src("features_view.py")
    assert "review_gen" in src
    i = src.index("key=f\"feat_editor_")
    assert "review_gen" in src[i:i + 200], "편집표 key 에 세대 번호가 없습니다"


def test_risk_check_is_memoized():
    """위험 점검은 상관행렬을 만든다. 매 재실행마다 돌면 안 된다."""
    src = _src("features_view.py")
    assert "_risks_cached" in src
    assert "features.selection_risks(" in src
    # 화면 본문에서는 캐시 함수만 부른다
    body = src[src.index("def _review_gate("):]
    assert "features.selection_risks(" not in body, \
        "검토 화면이 캐시를 건너뛰고 직접 호출합니다"


# ── 3. 차트 다운샘플링 ──────────────────────────────────────
def test_thin_reduces_points_and_keeps_order():
    idx = pd.date_range("2025-01-01", periods=12948, freq="5min")
    s = pd.Series(np.arange(12948, dtype=float), index=idx)
    out = plots.thin(s)
    assert len(out) <= plots.MAX_POINTS
    assert out.index.is_monotonic_increasing
    assert out.index[0] == s.index[0]
    assert out.iloc[0] == s.iloc[0]


def test_thin_is_noop_when_small():
    s = pd.Series(np.arange(100, dtype=float))
    assert plots.thin(s) is s


def test_thin_pair_keeps_series_aligned():
    """두 시계열을 따로 솎으면 x 축이 어긋나 그림이 틀어진다."""
    idx = pd.date_range("2025-01-01", periods=9000, freq="5min")
    a = pd.Series(np.arange(9000, dtype=float), index=idx)
    b = pd.Series(np.arange(9000, dtype=float) * 2, index=idx)
    ta, tb = plots.thin_pair(a, b)
    assert len(ta) == len(tb)
    assert ta.index.equals(tb.index)
    assert np.allclose(tb.to_numpy(), ta.to_numpy() * 2)


def test_thin_does_not_mutate_original():
    """그리기용 사본만 줄여야 한다. 통계·모델은 항상 전체를 쓴다."""
    idx = pd.date_range("2025-01-01", periods=9000, freq="5min")
    s = pd.Series(np.arange(9000, dtype=float), index=idx)
    before = len(s)
    plots.thin(s)
    assert len(s) == before


def test_webgl_kicks_in_for_large_series():
    class FakeGo:
        Scatter = "svg"
        Scattergl = "webgl"
    assert plots._line(FakeGo, 100) == "svg"
    assert plots._line(FakeGo, plots.GL_THRESHOLD + 1) == "webgl"


def test_timeseries_charts_thin_their_input():
    """시계열을 그리는 함수는 솎아내기를 거쳐야 한다."""
    src = (ROOT / "core" / "plots.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    need = {"actual_vs_pred", "residual_series", "residual_band",
            "scatter_actual_pred", "anomaly_timeline", "whatif_compare"}
    missing = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in need:
            body = ast.unparse(node)
            if "thin(" not in body and "thin_pair(" not in body:
                missing.append(node.name)
    assert not missing, f"솎아내기 없이 그리는 함수: {missing}"


# ── 4. 단계 이동 ────────────────────────────────────────────
def _nav_src() -> str:
    return (ROOT / "app" / "nav.py").read_text(encoding="utf-8")


def test_every_step_has_a_guide_line():
    """각 단계가 '여기서 뭘 하는 곳인지' 한 줄 설명을 가져야 한다."""
    tree = ast.parse((ROOT / "app" / "state.py").read_text(encoding="utf-8"))
    steps = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "STEPS" for t in node.targets):
            steps = ast.literal_eval(node.value)
    assert steps

    guide = None
    for node in ast.parse(_nav_src()).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "GUIDE":
            guide = ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "GUIDE" for t in node.targets):
            guide = ast.literal_eval(node.value)
    assert guide is not None, "nav.GUIDE 를 읽지 못했습니다"

    missing = [k for _, k in steps if k not in guide]
    assert not missing, f"설명이 없는 단계: {missing}"
    assert all(v.strip() for v in guide.values())


def test_rail_and_next_button_share_one_source_of_truth():
    """사이드바와 '다음 단계' 버튼이 다른 값을 보면 화면이 튄다."""
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "nav.current()" in main
    assert 'S["_rail"] = idx' in main, "위젯 생성 전에 라디오 값을 맞추지 않습니다"
    assert "nav.header(step)" in main and "nav.footer(step)" in main


def test_blocker_covers_the_confirm_gate():
    """확정 전에는 다음으로 못 넘어가고, 그 이유가 안내돼야 한다."""
    src = _nav_src()
    i = src.index('elif key == "features":')
    block = src[i:src.index('elif key == "train":')]
    assert "S.X is None" in block
    assert "확정" in block


# ── 5. Streamlit 위젯 인자 유효성 ───────────────────────────
def test_no_non_emoji_icon_argument():
    """st.info/warning/error 의 icon= 은 진짜 이모지만 받는다.

    "◧" 같은 기하도형 문자를 넣으면 화면 하단에 StreamlitAPIException 이 크게
    뜬다. 이 프로젝트는 이모지를 안 쓰기로 했으므로 파라미터 자체를 안 쓴다.
    """
    bad = []
    for f in sorted(VIEWS.glob("*.py")) + [ROOT / "app" / "nav.py"]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "icon=" in code and "page_icon" not in code:
                bad.append(f"{f.name}:{i}  {code.strip()}")
    assert not bad, "icon= 인자를 쓰고 있습니다 (이모지만 허용됨):\n" + "\n".join(bad)


# ── 6. SQream ───────────────────────────────────────────────
def test_sqream_is_a_supported_dialect():
    from core import datasource as d
    assert "SQream" in d.DIALECTS
    assert d.DIALECTS["SQream"] == "sqream"
    assert d.DEFAULT_PORTS["sqream"] == 3108
    assert d.DRIVER_PACKAGES["sqream"] == "pysqream-sqlalchemy"


def test_sqream_url_is_built_correctly():
    from core import datasource as d
    url = d.build_url("sqream", "dm.internal", 3108, "master", "svc_ml", "p@ss w0rd")
    assert url.startswith("sqream://svc_ml:")
    assert "@dm.internal:3108/master" in url
    assert "p%40ss+w0rd" in url, "비밀번호가 URL 인코딩되지 않았습니다"
    assert "p@ss w0rd" not in url


def test_sqream_password_is_masked_in_display():
    from core import datasource as d
    url = d.build_url("sqream", "h", 3108, "master", "u", "secret")
    assert "secret" not in d.mask_url(url)


def test_every_dialect_has_a_driver_package_hint():
    from core import datasource as d
    missing = [v for v in d.DIALECTS.values() if v not in d.DRIVER_PACKAGES]
    assert not missing, f"드라이버 패키지 안내가 없는 방언: {missing}"


def test_connect_args_reach_the_engine():
    """SQream 의 clustered 는 URL 이 아니라 connect_args 로 가야 한다."""
    import ast
    src = (ROOT / "core" / "datasource.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "SqlAlchemySource")
    engine = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_engine")
    assert "connect_args=self.connect_args" in ast.unparse(engine)

    view = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    assert view.count("connect_args=connect_args") >= 2, \
        "접속 확인과 실행 양쪽에 connect_args 를 넘겨야 합니다"
