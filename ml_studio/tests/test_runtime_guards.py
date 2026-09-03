"""실행 중 터지는 것들에 대한 회귀 테스트.

이 파일에 모인 것은 전부 "화면을 열면 예외가 뜬다" 부류다. 기능 테스트로는
안 잡힌다 — 코드는 멀쩡하고, 특정 데이터 모양이나 라이브러리 버전에서만
터지기 때문이다. 그래서 데이터 모양을 직접 만들어 보거나, 소스에서 위험한
호출 형태를 찾는 방식으로 잡는다.

실제로 사용자 PC 에서 터졌거나, 터지기 직전이었던 것들:
  1. plotly 5 에서 사라진 titlefont= (잔차 진단 화면 전체가 죽음)
  2. st.column_config 의 잘못된 format 문자열 "%.2%"
  3. 빈 DataFrame 이 시간축 화면까지 흘러가 df.index[-1] 에서 IndexError
  4. 위젯 기본값이 min/max 밖으로 나가는 경우 (피처 3개, 짧은 구간 등)
  5. 전부 결측인 컬럼을 What-if 슬라이더에 넘겨 NaN 예외
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import features, plots, profiling, validation  # noqa: E402

VIEWS = ROOT / "app" / "views"
APP = ROOT / "app"
CORE = ROOT / "core"


def _py_files(*dirs: Path):
    for d in dirs:
        yield from sorted(d.glob("*.py"))


# ── 1. plotly 구형 API ──────────────────────────────────────
# plotly 5 에서 제거된 속성들. 남아 있으면 그 차트를 여는 순간 ValueError 가 난다.
DEAD_PLOTLY_ATTRS = ("titlefont", "titleside", "titlefontsize")


def test_no_removed_plotly_attributes():
    bad = []
    for f in _py_files(CORE, VIEWS):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for attr in DEAD_PLOTLY_ATTRS:
                if re.search(rf"\b{attr}\s*=", code):
                    bad.append(f"{f.name}:{i}  {code.strip()}")
    assert not bad, ("plotly 5 에서 제거된 속성입니다. title=dict(font=...) 형식을 쓰세요:\n"
                     + "\n".join(bad))


def test_residual_drift_builds_without_error():
    """제거된 속성이 있으면 이 호출에서 바로 터진다.

    위의 문자열 검사만으로는 부족하다. 실제로 그려 봐야 plotly 가 그 속성을
    받아 주는지 알 수 있다. plotly 가 없는 환경(설치가 막힌 컨테이너)에서는
    건너뛰되, 건너뛴 사실이 실행 결과에 남는다.
    """
    pytest.importorskip("plotly")
    table = pd.DataFrame({"구간": [1, 2, 3], "MAE": [0.5, 0.6, 0.4],
                          "std": [1.0, 1.2, 0.9]})
    fig = plots.residual_drift(table)
    assert fig.layout.yaxis2.title.text == "표준편차"


def test_every_plot_function_builds_with_a_minimal_frame():
    """차트 함수는 그려 봐야 인자 오류가 드러난다.

    이 컨테이너에는 plotly 가 없어 여기서는 건너뛴다. 사용자 PC 에서 한 번
    돌려 보면 20여 개 함수의 인자 오류가 한꺼번에 드러난다.
    """
    pytest.importorskip("plotly")
    idx = pd.date_range("2025-01-01", periods=200, freq="5min")
    a = pd.Series(np.sin(np.linspace(0, 20, 200)), index=idx)
    b = a + np.random.default_rng(0).normal(scale=0.05, size=200)

    plots.actual_vs_pred(a, b, train_end=idx[150])
    plots.residual_series(a, b)
    plots.scatter_actual_pred(a, b)
    plots.residual_band(pd.DataFrame(
        {"residual": (a - b).to_numpy(), "roll_mean": 0.0, "roll_std": 0.1}, index=idx))
    plots.residual_drift(pd.DataFrame(
        {"구간": [1, 2], "MAE": [0.1, 0.2], "std": [0.3, 0.4]}))


# ── 2. column_config format 문자열 ──────────────────────────
# streamlit 의 NumberColumn format 은 sprintf 형식이다. 변환자로 끝나지 않는
# "%.2%" 같은 문자열은 프런트에서 예외가 된다.
# 리터럴 퍼센트는 정확히 %% 여야 한다. 폭·정밀도를 끼운 "%.2%" 는 sprintf 가
# 해석하지 못한다. 한 덩어리 정규식으로 쓰면 그게 "정밀도 .2 + 변환자 %" 로
# 잘못 읽혀 통과해 버린다 — 이 검사기의 첫 판이 실제로 그랬다.
_SPRINTF = re.compile(r"%%|%[-+ #0']*\d*(?:\.\d+)?[bcdieEfgGosuxX]")


def test_number_column_formats_are_valid_sprintf():
    bad = []
    for f in _py_files(VIEWS):
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if "column_config" not in line and "NumberColumn" not in line:
                continue
            for m in re.finditer(r'format="([^"]*)"', line):
                spec = m.group(1)
                if "%" not in spec:      # "YYYY-MM-DD" 같은 날짜 형식은 대상 아님
                    continue
                stripped = _SPRINTF.sub("", spec)
                if "%" in stripped:
                    bad.append(f"{f.name}:{i}  format=\"{spec}\"")
    assert not bad, ("sprintf 로 해석되지 않는 format 문자열입니다 "
                     "(퍼센트 기호는 %% 로 씁니다):\n" + "\n".join(bad))


# ── 3. 빈 데이터 방어 ───────────────────────────────────────
def test_timeseries_panel_guards_empty_frame():
    """빈 프레임이 오면 위젯을 만들기 전에 빠져나가야 한다.

    df.index[-1] · raw.columns[0] 은 빈 프레임에서 IndexError 를 낸다.
    WHERE 조건이 좁은 쿼리에서 실제로 나오는 상황이다.
    """
    # 화면 문구가 아니라 **코드**를 기준으로 잡는다. 라벨을 다듬을 때마다
    # 테스트가 깨지면, 문구를 고치는 일 자체가 무서워진다.
    src = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _timeseries_panel("):]

    head = body[:body.index("guess = datasource.guess_time_column")]
    assert "raw.empty" in head, "빈 raw 를 막는 검사가 없습니다"

    pre = body[:body.index("freq = datasource.infer_freq")]
    assert "df.empty" in pre, "빈 df 를 막는 검사가 없습니다"
    assert pre.index("df.empty") > pre.index("raw.empty"), \
        "df 검사가 raw 검사보다 앞에 있습니다"


def test_empty_frame_stays_empty_through_profiling():
    """빈 프레임이 진단 함수에 들어가도 예외 없이 빈 결과가 나와야 한다."""
    empty = pd.DataFrame()
    assert profiling.find_correlated_pairs(empty).empty


# ── 4. 위젯 기본값 범위 ─────────────────────────────────────
def _widget_calls(path: Path):
    """number_input / slider 호출을 (파일, 줄, 소스) 로 뽑는다."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in ("number_input", "slider"):
            yield node, ast.unparse(node)


def test_no_widget_uses_a_bare_shape_expression_as_bound():
    """min/max 를 계산해 넘길 때는 반대쪽 끝으로 한 번 더 눌러 줘야 한다.

    number_input(min, max, value) 에서 value 가 범위 밖이면 streamlit 은
    StreamlitAPIException 을 던진다. 피처가 3개뿐인 데이터, 200행보다 짧은
    구간에서 실제로 일어났다. 계산식이 들어간 자리에는 max()/min() 클램프가
    같이 있어야 한다.
    """
    suspicious = []
    for f in _py_files(VIEWS):
        for node, src in _widget_calls(f):
            args = node.args
            if len(args) < 4:                     # (label, min, max, value)
                continue
            bound_src = " ".join(ast.unparse(a) for a in args[1:4])
            has_shape = re.search(r"(shape\[1\]|len\(|\.size)", bound_src)
            has_clamp = ("max(" in bound_src or "min(" in bound_src)
            if has_shape and not has_clamp:
                suspicious.append(f"{f.name}: {src[:90]}")
    assert not suspicious, ("범위가 데이터 크기로 정해지는데 클램프가 없습니다:\n"
                            + "\n".join(suspicious))


def test_pca_slider_bounds_are_clamped():
    src = (VIEWS / "train_view.py").read_text(encoding="utf-8")
    i = src.index('n_comp = st.slider("주성분 개수"')
    window = src[max(0, i - 400):i + 120]
    assert "pc_max" in window and "min(3, pc_max)" in window


def test_shap_sample_cap_is_clamped():
    src = (VIEWS / "explain_view.py").read_text(encoding="utf-8")
    i = src.index('n_max = c2.number_input("표본 상한"')
    window = src[max(0, i - 300):i + 150]
    assert "n_hi" in window, "짧은 구간에서 기본값이 하한 아래로 내려갑니다"


# ── 5. NaN 이 위젯에 들어가는 경로 ──────────────────────────
def test_whatif_skips_all_nan_features():
    """전부 결측인 컬럼은 슬라이더를 만들지 않고 건너뛰어야 한다."""
    src = (VIEWS / "whatif_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _scenario("):src.index("def _sweep(")]
    assert "np.isfinite" in body, "NaN 검사가 없습니다"
    assert "continue" in body, "NaN 컬럼을 건너뛰지 않습니다"
    assert "skipped" in body, "건너뛴 피처를 사용자에게 알리지 않습니다"


def test_all_nan_column_produces_nan_bounds():
    """방어가 왜 필요한지 — 실제로 NaN 이 나오는지 확인한다."""
    s = pd.Series([np.nan] * 50, dtype=float)
    assert not np.isfinite(float(s.mean()))
    assert not np.isfinite(float(s.min()))


# ── 6. 예외를 삼키지 않는 경로 ──────────────────────────────
def test_long_running_view_calls_are_wrapped():
    """오래 도는 core 호출은 화면에서 try 로 감싸야 한다.

    감싸지 않으면 모델 하나가 터질 때 화면 전체가 빨간 트레이스로 덮이고,
    그때까지의 상태를 잃는다.
    """
    checks = [
        ("diagnostics_view.py", "train.random_vs_time("),
        ("features_view.py", "validation.build_split(cfg_split, X_all.index)"),
    ]
    for name, needle in checks:
        src = (VIEWS / name).read_text(encoding="utf-8")
        i = src.index(needle)
        before = src[max(0, i - 400):i]
        assert "try:" in before, f"{name} 의 {needle} 가 try 밖에 있습니다"


# ── 7. 성능 회귀 ────────────────────────────────────────────
def test_fold_selector_skips_mutual_info_when_no_top_k():
    """top_k 가 없으면 MI 는 컷오프에 안 쓰인다. 계산하면 안 된다.

    MI 는 k-NN 기반이라 행 수·열 수 모두에 비례한다. 50만 행 × 200열 폴드
    5개면 순수한 낭비가 수십 분이다.
    """
    src = (CORE / "features.py").read_text(encoding="utf-8")
    cls = src[src.index("class FoldSelector"):src.index("def jaccard_stability")]
    assert "compute_mi=bool(self.top_k)" in cls


def test_compute_mi_false_leaves_selection_unchanged():
    """MI 를 건너뛰어도 top_k 가 없으면 남는 피처가 같아야 한다.

    같지 않으면 폴드 내부 선별과 화면 선별의 결과가 갈린다.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(300, 12))
    A[:, 3] = A[:, 2] * 1.0001            # 상관 중복
    A[:, 7] = 5.0                          # 분산 0
    y = A[:, 0] * 2 + rng.normal(size=300) * 0.1
    names = [f"f{i}" for i in range(12)]

    with_mi, _ = features.select_core(A, y, names, compute_mi=True)
    without, _ = features.select_core(A, y, names, compute_mi=False)
    assert with_mi == without


def test_compute_mi_false_actually_skips_the_column():
    A = np.random.default_rng(1).normal(size=(200, 5))
    y = A[:, 0]
    names = [f"f{i}" for i in range(5)]
    _, rep = features.select_core(A, y, names, compute_mi=False)
    assert rep["mutual_info"].isna().all(), "MI 를 계산하지 않기로 했는데 값이 있습니다"

    _, rep2 = features.select_core(A, y, names, compute_mi=True)
    assert rep2["mutual_info"].notna().any(), "기본 경로에서는 MI 가 있어야 합니다"


def test_fold_selector_matches_the_mi_path_on_realistic_shapes():
    """실제 학습 경로(FoldSelector, top_k=None)가 MI 계산 경로와 같은 답을 내야 한다.

    합성 데이터 하나로는 부족하다. 상관 중복·분산 0·결측이 섞인 여러 크기에서
    한 번이라도 갈리면 폴드 내부 선별을 못 믿는다.
    """
    rng = np.random.default_rng(7)
    for n, k in ((400, 30), (1500, 60)):
        A = rng.normal(size=(n, k))
        A[:, 5] = A[:, 4] * 1.00001          # 상관 중복
        A[:, 9] = 3.0                         # 분산 0
        A[:20, 11] = np.nan                   # 결측
        y = A[:, 0] * 2 + A[:, 3] + rng.normal(size=n) * 0.2

        skipped = features.FoldSelector(top_k=None).fit(A, y).selected_index_
        with_mi, _ = features.select_core(
            A, y, [str(i) for i in range(k)], top_k=None, compute_mi=True)
        assert list(skipped) == list(with_mi), \
            f"{n}행 × {k}열 에서 선별 결과가 갈립니다"


def test_top_k_still_needs_mi():
    """top_k 가 있으면 MI 로 잘라야 하므로 계산을 건너뛰면 안 된다."""
    A = np.random.default_rng(2).normal(size=(200, 8))
    y = A[:, 3] * 3
    names = [f"f{i}" for i in range(8)]
    keep, rep = features.select_core(A, y, names, top_k=3)
    assert len(keep) == 3
    assert rep.loc[rep["kept"], "mutual_info"].notna().all()


def test_duplicated_index_is_hashed_once():
    """duplicated() 를 두 번 부르면 50만 행 해싱을 두 번 한다."""
    src = (CORE / "validation.py").read_text(encoding="utf-8")
    body = src[src.index("def leakage_checklist"):src.index("return pd.DataFrame(checks)")]
    assert body.count("index.duplicated()") == 1, \
        "duplicated() 를 한 번만 계산해 재사용하세요"


def test_leakage_checklist_result_is_unchanged_by_the_speedup():
    """빠르게 만든 뒤에도 판정이 같아야 한다. 격리 항목 두 개를 직접 확인한다."""
    idx = pd.date_range("2025-01-01", periods=1000, freq="5min")
    train_idx = np.arange(0, 600)
    test_idx = np.arange(620, 800)
    unseen_idx = np.arange(820, 1000)

    tbl = validation.leakage_checklist(
        idx, train_idx, test_idx, ["a", "b"], "y", None,
        gap=20, max_lookback=12,
        selection_idx=train_idx, unseen_idx=unseen_idx)
    got = dict(zip(tbl["항목"], tbl["결과"]))
    assert got["선별 구간 격리"] == "통과"
    assert got["Final Unseen 격리"] == "통과"
    assert got["gap 확보"] == "통과"

    # 선별이 검증 구간을 침범하면 잡아야 한다
    bad = validation.leakage_checklist(
        idx, train_idx, test_idx, ["a"], "y", None,
        gap=20, max_lookback=12,
        selection_idx=np.arange(0, 700), unseen_idx=unseen_idx)
    assert dict(zip(bad["항목"], bad["결과"]))["선별 구간 격리"] == "실패"

    # unseen 이 학습과 겹치면 잡아야 한다
    bad2 = validation.leakage_checklist(
        idx, np.arange(0, 900), test_idx, ["a"], "y", None,
        gap=20, max_lookback=12,
        selection_idx=None, unseen_idx=unseen_idx)
    assert dict(zip(bad2["항목"], bad2["결과"]))["Final Unseen 격리"] == "실패"


def test_correlation_sampling_is_opt_in_and_bounded():
    """표본은 명시적으로 켤 때만 쓰고, 켜면 실제로 줄어야 한다."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=20000)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(size=20000) * 0.001,
                       "c": rng.normal(size=20000)})
    full = profiling.find_correlated_pairs(df, 0.95)
    sampled = profiling.find_correlated_pairs(df, 0.95, max_rows=2000)
    assert set(zip(full["feature_a"], full["feature_b"])) == \
           set(zip(sampled["feature_a"], sampled["feature_b"])), \
        "표본으로도 같은 중복 쌍을 찾아야 합니다"


def test_prep_view_memoizes_correlation():
    src = (VIEWS / "prep_view.py").read_text(encoding="utf-8")
    assert "_pairs_cached" in src
    body = src[src.index("def render("):]
    assert "profiling.find_correlated_pairs(" not in body, \
        "화면 본문이 캐시를 건너뛰고 직접 호출합니다"


def test_csv_downloads_do_not_encode_on_every_rerun():
    """download_button 에 to_csv() 를 직접 넘기면 재실행마다 다시 만든다."""
    bad = []
    for f in _py_files(VIEWS):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "download_button" in code and "to_csv()" in code:
                bad.append(f"{f.name}:{i}")
    assert not bad, ("theme.csv_download() 를 쓰세요 (바이트를 캐시합니다): "
                     + ", ".join(bad))


# ── 8. Final Unseen 접근권 (누수 방지) ─────────────────────
# 여기 있는 것들은 "터지지 않고 틀린 숫자를 보여주는" 부류다. 화면이 정상으로
# 보이므로 사용자가 알아챌 방법이 없다. 그래서 코드로만 막을 수 있다.

def test_evaluate_unseen_refuses_to_run_without_a_guard():
    """가드 없이 여는 뒷문이 있으면 안 된다.

    예전에는 guard=None 이면 횟수 제한 없이 통과했다. 화면에서 챔피언을 바꾸면
    세션의 guard 가 None 이 되는 경로가 있었으므로, 그 상태로는 Final Unseen 을
    몇 번이든 열어 가장 좋은 점수가 나올 때까지 모델을 고를 수 있었다.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    from core import models, preprocess
    from core import train as T

    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=300, freq="5min")
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"), index=idx)
    y = pd.Series(X["a"] * 2 + rng.normal(scale=0.1, size=300), index=idx)
    num, cat = preprocess.split_column_types(X)
    pipe = Pipeline([
        ("prep", preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())),
        ("est", Ridge()),
    ]).fit(X.iloc[:200], y.iloc[:200])
    cfg = T.TrainConfig(task=models.TASK_REGRESSION)
    unseen = np.arange(250, 300)

    with pytest.raises(T.UnseenAccessError):
        T.evaluate_unseen(pipe, X, y, unseen, cfg, guard=None)

    # 가드를 주면 정확히 한 번만 통과한다
    guard = T.UnseenGuard(unseen)
    out = T.evaluate_unseen(pipe, X, y, unseen, cfg, guard, who="champ")
    assert out["unseen_rows"] == len(unseen)
    with pytest.raises(T.UnseenAccessError):
        T.evaluate_unseen(pipe, X, y, unseen, cfg, guard, who="champ2")

    # unseen 이 아예 없으면 (2분할 호환) 가드 없이도 조용히 빈 결과
    assert T.evaluate_unseen(pipe, X, y, np.array([], dtype=int), cfg, guard=None) == {}


def test_unseen_guard_belongs_to_the_split_not_the_training_run():
    """학습을 다시 돌린다고 접근권이 되살아나면 안 된다.

    guard 가 invalidate('train') 으로 지워지면, 사용자는 [학습 실행] 을 다시 눌러
    Final Unseen 을 원하는 만큼 열어 볼 수 있다. 그러면 '미접촉 구간' 이라는
    말 자체가 성립하지 않는다.
    """
    src = (APP / "state.py").read_text(encoding="utf-8")
    body = src[src.index("def invalidate("):src.index("def ready(")]

    sel = body[body.index("select_out = "):body.index("chains = ")]
    assert '"unseen_guard"' in sel, "unseen_guard 가 분할(select_out) 쪽에 없습니다"

    tr = body[body.index("train_out = "):body.index("select_out = ")]
    assert '"unseen_guard"' not in tr, \
        "unseen_guard 가 train_out 에 있습니다 — 학습마다 접근권이 되살아납니다"

    chain = body[body.index('"train": ['):]
    chain = chain[:chain.index("]")]
    assert "unseen_guard" not in chain, \
        "invalidate('train') 이 접근권을 되돌려 줍니다"


def test_only_the_split_step_creates_the_guard():
    """가드를 만드는 곳은 분할이 확정되는 3단계 한 곳이어야 한다."""
    makers = []
    for f in sorted(VIEWS.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if "UnseenGuard(" in code:
                makers.append(f.name)
    assert makers == ["features_view.py"], \
        f"가드를 만드는 화면이 3단계 말고 또 있습니다: {makers}"


def test_spent_guard_is_explained_before_the_button():
    """이미 쓴 접근권이면 버튼을 눌러 예외로 알게 하지 말고 미리 말해야 한다."""
    src = (VIEWS / "train_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _unseen_panel("):]
    assert "access_count" in body, "접근권이 남았는지 확인하지 않습니다"
    i = body.index("access_count")
    j = body.index('st.button("Final Unseen 평가 실행"')
    assert i < j, "버튼을 먼저 그리고 나중에 확인합니다"


def test_run_artifacts_are_cleared_when_upstream_changes():
    """리포트·manifest 는 완료된 실행 하나를 가리킨다. 앞 단계가 바뀌면 놓아야 한다.

    안 지우면 Champion-Challenger 가 예전 manifest 의 데이터 지문을 '지금 지문'
    으로 삼아 비교한다. 터지지 않고 틀린 답을 낸다.
    """
    src = (APP / "state.py").read_text(encoding="utf-8")
    body = src[src.index("def invalidate("):src.index("def ready(")]
    for key in ("run_dir", "saved", "manifest", "report_html", "challenger"):
        assert f'"{key}"' in body, f"{key} 를 어느 체인에서도 지우지 않습니다"

    chain = body[body.index('"train": ['):]
    chain = chain[:chain.index("]")]
    assert "report_out" in chain, "챔피언이 바뀌어도 리포트가 남습니다"


def test_auto_mode_clears_the_previous_run_first():
    """수동으로 돌린 뒤 Auto 를 돌리면 예전 산출물이 남으면 안 된다."""
    src = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _apply("):]
    head = body[:body.index("S.kept = res.kept")]
    assert 'state.invalidate("data")' in head, \
        "Auto 적용 전에 아래 단계를 비우지 않습니다"


def test_no_view_creates_its_own_split():
    """분할은 3단계에서 한 번만 정한다. 뒤 단계가 다시 나누면 선별 구간이 샌다."""
    offenders = []
    for f in sorted(VIEWS.glob("*.py")):
        if f.name in ("features_view.py", "data_view.py"):   # 3단계 · Auto
            continue
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"validation\.(build_split|three_way_split)\(", code):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"3단계 밖에서 분할을 다시 만듭니다: {offenders}"


# ── 8-b. 라이브러리 버전 호환 ───────────────────────────────
def test_preprocessing_matrix_builds_and_fits():
    """전처리 조합 8가지를 실제로 학습시켜 본다.

    가상 데이터에는 범주형이 거의 없어서 스모크 테스트가 OneHotEncoder 경로를
    거의 지나가지 않는다. 그런데 sklearn 은 이 언저리에서 인자를 자주 바꾼다
    (sparse= 는 1.4 에서 삭제됨). 조합을 직접 돌려 봐야 안다.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    from core import preprocess

    idx = pd.date_range("2025-01-01", periods=300, freq="5min")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=300),
        "b": rng.normal(size=300),
        "status": rng.choice(["NORMAL", "WARN", "TRIP"], 300),
    }, index=idx)
    X.loc[X.index[:10], "a"] = np.nan          # 결측 대치 경로도 지난다
    y = pd.Series(X["a"].fillna(0) * 2 + rng.normal(size=300), index=idx)

    num, cat = preprocess.split_column_types(X)
    assert cat == ["status"], "범주형 컬럼을 못 잡았습니다"

    for enc in ("onehot", "ordinal"):
        for scaler in ("standard", "robust", "minmax", "none"):
            cfg = preprocess.PreprocessConfig(categorical_encoding=enc,
                                              scaler=scaler, clip_outliers=True)
            pre = preprocess.build_preprocessor(num, cat, cfg)
            pipe = Pipeline([("prep", pre), ("est", Ridge())]).fit(X, y)
            pred = pipe.predict(X)
            assert len(pred) == len(X)
            assert np.isfinite(pred).all(), f"{enc}/{scaler} 에서 NaN 예측"
            assert len(pre.get_feature_names_out()) >= len(num)


def test_onehot_encoder_uses_the_current_argument_name():
    """sparse= 는 sklearn 1.4 에서 삭제됐다. sparse_output= 을 먼저 써야 한다."""
    src = (CORE / "preprocess.py").read_text(encoding="utf-8")
    i = src.index("OneHotEncoder(")
    j = src.index("sparse=", i)
    assert src.index("sparse_output=", i) < j, \
        "구버전 인자(sparse=)를 먼저 시도합니다"
    assert "except TypeError" in src[i:j], "구버전 대비 fallback 이 없습니다"


# ── 8-c. 스크립트가 gap 을 옳게 계산하는가 ──────────────────
def test_no_script_uses_max_lags_as_the_gap():
    """gap 은 파생 **전체**의 lookback 이어야 한다. max(lags) 만 보면 안 된다.

    lags=[1,3] · rolling=[6] 이면 max(lags)=3 인데 실제 lookback 은 6 이다.
    gap 3 으로 나누면 학습 마지막 행과 홀드아웃 첫 행의 입력 창이 같은 원자료를
    공유한다 — 그게 바로 누수다.

    smoke_test.py 는 진작 고쳤는데 quick_check.py 에 옛 계산이 남아 있었고,
    사용자 PC 에서 '누수 점검 실패' 로 터진 뒤에야 발견했다. 이 환경에서는
    --quick 옵션이 그 단계를 건너뛰고 있어서 한 번도 안 돌아 봤다.
    """
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")) + sorted((ROOT / "tests").glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"gap\s*=\s*max\s*\(", code):
                bad.append(f"{f.name}:{i}  {code.strip()}")
    assert not bad, ("gap 을 max(lags) 로 잡고 있습니다. "
                     "features.warmup_rows(cfg) 를 쓰세요:\n" + "\n".join(bad))


def test_warmup_rows_covers_every_lookback_source():
    """warmup_rows 가 lag·rolling·ewm·diff 를 모두 반영해야 한다."""
    cfg = features.FeatureConfig(lags=[1, 3], rolling_windows=[6],
                                 rolling_stats=["mean"], ewm_spans=[], diffs=[1])
    assert features.warmup_rows(cfg) >= 6, "rolling 창을 빼먹었습니다"

    cfg2 = features.FeatureConfig(lags=[1], rolling_windows=[], rolling_stats=[],
                                  ewm_spans=[24], diffs=[])
    assert features.warmup_rows(cfg2) >= 24, "ewm span 을 빼먹었습니다"

    cfg3 = features.FeatureConfig(lags=[48], rolling_windows=[6],
                                  rolling_stats=["mean"], ewm_spans=[], diffs=[])
    assert features.warmup_rows(cfg3) >= 48, "lag 를 빼먹었습니다"


def test_quick_check_passes_its_own_leakage_gate():
    """축소 점검이 자기 누수 점검을 통과하는 설정으로 짜여 있어야 한다.

    스크립트를 통째로 돌리면 몇 분이 걸리므로, 여기서는 같은 설정으로
    점검표만 만들어 본다.
    """
    src = (SCRIPTS / "quick_check.py").read_text(encoding="utf-8")
    assert "features.warmup_rows(fcfg)" in src, \
        "quick_check 가 gap 을 warmup_rows 로 잡지 않습니다"

    # 그 설정으로 실제 판정이 통과하는지 확인한다
    cfg = features.FeatureConfig(lags=[1, 3], rolling_windows=[6],
                                 rolling_stats=["mean"], ewm_spans=[], diffs=[1])
    gap = features.warmup_rows(cfg)
    idx = pd.date_range("2025-01-01", periods=2000, freq="5min")
    tr = np.arange(0, 1500)
    te = np.arange(1500 + gap, 2000)
    tbl = validation.leakage_checklist(idx, tr, te, ["a", "b"], "y", None,
                                       gap=gap, max_lookback=gap)
    assert not (tbl["결과"] == "실패").any(), \
        f"quick_check 설정이 누수 점검을 통과하지 못합니다:\n{tbl.to_string(index=False)}"


# ── 8-d. 점검 스크립트가 제품 규격과 맞는가 ─────────────────
def test_verify_env_uses_the_real_frame_shapes():
    """점검 스크립트의 가짜 데이터가 제품이 실제로 내는 컬럼과 같아야 한다.

    다르면 점검만 KeyError 로 실패하고 제품은 멀쩡하다 — 가짜 경보다.
    실제로 shap_dependence · whatif_compare · pdp_curve 세 개가 그랬다.
    """
    src = (SCRIPTS / "verify_env.py").read_text(encoding="utf-8")
    required = {
        "shap_dependence": ["timestamp", "feature_value", "shap_value"],
        "whatif_compare": ["baseline", "scenario"],
        "pdp_curve": ["prediction", "p10", "p90"],
    }
    missing = []
    for fn, cols in required.items():
        for c in cols:
            if f'"{c}"' not in src:
                missing.append(f"{fn} 에 필요한 '{c}' 컬럼이 점검 데이터에 없습니다")
    assert not missing, "\n".join(missing)


def test_missing_driver_is_a_skip_not_a_failure():
    """드라이버 미설치는 고칠 결함이 아니라 건너뛸 일이다."""
    src = (SCRIPTS / "verify_env.py").read_text(encoding="utf-8")
    body = src[src.index("def _is_missing_library"):src.index("def check(")]
    assert "NoSuchModuleError" in body, \
        "SQLAlchemy 의 드라이버 없음(NoSuchModuleError)을 실패로 잡고 있습니다"


# ── 8-e. 글씨가 보이는가 ────────────────────────────────────
def test_primary_button_text_is_forced_white():
    """남색 버튼에 검정 글씨가 나오던 문제.

    streamlit 은 버튼 라벨을 button 안쪽 <p>/<div> 에 넣고 거기에
    textColor(거의 검정)를 따로 먹인다. button 에만 color 를 주면 안쪽이
    이기고 **글자가 안 보인다.** 자손 선택자와 !important 가 둘 다 있어야 한다.
    """
    css = (APP / "theme.py").read_text(encoding="utf-8")
    i = css.index("/* 버튼")
    block = css[i:css.index("/* 입력 요소", i)]

    assert 'button[kind="primary"] *' in block, \
        "버튼 안쪽 요소까지 색을 주지 않습니다 — 라벨이 검정으로 남습니다"
    assert "color: #FFFFFF !important" in block, \
        "!important 가 없으면 streamlit 기본 색이 이깁니다"
    # 새 streamlit DOM 도 함께 잡아야 한다
    assert "stBaseButton-primary" in block, \
        "최신 streamlit 의 버튼 선택자를 잡지 않습니다"


def test_config_primary_color_is_dark_enough_to_need_white_text():
    """전제 확인 — primaryColor 가 어두우니 흰 글씨가 맞다.

    나중에 밝은 색으로 바꾸면 위 규칙이 오히려 안 보이게 만든다.
    """
    import re as _re

    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    m = _re.search(r'primaryColor\s*=\s*"#([0-9A-Fa-f]{6})"', cfg)
    assert m, "primaryColor 를 읽지 못했습니다"
    r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    assert lum < 128, (f"primaryColor 가 밝아졌습니다(밝기 {lum:.0f}). "
                       "흰 글씨 규칙이 오히려 안 보이게 만듭니다 — "
                       "theme.py 의 버튼 색 규칙을 함께 손보세요.")


def test_sidebar_buttons_stay_light_on_dark_rail():
    """사이드바는 어두운 배경이라 밝은 글씨여야 한다."""
    css = (APP / "theme.py").read_text(encoding="utf-8")
    i = css.index('[data-testid="stSidebar"] .stButton')
    block = css[i:i + 900]
    assert "!important" in block, "사이드바 버튼 색이 덮일 수 있습니다"


# ── 9. 한글 윈도우 콘솔 ─────────────────────────────────────
SCRIPTS = ROOT / "scripts"


def test_em_dash_is_not_encodable_in_cp949():
    """왜 아래 테스트가 필요한지 — 전제부터 확인한다.

    한글 윈도우의 기본 코덱 cp949 에는 em dash(—, U+2014)가 없다. 이 프로젝트는
    출력에서 이 문자를 아주 많이 쓴다.
    """
    with pytest.raises(UnicodeEncodeError):
        "—".encode("cp949")
    # 한글 자체는 cp949 에 있다. 문제는 기호다.
    assert "전 단계 통과".encode("cp949")


def test_every_printing_script_protects_its_console():
    """직접 실행되는 스크립트는 출력 인코딩을 스스로 지켜야 한다.

    콘솔 창에 바로 찍을 때는 파이썬이 UTF-16 경로를 써서 문제가 없다. 하지만
    출력을 파일이나 파이프로 넘기는 순간 cp949 로 떨어지고, em dash 하나에
    UnicodeEncodeError 로 죽는다. 결과를 로그로 남기려다 실행이 실패한다.
    """
    entries = [p for p in sorted(SCRIPTS.glob("*.py"))] + [ROOT / "tests" / "run_tests.py"]
    missing = []
    for f in entries:
        src = f.read_text(encoding="utf-8")
        if "__main__" not in src and "def main(" not in src:
            continue                                   # 직접 실행하는 파일이 아니다
        if not re.search(r"^\s*_enable_utf8\(\)\s*$", src, re.M):
            missing.append(f.name)
    assert not missing, ("출력 인코딩을 지키지 않는 실행 파일: " + ", ".join(missing))


def test_setup_passes_utf8_to_child_processes():
    """자식 프로세스는 부모의 reconfigure 를 물려받지 않는다."""
    src = (SCRIPTS / "setup.py").read_text(encoding="utf-8")
    body = src[src.index("def run("):src.index("def venv_python(")]
    assert "PYTHONIOENCODING" in body, "자식에게 출력 인코딩을 넘기지 않습니다"
    assert "env" in body


def test_project_files_are_written_with_an_explicit_encoding():
    """encoding 없이 파일을 쓰면 한글 윈도우에서 cp949 로 저장된다.

    그렇게 저장된 리포트·설정은 다른 PC 에서 열면 깨진다.
    """
    bad = []
    for f in sorted(list((ROOT / "core").glob("*.py")) + list(SCRIPTS.glob("*.py"))):
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\b(open|write_text|read_text)\(", code) and "encoding" not in code:
                # 여러 줄로 나뉜 호출은 다음 줄까지 본다
                nxt = src.splitlines()[i:i + 1]
                if nxt and "encoding" in nxt[0]:
                    continue
                if "def open(" in code or ".open(" in code:
                    continue                            # UnseenGuard.open 등
                bad.append(f"{f.name}:{i}  {code.strip()[:70]}")
    assert not bad, "encoding 을 지정하지 않은 파일 입출력:\n" + "\n".join(bad)


# ── 10. 사용자 안내 ─────────────────────────────────────────
def test_full_reset_needs_confirmation():
    """한 번 누르면 전부 사라지는 버튼은 두 단계여야 한다."""
    src = (APP / "main.py").read_text(encoding="utf-8")
    assert "_reset_armed" in src
    i = src.index("전체 초기화")
    assert "지웁니다" in src[max(0, i - 600):i + 600]


def test_invalidate_tells_the_user_what_was_lost():
    src = (APP / "state.py").read_text(encoding="utf-8")
    body = src[src.index("def invalidate("):src.index("def ready(")]
    assert "st.toast" in body, "무엇이 지워졌는지 알리지 않습니다"
    assert "챔피언" in body


def test_predict_does_not_block_the_explain_step():
    """SHAP 은 챔피언만 있으면 된다. 예측을 강제로 시키면 안 된다."""
    src = (APP / "nav.py").read_text(encoding="utf-8")
    blocker = src[src.index("def _blocker("):src.index("def _advice(")]
    assert "predictions" not in blocker, "예측이 아직 다음 단계를 막고 있습니다"
    advice = src[src.index("def _advice("):src.index("def _index(")]
    assert "predictions" in advice, "권유 문구로도 남아 있지 않습니다"


def test_csv_encoding_error_gets_a_useful_message():
    src = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    assert "UnicodeDecodeError" in src
    assert "cp949" in src


def test_unsupervised_warns_when_features_are_unconfirmed():
    src = (VIEWS / "train_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _unsupervised("):]
    assert "confirmed" in body
    assert "3단계" in body[:1500]


def test_unseen_scope_is_flagged_in_predict_view():
    src = (VIEWS / "predict_view.py").read_text(encoding="utf-8")
    assert "unseen_start" in src
    i = src.index('if st.button("예측 실행"')
    assert "미접촉" in src[max(0, i - 900):i]


def test_ensemble_panel_explains_oof_in_plain_words():
    src = (VIEWS / "train_view.py").read_text(encoding="utf-8")
    body = src[src.index("def _ensemble_panel("):]
    assert "out-of-fold" in body
    assert "학습에 쓰지 않은" in body


def test_leakage_badge_carries_a_tooltip():
    theme_src = (APP / "theme.py").read_text(encoding="utf-8")
    assert "def badge(text: str, kind: str = \"idle\", help: str = \"\")" in theme_src
    main_src = (APP / "main.py").read_text(encoding="utf-8")
    i = main_src.index("누수 가드")
    assert "help=" in main_src[i:i + 600]


# ── 11. 세로형(long) 데이터 ─────────────────────────────────
# PI · IP.21 같은 히스토리언은 long 이 기본이다. 사내 데이터마트에서 뽑으면
# 이 모양으로 나오는 경우가 흔한데, 모델은 wide 를 전제로 한다.

def _long_frame(n=400, tags=("FLOW_01", "TEMP_01", "PRESS_01", "Y_YIELD")):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    return pd.concat(
        [pd.DataFrame({"tag_time": idx, "tag_name": t,
                       "value": rng.normal(loc=m, size=n)})
         for t, m in zip(tags, (50, 65, 3, 90))],
        ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


def test_long_layout_is_detected():
    from core import datasource as d

    r = d.detect_layout(_long_frame())
    assert r["layout"] == "long", f"세로형을 못 알아봤습니다: {r}"
    assert r["time_col"] == "tag_time"
    assert r["tag_col"] == "tag_name"
    assert r["value_col"] == "value"
    assert r["n_tags"] == 4
    assert r["reasons"], "판단 근거를 남기지 않았습니다"


def test_wide_layout_is_not_mistaken_for_long():
    """오판이 더 나쁘다. 멀쩡한 wide 를 돌리면 데이터가 망가진다."""
    from core import datasource as d

    idx = pd.date_range("2025-01-01", periods=400, freq="5min")
    rng = np.random.default_rng(1)
    wide = pd.DataFrame({"tag_time": idx, "flow": rng.normal(size=400),
                         "temp": rng.normal(size=400), "y": rng.normal(size=400)})
    assert d.detect_layout(wide)["layout"] == "wide"

    # 상태 컬럼이 섞여 있어도 wide 다 — 시각이 안 겹치기 때문
    wide2 = wide.assign(status=rng.choice(["NORMAL", "WARN", "TRIP"], 400))
    assert d.detect_layout(wide2)["layout"] == "wide", \
        "상태 컬럼을 태그 컬럼으로 오해했습니다"


def test_long_to_wide_round_trips():
    from core import datasource as d

    long = _long_frame()
    wide = d.long_to_wide(long, "tag_time", "tag_name", "value")

    assert list(wide.columns)[0] == "tag_time"
    assert set(wide.columns) - {"tag_time"} == {"FLOW_01", "TEMP_01",
                                                "PRESS_01", "Y_YIELD"}
    assert len(wide) == 400
    assert wide["tag_time"].is_monotonic_increasing, "시간순 정렬이 안 됐습니다"

    # 값이 실제로 보존됐는지 — 한 시점을 직접 대조한다
    t0 = long["tag_time"].min()
    for tag in ("FLOW_01", "TEMP_01"):
        want = float(long[(long["tag_time"] == t0) &
                          (long["tag_name"] == tag)]["value"].iloc[0])
        got = float(wide.loc[wide["tag_time"] == t0, tag].iloc[0])
        assert abs(want - got) < 1e-9, f"{tag} 값이 바뀌었습니다"


def test_long_to_wide_feeds_to_timeseries():
    """돌린 결과가 다음 단계에 그대로 들어가야 한다."""
    from core import datasource as d

    wide = d.long_to_wide(_long_frame(), "tag_time", "tag_name", "value")
    ts = d.to_timeseries(wide, "tag_time")
    assert isinstance(ts.index, pd.DatetimeIndex)
    assert ts.index.is_monotonic_increasing
    assert "Y_YIELD" in ts.columns
    assert len(ts) == 400


def test_long_to_wide_handles_duplicate_timestamps():
    """히스토리언은 같은 초에 두 번 찍는 일이 있다. 그냥 pivot 하면 거기서 죽는다."""
    from core import datasource as d

    long = _long_frame()
    dup = pd.concat([long, long.head(40)], ignore_index=True)
    wide = d.long_to_wide(dup, "tag_time", "tag_name", "value", agg="mean")
    assert wide.attrs["long_to_wide"]["같은 시각·태그 중복"] == 40
    assert len(wide) == 400, "중복 때문에 시점이 늘어났습니다"


def test_long_to_wide_can_pick_tags():
    from core import datasource as d

    wide = d.long_to_wide(_long_frame(), "tag_time", "tag_name", "value",
                          tags=["FLOW_01", "Y_YIELD"])
    assert set(wide.columns) == {"tag_time", "FLOW_01", "Y_YIELD"}


def test_long_to_wide_rejects_bad_input():
    from core import datasource as d

    long = _long_frame()
    with pytest.raises(d.LayoutError):
        d.long_to_wide(long, "tag_time", "없는칸", "value")
    with pytest.raises(d.LayoutError):
        d.long_to_wide(long, "tag_time", "tag_name", "tag_name")
    with pytest.raises(d.LayoutError):
        d.long_to_wide(long.assign(value="글자"), "tag_time", "tag_name", "value")
    with pytest.raises(d.LayoutError):
        d.long_to_wide(long, "tag_time", "tag_name", "value", tags=["없는태그"])


def test_tag_inventory_lists_every_tag():
    from core import datasource as d

    inv = d.tag_inventory(_long_frame(), "tag_time", "tag_name", "value")
    assert len(inv) == 4
    assert set(inv["태그"]) == {"FLOW_01", "TEMP_01", "PRESS_01", "Y_YIELD"}
    assert (inv["줄 수"] == 400).all()
    assert (inv["숫자로 읽힌 비율"] == 1.0).all()


def test_layout_panel_blocks_until_converted():
    """세로형인 채로 다음 단계에 넘어가면 안 된다.

    시각이 중복된 표를 시계열로 만들면 태그가 서로 덮어써서 데이터가 망가진다.
    """
    src = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    assert "_layout_panel" in src
    body = src[src.index("def _layout_panel("):src.index("def _timeseries_panel(")]
    assert "return False" in body, "돌리기 전에 막지 않습니다"
    # render 는 통과했을 때만 다음 패널을 그려야 한다
    head = src[src.index("def render("):src.index("def _csv_panel(")]
    assert "if _layout_panel():" in head, \
        "돌리기 전에도 시간축 화면이 그려집니다"


def test_time_column_guess_lives_in_core_only():
    """추측 규칙이 두 군데면 판정과 화면이 서로 다른 컬럼을 가리킬 수 있다."""
    view = (VIEWS / "data_view.py").read_text(encoding="utf-8")
    assert "def _guess_time_col" not in view, \
        "화면에 시간 컬럼 추측이 따로 남아 있습니다 (core 것을 쓰세요)"
    assert "datasource.guess_time_column" in view


# ── 실행 편의 — 브라우저 자동 열기 ───────────────────────────
def test_free_port_skips_a_port_already_in_use():
    """8501 이 이미 쓰이고 있으면 다음 포트를 잡아야 한다.

    streamlit 도 알아서 넘어가지만, 그러면 우리가 열어야 할 주소를 모른다.
    그래서 포트는 우리가 먼저 정한다.

    **회사 PC 에서 이 테스트가 실제 결함을 잡았다.** free_port 가 SO_REUSEADDR
    을 쓰고 있었는데, 그 옵션은 윈도우에서 의미가 반대다 — 리눅스에서는
    "TIME_WAIT 재사용", 윈도우에서는 "이미 잡혀 있어도 같이 잡기" 에 가깝다.
    그래서 streamlit 이 돌고 있는 포트를 "비었다" 고 돌려줬다.
    리눅스에서는 통과하고 윈도우에서만 실패하는 종류라, 여기서만 드러났다.

    검사용 소켓에도 SO_REUSEADDR 을 걸지 않는다 — 걸면 두 OS 의 차이가
    가려져서 다시 리눅스에서만 통과하는 테스트가 된다.
    """
    import socket
    from scripts import launch

    base = launch.free_port()
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", base))
        busy.listen(1)
        got = launch.free_port(base)
        assert got != base, "이미 듣고 있는 포트를 다시 골랐습니다"
        assert not launch._alive(got), "고른 포트에서 이미 누가 듣고 있습니다"


def test_browser_does_not_open_when_the_server_dies():
    """서버가 못 떴는데 브라우저를 열면 '연결할 수 없음' 화면이 뜬다.

    사용자는 그걸 실패로 읽는다. 포트가 응답할 때만 열어야 한다.
    """
    import types
    from scripts import launch

    opened = []
    real = launch.webbrowser
    launch.webbrowser = types.SimpleNamespace(open=opened.append)
    dead = types.SimpleNamespace(poll=lambda: 1)          # 이미 끝난 프로세스
    try:
        launch.WAIT_SECONDS, keep = 2, launch.WAIT_SECONDS
        launch._open_when_ready("http://localhost:9999", 9999, dead)
        launch.WAIT_SECONDS = keep
    finally:
        launch.webbrowser = real
    assert not opened, "죽은 서버인데 브라우저를 열었습니다"


def test_browser_opens_once_the_port_answers():
    import socket
    import types
    from scripts import launch

    opened = []
    real = launch.webbrowser
    launch.webbrowser = types.SimpleNamespace(open=opened.append)
    alive = types.SimpleNamespace(poll=lambda: None)
    port = launch.free_port(8700)
    try:
        with socket.socket() as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", port))
            srv.listen(1)
            launch.WAIT_SECONDS, keep = 5, launch.WAIT_SECONDS
            launch._open_when_ready(f"http://localhost:{port}", port, alive)
            launch.WAIT_SECONDS = keep
    finally:
        launch.webbrowser = real
    assert opened == [f"http://localhost:{port}"], opened


def test_setup_fast_path_requires_a_working_venv():
    """도장만 믿으면 안 된다 — 사용자가 .venv 를 지운 뒤 엉뚱한 오류로 죽는다."""
    from scripts import setup

    assert setup._ready() is False, ("이 환경에는 .venv 가 없는데 "
                                     "설치가 끝났다고 판단했습니다")


def test_sqream_driver_is_installed_automatically():
    """SQream 드라이버가 자동 설치 목록에 있어야 한다.

    이 도구를 실제로 쓰는 곳이 SQream 인데, 예전에는 "쓰는 DBMS 것만 골라 깔라"
    며 빼 둬서 환경 점검이 매번 '건너뜀 1' 로 끝났다. 그 한 건이 정작 제일
    확인해야 할 경로였다.
    """
    from scripts import setup

    assert any("pysqream" in p for p in setup.EXTRAS), setup.EXTRAS
    extra = (ROOT / "requirements-extra.txt").read_text(encoding="utf-8")
    assert "pysqream-sqlalchemy" in extra
    # core 쪽 **설치 목록**에는 없어야 한다 — 두 군데에 적히면 버전이 갈라진다.
    # 주석에서 이름을 언급하는 것은 괜찮다 (왜 여기 없는지 설명하는 자리다).
    core_lines = [l.split("#")[0].strip()
                  for l in (ROOT / "requirements-core.txt")
                  .read_text(encoding="utf-8").splitlines()]
    assert not any("pysqream" in l for l in core_lines if l), \
        "설치 목록이 두 파일로 갈렸습니다"


def test_changing_the_install_list_forces_a_full_setup():
    """설치 목록이 바뀌면 빠른 경로로 새면 안 된다.

    **이게 없으면 조용한 함정이 된다.** 새 버전을 기존 폴더에 덮어썼을 때,
    도장만 보고 설치를 건너뛰면 새로 추가된 패키지가 영영 안 깔린다.
    사용자는 "받았는데 왜 그대로지" 가 되고, 원인을 찾을 단서가 없다.
    """
    from scripts import setup

    before = setup._requirements_fingerprint()
    keep = setup.EXTRAS
    try:
        setup.EXTRAS = keep + ["새-패키지"]
        assert setup._requirements_fingerprint() != before, (
            "설치 목록을 바꿨는데 지문이 그대로입니다 — 빠른 경로로 새 버립니다")
    finally:
        setup.EXTRAS = keep
    assert setup._requirements_fingerprint() == before, "지문이 안정적이지 않습니다"


def test_verify_env_tells_you_how_to_install_what_it_skipped():
    """무엇이 없다만 알려 주면 패키지 이름을 사용자가 다시 찾아야 한다.

    특히 SQream 드라이버는 이름이 pysqream-sqlalchemy 라 짐작이 안 된다.
    그리고 **가상환경 파이썬으로** 안내해야 한다 — 그냥 pip install 하면
    시스템 파이썬에 들어가서 다시 돌려도 여전히 건너뜀이다.
    """
    from scripts import verify_env

    hint = verify_env._install_hint("SQL · SQream 드라이버 로드")
    assert "pysqream-sqlalchemy" in hint, hint
    assert verify_env.VENV_PIP in hint, "가상환경이 아닌 파이썬을 안내합니다"
    assert verify_env._install_hint("시간축 정리") == ""      # 라이브러리 문제가 아닌 것


def test_stale_stamp_does_not_take_the_fast_path(monkeypatch):
    """예전 도장이 남아 있으면 빠른 경로로 가면 안 된다 — 실제 _ready() 로 확인.

    **처음 쓴 판은 엉뚱한 이유로 통과했다.** 가짜 파이썬을 물려 놨더니 지문
    검사를 통째로 빼도 임포트 확인에서 False 가 나와서, 변형을 주입해도 테스트가
    그대로 통과했다. 두 관문이 각각 제 몫을 하는지 갈라서 봐야 한다.
    """
    import subprocess as _sp
    import tempfile
    import types
    from pathlib import Path as _P
    from scripts import setup

    ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    bad = types.SimpleNamespace(returncode=1, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as d:
        stamp = _P(d) / ".setup_ok"
        fake_py = _P(d) / "python"
        fake_py.write_text("", encoding="utf-8")
        monkeypatch.setattr(setup, "STAMP", stamp)
        monkeypatch.setattr(setup, "venv_python", lambda: fake_py)

        # 관문 1 — 지문. 임포트 확인은 통과시켜 놓고 지문만 본다.
        monkeypatch.setattr(setup, "subprocess",
                            types.SimpleNamespace(run=lambda *a, **k: ok,
                                                  SubprocessError=_sp.SubprocessError))
        stamp.write_text(setup._requirements_fingerprint(), encoding="utf-8")
        assert setup._ready() is True, "정상 상태인데 다시 설치하려 합니다"

        stamp.write_text("옛-지문", encoding="utf-8")
        assert setup._ready() is False, (
            "설치 목록이 바뀌었는데 빠른 경로로 갔습니다 — "
            "새 패키지가 영영 안 깔립니다")

        # 관문 2 — 실제 임포트. 지문이 맞아도 패키지가 없으면 다시 깔아야 한다.
        stamp.write_text(setup._requirements_fingerprint(), encoding="utf-8")
        monkeypatch.setattr(setup, "subprocess",
                            types.SimpleNamespace(run=lambda *a, **k: bad,
                                                  SubprocessError=_sp.SubprocessError))
        assert setup._ready() is False, "도장만 믿고 패키지 확인을 건너뛰었습니다"


    # 도장이 없으면 당연히 전체 설치
    assert setup._ready() is False


def test_explain_view_asks_core_which_explainer_it_will_use():
    """**화면이 방법을 짐작하면 안 된다.**

    예전 explain_view 는 챔피언 이름에 "Ensemble" 이 들어 있으면 "트리 계열이라
    수 초~1분" 이라고 안내했다. 그런데 core 는 앙상블을 트리로 보지 않아 커널
    근사로 갔다. 안내와 실제가 갈린 탓에 사용자는 몇 시간짜리 계산을 1분짜리로
    알고 20분을 기다리다 포기했다.

    비용이 100배 다른 두 경로를 이름으로 짐작하는 것은 반드시 어긋난다.
    판단은 core.explain.plan() 한 곳에서만 한다.
    """
    src = (ROOT / "app" / "views" / "explain_view.py").read_text(encoding="utf-8")
    assert "explain.plan(" in src, "화면이 core 에 물어보지 않습니다"
    assert "fast = any(" not in src, "이름으로 빠른 경로를 짐작하고 있습니다"
    # 모델 이름 목록으로 분기하는 흔적이 남아 있으면 다시 갈라진다
    for name in ("LGBM", "CatBoost", "XGB"):
        assert f'"{name}"' not in src, (
            f"화면이 모델 이름({name})으로 분기합니다 — core 에 물어보세요")


def test_every_plan_method_is_handled_by_compute_shap():
    """plan() 이 내놓는 방법을 _explain 이 전부 처리해야 한다.

    새 방법을 추가하고 처리를 빠뜨리면 그 모델에서만 조용히 커널로 떨어진다.
    """
    import ast

    from core import explain

    src = (ROOT / "core" / "explain.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    plan_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "plan")
    methods = {n.value for n in ast.walk(plan_fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value in ("blend", "tree", "linear", "kernel")}
    assert methods == {"blend", "tree", "linear", "kernel"}, methods

    explain_fn = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_explain")
    body = ast.dump(explain_fn)
    for m in ("blend", "tree", "linear"):
        assert f"'{m}'" in body or f'"{m}"' in body, f"_explain 이 {m} 을 안 봅니다"
    assert hasattr(explain, "_explain_blend")
