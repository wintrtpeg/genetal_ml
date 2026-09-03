"""streamlit 대역 — 화면 코드를 실제로 실행해 보기 위한 것.

왜 필요한가
-----------
이 도구를 만든 환경에는 streamlit·plotly 가 설치돼 있지 않다 (패키지 저장소가
정책으로 막혀 있다). 그래서 지금까지 화면 코드는 **소스를 눈으로 읽어서만**
점검했다. 그 방식으로 잡은 결함이 10차까지 여러 건이지만, 눈으로 읽는 것은
한 줄씩 훑는 일이라 반드시 새는 곳이 생긴다.

이 대역은 다르다. **화면 함수를 진짜로 실행한다.** 그러면 파이썬 수준의 오류는
라이브러리 없이도 전부 드러난다 — KeyError, IndexError, AttributeError,
TypeError, 없는 컬럼 참조, None 에 대한 연산. 이런 것들은 streamlit 이
있든 없든 똑같이 터지는 진짜 결함이다.

거기에 더해, **진짜 streamlit 이 던지는 예외를 흉내 낸다.** 위젯 기본값이
범위 밖이면 StreamlitAPIException 을 던지고, icon= 에 이모지가 아닌 것이 오면
막고, column_config 의 format 문자열을 검사한다. 지금까지 손으로 하나씩
찾아낸 결함들이 바로 이 부류였다.

한계 — 정직하게
---------------
- 렌더링 결과(레이아웃·색·간격)는 확인할 수 없다. 그건 브라우저가 필요하다.
- 버튼 안쪽 코드는 기본적으로 실행되지 않는다. 진짜 streamlit 도 그렇다.
  `clicks=` 로 특정 버튼을 눌린 것으로 만들어 그 경로도 지나갈 수 있다.
- 재실행(rerun) 의미론은 흉내만 낸다. 실제 위젯 상태 보존까지는 아니다.
- 여기서 통과한다고 화면이 예쁘다는 뜻은 아니다. **터지지 않는다**는 뜻이다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


class StreamlitAPIException(Exception):
    """진짜 streamlit 이 잘못된 인자에 던지는 예외와 같은 자리."""


class RerunException(Exception):
    """st.rerun() — 진짜는 스크립트를 여기서 끊는다."""


class StopException(Exception):
    """st.stop()"""


# ── 검증 헬퍼 ────────────────────────────────────────────────
# 리터럴 퍼센트는 **정확히 %%** 여야 한다. 그 사이에 폭이나 정밀도를 끼운
# "%.2%" 는 sprintf 가 해석하지 못한다. 두 갈래를 따로 두지 않고 한 덩어리로
# 쓰면 "%.2%" 가 "정밀도 .2 에 변환자 %" 로 잘못 읽혀 통과해 버린다 —
# 실제로 이 검사기의 첫 판이 그랬고, 자기점검 테스트가 그걸 잡아냈다.
_SPRINTF = re.compile(r"%%|%[-+ #0']*\d*(?:\.\d+)?[bcdieEfgGosuxX]")


def _check_format(spec: Any, where: str) -> None:
    """NumberColumn 의 format 은 sprintf 형식이다. "%.2%" 같은 건 프런트에서 죽는다."""
    if not isinstance(spec, str) or "%" not in spec:
        return
    if "%" in _SPRINTF.sub("", spec):
        raise StreamlitAPIException(
            f'{where}: format="{spec}" 을 sprintf 로 해석할 수 없습니다. '
            "퍼센트 기호는 %% 로 씁니다.")


def _check_icon(icon: Any) -> None:
    """icon= 은 진짜 이모지 한 글자만 받는다. 기하도형 문자는 거부한다."""
    if icon is None:
        return
    if not isinstance(icon, str) or len(icon) == 0:
        raise StreamlitAPIException(f"icon={icon!r} 은 이모지가 아닙니다.")
    # 이모지는 So 범주이면서 Emoji_Presentation 을 갖는다. 여기서는
    # "기하도형·괘선 블록이면 거부" 라는 실용적 기준을 쓴다.
    cp = ord(icon[0])
    if 0x2190 <= cp <= 0x2BFF and not (0x2600 <= cp <= 0x27BF):
        raise StreamlitAPIException(
            f'The value "{icon}" is not a valid emoji. '
            "Shortcodes are not allowed, please use a single character instead.")
    if unicodedata.category(icon[0]) not in ("So", "Sk", "Cs") and cp < 0x1F000:
        raise StreamlitAPIException(f'The value "{icon}" is not a valid emoji.')


def _num(x):
    """비교 가능한 수치로. datetime/date/Timestamp 도 그대로 비교된다."""
    return x


def _check_bounds(label, lo, hi, value, kind):
    """진짜 streamlit 은 기본값이 범위 밖이면 예외를 던진다.

    지금까지 손으로 찾아낸 결함(PCA 슬라이더, SHAP 표본 상한, top_k)이 전부
    이 부류였다. 여기서는 자동으로 잡힌다.
    """
    if lo is None or hi is None:
        return
    try:
        if lo > hi:
            raise StreamlitAPIException(
                f"{kind}('{label}'): min_value({lo}) 가 max_value({hi}) 보다 큽니다.")
        if value is None:
            return
        vals = value if isinstance(value, (tuple, list)) else [value]
        for v in vals:
            if v is None:
                continue
            if v < lo or v > hi:
                raise StreamlitAPIException(
                    f"{kind}('{label}'): 기본값 {v} 가 범위 [{lo}, {hi}] 밖입니다.")
    except TypeError:
        return          # 비교 불가한 타입은 검사하지 않는다


# ── column_config ────────────────────────────────────────────
class _Col:
    def __init__(self, kind, label=None, **kw):
        self.kind, self.label, self.kw = kind, label, kw
        _check_format(kw.get("format"), f"{kind}({label!r})")


class _ColumnConfig:
    def NumberColumn(self, label=None, **kw):   # noqa: N802
        return _Col("NumberColumn", label, **kw)

    def TextColumn(self, label=None, **kw):     # noqa: N802
        return _Col("TextColumn", label, **kw)

    def CheckboxColumn(self, label=None, **kw):  # noqa: N802
        return _Col("CheckboxColumn", label, **kw)

    def DatetimeColumn(self, label=None, **kw):  # noqa: N802
        return _Col("DatetimeColumn", label, **kw)

    def SelectboxColumn(self, label=None, **kw):  # noqa: N802
        return _Col("SelectboxColumn", label, **kw)

    def ProgressColumn(self, label=None, **kw):  # noqa: N802
        return _Col("ProgressColumn", label, **kw)

    def BarChartColumn(self, label=None, **kw):  # noqa: N802
        return _Col("BarChartColumn", label, **kw)


# ── 세션 상태 ────────────────────────────────────────────────
class SessionState(dict):
    """진짜 st.session_state 처럼 속성 접근과 딕셔너리 접근을 둘 다 받는다."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(
                f"st.session_state 에 '{k}' 가 없습니다 "
                "(state.DEFAULTS 에 넣었는지 확인하세요)") from e

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


# ── 기록 ─────────────────────────────────────────────────────
@dataclass
class Recorder:
    calls: list[tuple[str, tuple, dict]] = field(default_factory=list)
    widgets: list[str] = field(default_factory=list)
    charts: int = 0
    frames: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    # 위젯이 **무엇으로 시작하는가**. 추천 기능이 생기면서 필요해졌다 —
    # "추천은 median 인데 위젯은 ffill 로 시작한다" 는 화면이 그려지기만 하면
    # 통과해 버리기 때문이다. 추천을 읽지 않는 사람에게는 기본값이 곧 결정이다.
    defaults: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)   # multiselect 가 무엇을 내주는가

    def summary(self) -> str:
        return (f"위젯 {len(self.widgets)} · 차트 {self.charts} · 표 {self.frames} "
                f"· error {len(self.errors)} · warning {len(self.warnings)}")


# ── 본체 ─────────────────────────────────────────────────────
class FakeStreamlit:
    """화면 코드가 부르는 것만 구현한다. 모르는 호출은 조용히 받아 넘긴다."""

    def __init__(self, rec: Recorder, state: SessionState,
                 clicks: set[str] | None = None, values: dict | None = None):
        self._rec = rec
        self.session_state = state
        self._clicks = clicks or set()
        self._values = values or {}
        self.column_config = _ColumnConfig()
        # 진짜 st.sidebar 는 `with st.sidebar:` 로도 쓰인다. 자기 자신을 넣으면
        # 컨텍스트 매니저가 아니라서 TypeError 가 난다.
        self.sidebar = _Child(self)

    # -- 내부 --
    def _log(self, name, *a, **kw):
        self._rec.calls.append((name, a, kw))

    def _pick(self, label, default):
        """values 로 특정 위젯의 반환값을 지정할 수 있게 한다."""
        return self._values.get(label, default)

    # -- 텍스트 --
    def title(self, *a, **kw): self._log("title", *a, **kw)
    def header(self, *a, **kw): self._log("header", *a, **kw)
    def subheader(self, *a, **kw): self._log("subheader", *a, **kw)
    def caption(self, *a, **kw): self._log("caption", *a, **kw)
    def code(self, *a, **kw): self._log("code", *a, **kw)
    def divider(self, *a, **kw): self._log("divider", *a, **kw)
    def text(self, *a, **kw): self._log("text", *a, **kw)
    def help(self, *a, **kw): self._log("help", *a, **kw)
    def write(self, *a, **kw): self._log("write", *a, **kw)
    def json(self, *a, **kw): self._log("json", *a, **kw)
    def latex(self, *a, **kw): self._log("latex", *a, **kw)
    def toast(self, *a, **kw): self._log("toast", *a, **kw)
    def balloons(self, *a, **kw): self._log("balloons")
    def set_page_config(self, *a, **kw): self._log("set_page_config", **kw)

    def markdown(self, body="", **kw):
        self._log("markdown", body, **kw)
        if isinstance(body, str) and "<" in body and not kw.get("unsafe_allow_html"):
            # 진짜는 태그를 그대로 글자로 보여준다 — 화면이 깨져 보인다
            if re.search(r"<(p|div|span|b|br)\b", body):
                raise StreamlitAPIException(
                    f"HTML 을 넣으면서 unsafe_allow_html 을 안 켰습니다: {body[:60]}")

    # -- 알림 (icon= 검사) --
    def _alert(self, kind, body="", icon=None, **kw):
        _check_icon(icon)
        self._log(kind, body, **kw)
        getattr(self._rec, {"error": "errors", "warning": "warnings",
                            "info": "infos", "success": "infos"}[kind]).append(str(body)[:200])

    def error(self, body="", icon=None, **kw): self._alert("error", body, icon, **kw)
    def warning(self, body="", icon=None, **kw): self._alert("warning", body, icon, **kw)
    def info(self, body="", icon=None, **kw): self._alert("info", body, icon, **kw)
    def success(self, body="", icon=None, **kw): self._alert("success", body, icon, **kw)
    def exception(self, e): self._rec.errors.append(str(e))

    # -- 표 / 차트 --
    def dataframe(self, data=None, **kw):
        self._rec.frames += 1
        self._log("dataframe", **kw)
        self._check_col_config(data, kw.get("column_config"))

    def table(self, data=None, **kw):
        self._rec.frames += 1

    def _check_col_config(self, data, cfg):
        """설정한 컬럼이 실제로 있는지 본다. 없으면 조용히 무시되지만 의도와 다르다."""
        if not cfg or data is None or not hasattr(data, "columns"):
            return
        missing = [k for k in cfg if k not in data.columns]
        if missing:
            raise StreamlitAPIException(
                f"column_config 에 있는 컬럼이 데이터에 없습니다: {missing} "
                f"(실제 컬럼: {list(data.columns)[:8]})")

    def data_editor(self, data=None, **kw):
        self._rec.frames += 1
        self._rec.widgets.append("data_editor")
        self._check_col_config(data, kw.get("column_config"))
        if kw.get("key") is None:
            raise StreamlitAPIException("data_editor 에 key 가 없으면 편집 상태가 꼬입니다.")
        return data

    def plotly_chart(self, fig=None, **kw):
        self._rec.charts += 1
        if fig is None:
            raise StreamlitAPIException("plotly_chart 에 None 을 넘겼습니다.")

    def line_chart(self, data=None, **kw): self._rec.charts += 1
    def bar_chart(self, data=None, **kw): self._rec.charts += 1
    def area_chart(self, data=None, **kw): self._rec.charts += 1
    def pyplot(self, *a, **kw): self._rec.charts += 1
    def image(self, *a, **kw): self._log("image")

    def metric(self, label, value=None, delta=None, **kw):
        self._log("metric", label, value, delta, **kw)

    def progress(self, value=0.0, **kw):
        if isinstance(value, (int, float)) and not (0.0 <= value <= 1.0):
            raise StreamlitAPIException(f"progress 값 {value} 가 0~1 밖입니다.")
        return _Progress()

    # -- 입력 위젯 --
    def button(self, label="", **kw):
        self._rec.widgets.append(f"button:{label}")
        return label in self._clicks

    def form_submit_button(self, label="", **kw):
        return label in self._clicks

    def download_button(self, label="", data=None, **kw):
        self._rec.widgets.append(f"download:{label}")
        if data is None:
            raise StreamlitAPIException("download_button 에 데이터가 없습니다.")
        return False

    def checkbox(self, label="", value=False, **kw):
        self._rec.widgets.append(f"checkbox:{label}")
        self._rec.defaults[label] = bool(value)
        return bool(self._pick(label, value))

    def toggle(self, label="", value=False, **kw):
        return bool(self._pick(label, value))

    def radio(self, label="", options=(), index=0, **kw):
        opts = list(options)
        self._rec.widgets.append(f"radio:{label}")
        if not opts:
            raise StreamlitAPIException(f"radio('{label}'): 선택지가 비었습니다.")
        if index is not None and not (0 <= index < len(opts)):
            raise StreamlitAPIException(
                f"radio('{label}'): index {index} 가 선택지 {len(opts)}개 밖입니다.")
        self._rec.defaults[label] = opts[index or 0]
        return self._pick(label, opts[index or 0])

    def selectbox(self, label="", options=(), index=0, **kw):
        opts = list(options)
        self._rec.widgets.append(f"selectbox:{label}")
        if not opts:
            raise StreamlitAPIException(f"selectbox('{label}'): 선택지가 비었습니다.")
        if index is not None and not (0 <= index < len(opts)):
            raise StreamlitAPIException(
                f"selectbox('{label}'): index {index} 가 선택지 {len(opts)}개 밖입니다.")
        self._rec.defaults[label] = opts[index or 0]
        return self._pick(label, opts[index or 0])

    def multiselect(self, label="", options=(), default=None, **kw):
        opts = list(options)
        self._rec.widgets.append(f"multiselect:{label}")
        d = list(default) if default is not None else []
        bad = [x for x in d if x not in opts]
        if bad:
            raise StreamlitAPIException(
                f"multiselect('{label}'): 기본값 {bad} 가 선택지에 없습니다.")
        self._rec.defaults[label] = list(d)
        self._rec.options[label] = opts
        return self._pick(label, d)

    def number_input(self, label="", min_value=None, max_value=None, value=None, **kw):
        self._rec.widgets.append(f"number_input:{label}")
        _check_bounds(label, min_value, max_value, value, "number_input")
        v = value if value is not None else (min_value if min_value is not None else 0)
        self._rec.defaults[label] = v
        return self._pick(label, v)

    def slider(self, label="", min_value=None, max_value=None, value=None, step=None, **kw):
        self._rec.widgets.append(f"slider:{label}")
        _check_bounds(label, min_value, max_value, value, "slider")
        if value is None:
            value = min_value
        return self._pick(label, value)

    def select_slider(self, label="", options=(), value=None, **kw):
        opts = list(options)
        self._rec.widgets.append(f"select_slider:{label}")
        if len(opts) > 2000:
            raise StreamlitAPIException(
                f"select_slider('{label}'): 선택지가 {len(opts):,}개입니다. "
                "브라우저로 그대로 전송되어 화면이 멈춥니다.")
        return self._pick(label, value if value is not None else (opts[0] if opts else None))

    def text_input(self, label="", value="", **kw):
        self._rec.widgets.append(f"text_input:{label}")
        return self._pick(label, value)

    def text_area(self, label="", value="", **kw):
        self._rec.widgets.append(f"text_area:{label}")
        return self._pick(label, value)

    def date_input(self, label="", value=None, min_value=None, max_value=None, **kw):
        self._rec.widgets.append(f"date_input:{label}")
        _check_bounds(label, min_value, max_value, value, "date_input")
        return self._pick(label, value)

    def time_input(self, label="", value=None, **kw):
        return self._pick(label, value)

    def file_uploader(self, label="", **kw):
        self._rec.widgets.append(f"file_uploader:{label}")
        return self._pick(label, None)

    def color_picker(self, label="", value="#000000", **kw):
        return self._pick(label, value)

    # -- 레이아웃 --
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        if n < 1:
            raise StreamlitAPIException(f"columns({spec}): 열이 1개 이상이어야 합니다.")
        return [_Child(self) for _ in range(n)]

    def tabs(self, labels, **kw):
        return [_Child(self) for _ in labels]

    def expander(self, label="", **kw):
        return _Child(self)

    def container(self, **kw):
        return _Child(self)

    def form(self, key=None, **kw):
        return _Child(self)

    def empty(self):
        return _Child(self)

    def spinner(self, text="", **kw):
        return _Child(self)

    def status(self, label="", **kw):
        return _Child(self)

    def popover(self, label="", **kw):
        return _Child(self)

    # -- 흐름 --
    def rerun(self, **kw):
        raise RerunException()

    def stop(self):
        raise StopException()

    def cache_data(self, fn=None, **kw):
        return fn if fn is not None else (lambda f: f)

    cache_resource = cache_data

    def __getattr__(self, name):
        """모르는 호출은 아무것도 안 하는 함수로. 대역이 못 따라간다고 죽지는 않게."""
        def _noop(*a, **kw):
            self._log(f"?{name}", *a, **kw)
        return _noop


class _Progress:
    def progress(self, *a, **kw): pass
    def empty(self): pass
    def caption(self, *a, **kw): pass
    def markdown(self, *a, **kw): pass
    def text(self, *a, **kw): pass


class _Child:
    """columns / expander / tabs 가 돌려주는 것. 부모와 같은 API 를 쓴다."""

    def __init__(self, parent: FakeStreamlit):
        self._p = parent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._p, name)
