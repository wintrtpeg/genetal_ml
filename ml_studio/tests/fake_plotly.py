"""plotly.graph_objects 대역.

목적은 두 가지다.

1. **차트 함수를 실제로 실행한다.** 그러면 파이썬 수준 오류가 드러난다 —
   없는 컬럼 참조, 빈 프레임에서의 IndexError, None 연산, 잘못된 zip.
   이건 plotly 가 있든 없든 똑같이 터지는 진짜 결함이다.

2. **plotly 5 에서 제거된 속성을 거부한다.** 아는 만큼만 엄격하게 본다.
   모르는 속성은 통과시킨다 — 여기서 오탐이 나면 진짜 결함을 가리게 되므로,
   "확실히 틀린 것만 잡는다" 를 원칙으로 한다.

한계 — 이 대역이 통과해도 진짜 plotly 가 통과한다는 보장은 없다. 진짜 스키마는
훨씬 넓다. `scripts/verify_env.py` 가 설치된 PC 에서 진짜로 그려 보는 이유다.
"""

from __future__ import annotations

# plotly 5 에서 제거됐거나 이름이 바뀐 속성. 이게 남아 있으면 그 차트는 죽는다.
REMOVED = {
    "titlefont": "title=dict(font=...)",
    "titleside": "title=dict(side=...)",
    "titlefontsize": "title=dict(font=dict(size=...))",
    "plot_bgcolor_alpha": "(그런 속성 없음)",
}


def _scan(where: str, kw: dict) -> None:
    for k, v in kw.items():
        if k in REMOVED:
            raise ValueError(
                f"{where}: '{k}' 은 plotly 5 에서 제거된 속성입니다. "
                f"{REMOVED[k]} 형식을 쓰세요.")
        if isinstance(v, dict):
            _scan(f"{where}.{k}", v)


class _Node(dict):
    """layout.yaxis2.title.text 같은 점 접근을 받아 준다."""

    def __getattr__(self, k):
        v = self.get(k)
        if isinstance(v, dict) and not isinstance(v, _Node):
            v = _Node(v)
            self[k] = v
        if v is None:
            v = _Node()
            self[k] = v
        return v

    def __setattr__(self, k, v):
        self[k] = v


class _Trace(_Node):
    def __init__(self, kind, **kw):
        _scan(kind, kw)
        super().__init__(kw)
        self["_kind"] = kind


def _trace_factory(kind):
    def make(**kw):
        return _Trace(kind, **kw)
    return make


class Figure:
    def __init__(self, data=None, layout=None):
        self.data = []
        self.layout = _Node()
        self.shapes: list = []
        self.annotations: list = []
        if data is not None:
            self.add_trace(data)
        if layout:
            self.update_layout(**layout)

    def add_trace(self, tr, **kw):
        if tr is None:
            raise ValueError("add_trace 에 None 을 넘겼습니다.")
        self.data.append(tr)
        return self

    def add_traces(self, trs, **kw):
        for t in trs:
            self.add_trace(t)
        return self

    def _line_arg(self, name, val, kw):
        if val is None:
            raise ValueError(f"{name}: 위치 값이 None 입니다.")
        # 진짜 plotly 는 datetime 을 받지만 NaN/None 은 못 받는다
        try:
            if val != val:            # NaN
                raise ValueError(f"{name}: 위치 값이 NaN 입니다.")
        except (TypeError, ValueError) as e:
            if "NaN" in str(e):
                raise
        _scan(name, kw)

    def add_hline(self, y=None, **kw):
        self._line_arg("add_hline", y, kw)
        self.shapes.append(("hline", y))
        return self

    def add_vline(self, x=None, **kw):
        self._line_arg("add_vline", x, kw)
        self.shapes.append(("vline", x))
        return self

    def add_hrect(self, **kw):
        _scan("add_hrect", kw); return self

    def add_vrect(self, **kw):
        _scan("add_vrect", kw); return self

    def add_annotation(self, **kw):
        _scan("add_annotation", kw)
        self.annotations.append(kw)
        return self

    def update_layout(self, **kw):
        _scan("update_layout", kw)
        for k, v in kw.items():
            self.layout[k] = _Node(v) if isinstance(v, dict) else v
        return self

    def update_xaxes(self, **kw):
        _scan("update_xaxes", kw); return self

    def update_yaxes(self, **kw):
        _scan("update_yaxes", kw); return self

    def update_traces(self, **kw):
        _scan("update_traces", kw); return self

    def update_coloraxes(self, **kw):
        _scan("update_coloraxes", kw); return self

    def to_html(self, **kw):
        return f"<div>fake figure with {len(self.data)} traces</div>"

    def to_json(self, **kw):
        return "{}"

    def write_html(self, path, **kw):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())


# graph_objects 의 트레이스들
Scatter = _trace_factory("Scatter")
Scattergl = _trace_factory("Scattergl")
Bar = _trace_factory("Bar")
Heatmap = _trace_factory("Heatmap")
Histogram = _trace_factory("Histogram")
Box = _trace_factory("Box")
Violin = _trace_factory("Violin")
Pie = _trace_factory("Pie")
Waterfall = _trace_factory("Waterfall")
Scatterpolar = _trace_factory("Scatterpolar")
Table = _trace_factory("Table")


def install() -> None:
    """sys.modules 에 가짜 plotly 를 꽂는다. core/plots.py 의 _px() 가 이걸 집는다."""
    import sys
    import types

    go = types.ModuleType("plotly.graph_objects")
    for name in ("Figure", "Scatter", "Scattergl", "Bar", "Heatmap", "Histogram",
                 "Box", "Violin", "Pie", "Waterfall", "Scatterpolar", "Table"):
        setattr(go, name, globals()[name])

    io = types.ModuleType("plotly.io")
    io.to_html = lambda fig, **kw: fig.to_html(**kw)
    io.templates = types.SimpleNamespace(default="plotly")

    root = types.ModuleType("plotly")
    root.graph_objects = go
    root.io = io
    root.__version__ = "5.0.0-fake"

    sys.modules["plotly"] = root
    sys.modules["plotly.graph_objects"] = go
    sys.modules["plotly.io"] = io
