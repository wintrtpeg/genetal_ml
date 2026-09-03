"""화면 디자인 토큰과 공용 UI 조각.

설계 의도
---------
계측실 트렌드 화면의 문법을 따른다. 색은 장식이 아니라 의미를 나른다.
실측은 짙은 계기 남색, 예측은 앰버, 나머지 UI 는 잉크와 괘선만 쓴다.
색 예산 전부를 차트에 쓰고 화면 껍데기는 무채색으로 조용히 둔다.

좌측 레일은 파이프라인 단계다. 실제 순서가 있는 흐름이라 번호를 붙였고,
각 단계가 끝났는지 여부를 점으로 표시한다. 상단 상태바(runbar)가 이 화면의
중심 장치다 — 어느 단계에 있든 데이터·타겟·분할·챔피언·누수 가드 상태를
한 줄로 계속 보여준다.

웹폰트는 쓰지 않는다. 폐쇄망에서 CDN 이 막히면 레이아웃이 무너지므로
시스템 폰트 스택에 기대고, 개성은 자간·크기 단계·괘선으로 만든다.
"""

from __future__ import annotations

import inspect

import streamlit as st

# ── 디자인 토큰 ──────────────────────────────────────────────────────────
CANVAS = "#EDF0F4"
PANEL = "#FFFFFF"
PANEL_2 = "#F5F7FA"
RULE = "#D6DCE4"
RULE_SOFT = "#E7EBF0"
INK = "#0E1620"
INK_2 = "#33414F"
MUTED = "#66768A"
RAIL_BG = "#141C26"
RAIL_INK = "#DFE6EE"
RAIL_MUTED = "#8595A6"

MEASURED = "#0B4F8C"   # 실측
PREDICTED = "#C77B02"  # 예측
OK = "#1F6F5C"
WARN = "#B4632A"
BAD = "#A32015"

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", '
        '"Apple SD Gothic Neo", "Noto Sans KR", sans-serif')
MONO = '"SFMono-Regular", Consolas, "D2Coding", "Courier New", monospace'


CSS = f"""
<style>
  :root {{
    --canvas: {CANVAS}; --panel: {PANEL}; --panel2: {PANEL_2};
    --rule: {RULE}; --rule-soft: {RULE_SOFT};
    --ink: {INK}; --ink2: {INK_2}; --muted: {MUTED};
    --measured: {MEASURED}; --predicted: {PREDICTED};
    --ok: {OK}; --warn: {WARN}; --bad: {BAD};
  }}

  html, body, [class*="css"] {{ font-family: {FONT}; }}
  .stApp {{ background: var(--canvas); }}

  .block-container {{
    padding-top: 1.9rem; padding-bottom: 4rem;
    max-width: 1360px;
  }}

  /* 표제 — 자간을 조이고 무게 대비로 단계를 만든다 */
  h1 {{ font-size: 1.55rem !important; font-weight: 680 !important;
       letter-spacing: -0.021em; color: var(--ink); margin-bottom: 0.15rem !important; }}
  h2 {{ font-size: 1.08rem !important; font-weight: 640 !important;
       letter-spacing: -0.012em; color: var(--ink);
       margin-top: 1.6rem !important; padding-bottom: 0.45rem;
       border-bottom: 1px solid var(--rule); }}
  h3 {{ font-size: 0.94rem !important; font-weight: 620 !important;
       letter-spacing: -0.008em; color: var(--ink2); margin-top: 1.1rem !important; }}

  p, li, label, .stMarkdown {{ color: var(--ink2); }}
  .caption {{ color: var(--muted); font-size: 0.845rem; line-height: 1.6;
              max-width: 78ch; margin: 0.2rem 0 0.9rem 0; }}
  [data-testid="stCaptionContainer"] p {{ color: var(--muted); font-size: 0.82rem; }}

  /* 숫자는 자릿수를 맞춘다 */
  [data-testid="stMetricValue"], .stDataFrame, .rb-v, table {{
    font-variant-numeric: tabular-nums; }}

  /* 지표 — 카드 대신 상단 굵은 괘선 */
  [data-testid="stMetric"] {{
    background: var(--panel); padding: 0.7rem 0.9rem 0.75rem 0.9rem;
    border: 1px solid var(--rule-soft); border-top: 2px solid var(--measured);
    border-radius: 2px; }}
  [data-testid="stMetricLabel"] p {{
    font-size: 0.76rem !important; color: var(--muted); font-weight: 500; }}
  [data-testid="stMetricValue"] {{
    font-size: 1.42rem !important; font-weight: 620; color: var(--ink); }}
  [data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}

  /* 상단 상태바 — 이 화면의 중심 장치 */
  .runbar {{
    display: flex; flex-wrap: wrap; align-items: stretch;
    background: var(--panel); border: 1px solid var(--rule);
    border-left: 3px solid var(--measured); border-radius: 2px;
    margin: 0.55rem 0 1.5rem 0; overflow: hidden; }}
  .rb-item {{ padding: 0.55rem 1.05rem; border-right: 1px solid var(--rule-soft);
              min-width: 108px; flex: 0 1 auto; }}
  .rb-item:last-child {{ border-right: none; }}
  .rb-k {{ font-size: 0.685rem; color: var(--muted); letter-spacing: 0.02em;
           display: block; margin-bottom: 0.16rem; }}
  .rb-v {{ font-size: 0.895rem; color: var(--ink); font-weight: 590;
           display: block; white-space: nowrap; overflow: hidden;
           text-overflow: ellipsis; max-width: 240px; }}
  .rb-v.dim {{ color: #9AA7B6; font-weight: 480; }}
  .rb-item.flag {{ margin-left: auto; border-right: none;
                   border-left: 1px solid var(--rule-soft); background: var(--panel2); }}
  /* 누수 가드는 margin-left:auto 로 오른쪽 끝에 붙는다. 그러면 그 앞 칸의
     오른쪽 괘선과 가드 칸의 왼쪽 괘선 사이에 빈 공간이 생겨, 값이 안 채워진
     빈 칸이 하나 있는 것처럼 보인다. 앞 칸의 괘선을 지워 선을 하나만 남긴다. */
  .rb-item:has(+ .flag) {{ border-right: none; }}

  .badge {{ display: inline-block; font-size: 0.735rem; font-weight: 600;
            padding: 0.14rem 0.5rem; border-radius: 2px; line-height: 1.5; }}
  .badge-ok   {{ color: var(--ok);   background: #E8F2EE; }}
  .badge-warn {{ color: var(--warn); background: #FBEFE4; }}
  .badge-bad  {{ color: var(--bad);  background: #F8E7E5; }}
  .badge-idle {{ color: var(--muted); background: var(--panel2); }}

  /* 추천 근거 — 위젯 바로 밑에 붙어 "왜 이 값인지" 를 항상 보이게 둔다.
     접어 두면 아무도 안 편다. 왼쪽 색띠로 확신도를 같이 알린다. */
  .advice {{ font-size: 0.815rem; line-height: 1.62; color: var(--ink);
             background: var(--panel2); border-left: 3px solid var(--muted);
             padding: 0.52rem 0.72rem; margin: 0.3rem 0 0.45rem;
             border-radius: 0 3px 3px 0; }}
  .advice-ok   {{ border-left-color: var(--ok); }}
  .advice-warn {{ border-left-color: var(--warn); }}
  .advice-bad  {{ border-left-color: var(--bad); }}
  .advice-tag {{ display: inline-block; font-size: 0.7rem; font-weight: 700;
                 letter-spacing: 0.02em; color: var(--muted);
                 margin-right: 0.4rem; }}
  .advice b, .advice strong {{ font-weight: 680; }}

  /* 좌측 레일 — 어두운 면으로 본문과 분리한다 */
  [data-testid="stSidebar"] {{ background: {RAIL_BG}; border-right: none; }}
  [data-testid="stSidebar"] * {{ color: {RAIL_INK}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
  .rail-title {{ font-size: 0.98rem; font-weight: 650; letter-spacing: -0.012em;
                 color: #FFFFFF; margin-bottom: 0.1rem; }}
  .rail-sub {{ font-size: 0.755rem; color: {RAIL_MUTED}; line-height: 1.5;
               margin-bottom: 1.1rem; }}
  .rail-note {{ font-size: 0.745rem; color: {RAIL_MUTED}; line-height: 1.6;
                border-left: 2px solid #2C3A49; padding-left: 0.6rem;
                margin: 0.5rem 0 0.9rem 0; }}
  .rail-meta {{ font-size: 0.775rem; color: {RAIL_INK}; line-height: 1.85; }}
  .rail-meta span {{ color: {RAIL_MUTED}; }}
  /* 레일 안 소제목 — 모드와 단계가 각각 무엇인지 구분되게 한다.
     예전엔 라벨 없이 라디오만 두 벌 있어서 무엇을 고르는 자리인지 안 보였다. */
  .rail-section {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
                   color: {RAIL_MUTED}; text-transform: uppercase;
                   margin: 0 0 0.35rem; }}
  [data-testid="stSidebar"] hr {{ border-color: #2C3A49; margin: 0.9rem 0; }}

  /* 레일 항목: 라디오를 목록처럼 보이게 */
  [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.08rem; }}
  [data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.4rem 0.55rem; border-radius: 2px; width: 100%;
    border-left: 2px solid transparent; transition: background 0.12s ease; }}
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #1D2833; }}
  [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
  [data-testid="stSidebar"] [role="radiogroup"] label p {{
    font-size: 0.83rem !important; color: {RAIL_MUTED}; margin: 0; }}
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: #1F2B37; border-left-color: var(--predicted); }}
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
    color: #FFFFFF; font-weight: 620 !important; }}

  /* 버튼 — 각지게, 그림자 없이
     주의: streamlit 은 버튼 라벨을 button 안쪽 <p>/<div> 에 넣고 거기에
     textColor(거의 검정)를 따로 먹인다. button 에만 color 를 주면 안쪽이
     이기고, 남색 배경에 검정 글씨가 되어 **글자가 안 보인다.**
     그래서 자손까지 !important 로 눌러야 한다. DOM 이 버전마다 달라서
     구·신 선택자를 함께 적는다. */
  .stButton > button, .stDownloadButton > button {{
    border-radius: 2px; font-weight: 570; font-size: 0.86rem;
    border: 1px solid var(--rule); box-shadow: none; transition: none; }}

  .stButton > button[kind="primary"],
  .stDownloadButton > button[kind="primary"],
  [data-testid="stBaseButton-primary"],
  [data-testid="baseButton-primary"],
  [data-testid="stBaseButton-primaryFormSubmit"],
  [data-testid="baseButton-primaryFormSubmit"] {{
    background: var(--measured) !important;
    border-color: var(--measured) !important; }}

  .stButton > button[kind="primary"], .stButton > button[kind="primary"] *,
  .stDownloadButton > button[kind="primary"],
  .stDownloadButton > button[kind="primary"] *,
  [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] *,
  [data-testid="baseButton-primary"], [data-testid="baseButton-primary"] *,
  [data-testid="stBaseButton-primaryFormSubmit"],
  [data-testid="stBaseButton-primaryFormSubmit"] *,
  [data-testid="baseButton-primaryFormSubmit"],
  [data-testid="baseButton-primaryFormSubmit"] * {{
    color: #FFFFFF !important; fill: #FFFFFF !important; }}

  .stButton > button[kind="primary"]:hover,
  [data-testid="stBaseButton-primary"]:hover,
  [data-testid="baseButton-primary"]:hover {{
    background: #093F70 !important; border-color: #093F70 !important; }}

  /* 보조 버튼 — 흰 배경이므로 글자는 잉크색이어야 한다 */
  .stButton > button[kind="secondary"], .stButton > button[kind="secondary"] *,
  [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondary"] *,
  [data-testid="baseButton-secondary"], [data-testid="baseButton-secondary"] * {{
    color: var(--ink) !important; }}

  /* 사이드바는 어두운 배경 — 여기만 밝은 글자 */
  [data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; border-color: #33414F !important; }}
  [data-testid="stSidebar"] .stButton > button,
  [data-testid="stSidebar"] .stButton > button * {{
    color: {RAIL_MUTED} !important; }}
  [data-testid="stSidebar"] .stButton > button:hover,
  [data-testid="stSidebar"] .stButton > button:hover * {{
    border-color: #556575 !important; color: #FFFFFF !important; }}
  [data-testid="stSidebar"] .stButton > button[kind="primary"],
  [data-testid="stSidebar"] .stButton > button[kind="primary"] * {{
    background: var(--measured) !important; color: #FFFFFF !important; }}

  /* 입력 요소 */
  .stTextInput input, .stNumberInput input, .stTextArea textarea,
  [data-baseweb="select"] > div {{ border-radius: 2px !important; }}
  .stTextArea textarea {{ font-family: {MONO}; font-size: 0.83rem; }}
  code {{ font-family: {MONO}; font-size: 0.84em;
          background: var(--panel2); padding: 0.1em 0.35em; border-radius: 2px;
          color: var(--ink); }}

  /* 탭 — 밑줄만 */
  .stTabs [data-baseweb="tab-list"] {{ gap: 1.35rem; border-bottom: 1px solid var(--rule); }}
  .stTabs [data-baseweb="tab"] {{
    padding: 0.4rem 0; font-size: 0.875rem; font-weight: 560; color: var(--muted); }}
  .stTabs [aria-selected="true"] {{ color: var(--ink); }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: var(--measured); height: 2px; }}

  /* 표·확장·알림 */
  [data-testid="stExpander"] details {{
    border: 1px solid var(--rule-soft); border-radius: 2px; background: var(--panel); }}
  [data-testid="stExpander"] summary {{ font-size: 0.86rem; font-weight: 560; }}
  [data-testid="stDataFrame"] {{ border: 1px solid var(--rule-soft); border-radius: 2px; }}
  [data-testid="stAlert"] {{ border-radius: 2px; border-left-width: 3px; }}
  hr {{ border-color: var(--rule); }}
  [data-testid="stElementToolbar"] {{ display: none; }}

  /* 차트를 흰 판 위에 올린다 */
  [data-testid="stPlotlyChart"] {{
    background: var(--panel); border: 1px solid var(--rule-soft);
    border-radius: 2px; padding: 0.35rem 0.5rem 0.1rem 0.5rem; }}

  /* 구간 표식 — SHAP 기간 비교에서 쓴다 */
  .swatch {{ display: inline-block; width: 9px; height: 9px; border-radius: 1px;
             margin-right: 0.4rem; vertical-align: middle; }}
</style>
"""


# ── 폭 지정 — streamlit 버전 호환 ────────────────────────────────────────
# streamlit 이 `use_container_width` 를 `width` 로 바꾸면서 **"2025-12-31 이후
# 제거"** 를 예고했다. 예고한 날짜는 이미 지났다. 지워지는 순간 이 도구는
# 화면 한 장도 못 띄운다 — 표·차트·버튼 100군데가 전부 그 인자를 쓴다.
#
# 그렇다고 `width` 로 그냥 갈아타면 구버전에서 깨진다. 옛 streamlit 에도
# `width` 라는 인자가 있었지만 **픽셀 정수**였다. 이름만 보고 판단하면
# `width="stretch"` 를 정수 자리에 넣는 셈이라 조용히 다르게 그려진다.
#
# 그래서 이름이 아니라 **기본값의 타입**으로 가른다. 새 API 는 기본값이
# "stretch"·"content" 라는 문자열이고, 옛 API 는 None 이다. 판정은 import 할 때
# 한 번만 하고, 화면 쪽은 `**theme.WIDE` 한 가지만 쓴다.
def _wide_kwargs() -> dict:
    try:
        default = inspect.signature(st.dataframe).parameters["width"].default
    except (AttributeError, KeyError, TypeError, ValueError):
        return {"use_container_width": True}
    return {"width": "stretch"} if isinstance(default, str) else {"use_container_width": True}


WIDE = _wide_kwargs()


def inject() -> None:
    """페이지마다 한 번 호출한다."""
    st.markdown(CSS, unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def badge(text: str, kind: str = "idle", help: str = "") -> str:
    """kind: ok | warn | bad | idle

    help 를 주면 브라우저 기본 툴팁(title 속성)으로 붙는다. 상태바 배지는
    st.help 를 쓸 수 없는 HTML 조각이라 이 방법 말고는 설명을 붙일 자리가 없다.
    """
    tip = f' title="{help}"' if help else ""
    return f'<span class="badge badge-{kind}"{tip}>{text}</span>'


def runbar(items: list[tuple[str, str, bool]], flag: tuple[str, str] | None = None) -> None:
    """상단 상태바.

    items : (라벨, 값, 값이 채워졌는지) 목록
    flag  : (라벨, badge HTML) — 오른쪽 끝에 붙는 상태 표식
    """
    cells = []
    for key, val, filled in items:
        dim = "" if filled else " dim"
        cells.append(
            f'<div class="rb-item"><span class="rb-k">{key}</span>'
            f'<span class="rb-v{dim}">{val}</span></div>')
    if flag is not None:
        cells.append(
            f'<div class="rb-item flag"><span class="rb-k">{flag[0]}</span>'
            f'<span class="rb-v">{flag[1]}</span></div>')
    st.markdown(f'<div class="runbar">{"".join(cells)}</div>', unsafe_allow_html=True)


def page_head(title: str, description: str = "") -> None:
    st.title(title)
    if description:
        caption(description)


# ── 내려받기 ─────────────────────────────────────────────────────────────
def csv_download(label: str, df, file_name: str, key: str, **kw) -> None:
    """CSV 내려받기 버튼. 바이트를 만드는 일은 한 번만 한다.

    download_button 은 눌리기 전에도 데이터를 미리 갖고 있어야 해서, 그냥 쓰면
    재실행마다 to_csv() 가 다시 돈다. expander 안에 있어도 마찬가지다 —
    접혀 있는 expander 의 본문도 실행되기 때문이다. 50만 행이면 위젯 하나 만질
    때마다 수 초가 그냥 사라진다. 프레임이 그대로면 지난 바이트를 재사용한다.
    """
    cache = st.session_state.setdefault("_csv_cache", {})
    stamp = (df.shape, tuple(map(str, df.columns)),
             str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "")
    hit = cache.get(key)
    if hit is None or hit[0] != stamp:
        hit = (stamp, df.to_csv().encode("utf-8-sig"))
        cache[key] = hit
    st.download_button(label, hit[1], file_name=file_name, mime="text/csv",
                       key=f"dl_{key}", **kw)
