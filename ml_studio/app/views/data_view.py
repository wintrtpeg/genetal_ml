"""1단계. 데이터 확보와 타겟 지정."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app import state
from core import datasource, pipeline, plots

SAMPLE_QUERY = """-- 시간 컬럼을 반드시 함께 뽑고 시간순으로 정렬하세요.
SELECT
    tag_time,
    flow_rate,
    inlet_temp,
    outlet_temp,
    differential_pressure,
    y_output
FROM  dm_utility.operation_5min
WHERE tag_time >= '2025-01-01'
  AND tag_time <  '2025-07-01'
ORDER BY tag_time
"""


def render() -> None:
    S = st.session_state
    st.title("1. 데이터")
    st.markdown('<p class="caption">CSV 파일을 올리거나 데이터마트에 쿼리를 던져 '
                '시계열을 가져옵니다. 그다음 시간축과 타겟(Y)을 정합니다.</p>',
                unsafe_allow_html=True)

    tab_csv, tab_sql = st.tabs(["로컬 CSV", "SQL 쿼리"])

    with tab_csv:
        _csv_panel()
    with tab_sql:
        _sql_panel()

    if S.raw is not None:
        st.divider()
        if _layout_panel():          # long 이면 여기서 돌려 세운다
            st.divider()
            _timeseries_panel()


# ─────────────────────────────────────────────────────────────
def _csv_panel() -> None:
    S = st.session_state
    st.markdown("**CSV 파일을 올리세요.** 시간 컬럼이 들어 있어야 합니다.")
    up = st.file_uploader("CSV 파일", type=["csv", "txt"], label_visibility="collapsed")

    # 아래 세 가지는 대부분 손댈 일이 없다. 기본값으로 안 읽힐 때만 열어 보게
    # 접어 둔다. 처음 열었을 때 알 수 없는 용어 세 개가 먼저 보이면
    # 파일을 올리기도 전에 막힌다.
    with st.expander("읽기 옵션 (기본값으로 안 읽힐 때만)"):
        c1, c2, c3 = st.columns(3)
        sep = c1.selectbox(
            "구분자", [",", ";", "\t", "|"], index=0,
            format_func=lambda s: {",": "쉼표  ,", ";": "세미콜론  ;",
                                   "\t": "탭", "|": "막대  |"}[s],
            help="값과 값 사이를 무엇으로 나눴는지. 엑셀에서 CSV 로 저장했으면 "
                 "쉼표입니다. 미리보기에서 여러 컬럼이 한 칸에 몰려 보이면 "
                 "이걸 바꿔 보세요.")
        enc = c2.selectbox(
            "인코딩", ["utf-8", "utf-8-sig", "cp949", "euc-kr"], index=0,
            help="한글이 깨져 보이면 바꾸세요. 한글 윈도우 엑셀로 저장한 파일은 "
                 "대개 cp949, 설비 시스템에서 바로 받은 파일은 대개 utf-8 입니다.")
        nrows = c3.number_input(
            "최대 행 (0=전체)", min_value=0, value=0, step=10000,
            help="파일이 아주 크면 먼저 10만 행으로 흐름을 확인한 뒤 0 으로 "
                 "바꾸는 편이 빠릅니다.")

    if up is not None and st.button("불러오기", type="primary", key="csv_load"):
        try:
            df = datasource.CsvSource(up, sep=sep, encoding=enc,
                                      nrows=nrows or None).load()
            S.raw = df
            S.source_desc = up.name
            state.invalidate("data")
            st.success(f"{len(df):,}행 × {df.shape[1]}열을 읽었습니다.")
        except UnicodeDecodeError:
            # 인코딩만 바꾸면 되는데 파이썬 원문 메시지는 그걸 알려주지 않는다.
            alt = "cp949" if enc.startswith("utf") else "utf-8"
            st.error(f"인코딩이 맞지 않습니다. 지금 **{enc}** 로 읽었는데 "
                     f"**{alt}** 일 가능성이 큽니다. 위 인코딩 칸을 바꿔 다시 눌러 보세요.")
        except Exception as e:  # noqa: BLE001
            st.error(f"읽지 못했습니다 — {e}")

    st.divider()
    st.caption("실데이터 없이 둘러보실 때. 유량·온도·압력·밸브개도로 만든 "
               "45일치 가상 설비 데이터입니다.")

    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[2] / "data"

    def _load_demo(name: str, desc: str) -> None:
        demo = data_dir / name
        if not demo.exists():
            st.warning("가상 데이터가 없습니다. 명령창에서 "
                       "`python scripts/make_demo_data.py` 를 한 번 실행하세요.")
            return
        S.raw = pd.read_csv(demo)
        S.source_desc = desc
        state.invalidate("data")
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("가상 데이터 (wide)", key="demo_load", use_container_width=True,
                 help="한 행이 한 시점. 가장 흔한 형태입니다."):
        _load_demo("demo_timeseries.csv", "가상 데이터 (wide)")
    if c2.button("가상 데이터 (long)", key="demo_load_long", use_container_width=True,
                 help="한 행이 '한 시점의 한 태그'. PI · IP.21 같은 히스토리언 "
                      "형태입니다. pivot 화면을 확인하실 수 있습니다."):
        _load_demo("demo_timeseries_long.csv", "가상 데이터 (long)")


def _sql_panel() -> None:
    S = st.session_state
    st.markdown("**접속 정보**")
    st.caption("평소 쓰시던 SQL 툴(DBeaver·DataGrip 등)의 접속 설정에 적힌 값 "
               "그대로입니다. 계정·비밀번호는 화면 안에서만 쓰이고 실행 결과에 "
               "저장되지 않습니다.")

    mode = st.radio("입력 방식", ["항목별 입력", "접속 URL 직접 입력"],
                    horizontal=True, label_visibility="collapsed",
                    help="이미 접속 문자열을 갖고 계시면 두 번째를 쓰세요.")

    connect_args: dict = {}

    if mode == "항목별 입력":
        c1, c2 = st.columns([2, 1])
        dialect_name = c1.selectbox(
            "DBMS", list(datasource.DIALECTS),
            help="사내에서 쓰는 데이터베이스 종류입니다. 모르시면 DB 담당자에게 "
                 "물어보세요.")
        driver = datasource.DIALECTS[dialect_name]
        c2.text_input("드라이버", value=driver, disabled=True,
                      help="DBMS 에 맞춰 자동으로 정해집니다.")

        pkg = datasource.DRIVER_PACKAGES.get(driver)
        if pkg and pkg != "(내장)":
            st.caption(f"이 DB 에 붙으려면 프로그램이 하나 더 필요합니다 — "
                       f"명령창에서 `pip install {pkg}`")

        default_port = datasource.DEFAULT_PORTS.get(driver, "")
        c1, c2, c3 = st.columns([3, 1, 2])
        host = c1.text_input(
            "호스트", placeholder="dm.internal.example.com",
            help="데이터마트 서버 주소. 사내 도메인이거나 192.168.x.x 형태입니다. "
                 "DBeaver 등의 접속 설정에 적혀 있는 값 그대로입니다.")
        port = c2.text_input(
            "포트", value=str(default_port) if default_port else "",
            help="DBMS 기본값이 자동으로 채워집니다. 담당자가 다른 번호를 "
                 "알려주지 않았다면 그대로 두세요. (SQream 기본 3108)")
        database = c3.text_input(
            "데이터베이스", placeholder="master",
            help="접속할 DB 이름. 쿼리에 쓰는 '스키마.테이블' 의 스키마와는 "
                 "다를 수 있습니다 — 스키마는 쿼리 안에서 지정하세요.")

        c1, c2 = st.columns(2)
        user = c1.text_input("계정")
        password = c2.text_input("비밀번호", type="password",
                                 help="화면 안에서만 쓰이고 runs/ 에 저장되지 않습니다.")

        extra = st.text_input(
            "URL 추가 파라미터 (선택)", value="",
            help="담당자가 알려준 옵션이 있을 때만. 형식: key=value, key=value")
        params = {}
        for part in extra.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()

        # 일부 드라이버는 URL 이 아니라 connect_args 로 옵션을 받는다.
        # SQream 의 clustered 가 그렇다 — 로드밸런서를 거치면 True 여야 한다.
        defaults = datasource.DEFAULT_CONNECT_ARGS.get(driver)
        if defaults:
            with st.expander("고급 접속 옵션 (connect_args)", expanded=False):
                st.caption("URL 이 아니라 드라이버에 직접 넘기는 값입니다. "
                           "SQream 은 로드밸런서를 거칠 때 clustered 가 true 여야 "
                           "합니다 — 기본값이 이미 채워져 있습니다.")
                raw_args = st.text_area(
                    "connect_args", value=json.dumps(defaults, ensure_ascii=False),
                    height=80, label_visibility="collapsed")
                try:
                    connect_args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError as e:
                    st.error(f"JSON 형식이 아닙니다 — {e}")

        url = datasource.build_url(driver, host, port, database, user, password, params or None)
    else:
        url = st.text_input(
            "SQLAlchemy 접속 URL", value=S.sql_url or "",
            placeholder="sqream://user:password@host:3108/master",
            type="password",
            help="쓰시던 접속 문자열을 그대로 붙여넣으세요. "
                 "비밀번호가 들어 있어 가려서 표시됩니다.")
        raw_args = st.text_input("connect_args (JSON, 선택)", value="")
        if raw_args.strip():
            try:
                connect_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                st.error(f"JSON 형식이 아닙니다 — {e}")

    if url:
        shown = f'sa.create_engine("{datasource.mask_url(url)}"'
        shown += f", connect_args={connect_args})" if connect_args else ")"
        st.code(f"engine = {shown}", language="python")

    st.markdown("**쿼리**")
    st.caption("평소 쓰시던 SELECT 문을 그대로 붙여넣으시면 됩니다. "
               "**시간 컬럼을 꼭 함께 뽑고 시간순으로 정렬**해 주세요 — "
               "시계열이라 순서가 틀리면 아무것도 못 합니다. "
               "안전을 위해 조회(SELECT·WITH)만 실행되고 데이터를 바꾸는 문장은 막습니다.")
    query = st.text_area("SQL", value=S.sql_query, height=260,
                         label_visibility="collapsed")
    if st.button("예시 쿼리 넣기", key="sql_sample"):
        S.sql_query = SAMPLE_QUERY
        st.rerun()

    c1, c2, c3 = st.columns([1, 1, 2])
    chunk = c3.number_input(
        "청크 크기 (0=한 번에)", min_value=0, value=100000, step=10000,
        help="큰 결과를 나눠서 받습니다. 메모리가 부족할 때 줄이세요.")

    if c1.button("접속 확인", key="sql_test"):
        try:
            msg = datasource.SqlAlchemySource(
                url, "SELECT 1", connect_args=connect_args).test_connection()
            st.success(msg)
        except Exception as e:  # noqa: BLE001
            st.error(f"접속 실패 — {type(e).__name__}: {e}")

    if c2.button("실행", type="primary", key="sql_run"):
        S.sql_query, S.sql_url = query, url
        try:
            src = datasource.SqlAlchemySource(url, query, chunksize=chunk or None,
                                              connect_args=connect_args)
            with st.spinner("조회 중"):
                df = src.load()
            if df.empty:
                st.warning("결과가 0행입니다. 조건을 확인해 주세요.")
            else:
                S.raw = df
                S.source_desc = "데이터마트 조회"
                state.invalidate("data")
                st.success(f"{len(df):,}행 × {df.shape[1]}열을 읽었습니다.")
        except datasource.QueryNotAllowed as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"조회 실패 — {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────
def _layout_panel() -> bool:
    """표가 세로로 긴 모양(long)이면 가로로 돌려 세운다.

    설비 데이터는 두 모양으로 나온다. 모델은 wide 를 전제로 하므로 long 이면
    먼저 돌려야 한다. PI·IP.21 같은 히스토리언은 long 이 기본이라 사내
    데이터마트에서 뽑으면 이쪽이 흔하다.

    반환: 다음 단계로 진행해도 되는가 (long 인데 아직 안 돌렸으면 False)
    """
    S = st.session_state
    raw = S.raw
    if raw is None or raw.empty:
        return True

    det = datasource.detect_layout(raw)
    S["_layout_det"] = det

    st.header("표 형태 (wide / long)")
    with st.expander("wide 와 long 이 무엇인가", expanded=False):
        st.markdown(
            "설비 데이터는 보통 둘 중 하나입니다.\n\n"
            "**wide** — 한 행이 한 시점. 태그가 각각 컬럼을 차지합니다.\n"
            "```\n"
            "tag_time            flow   temp   press\n"
            "2025-01-01 00:00    52.3   65.1   3.2\n"
            "2025-01-01 00:05    52.9   65.4   3.2\n"
            "```\n"
            "**long** — 한 행이 '한 시점의 한 태그'.\n"
            "```\n"
            "tag_time            tag_name   value\n"
            "2025-01-01 00:00    flow       52.3\n"
            "2025-01-01 00:00    temp       65.1\n"
            "2025-01-01 00:00    press      3.2\n"
            "```\n"
            "PI · IP.21 같은 히스토리언은 **long 이 기본**입니다 — 태그를 추가해도 "
            "테이블 구조를 안 바꿔도 되기 때문입니다.\n\n"
            "모델은 '한 행 = 그 시점의 설비 상태' 를 전제로 하므로 **wide 가 "
            "필요**합니다. long 이면 여기서 pivot 합니다 — 값은 그대로이고 "
            "배치만 바뀝니다.")

    manual = st.radio(
        "표 형태", ["자동 판정", "wide", "long"], horizontal=True,
        help="자동 판정이 틀렸다고 보이면 직접 고르세요.")

    if manual == "wide":
        layout = "wide"
    elif manual == "long":
        layout = "long"
    else:
        layout = det["layout"]

    if layout == "wide":
        if manual == "자동 판정":
            st.success(f"**wide** 로 판정했습니다 — {len(raw):,}행 × {raw.shape[1]}열. "
                       "그대로 진행하시면 됩니다.")
        else:
            st.info("wide 로 진행합니다.")
        st.dataframe(raw.head(5), use_container_width=True)
        return True

    # ── 세로형 ────────────────────────────────────────────
    st.warning(f"**long** 으로 보입니다 (확신 {det['confidence']:.0%}). "
               "아래에서 확인하고 wide 로 pivot 해 주세요.")
    for r in det["reasons"]:
        st.caption(f"· {r}")

    cols = list(raw.columns)

    def _idx(name, fallback=0):
        return cols.index(name) if name in cols else fallback

    c1, c2, c3 = st.columns(3)
    time_col = c1.selectbox("시간 컬럼", cols, index=_idx(det["time_col"]),
                            key="lw_time")
    tag_col = c2.selectbox("태그 컬럼", cols,
                           index=_idx(det["tag_col"], 1), key="lw_tag",
                           help="FLOW_01 · TI-101 같은 태그명이 담긴 컬럼입니다. "
                                "이 컬럼의 고유값 하나하나가 pivot 후 컬럼이 됩니다.")
    value_col = c3.selectbox("값 컬럼", cols,
                             index=_idx(det["value_col"], 2), key="lw_value",
                             help="숫자 측정값이 담긴 컬럼입니다.")

    if len({time_col, tag_col, value_col}) < 3:
        st.error("세 컬럼은 서로 달라야 합니다.")
        return False

    try:
        inv = datasource.tag_inventory(raw, time_col, tag_col, value_col)
    except Exception as e:  # noqa: BLE001
        st.error(f"태그 목록을 만들지 못했습니다 — {e}")
        return False

    st.markdown(f"**태그 {len(inv):,}개**")
    st.caption("pivot 하면 각 태그가 컬럼 하나가 됩니다. 행 수가 유독 적은 태그는 "
               "일부 기간에만 계측된 것이라 결측이 많아집니다.")
    st.dataframe(inv, use_container_width=True, hide_index=True, height=240,
                 column_config={
                     "줄 수": st.column_config.NumberColumn("행 수", format="%d"),
                     "숫자로 읽힌 비율": st.column_config.NumberColumn(
                         "수치 비율", format="%.1f%%",
                         help="1.0 이 아니면 숫자가 아닌 값이 섞여 있습니다."),
                 })

    all_tags = list(inv["태그"])
    c1, c2 = st.columns([3, 1])
    picked = c1.multiselect("사용할 태그 (비우면 전부)", all_tags, default=[],
                            help="태그가 아주 많으면 필요한 것만 고르는 편이 빠릅니다. "
                                 "타겟(Y)이 될 태그를 반드시 포함하세요.")
    agg = c2.selectbox(
        "중복 시각 처리", ["mean", "last", "max", "min"],
        format_func=lambda k: {"mean": "mean (평균)", "last": "last (마지막)",
                               "max": "max", "min": "min"}[k],
        help="같은 (시각, 태그) 가 두 번 이상 있을 때 합치는 방법입니다. "
             "히스토리언에서 같은 초에 두 번 기록되는 일이 있습니다.")

    if st.button("wide 로 pivot", type="primary"):
        try:
            with st.spinner("pivot 중"):
                wide = datasource.long_to_wide(
                    raw, time_col, tag_col, value_col,
                    agg=agg, tags=picked or None)
        except datasource.LayoutError as e:
            st.error(str(e))
            return False
        except Exception as e:  # noqa: BLE001
            st.error(f"pivot 실패 — {type(e).__name__}: {e}")
            return False

        info = wide.attrs.get("long_to_wide", {})
        S.raw = wide
        S.source_desc = f"{S.source_desc or '데이터'} (long→wide)"
        state.invalidate("data")
        st.success(f"태그 **{info.get('태그 수', 0)}개** × "
                   f"**{info.get('시점 수', 0):,}시점** 으로 pivot 했습니다.")
        st.rerun()

    st.caption("pivot 전 미리보기 (앞 5행)")
    st.dataframe(raw.head(5), use_container_width=True)
    return False


def _timeseries_panel() -> None:
    S = st.session_state
    raw = S.raw
    st.header("시간축 정리")
    st.caption("시간 컬럼을 골라 DatetimeIndex 로 만듭니다. "
               "시간순 정렬하고 중복 시각은 마지막 값만 남깁니다.")

    # 빈 결과가 여기까지 오면 아래 위젯이 전부 IndexError 를 낸다.
    # (WHERE 조건이 과하게 좁은 쿼리에서 실제로 나온다)
    if raw.empty or raw.shape[1] == 0:
        st.warning("읽어온 데이터가 비어 있습니다. 쿼리 조건이나 CSV 내용을 확인하세요.")
        return

    guess = datasource.guess_time_column(raw)
    c1, c2, c3 = st.columns([2, 1, 1])
    time_col = c1.selectbox(
        "시간 컬럼", list(raw.columns),
        index=list(raw.columns).index(guess) if guess else 0,
        help="tag_time · timestamp · 일시 같은 컬럼입니다. 이름으로 자동 추측해 "
             "미리 골라 뒀으니 맞는지만 확인해 주세요.")
    do_res = c2.checkbox(
        "리샘플링", value=False,
        help="간격이 제각각이거나 너무 촘촘할 때 일정 간격으로 묶어 평균을 냅니다. "
             "1초 데이터를 5분 평균으로 줄이면 노이즈가 줄고 계산도 빨라집니다.")
    rule = c3.text_input(
        "주기", value="5min", disabled=not do_res,
        help="5min = 5분, 1h = 1시간, 1D = 하루.")

    if st.button("시계열로 변환", type="primary"):
        try:
            df = datasource.to_timeseries(raw, time_col,
                                          resample=rule if do_res else None)
            S.df, S.time_col = df, time_col
            state.invalidate("data")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"변환 실패 — {e}")

    st.dataframe(raw.head(50), use_container_width=True, height=240)

    if S.df is None:
        return

    st.header("타겟 지정")
    st.caption("예측하려는 값 하나를 고르세요 — 수율·품질지표·소비전력 같은 것입니다. "
               "**나머지 컬럼이 전부 X 후보로 등록됩니다.**")
    df = S.df
    if df.empty or df.shape[1] == 0:
        st.error("시계열로 바꾸고 나니 남은 행이 없습니다. "
                 "시간 컬럼이 잘못됐거나 시간값이 전부 비어 있는 경우입니다.")
        return

    freq = datasource.infer_freq(df.index)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행", f"{len(df):,}")
    c2.metric("열", df.shape[1])
    c3.metric("추정 주기", freq or "불규칙",
              help="'불규칙' 이면 간격이 일정하지 않다는 뜻입니다. "
                   "위에서 리샘플링을 쓰면 맞출 수 있습니다.")
    c4.metric("기간", f"{(df.index[-1] - df.index[0]).days}일")

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target = st.selectbox("타겟 (Y)", numeric or list(df.columns),
                          index=(numeric.index(S.target) if S.target in numeric else
                                 (len(numeric) - 1 if numeric else 0)))
    if target != S.target:
        S.target = target
        S.candidates = [c for c in df.columns if c != target]
        state.invalidate("data")

    if S.target:
        st.success(f"타겟 **{S.target}** · 나머지 **{len(S.candidates)}개**를 "
                   "X 후보로 등록했습니다.")
        y = df[S.target]
        # 12,000점을 그대로 넘기면 브라우저가 버틴다 해도 느려진다.
        st.line_chart(plots.thin(y, 2000), height=220)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("평균", f"{y.mean():,.4g}")
        c2.metric("표준편차", f"{y.std():,.4g}")
        c3.metric("결측", f"{y.isna().mean():.2%}")
        c4.metric("고유값", f"{y.nunique():,}",
                  help="아주 작으면(2~5개) 연속값이 아니라 등급·상태일 수 있습니다. "
                       "그러면 분류 문제로 잡힙니다.")

        if state.mode() == "Auto":
            st.divider()
            _auto_panel()


def _auto_panel() -> None:
    """Auto 모드 — 타겟만 고르면 챔피언까지 한 번에.

    자동화되는 것은 '사람이 버튼을 누르는 일' 이지 '검증을 생략하는 일' 이 아니다.
    3분할·gap 점검·선별구간 추적·폴드 내부 선별·Unseen 1회 접근은 그대로 돈다.
    """
    S = st.session_state
    st.header("한 번에 실행")
    st.markdown('<p class="caption">1~4단계를 이어서 돌립니다. 누수 방지 장치는 그대로 '
                '작동하고, 점검을 통과하지 못하면 중단합니다. 사람이 고르던 값은 '
                '기본값으로 대체되며 무엇을 어떻게 정했는지 표로 남깁니다.</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    top_k = c1.number_input("선별 상한", 5, 200, 40)
    unseen = c2.slider("Final Unseen 비율", 0.0, 0.4, 0.15, 0.05)
    quick = c3.checkbox(
        "빠르게 훑기", value=True,
        help="가벼운 모델 3종만 돌립니다. 화면이 제대로 도는지 먼저 볼 때 쓰세요. "
             "끄면 설치된 모델을 전부 비교합니다 — 데이터가 크면 몇 분 걸립니다.")
    heavy = False if quick else st.checkbox(
        "느린 모델(KNN·SVM·MLP)까지 포함", value=False)

    n_rows = len(S.df) if S.df is not None else 0
    if not quick and n_rows > 5000:
        st.warning(f"{n_rows:,}행에 전체 모델을 돌리면 몇 분 동안 화면이 멈춘 것처럼 "
                   "보입니다. 진행 표시가 올라가면 도는 중입니다.")

    if st.button("전체 자동 실행", type="primary", use_container_width=True):
        bar, label = st.progress(0.0), st.empty()

        def tick(i, total, msg):
            bar.progress(i / total)
            label.caption(f"{i}/{total} · {msg}")

        try:
            with st.spinner("실행 중"):
                res = pipeline.run_auto(
                    S.df, S.target,
                    pipeline.AutoConfig(top_k=int(top_k), unseen_ratio=float(unseen),
                                        include_heavy=bool(heavy),
                                        max_models=3 if quick else None,
                                        n_splits=3 if quick else 4),
                    progress=tick)
        except pipeline.AutoRunError as e:
            bar.empty(); label.empty()
            st.error(f"중단됨 — {e}")
            return
        except Exception as e:  # noqa: BLE001
            bar.empty(); label.empty()
            st.error(f"실패 — {type(e).__name__}: {e}")
            return

        bar.progress(1.0); label.empty()
        _apply(res)
        st.rerun()

    if S.get("auto_decisions") is not None:
        st.success(f"챔피언 **{S.champion}** · X 피처 {len(S.selected_features):,}개")
        if S.unseen_scores:
            cols = st.columns(len(S.unseen_scores) - 1 or 1)
            for c, (k, v) in zip(cols, [(k, v) for k, v in S.unseen_scores.items()
                                        if k != "unseen_rows"]):
                c.metric(f"Final Unseen {k.replace('unseen_', '')}", f"{v:.4f}")
        st.markdown("**자동으로 정해진 것들**")
        st.caption("자동으로 돌렸어도 무엇이 어떻게 결정됐는지 되짚을 수 있어야 합니다. "
                   "바꾸고 싶은 항목이 있으면 Guided 나 Expert 모드로 그 단계만 다시 하세요.")
        st.dataframe(S.auto_decisions, use_container_width=True, hide_index=True)
        st.info("5단계 예측 · 6단계 SHAP · 8단계 진단으로 바로 넘어가실 수 있습니다.")


def _apply(res) -> None:
    """AutoResult 를 화면 상태에 꽂는다. 수동 경로와 같은 키를 쓴다."""
    S = st.session_state
    # 먼저 아래 단계를 전부 비운다. 수동으로 한 번 돌리고 리포트까지 저장한 뒤
    # Auto 를 돌리면, 예전 실행의 예측·리포트·manifest 가 남아 새 실행의 것처럼
    # 보인다. 터지지 않고 틀린 것을 보여주는 쪽이라 더 나쁘다.
    state.invalidate("data")
    S.kept = res.kept
    S.feat_df, S.provenance, S.feature_config = res.feat_df, res.provenance, res.feature_config
    S.prep_config = res.prep_config
    S.selection_report = res.selection_report
    S.feature_review = res.selection_report
    S.review_picks = res.selected_features
    S.selected_features = res.selected_features
    S.X_pool = res.X            # Auto 는 확정까지 끝낸 상태라 pool 과 X 가 같다
    S.X, S.y = res.X, res.y
    S.split, S.split_config = res.split, res.split_config
    S.train_idx, S.test_idx = res.split.train, res.split.valid
    S.unseen_idx = res.split.unseen
    S.selection_train_idx = res.split.train
    S.task = res.task
    S.train_config = res.train_config
    S.leaderboard, S.detail, S.champion = res.leaderboard, res.detail, res.champion
    S.unseen_scores = res.unseen_scores
    S.unseen_guard = res.unseen_guard
    S.ensemble_report = res.ensemble_report
    S.auto_decisions = res.decisions
    S.learning_mode = "지도학습"

