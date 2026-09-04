"""단독 HTML 리포트.

파일 하나로 떨어지므로 메일 첨부나 사내 공유가 된다.
Plotly 는 CDN 대신 파일에 함께 담는 것이 기본이다 — 폐쇄망에서 CDN 은 로딩되지 않는다.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .plots import ACTUAL, FONT_STACK, GRID, INK, MUTED, PREDICTED

_CSS = f"""
:root {{
  --ink: {INK};
  --muted: {MUTED};
  --grid: {GRID};
  --actual: {ACTUAL};
  --predicted: {PREDICTED};
  --paper: #FFFFFF;
  --panel: #F7F8FA;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--panel); color: var(--ink);
  font-family: {FONT_STACK}; font-size: 15px; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}}
.sheet {{ max-width: 1120px; margin: 0 auto; background: var(--paper);
  border-left: 3px solid var(--actual); }}
header {{ padding: 40px 48px 28px; border-bottom: 1px solid var(--grid); }}
h1 {{ margin: 0 0 6px; font-size: 26px; font-weight: 650; letter-spacing: -0.01em; }}
.sub {{ color: var(--muted); font-size: 13.5px; margin: 0; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 28px; margin-top: 22px; }}
.meta div {{ min-width: 120px; }}
.meta dt {{ font-size: 12px; color: var(--muted); margin-bottom: 2px; }}
.meta dd {{ margin: 0; font-size: 14px; font-variant-numeric: tabular-nums; }}
.scores {{ display: flex; flex-wrap: wrap; gap: 1px; background: var(--grid);
  border-top: 1px solid var(--grid); border-bottom: 1px solid var(--grid); }}
.score {{ flex: 1 1 150px; background: var(--paper); padding: 20px 24px; }}
.score b {{ display: block; font-size: 28px; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; }}
.score span {{ font-size: 12px; color: var(--muted); }}
section {{ padding: 34px 48px; border-bottom: 1px solid var(--grid); }}
section:last-child {{ border-bottom: 0; }}
h2 {{ font-size: 17px; font-weight: 620; margin: 0 0 4px; display: flex;
  align-items: baseline; gap: 12px; }}
h2 i {{ font-style: normal; font-size: 12px; color: var(--muted);
  font-variant-numeric: tabular-nums; min-width: 22px; }}
.note {{ color: var(--muted); font-size: 13.5px; margin: 0 0 18px 34px; max-width: 68ch; }}
.body {{ margin-left: 34px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px;
  font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--muted); font-weight: 550; }}
tbody tr:hover {{ background: var(--panel); }}
.pass {{ color: #2E7D5B; font-weight: 600; }}
.fail {{ color: #A8322D; font-weight: 600; }}
footer {{ padding: 26px 48px 44px; color: var(--muted); font-size: 12.5px; }}
@media (max-width: 700px) {{
  header, section, footer {{ padding-left: 22px; padding-right: 22px; }}
  .note, .body {{ margin-left: 0; }}
}}
@media print {{
  body {{ background: #fff; }}
  .sheet {{ max-width: none; }}
  section {{ break-inside: avoid; }}
}}
"""


def _fig_html(fig, first: bool, embed: bool) -> str:
    if fig is None:
        return ""
    include = ("cdn" if not embed else True) if first else False
    return fig.to_html(full_html=False, include_plotlyjs=include,
                       config={"displaylogo": False, "responsive": True})


def _table(df: pd.DataFrame | None, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return '<p class="note">표시할 내용이 없습니다.</p>'
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.4g}")
    html = d.to_html(index=False, escape=True, border=0)
    return (html.replace(">통과<", ' class="pass">통과<')
                .replace(">실패<", ' class="fail">실패<'))


def build_report(
    title: str,
    meta: dict,
    scores: dict,
    sections: list[dict],
    embed_plotly: bool = True,
) -> str:
    """sections 는 [{'title','note','figures':[fig],'tables':[df]}] 형태."""
    meta_html = "".join(
        f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in meta.items()
    )
    score_html = "".join(
        f'<div class="score"><b>{v}</b><span>{k}</span></div>' for k, v in scores.items()
    )

    body, first = [], True
    for i, sec in enumerate(sections, start=1):
        figs = []
        for fig in sec.get("figures") or []:
            figs.append(_fig_html(fig, first, embed_plotly))
            first = False
        tables = "".join(_table(t) for t in (sec.get("tables") or []))
        note = f'<p class="note">{sec["note"]}</p>' if sec.get("note") else ""
        body.append(
            f'<section><h2><i>{i:02d}</i>{sec.get("title","")}</h2>{note}'
            f'<div class="body">{"".join(figs)}{tables}'
            f'{sec.get("html","")}</div></section>'
        )

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head>
<body><div class="sheet">
<header>
  <h1>{title}</h1>
  <p class="sub">시계열 머신러닝 실행 결과 · {datetime.now():%Y-%m-%d %H:%M}</p>
  <dl class="meta">{meta_html}</dl>
</header>
<div class="scores">{score_html}</div>
{''.join(body)}
<footer>
  구간은 시간순으로 학습 · 검증 · Final Unseen 으로 나뉩니다. 검증 구간은 모델을
  고르는 데 쓰였고, Final Unseen 은 학습·선별·모델선택 어디에도 쓰이지 않은 구간이라
  최종 성능 보고값입니다. 2분할로 실행한 경우에는 홀드아웃이 두 역할을 겸합니다.
  SHAP 결과는 모델이 학습한 통계적 관계이며, 설비의 인과 관계와는 다를 수 있습니다.
</footer>
</div></body></html>"""


def save_report(html: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
