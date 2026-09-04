"""8단계. 결과 저장과 단독 HTML 리포트."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import state, theme
from core import diagnostics, explain, persist, plots, report, train


def render() -> None:
    S = st.session_state
    st.title("9. 리포트")
    if not state.guard("report", "먼저 학습을 실행해 주세요."):
        return

    st.markdown('<p class="caption">파일 하나로 떨어지는 HTML 입니다. '
                'Plotly 를 파일에 함께 담으므로 인터넷 없이도 열립니다.</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    title = c1.text_input("제목", value=f"{S.target} 예측 모델 — {S.champion or '분석'}")
    embed = c2.checkbox("Plotly 포함", value=True,
                        help="끄면 파일이 작아지지만 인터넷 연결이 필요합니다.")

    ALL = ["Final Unseen", "구간 분할", "리더보드", "앙상블 판정", "누수 점검",
           "폴드 안정성", "실측 대비 예측", "잔차", "잔차 진단", "Rolling Backtest",
           "분할 방식 진단", "SHAP 중요도", "SHAP dependence", "피처 선별 이력",
           "피처 출처", "재현 기록"]
    DEFAULT = ["Final Unseen", "구간 분할", "리더보드", "누수 점검", "실측 대비 예측",
               "잔차 진단", "SHAP 중요도", "SHAP dependence", "피처 선별 이력", "재현 기록"]
    include = st.multiselect("포함할 내용", ALL, default=DEFAULT)
    st.caption("있는 것만 담깁니다. 실행하지 않은 항목(backtest·분할 진단 등)은 "
               "골라도 그냥 넘어갑니다.")

    if st.button("리포트 만들기", type="primary"):
        with st.spinner("생성 중"):
            html = _build(title, include, embed)
        S.report_html = html
        run_dir = persist.new_run_dir(tag=(S.champion or "run").replace(" ", ""))
        pipe = state.champion_pipeline()
        manifest = persist.build_manifest(
            run_id=run_dir.name, target=S.target, df=S.df, split=S.split,
            index=S.X.index if S.X is not None else None,
            seed=getattr(S.train_config, "seed", None),
            champion=S.champion, configs=_config_snapshot(),
            selection_report=S.selection_report, unseen_scores=S.unseen_scores,
            source_desc=S.source_desc or "")
        S.manifest = manifest
        S.saved = persist.save_run(
            run_dir, pipeline=pipe, leaderboard=S.leaderboard,
            predictions=S.predictions, provenance=S.provenance,
            config=_config_snapshot(), report_html=html,
            manifest=manifest, selection_report=S.selection_report)
        S.run_dir = str(run_dir)
        st.rerun()

    if not S.get("saved"):
        return

    st.success(f"저장 위치 · `{S.run_dir}`")
    st.dataframe(pd.DataFrame([{"항목": k, "경로": v} for k, v in S.saved.items()]),
                 **theme.WIDE, hide_index=True)

    st.download_button("HTML 리포트 내려받기", S.report_html.encode("utf-8"),
                       file_name="ml_report.html", mime="text/html", type="primary")

    _reproducibility_panel()

    with st.expander("지난 실행 목록"):
        runs = persist.list_runs()
        st.dataframe(runs, **theme.WIDE, hide_index=True) if len(runs) \
            else st.caption("아직 없습니다.")


def _reproducibility_panel() -> None:
    """SPEC §26 — 몇 달 뒤 같은 결과가 안 나올 때 첫 번째로 볼 곳."""
    S = st.session_state
    man = S.get("manifest")
    if not man:
        return

    st.divider()
    st.header("재현 기록")
    st.markdown('<p class="caption">이 run 을 나중에 그대로 되살리는 데 필요한 것들입니다. '
                'manifest.json 한 파일에 모두 담깁니다.</p>', unsafe_allow_html=True)

    ds = man.get("dataset", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", man.get("run_id", "-"))
    c2.metric("데이터 지문", (ds.get("sha256") or "")[:12] or "-",
              help="같은 파일인지 확인하는 해시입니다. 값이 하나라도 바뀌면 달라집니다.")
    c3.metric("행 × 열", f"{ds.get('rows', 0):,} × {ds.get('columns', 0)}")
    c4.metric("seed", str(man.get("seed", "-")))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**구간 경계**")
        bounds = man.get("split_bounds", {})
        if bounds:
            st.dataframe(pd.DataFrame([
                {"구간": k, "행수": v["rows"], "시작": v["start"], "끝": v["end"]}
                for k, v in bounds.items()
            ]), **theme.WIDE, hide_index=True)
        else:
            st.caption("기록 없음")
    with c2:
        st.markdown("**패키지 버전**")
        pk = man.get("packages", {})
        st.dataframe(pd.DataFrame([{"패키지": k, "버전": v} for k, v in pk.items()]),
                     **theme.WIDE, hide_index=True, height=240)

    excluded = man.get("features_excluded", [])
    if excluded:
        with st.expander(f"제외 피처 사유 {len(excluded):,}건"):
            st.dataframe(pd.DataFrame(excluded), **theme.WIDE,
                         hide_index=True, height=300)

    with st.expander("지난 run 과 대조"):
        st.caption("\"지난달 결과가 안 나온다\" 의 원인은 대개 데이터 지문이나 패키지 버전입니다.")
        runs = persist.list_runs()
        others = [r for r in runs["run"].tolist() if r != man.get("run_id")] if len(runs) else []
        if not others:
            st.caption("비교할 지난 run 이 없습니다.")
        else:
            pick = st.selectbox("비교 대상", others)
            loaded = persist.load_run(persist.RUNS_DIR / pick)
            other_man = loaded.get("manifest")
            if not other_man:
                st.warning("그 run 에는 manifest.json 이 없습니다 (구버전 실행).")
            else:
                cmp = persist.compare_manifests(man, other_man)
                diff = cmp[cmp["일치"] == "✕"]
                if diff.empty:
                    st.success("대조한 항목이 모두 일치합니다.")
                else:
                    st.warning(f"{len(diff)}개 항목이 다릅니다.")
                st.dataframe(cmp, **theme.WIDE, hide_index=True, height=320)


def _config_snapshot() -> dict:
    S = st.session_state
    return {
        "source": S.source_desc,
        "target": S.target,
        "time_col": S.time_col,
        "n_rows": int(len(S.df)) if S.df is not None else None,
        "kept_columns": S.kept,
        "selected_features": S.selected_features,
        "feature_config": S.feature_config,
        "prep_config": S.prep_config,
        "split_config": S.split_config,
        "champion": S.champion,
        "learning_mode": S.learning_mode,
        # 접속 URL·계정은 남기지 않는다. 쿼리 본문만 재현용으로 보관.
        "sql_query": S.sql_query if S.source_desc == "데이터마트 조회" else None,
    }


def _build(title: str, include: list[str], embed: bool) -> str:
    S = st.session_state
    three_way = S.unseen_idx is not None and len(S.unseen_idx) > 0
    metric = S.train_config.champion_metric if S.train_config else "R2"

    meta = {
        "데이터": S.source_desc or "-",
        "타겟": S.target or "-",
        "행 수": f"{len(S.df):,}" if S.df is not None else "-",
        "X 피처": f"{len(S.selected_features):,}",
        "챔피언": S.champion or "-",
        "분할": ("3분할 (학습 / 검증 / Final Unseen)" if three_way
               else "2분할 (구버전 호환 — 검증이 보고를 겸함)"),
    }

    # 최종 성능은 Final Unseen 값이다. 검증 점수는 모델 선택에 이미 쓰였으므로
    # 같은 자리에 놓으면 어느 쪽이 보고값인지 흐려진다.
    scores: dict = {}
    if three_way and S.unseen_scores:
        for k, v in S.unseen_scores.items():
            if k != "unseen_rows" and pd.notna(v):
                scores[f"Final Unseen {k.replace('unseen_', '')}"] = f"{v:,.4f}"

    metric_cols = [c for c in (S.leaderboard.columns if S.leaderboard is not None else [])
                   if c.startswith("holdout_")]
    if S.leaderboard is not None and S.champion:
        row = S.leaderboard[S.leaderboard["model"] == S.champion]
        if not row.empty:
            label = "검증 " if three_way else "홀드아웃 "
            for c in metric_cols:
                v = row.iloc[0][c]
                if pd.notna(v):
                    scores[label + c.replace("holdout_", "")] = f"{v:,.4f}"

    sections: list[dict] = []

    # ── Final Unseen 을 맨 앞에 둔다. 이것이 보고값이다 ──
    if "Final Unseen" in include and three_way and S.unseen_scores:
        tables = [pd.DataFrame([{
            k.replace("unseen_", ""): (round(v, 6) if k != "unseen_rows" else int(v))
            for k, v in S.unseen_scores.items()}])]
        if S.leaderboard is not None and S.champion:
            tables.append(train.selection_bias_report(
                S.leaderboard, S.champion, S.unseen_scores, metric))
        sections.append({
            "title": "최종 성능 (Final Unseen)",
            "note": "학습·피처선별·모델선택 어디에도 쓰이지 않은 구간입니다. "
                    "이 값이 최종 일반화 성능 보고값입니다. 아래 검증 점수는 챔피언을 "
                    "고르는 데 이미 쓰였으므로 모델 수만큼 선택 편향이 들어 있습니다.",
            "tables": tables,
        })

    if "구간 분할" in include and S.split is not None and S.X is not None:
        sections.append({
            "title": "구간 분할",
            "note": "분할은 파생변수 단계에서 한 번만 정하고 학습 단계는 읽기만 합니다. "
                    "단계마다 다시 나누면 피처 선별에 쓴 구간이 평가 구간으로 넘어갑니다.",
            "tables": [S.split.describe(S.X.index)],
        })

    if "리더보드" in include and S.leaderboard is not None:
        cols = ["rank", "model", "family"] + metric_cols + [
            "cv_" + metric, "fit_seconds", "status"]
        sections.append({
            "title": "모델 비교",
            "note": "모든 모델이 같은 분할·같은 전처리로 학습됐습니다. "
                    + ("holdout_ 열은 <b>검증 구간</b> 성능이며 챔피언 선정에 쓰였습니다."
                       if three_way else
                       "holdout_ 열은 마지막 구간 성능이며 선정과 보고를 겸합니다."),
            "figures": [plots.leaderboard_bar(S.leaderboard, metric)],
            "tables": [S.leaderboard[[c for c in cols if c in S.leaderboard.columns]]],
        })

    if "앙상블 판정" in include and S.ensemble_report is not None \
            and not S.ensemble_report.empty:
        sections.append({
            "title": "앙상블 자동채택 판정",
            "note": "단일 최고 모델 대비 임계값 이상 좋아진 앙상블만 챔피언으로 삼습니다. "
                    "미미한 개선으로 복잡한 모델을 고르면 해석과 운영 비용만 늘어납니다.",
            "tables": [S.ensemble_report],
        })

    if "누수 점검" in include and S.X is not None:
        from core import features, validation
        chk = validation.leakage_checklist(
            S.X.index, S.train_idx, S.test_idx, list(S.X.columns), S.target,
            S.provenance, S.split_config.gap,
            features.warmup_rows(S.feature_config, S.X.index if S.X is not None else None)
            if S.feature_config else 0,
            selection_idx=S.selection_train_idx, unseen_idx=S.unseen_idx)
        sections.append({
            "title": "누수 점검",
            "note": "지도학습 경로에서 이 점검을 통과하지 못하면 학습이 진행되지 않습니다. "
                    "선별 구간 격리와 Final Unseen 격리는 시간 순서가 아니라 실제 행 "
                    "인덱스의 교집합을 봅니다.",
            "tables": [chk],
        })

    if "폴드 안정성" in include and S.fold_stability is not None \
            and not S.fold_stability.empty:
        sections.append({
            "title": "폴드 내부 선별 안정성",
            "note": "CV 폴드마다 다시 선별한 결과의 중복도입니다. Jaccard 가 낮으면 "
                    "'어떤 피처가 중요한가'의 답이 구간마다 달라진다는 뜻이라, "
                    "아래 SHAP 해석도 그만큼 조심해서 읽어야 합니다.",
            "tables": [S.fold_stability],
        })

    if S.predictions is not None:
        res = S.predictions.dropna()
        valid_start = S.X.index[S.test_idx[0]]
        if "실측 대비 예측" in include:
            sections.append({
                "title": "실측 대비 예측",
                "note": f"점선 오른쪽이 검증 구간입니다 ({valid_start:%Y-%m-%d %H:%M} 이후)."
                        + (f" 그 뒤 {S.X.index[S.unseen_idx[0]]:%Y-%m-%d %H:%M} 부터는 "
                           "Final Unseen 구간입니다." if three_way else ""),
                "figures": [plots.actual_vs_pred(res["actual"], res["predicted"],
                                                 valid_start, ylabel=S.target)],
            })
        if "잔차" in include:
            sections.append({
                "title": "잔차",
                "note": "특정 구간에서 잔차가 한쪽으로 쏠린다면 그 구간의 운전 조건이 "
                        "학습 구간과 다르다는 신호입니다.",
                "figures": [plots.residual_series(res["actual"], res["predicted"]),
                            plots.scatter_actual_pred(res["actual"], res["predicted"])],
            })
        if "잔차 진단" in include:
            r = diagnostics.residuals(res["actual"], res["predicted"])
            if len(r) >= 10:
                dcfg = diagnostics.ResidualConfig(window=min(96, max(6, len(r) // 10)))
                drift = diagnostics.drift_table(r, dcfg)
                summ = diagnostics.summary(r, dcfg)
                note = ("R2 한 숫자로는 '어디서 어떻게 틀렸는지'가 안 보입니다. "
                        f"잔차 평균 {summ['mean']:+.4f} · lag1 자기상관 {summ['lag1_acf']:.3f} "
                        f"· 이상점 {summ['outliers']:,}건. ")
                if not drift.empty:
                    note += diagnostics.drift_verdict(drift)["message"]
                figs = [plots.residual_band(diagnostics.rolling_stats(r, dcfg),
                                            diagnostics.outliers(r, dcfg))]
                tables = []
                if not drift.empty:
                    figs.append(plots.residual_drift(drift))
                    tables.append(drift[["구간", "행수", "mean", "std", "MAE", "MAE_배율"]]
                                  .round(4))
                acf = diagnostics.autocorrelation(r, dcfg)
                if not acf.empty:
                    figs.append(plots.residual_acf(acf, len(r)))
                sections.append({"title": "잔차 진단", "note": note,
                                 "figures": figs, "tables": tables})

    if "Rolling Backtest" in include and S.backtest:
        bt = S.backtest["table"]
        summ = train.backtest_summary(bt, metric)
        note = ("시기를 굴리며 같은 절차로 재학습·평가한 결과입니다. "
                "한 번의 검증 점수가 운이었는지 실력이었는지 가릅니다. ")
        if summ:
            note += (f"{metric} 평균 {summ['평균']:.4f} · 표준편차 {summ['표준편차']:.4f} "
                     f"· 최저 {summ['최저']:.4f} ({summ['최악구간']}구간).")
        show = [c for c in ("구간", "학습", "평가시작", "평가끝", "n_train", "n_test",
                            metric, "RMSE", "MAE", "status") if c in bt.columns]
        sections.append({
            "title": f"Rolling Backtest — {S.backtest['model']}",
            "note": note,
            "figures": [plots.backtest_series(bt, metric)],
            "tables": [bt[show]],
        })

    if "분할 방식 진단" in include and S.split_diag:
        d = S.split_diag
        v = d["verdict"]
        m = d["metric"]
        cols = [c for c in ("model", f"time_{m}", f"random_{m}", "격차")
                if c in d["table"].columns]
        tables = [d["table"][cols].round(4)]
        if v.get("causes"):
            tables.append(pd.DataFrame(v["causes"]))
        sections.append({
            "title": "Random vs Time 진단",
            "note": "무작위 분할은 검증 행의 바로 앞뒤를 학습에 넣으므로 그 점수는 미래 "
                    "성능이 아닙니다. 진단 목적으로만 계산했고 챔피언 선정·최종 보고 "
                    f"어디에도 쓰이지 않았습니다. 평균 격차 {d['mean_gap']:+.4f}.",
            "tables": tables,
        })

    if S.shap_result is not None:
        imp = explain.importance(S.shap_result)
        if "SHAP 중요도" in include:
            sections.append({
                "title": "피처 기여도",
                "note": "학습 구간에 챔피언 모델을 대입해 구한 평균 절대 SHAP 값입니다. "
                        "모델이 학습한 통계적 관계이며 인과 관계와는 다릅니다.",
                "figures": [plots.shap_importance_bar(imp)],
                "tables": [imp[["feature", "mean_abs_shap", "contribution_pct"]].head(20)],
            })
        if "SHAP dependence" in include:
            figs = []
            for f in list(imp["feature"])[:4]:
                try:
                    i = explain.auto_interaction(S.shap_result, f)
                    figs.append(plots.shap_dependence(
                        explain.dependence_data(S.shap_result, f, i), f, i))
                except Exception:  # noqa: BLE001
                    continue
            sections.append({
                "title": "Dependence plot",
                "note": "가로축은 피처 값, 세로축은 그 값이 예측을 얼마나 밀었는지입니다.",
                "figures": figs,
            })

    if "피처 선별 이력" in include and S.selection_report is not None:
        rep = S.selection_report
        tables = []
        if "status" in rep.columns:
            st_col = rep["status"].astype(str)
            manual = rep[st_col.str.contains("수동", na=False)]
            if not manual.empty:
                tables.append(manual[["feature", "status", "reason"]])
            excluded = rep[st_col.str.startswith("removed")]
            if not excluded.empty:
                tables.append(excluded[["feature", "reason"]].head(60))
        if tables:
            n_manual = len(tables[0]) if "status" in tables[0].columns else 0
            sections.append({
                "title": "피처 선별 이력",
                "note": ("자동 선별은 추천이고 확정은 사람이 합니다. "
                         + (f"이 실행에서는 사용자가 {n_manual}건을 직접 바꿨습니다. "
                            if n_manual else "이 실행에서는 자동 추천을 그대로 확정했습니다. ")
                         + "선별 통계는 학습 구간에서만 계산했습니다 — 전체 구간 통계를 "
                           "보고 고르면 사람이 누수 경로가 됩니다."),
                "tables": tables,
            })

    if "피처 출처" in include and S.provenance is not None:
        prov = S.provenance[S.provenance["feature"].isin(S.selected_features)]
        sections.append({
            "title": "피처 출처",
            "note": "각 피처가 어떤 원본 태그에서 어떤 변환으로 나왔는지입니다.",
            "tables": [prov],
        })

    if "재현 기록" in include and S.manifest:
        man = S.manifest
        ds = man.get("dataset", {})
        rows = [
            {"항목": "Run ID", "값": man.get("run_id", "-")},
            {"항목": "데이터 지문 (sha256)", "값": (ds.get("sha256") or "-")[:32]},
            {"항목": "행 × 열", "값": f"{ds.get('rows', 0):,} × {ds.get('columns', 0)}"},
            {"항목": "seed", "값": str(man.get("seed", "-"))},
            {"항목": "생성 시각", "값": man.get("created_at", "-")},
        ]
        for k, v in (man.get("packages") or {}).items():
            rows.append({"항목": f"pkg {k}", "값": v})
        tables = [pd.DataFrame(rows)]
        bounds = man.get("split_bounds") or {}
        if bounds:
            tables.append(pd.DataFrame([
                {"구간": k, "행수": v["rows"], "시작": v["start"], "끝": v["end"]}
                for k, v in bounds.items()]))
        sections.append({
            "title": "재현 기록",
            "note": "이 실행을 나중에 그대로 되살리는 데 필요한 것들입니다. "
                    "'지난달 결과가 안 나온다' 의 원인은 대개 데이터 지문 아니면 패키지 버전입니다.",
            "tables": tables,
        })

    return report.build_report(title, meta, scores, sections, embed_plotly=embed)
