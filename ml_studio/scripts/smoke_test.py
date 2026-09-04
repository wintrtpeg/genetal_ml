"""UI 없이 코어만으로 1~8단계를 끝까지 돌려본다.

  python scripts/make_demo_data.py
  python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import (  # noqa: E402
    config, datasource, diagnostics, ensemble, features, models, persist,
    preprocess, profiling, train, validation,
)


def _enable_utf8() -> None:
    """윈도우 콘솔에서 한글·기호가 깨지거나 죽지 않게 한다.

    한글 윈도우의 기본 코덱은 cp949 이고, 여기에는 em dash(—, U+2014)가 없다.
    콘솔 창에 바로 찍을 때는 파이썬이 UTF-16 경로를 쓰므로 문제가 없지만,
    출력을 파일이나 파이프로 넘기는 순간 cp949 로 떨어져 UnicodeEncodeError 로
    죽는다. 결과를 로그로 남기려다 실행 자체가 실패하는 셈이라 미리 막는다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def line(t: str) -> None:
    print(f"\n{'─' * 68}\n{t}\n{'─' * 68}")


def main() -> None:
    _enable_utf8()
    csv = ROOT / "data" / "demo_timeseries.csv"
    if not csv.exists():
        raise SystemExit("먼저 scripts/make_demo_data.py 를 실행하세요.")

    # 1. 데이터
    line("1. 데이터 로드")
    raw = datasource.CsvSource(csv).load()
    df = datasource.to_timeseries(raw, "timestamp")
    print(f"{df.shape[0]:,}행 × {df.shape[1]}열, 주기 {datasource.infer_freq(df.index)}")

    # 2. 타겟
    target = "y_output"
    candidates = [c for c in df.columns if c != target]
    print(f"타겟 {target} / X 후보 {len(candidates)}개")

    # 3. 품질 진단
    line("3. 품질 진단")
    prof = profiling.profile(df)
    rule = profiling.QualityRule()
    pairs = profiling.find_correlated_pairs(df, rule.max_corr)
    drops = profiling.suggest_drops(prof, rule, protect=[target], corr_pairs=pairs)
    print(drops.to_string(index=False) if len(drops) else "제외 후보 없음")
    keep = [c for c in candidates if c not in set(drops["column"])]
    print(f"→ 유지 {len(keep)}개: {keep}")

    # 4. 파생변수
    line("4. 파생변수 (Y lag 차단)")
    fcfg = features.FeatureConfig(
        lags=[1, 2, 3, 6], rolling_windows=[6, 12], rolling_stats=["mean", "std"],
        ewm_spans=[12], diffs=[1], allow_target_derived=False,
    )
    feat, prov = features.generate(df, target, keep, fcfg)
    feat = features.drop_warmup(feat, fcfg)
    made = [c for c in feat.columns if c not in df.columns]
    print(f"생성 {len(made)}개 / 총 {feat.shape[1] - 1}개 X 후보")
    print(f"warm-up {features.warmup_rows(fcfg)}행 제거 → {len(feat):,}행")

    try:
        features.assert_no_target_derived([f"{target}__lag1"], target,
                                          pd.DataFrame([{"feature": f"{target}__lag1",
                                                         "origin": target}]))
        print("!! Y lag 가드가 동작하지 않았습니다")
    except features.TargetLeakage as e:
        print(f"Y lag 가드 정상: {e}")

    # 5. 분할 (3분할 — Train / Validation / Final Unseen)
    line("5. 분할 및 누수 점검")
    X_all, y_all = preprocess.prepare_xy(feat, target, [c for c in feat.columns if c != target])
    # gap 은 파생 최대 lookback 이상이어야 한다. lag 최대값이 아니라 warmup_rows 를 쓴다.
    gap = features.warmup_rows(fcfg)
    split = validation.SplitConfig(holdout_ratio=0.2, unseen_ratio=0.15, n_splits=4, gap=gap)
    sp = validation.build_split(split, X_all.index)
    validation.assert_disjoint(sp)
    tr, te, un = sp.train, sp.valid, sp.unseen
    print(sp.describe(X_all.index).to_string(index=False))

    sel, rep = features.select_features(X_all.iloc[tr], y_all.iloc[tr],
                                        top_k=40, corr_threshold=0.98)
    print(f"\n학습 구간에서만 선별 → {len(sel)}개 "
          f"(탈락 {int((rep['status'] == 'removed').sum())}개, 사유 기록됨)")
    X, y = X_all[sel], y_all

    # 5-b. 검토 게이트 — 사람이 X 를 확정하는 관문 (여기서는 자동 추천대로 확정)
    line("5-b. 피처 품질 리포트 (검토 게이트)")
    review = features.feature_report(rep, prov, X_train=X_all.iloc[tr])
    print(features.origin_rollup(review).to_string(index=False))
    print("\n검토 표 상위 5")
    rcols = ["feature", "kept", "mutual_info", "MI순위", "origin", "transform", "reason"]
    print(review.sort_values("mutual_info", ascending=False)[rcols].head(5).to_string(index=False))
    final_sel, review = features.apply_manual_selection(review, sel)
    risks = features.selection_risks(final_sel, review, X_all.iloc[tr])
    print(f"\n확정 {len(final_sel)}개 · 위험 경고 {len(risks)}건 "
          f"· 수동 변경 {int(review['status'].astype(str).str.contains('수동').sum())}건")
    rep = review

    check = validation.leakage_checklist(X.index, tr, te, sel, target, prov,
                                         split.gap, features.warmup_rows(fcfg),
                                         selection_idx=tr, unseen_idx=un)
    print(check.to_string(index=False))
    if (check["결과"] == "실패").any():
        raise SystemExit("누수 점검 실패")

    print(validation.audit_splits(X.index, validation.make_cv(split), len(X.iloc[tr])).to_string(index=False))

    # 6. 학습
    line("6. 병렬 학습")
    task = models.detect_task(y)
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, preprocess.PreprocessConfig())
    zoo = models.get_model_zoo(task, include_heavy=False)
    tcfg = train.TrainConfig(task=task, split=split, n_jobs=-1, champion_metric="R2")
    board, detail = train.train_all(X, y, tr, te, pre, zoo, list(zoo), tcfg)

    show = [c for c in ["rank", "model", "cv_R2", "holdout_R2", "holdout_RMSE",
                        "holdout_MAE", "fit_seconds", "status"] if c in board.columns]
    print(board[show].to_string(index=False))

    # 6-b. 폴드 내부 선별 안정성
    stab = [(n, r["fold_jaccard"], r["fold_features_mean"])
            for n, r in detail.items() if r.get("status") == "ok" and "fold_jaccard" in r]
    if stab:
        line("6-b. 폴드 내부 선별 안정성 (Jaccard)")
        print(pd.DataFrame(stab, columns=["model", "폴드간_Jaccard", "폴드평균_피처수"])
              .to_string(index=False))

    # 7. 앙상블 — OOF 재사용
    line("7. 앙상블 (보팅 / 가중 / 스태킹)")
    top = [m for m in board[board["status"] == "ok"]["model"].head(3)]
    eb, ed = train.build_ensembles(X, y, tr, te, pre, zoo, top, tcfg, detail=detail)
    if not eb.empty:
        print(eb[[c for c in ["model", "holdout_R2", "holdout_RMSE", "oof_rows",
                              "fit_seconds", "status"] if c in eb.columns]].to_string(index=False))
        if "weights" in eb.columns:
            wrow = eb[eb["model"] == "Ensemble_Weighted"]
            if len(wrow):
                print(f"가중치: {wrow['weights'].iloc[0]}")

    full = train.sort_leaderboard(pd.concat([board, eb], ignore_index=True), "R2")
    champ, adopt = ensemble.adopt_ensemble(full, "R2", tcfg.ensemble_threshold, prefix="holdout_")
    if not adopt.empty:
        print("\n자동채택 판정 (SPEC §17)")
        print(adopt.to_string(index=False))
    print(f"\n챔피언: {champ}")

    # 8. 예측
    line("8. 예측")
    all_detail = {**detail, **ed}
    pipe = all_detail[champ]["_pipeline"]
    pred = train.predict_range(pipe, X)
    res = pd.DataFrame({"actual": y, "pred": pred}).dropna()
    print(f"전체 {len(res):,}행 예측 완료")
    print(f"검증(모델선택) R2 "
          f"{train.regression_scores(y.iloc[te], pipe.predict(X.iloc[te]))['R2']:.4f}")

    # 8-b. Final Unseen — 챔피언 확정 뒤 단 한 번
    line("8-b. Final Unseen (1회 접근)")
    guard = train.UnseenGuard(un)
    unseen = train.evaluate_unseen(pipe, X, y, un, tcfg, guard, who=champ)
    print(" · ".join(f"{k.replace('unseen_', '')} {v:.4f}"
                     for k, v in unseen.items() if k != "unseen_rows"))
    print(train.selection_bias_report(full, champ, unseen, "R2").to_string(index=False))
    try:
        guard.open("두번째")
        print("!! Unseen 재접근이 막히지 않았습니다")
    except train.UnseenAccessError as e:
        print(f"재접근 차단 정상: {e}")

    # 9. 진단
    line("9. 진단 — 잔차")
    r = diagnostics.residuals(y, pred)
    dcfg = diagnostics.ResidualConfig(window=96, n_segments=5)
    s = diagnostics.summary(r, dcfg)
    print(f"평균 {s['mean']:+.4f} · 표준편차 {s['std']:.4f} · MAE {s['MAE']:.4f} "
          f"· lag1 자기상관 {s['lag1_acf']:.3f} · 이상점 {s['outliers']:,}건")
    drift = diagnostics.drift_table(r, dcfg)
    print(drift[["구간", "행수", "mean", "std", "MAE", "MAE_배율"]].round(4).to_string(index=False))
    print(diagnostics.drift_verdict(drift)["message"])

    line("9-b. Rolling Backtest")
    bt, _ = train.rolling_backtest(X, y, pre, zoo[champ] if champ in zoo else zoo["Ridge"],
                                   tcfg, n_folds=4)
    print(bt[["구간", "평가시작", "n_train", "n_test", "R2", "RMSE", "status"]].to_string(index=False))
    bs = train.backtest_summary(bt, "R2")
    print(f"평균 {bs['평균']:.4f} · 표준편차 {bs['표준편차']:.4f} "
          f"· 최저 {bs['최저']:.4f} (최악 {bs['최악구간']}구간)")

    line("9-c. Random vs Time (진단 전용)")
    diag = train.random_vs_time(X, y, pre, zoo, ["Ridge", "DecisionTree"], tcfg)
    m = diag["metric"]
    print(diag["table"][["model", f"time_{m}", f"random_{m}", "격차"]].round(4).to_string(index=False))
    v = diag["verdict"]
    print(f"평균 격차 {diag['mean_gap']:+.4f} · lag1 자기상관 {v['lag1_acf']} "
          f"· 타겟 이동 {v['y_shift_sd']}σ")
    for c in v["causes"]:
        print(f"  · {c['원인 후보']} — {c['근거']}")
    try:
        train.assert_not_diagnostic(diag)
        print("!! 진단 결과가 평가 경로로 새는 것을 막지 못했습니다")
    except ValueError:
        print("진단 결과 격리 정상: 평가 경로 유입이 예외로 막힙니다")

    # 10. 재현성
    line("10. 재현 기록")
    man = persist.build_manifest(
        run_id="smoke", target=target, df=feat, split=sp, index=X.index,
        seed=tcfg.seed, champion=champ, selection_report=rep, unseen_scores=unseen)
    print(f"run_id {man['run_id']} · 지문 {man['dataset']['sha256'][:16]} "
          f"· {man['dataset']['rows']:,}행 × {man['dataset']['columns']}열")
    print("구간 경계:")
    for k, v2 in man["split_bounds"].items():
        print(f"  {k:<12} {v2['rows']:>6,}행  {v2['start']} ~ {v2['end']}")
    print(f"제외 피처 사유 {len(man['features_excluded']):,}건 기록됨")
    print("패키지:", ", ".join(f"{k} {v2}" for k, v2 in list(man["packages"].items())[:5]))

    line("10-b. 설정 왕복")
    scfg = config.StudioConfig(features=fcfg, preprocess=preprocess.PreprocessConfig(),
                               split=split, train=tcfg, meta={"target": target})
    text = config.dumps(scfg)
    back, warns = config.loads(text)
    same = back == scfg
    print(f"직렬화 {len(text):,}자 · 경고 {len(warns)}건 · 왕복 동일 {same}")
    if not same:
        raise SystemExit("설정 왕복이 깨졌습니다")

    # 참고: 설계상 회복돼야 할 신호가 상위에 오는지
    line("참고: 순열 중요도 상위 10")
    from core.explain import permutation_importance_fallback
    imp = permutation_importance_fallback(pipe, X.iloc[te], y.iloc[te], n_repeats=3)
    print(imp.head(10).to_string(index=False))

    print("\n전 단계 통과")


if __name__ == "__main__":
    main()
