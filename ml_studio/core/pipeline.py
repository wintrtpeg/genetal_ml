"""전 단계를 한 번에 돌리는 오케스트레이션.

화면에서 Auto 모드가 쓰고, Dataiku Scenario 처럼 UI 가 없는 곳에서도 그대로 쓴다.
그래서 core 에 둔다 — streamlit 을 import 하지 않는다.

**자동이라고 안전 장치를 건너뛰지 않는다.** 3분할, gap 점검, 선별 구간 추적,
폴드 내부 선별, Final Unseen 1회 접근은 전부 그대로 돈다. 자동화되는 것은
"사람이 버튼을 누르는 일"이지 "검증을 생략하는 일"이 아니다.

사람이 개입하던 지점은 기본값으로 대체되고, 무엇이 기본값으로 결정됐는지
`AutoResult.decisions` 에 남는다. 자동으로 돌렸어도 나중에 되짚을 수 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import features, models, preprocess, profiling, train, validation

__all__ = ["AutoConfig", "AutoResult", "run_auto"]


@dataclass
class AutoConfig:
    """Auto 모드가 사람 대신 쓰는 기본값."""

    # 품질 판정
    quality: profiling.QualityRule = field(default_factory=profiling.QualityRule)
    # 파생변수
    feature: features.FeatureConfig | None = None
    # 전처리
    prep: preprocess.PreprocessConfig = field(default_factory=preprocess.PreprocessConfig)
    # 분할 — Final Unseen 을 기본으로 확보한다
    valid_ratio: float = 0.2
    unseen_ratio: float = 0.15
    n_splits: int = 4
    # 선별
    top_k: int = 40
    corr_threshold: float = 0.98
    # 학습
    metric: str = "R2"
    n_jobs: int = -1
    include_heavy: bool = False       # 자동 실행에서는 느린 모델을 뺀다
    max_models: int | None = None
    fold_selection: bool = True
    ensemble: bool = True
    ensemble_threshold: float = 0.03
    evaluate_unseen: bool = True
    seed: int = 42


@dataclass
class AutoResult:
    """한 번에 돌린 결과. 화면은 이걸 받아 상태에 꽂기만 하면 된다."""

    df: pd.DataFrame
    target: str
    kept: list[str]
    feat_df: pd.DataFrame
    provenance: pd.DataFrame
    selection_report: pd.DataFrame
    selected_features: list[str]
    X: pd.DataFrame
    y: pd.Series
    split: validation.Split
    split_config: validation.SplitConfig
    feature_config: features.FeatureConfig
    prep_config: preprocess.PreprocessConfig
    train_config: train.TrainConfig
    leaderboard: pd.DataFrame
    detail: dict
    champion: str | None
    checklist: pd.DataFrame
    decisions: pd.DataFrame          # 무엇이 기본값으로 정해졌는지
    unseen_scores: dict = field(default_factory=dict)
    unseen_guard: object | None = None
    ensemble_report: pd.DataFrame | None = None
    task: str = models.TASK_REGRESSION


class AutoRunError(RuntimeError):
    """자동 실행이 안전 점검에서 멈췄을 때."""


def _default_feature_config(index: pd.DatetimeIndex) -> features.FeatureConfig:
    """샘플링 간격을 보고 무난한 lag·창을 고른다.

    간격을 모르면 행 단위 기본값을 쓴다. 어림짐작으로 분 단위를 환산하면
    lookback 이 실제와 어긋나고, 그 값이 그대로 gap 계산에 들어간다.
    """
    return features.FeatureConfig(
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[6, 12, 24],
        rolling_stats=["mean", "std"],
        ewm_spans=[12],
        diffs=[1],
        time_features=True,
        cyclical=True,
        allow_target_derived=False,     # ★ 자동이어도 Y 파생은 막는다
    )


def run_auto(
    df: pd.DataFrame,
    target: str,
    cfg: AutoConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> AutoResult:
    """데이터와 타겟만 받아 챔피언까지 만든다.

    각 단계에서 사람이 하던 선택은 기본값으로 대체하되, 무엇을 어떻게 정했는지
    decisions 표에 남긴다.
    """
    cfg = cfg or AutoConfig()
    steps, total = 0, 8
    decisions: list[dict] = []

    def tick(msg: str) -> None:
        nonlocal steps
        steps += 1
        if progress is not None:
            progress(steps, total, msg)

    def decide(step: str, what: str, value, why: str) -> None:
        decisions.append({"단계": step, "항목": what, "결정": str(value), "근거": why})

    if target not in df.columns:
        raise AutoRunError(f"타겟 '{target}' 이 데이터에 없습니다.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AutoRunError("DatetimeIndex 가 필요합니다. to_timeseries() 를 먼저 적용하세요.")

    # 1. 품질 진단 → 제외
    tick("품질 진단")
    candidates = [c for c in df.columns if c != target]
    prof = profiling.profile(df)
    pairs = profiling.find_correlated_pairs(df[candidates], cfg.quality.max_corr)
    drops = profiling.suggest_drops(prof, cfg.quality, protect=[target], corr_pairs=pairs)
    dropped = set(drops["column"]) if not drops.empty else set()
    kept = [c for c in candidates if c not in dropped]
    if not kept:
        raise AutoRunError("품질 규칙을 통과한 컬럼이 없습니다. 기준을 완화하세요.")
    decide("품질", "제외 컬럼", f"{len(dropped)}개",
           "결측·상수·편중·중복 규칙에 걸린 컬럼" if dropped else "규칙에 걸린 컬럼 없음")

    # 2. 파생변수
    tick("파생변수 생성")
    fcfg = cfg.feature or _default_feature_config(df.index)
    feat, prov = features.generate(df, target, kept, fcfg)
    feat = features.drop_warmup(feat, fcfg)
    lookback = features.warmup_rows(fcfg, df.index)
    decide("파생", "lag / rolling", f"lag {fcfg.lags} · roll {fcfg.rolling_windows}",
           "샘플링 간격과 무관하게 통용되는 기본 격자")
    decide("파생", "최대 lookback", f"{lookback}행", "gap 을 이 값으로 맞춥니다")

    # 3. 분할 — gap 은 lookback 으로 자동 설정
    tick("구간 분할")
    X_all, y_all = preprocess.prepare_xy(feat, target, [c for c in feat.columns if c != target])
    scfg = validation.SplitConfig(
        holdout_ratio=cfg.valid_ratio, unseen_ratio=cfg.unseen_ratio,
        n_splits=cfg.n_splits, gap=lookback)
    split = validation.build_split(scfg, X_all.index)
    validation.assert_disjoint(split)
    decide("분할", "구간", f"학습 {len(split.train):,} / 검증 {len(split.valid):,} "
                        f"/ Unseen {len(split.unseen):,}",
           "Final Unseen 을 확보해 모델 선택과 최종 보고를 분리")
    decide("분할", "gap", f"{lookback}행", "파생 lookback 만큼 벌려 창 겹침을 막음")

    # 4. 선별 — 학습 구간에서만
    tick("피처 선별")
    task = models.detect_task(y_all)
    sel, rep = features.select_features(
        X_all.iloc[split.train], y_all.iloc[split.train], task=task,
        top_k=cfg.top_k, corr_threshold=cfg.corr_threshold)
    if not sel:
        raise AutoRunError("선별을 통과한 피처가 없습니다.")
    review = features.feature_report(rep, prov, X_train=X_all.iloc[split.train])
    sel, review = features.apply_manual_selection(review, sel)
    X, y = X_all[sel], y_all
    decide("선별", "선택 피처", f"{len(sel)}개 / {X_all.shape[1]}개 후보",
           f"분산 → 상관중복({cfg.corr_threshold}) → MI 상위 {cfg.top_k}")

    # 5. 누수 점검 — 통과 못 하면 여기서 멈춘다
    tick("누수 점검")
    checklist = validation.leakage_checklist(
        X.index, split.train, split.valid, sel, target, prov,
        scfg.gap, lookback, selection_idx=split.train, unseen_idx=split.unseen)
    if (checklist["결과"] == "실패").any():
        bad = checklist[checklist["결과"] == "실패"]
        raise AutoRunError(
            "누수 점검을 통과하지 못해 학습을 중단했습니다: "
            + " / ".join(f"{r['항목']} — {r['내용']}" for _, r in bad.iterrows()))

    # 6. 학습
    tick("모델 학습")
    num, cat = preprocess.split_column_types(X)
    pre = preprocess.build_preprocessor(num, cat, cfg.prep)
    zoo = models.get_model_zoo(task, seed=cfg.seed, include_heavy=cfg.include_heavy)
    picked = models.default_selection(zoo, len(X))
    if cfg.max_models:
        picked = picked[: cfg.max_models]
    tcfg = train.TrainConfig(
        task=task, split=scfg, n_jobs=cfg.n_jobs, seed=cfg.seed,
        champion_metric=cfg.metric if task == models.TASK_REGRESSION else "F1",
        fold_selection=cfg.fold_selection, ensemble_threshold=cfg.ensemble_threshold)
    board, detail = train.train_all(X, y, split.train, split.valid, pre, zoo, picked, tcfg)
    champion = train.pick_champion(board, tcfg.champion_metric)
    decide("학습", "모델", f"{len(picked)}종", "설치된 라이브러리 중 데이터 크기에 맞는 것")
    decide("학습", "폴드 내부 선별", "ON" if cfg.fold_selection else "OFF",
           "CV 점수의 선별 편향을 없애기 위해")

    # 7. 앙상블 — 임계값을 넘을 때만 챔피언 교체
    tick("앙상블")
    ens_report = None
    if cfg.ensemble and task == models.TASK_REGRESSION:
        from . import ensemble as ens_mod
        ok = board[board["status"] == "ok"]
        bases = [m for m in ok["model"].head(3) if m in zoo]
        if len(bases) >= 2:
            eb, ed = train.build_ensembles(
                X, y, split.train, split.valid, pre, zoo, bases, tcfg, detail=detail)
            if not eb.empty:
                board = train.sort_leaderboard(
                    pd.concat([board, eb], ignore_index=True), tcfg.champion_metric)
                detail = {**detail, **ed}
                champ2, ens_report = ens_mod.adopt_ensemble(
                    board, tcfg.champion_metric, cfg.ensemble_threshold, prefix="holdout_")
                champion = champ2 or champion
                decide("앙상블", "자동채택", champion,
                       f"단일 최고 대비 {cfg.ensemble_threshold:.0%} 이상일 때만 채택")

    # 8. Final Unseen — 챔피언 확정 뒤 한 번만
    tick("Final Unseen 평가")
    unseen_scores: dict = {}
    guard = None
    if len(split.unseen):
        guard = train.UnseenGuard(split.unseen)
        if cfg.evaluate_unseen and champion:
            unseen_scores = train.evaluate_unseen(
                detail[champion]["_pipeline"], X, y, split.unseen, tcfg, guard,
                who=champion)
            decide("보고", "최종 성능", f"Final Unseen {tcfg.champion_metric} "
                                    f"{unseen_scores.get(f'unseen_{tcfg.champion_metric}', float('nan')):.4f}",
                   "학습·선별·모델선택 어디에도 쓰이지 않은 구간")

    return AutoResult(
        df=df, target=target, kept=kept, feat_df=feat, provenance=prov,
        selection_report=review, selected_features=sel, X=X, y=y,
        split=split, split_config=scfg, feature_config=fcfg, prep_config=cfg.prep,
        train_config=tcfg, leaderboard=board, detail=detail, champion=champion,
        checklist=checklist, decisions=pd.DataFrame(decisions),
        unseen_scores=unseen_scores, unseen_guard=guard,
        ensemble_report=ens_report, task=task,
    )
