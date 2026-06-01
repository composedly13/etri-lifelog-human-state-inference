# -*- coding: utf-8 -*-
"""
피처 엔지니어링 모듈 (src/features.py)

현재 구현:
  - Baseline 1 : 원본 피처 (전처리만)
  - Baseline 2 : 날짜/요일/주말 피처 → preprocess.py에서 처리됨

TODO (Proposed 1): lag / rolling mean / diff 등 시간 맥락 피처
TODO (Proposed 2): 개인별 평균 대비 deviation / z-score 피처
TODO (Proposed 3): LightGBM + CatBoost + XGBoost 앙상블

논문 연구 질문:
  "시간적 맥락(lag/rolling)과 개인별 기준선(deviation)을 반영한 피처가
   수면·피로·스트레스 예측 성능을 향상시키는가?"
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from src.config import Config


# ── TODO (Proposed 1): 시간 맥락 피처 ─────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str],
    lag_windows: List[int],
    sort_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    TODO (Proposed 1): lag 피처를 생성합니다.
    각 피처의 과거 N일 값을 새 컬럼으로 추가합니다.

    Args:
        df          : DataFrame
        target_cols : lag를 생성할 컬럼 리스트 (수치형 피처)
        group_col   : 개인별 lag 계산 시 그룹 컬럼 (e.g. 'user_id')
        lag_windows : lag 윈도우 리스트 (e.g. [1, 2, 3, 7])
        sort_col    : 정렬 기준 컬럼 (날짜 컬럼 등)
    """
    # TODO: 아래 주석을 해제하고 구현하세요
    #
    # df = df.copy()
    # new_cols = []
    # if sort_col:
    #     by = [group_col, sort_col] if group_col else [sort_col]
    #     df = df.sort_values(by).reset_index(drop=True)
    # for col in target_cols:
    #     for lag in lag_windows:
    #         new_col = f"{col}_lag{lag}"
    #         if group_col and group_col in df.columns:
    #             df[new_col] = df.groupby(group_col)[col].shift(lag)
    #         else:
    #             df[new_col] = df[col].shift(lag)
    #         new_cols.append(new_col)
    # return df, new_cols
    raise NotImplementedError(
        "TODO: Proposed 1 - add_lag_features 미구현\n"
        "configs/base.yaml의 features.use_lag_features를 false로 유지하거나\n"
        "이 함수를 구현한 후 사용하세요."
    )


def add_rolling_features(
    df: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str],
    rolling_windows: List[int],
    agg_funcs: Optional[List[str]] = None,
    sort_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    TODO (Proposed 1): rolling 통계 피처를 생성합니다.
    각 피처의 과거 N일 평균/표준편차를 새 컬럼으로 추가합니다.

    Args:
        agg_funcs: 집계 함수 리스트 (기본: ['mean', 'std'])
    """
    # TODO: 아래 주석을 해제하고 구현하세요
    #
    # df = df.copy()
    # new_cols = []
    # agg_funcs = agg_funcs or ["mean", "std"]
    # if sort_col:
    #     by = [group_col, sort_col] if group_col else [sort_col]
    #     df = df.sort_values(by).reset_index(drop=True)
    # for col in target_cols:
    #     for w in rolling_windows:
    #         for agg in agg_funcs:
    #             new_col = f"{col}_roll{w}_{agg}"
    #             if group_col and group_col in df.columns:
    #                 df[new_col] = df.groupby(group_col)[col].transform(
    #                     lambda x: x.rolling(w, min_periods=1).agg(agg)
    #                 )
    #             else:
    #                 df[new_col] = df[col].rolling(w, min_periods=1).agg(agg)
    #             new_cols.append(new_col)
    # return df, new_cols
    raise NotImplementedError("TODO: Proposed 1 - add_rolling_features 미구현")


def add_diff_features(
    df: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str],
    diff_periods: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    TODO (Proposed 1): diff 피처를 생성합니다.
    전일 대비 변화량을 새 컬럼으로 추가합니다.
    """
    # TODO: 아래 주석을 해제하고 구현하세요
    #
    # diff_periods = diff_periods or [1]
    # df = df.copy()
    # new_cols = []
    # for col in target_cols:
    #     for p in diff_periods:
    #         new_col = f"{col}_diff{p}"
    #         if group_col and group_col in df.columns:
    #             df[new_col] = df.groupby(group_col)[col].diff(p)
    #         else:
    #             df[new_col] = df[col].diff(p)
    #         new_cols.append(new_col)
    # return df, new_cols
    raise NotImplementedError("TODO: Proposed 1 - add_diff_features 미구현")


# ── TODO (Proposed 2): 개인별 기준선 피처 ─────────────────────────────────────

def add_personal_baseline_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    TODO (Proposed 2): 개인별 평균 대비 deviation 피처를 생성합니다.

    각 수치형 피처에 대해 다음 컬럼을 생성합니다:
      - {col}_personal_mean : 개인별 평균 (train 기준)
      - {col}_personal_std  : 개인별 표준편차
      - {col}_deviation     : 개인 평균 대비 절대 편차
      - {col}_zscore        : 개인 기준 z-score

    논문 핵심 아이디어: 동일한 수면 시간이라도 개인마다 기준선이 다르므로
    절대값이 아닌 개인 평균 대비 편차가 더 의미 있는 피처가 됩니다.
    """
    # TODO: 아래 주석을 해제하고 구현하세요
    #
    # train = train.copy()
    # test  = test.copy()
    # new_cols = []
    # num_cols = [
    #     c for c in feature_cols
    #     if c in train.columns and train[c].dtype.kind in ("i", "f", "u")
    # ]
    # # train 기준 개인별 통계 계산
    # personal_stats = train.groupby(group_col)[num_cols].agg(["mean", "std"])
    # personal_stats.columns = [f"{col}_{agg}" for col, agg in personal_stats.columns]
    # personal_stats = personal_stats.reset_index()
    #
    # for col in num_cols:
    #     mean_col = f"{col}_personal_mean"
    #     std_col  = f"{col}_personal_std"
    #     dev_col  = f"{col}_deviation"
    #     z_col    = f"{col}_zscore"
    #
    #     for df_ref in [train, test]:
    #         df_ref = df_ref.merge(
    #             personal_stats[[group_col, f"{col}_mean", f"{col}_std"]],
    #             on=group_col, how="left"
    #         )
    #         df_ref.rename(columns={f"{col}_mean": mean_col, f"{col}_std": std_col}, inplace=True)
    #         df_ref[dev_col] = df_ref[col] - df_ref[mean_col]
    #         df_ref[z_col]   = df_ref[dev_col] / (df_ref[std_col] + 1e-8)
    #     new_cols += [mean_col, std_col, dev_col, z_col]
    #
    # return train, test, new_cols
    raise NotImplementedError(
        "TODO: Proposed 2 - add_personal_baseline_features 미구현\n"
        "configs/base.yaml의 features.use_personal_baseline을 false로 유지하거나\n"
        "이 함수를 구현한 후 사용하세요."
    )


# ── 통합 피처 엔지니어링 파이프라인 ───────────────────────────────────────────

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    피처 엔지니어링 파이프라인을 실행합니다.
    현재는 Baseline 피처만 반환하며, cfg 설정에 따라 Proposed 피처를 추가합니다.

    Returns:
        train, test, 최종 feature_cols
    """
    print("[features] 피처 엔지니어링 시작...")

    # ── TODO (Proposed 1): 시간 맥락 피처 ──────────────────────────────────
    if cfg.use_lag_features:
        print("[features] TODO: lag/rolling 피처 생성 (미구현 — use_lag_features=false 유지)")
        # sort_col = cfg.date_cols[0] if cfg.date_cols else None
        # train, new_cols = add_lag_features(
        #     train, feature_cols, cfg.feature_group_col, cfg.lag_windows, sort_col
        # )
        # feature_cols += new_cols
        # 주의: test에도 동일하게 적용 필요 (train+test concat 후 split 방식 권장)

    # ── TODO (Proposed 2): 개인별 기준선 피처 ──────────────────────────────
    if cfg.use_personal_baseline and cfg.feature_group_col:
        print("[features] TODO: 개인별 기준선 피처 생성 (미구현 — use_personal_baseline=false 유지)")
        # train, test, new_cols = add_personal_baseline_features(
        #     train, test, feature_cols, cfg.feature_group_col
        # )
        # feature_cols += new_cols

    print(f"[features] 피처 엔지니어링 완료. 최종 피처 수: {len(feature_cols)}")
    return train, test, feature_cols


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import Config
    from src.load_data import load_all
    from src.preprocess import preprocess

    cfg = Config()
    train, test, sub, target_cols, feature_cols = load_all(cfg)
    train, test, feature_cols, cat_cols = preprocess(train, test, feature_cols, cfg)
    train, test, feature_cols = build_features(train, test, feature_cols, cfg)
    print(f"최종 피처 수: {len(feature_cols)}")
