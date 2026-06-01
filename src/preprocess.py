# -*- coding: utf-8 -*-
"""
데이터 전처리 모듈 (src/preprocess.py)
결측치 처리, 범주형 인코딩(category dtype), 날짜 파생 피처 생성을 담당합니다.

처리 순서:
  1. 날짜 피처 파싱 (Baseline 2 - use_date_features=true 시 활성화)
  2. 결측치 처리 (수치형: train 기준 median, 범주형: "missing" 문자열)
  3. 범주형 인코딩 (Categorical dtype → LightGBM 직접 처리)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from src.config import Config


# ── 결측치 처리 ───────────────────────────────────────────────────────────────

def handle_missing_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    결측치를 처리합니다.
      - 수치형 컬럼: train 기준 통계값(median/mean)으로 채움
      - 범주형 컬럼: 'missing' 문자열로 채움
    train 기준 통계를 test에도 동일하게 적용하여 데이터 누수를 방지합니다.

    Args:
        train, test  : DataFrame
        feature_cols : 처리 대상 피처 컬럼 리스트
        strategy     : 수치형 처리 전략 ('median' | 'mean')

    Returns:
        처리된 train, test
    """
    train = train.copy()
    test  = test.copy()

    # dtype으로 수치/범주 구분
    num_cols = [
        c for c in feature_cols
        if c in train.columns and train[c].dtype.kind in ("i", "f", "u")
    ]
    cat_cols = [
        c for c in feature_cols
        if c in train.columns and train[c].dtype == object
    ]

    # 수치형 결측치: train 기준 통계로 채움
    for col in num_cols:
        if train[col].isna().any():
            fill_val = train[col].median() if strategy == "median" else train[col].mean()
            train[col] = train[col].fillna(fill_val)
            if col in test.columns:
                test[col] = test[col].fillna(fill_val)

    # 범주형 결측치: 'missing' 문자열로 채움
    for col in cat_cols:
        if train[col].isna().any():
            train[col] = train[col].fillna("missing")
            if col in test.columns:
                test[col] = test[col].fillna("missing")

    remaining = train[feature_cols].isna().sum().sum()
    print(f"[preprocess] 결측치 처리 완료. 남은 결측치 수: {remaining}")
    return train, test


# ── 범주형 인코딩 ─────────────────────────────────────────────────────────────

def encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    범주형 컬럼을 pandas Categorical dtype으로 변환합니다.
    LightGBM은 category dtype을 직접 처리하므로 레이블 인코딩 불필요.
    train + test 전체를 기준으로 category를 정의하여 unseen 카테고리를 방지합니다.

    Args:
        categorical_cols: None이면 object dtype 컬럼 자동 감지

    Returns:
        처리된 train, test, 사용된 범주형 컬럼 리스트
    """
    train = train.copy()
    test  = test.copy()

    if categorical_cols is None or len(categorical_cols) == 0:
        categorical_cols = [
            c for c in feature_cols
            if c in train.columns and train[c].dtype == object
        ]

    used_cats = []
    for col in categorical_cols:
        if col not in train.columns:
            continue
        test_series = test[col] if col in test.columns else pd.Series(dtype=str)
        all_values  = pd.concat([train[col], test_series]).dropna().unique().tolist()

        train[col] = pd.Categorical(train[col], categories=all_values)
        if col in test.columns:
            test[col] = pd.Categorical(test[col], categories=all_values)
        used_cats.append(col)

    if used_cats:
        print(f"[preprocess] 범주형 인코딩 완료: {used_cats}")
    return train, test, used_cats


# ── 날짜 피처 파싱 ────────────────────────────────────────────────────────────

def parse_date_features(
    df: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
    auto_detect: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    날짜 컬럼을 파싱하여 year / month / day / dayofweek / is_weekend / hour / quarter
    파생 피처를 생성합니다. (Baseline 2)

    자동 감지: 컬럼명에 'date', 'time', 'dt', 'timestamp', '날짜', '일자' 포함 시 시도.

    Returns:
        처리된 DataFrame, 새로 생성된 파생 피처 컬럼 리스트
    """
    df       = df.copy()
    new_cols = []
    date_cols = list(date_cols or [])

    if auto_detect:
        keywords = ["date", "time", "dt", "timestamp", "날짜", "일자"]
        for col in df.columns:
            if col in date_cols:
                continue
            if any(kw in col.lower() for kw in keywords):
                # 실제로 파싱 가능한지 샘플로 확인
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    date_cols.append(col)
                except Exception:
                    pass

    for col in date_cols:
        if col not in df.columns:
            continue
        try:
            dt     = pd.to_datetime(df[col], errors="coerce")
            prefix = f"{col}_"

            df[f"{prefix}year"]      = dt.dt.year
            df[f"{prefix}month"]     = dt.dt.month
            df[f"{prefix}day"]       = dt.dt.day
            df[f"{prefix}dayofweek"] = dt.dt.dayofweek   # 0=월요일, 6=일요일
            df[f"{prefix}is_weekend"]= (dt.dt.dayofweek >= 5).astype(int)
            df[f"{prefix}hour"]      = dt.dt.hour        # 시간 정보 없으면 0
            df[f"{prefix}quarter"]   = dt.dt.quarter

            added = [
                f"{prefix}year", f"{prefix}month", f"{prefix}day",
                f"{prefix}dayofweek", f"{prefix}is_weekend",
                f"{prefix}hour", f"{prefix}quarter",
            ]
            new_cols.extend(added)
            print(f"[preprocess] 날짜 피처 생성: '{col}' → {len(added)}개 파생 컬럼")
        except Exception as e:
            print(f"[preprocess] ⚠ '{col}' 날짜 파싱 실패: {e}")

    return df, new_cols


# ── 전체 전처리 파이프라인 ────────────────────────────────────────────────────

def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    전체 전처리 파이프라인을 실행합니다.

    Steps:
      1. 날짜 피처 파싱     (Baseline 2, cfg.use_date_features=true 시)
      2. 결측치 처리
      3. 범주형 인코딩

    Returns:
        train, test, 최종 feature_cols, categorical_cols
    """
    print("\n" + "=" * 55)
    print("[preprocess] 전처리 시작")
    print("=" * 55)

    # Step 1: 날짜 피처 파싱 (Baseline 2)
    if cfg.use_date_features:
        train, new_cols_train = parse_date_features(
            train, date_cols=cfg.date_cols.copy(), auto_detect=True
        )
        test, new_cols_test = parse_date_features(
            test, date_cols=cfg.date_cols.copy(), auto_detect=True
        )
        # train/test 공통으로 생성된 컬럼만 feature_cols에 추가
        common_new = [c for c in new_cols_train if c in test.columns]
        feature_cols = feature_cols + [c for c in common_new if c not in feature_cols]
        print(f"[preprocess] 날짜 파생 피처 {len(common_new)}개 추가됨")

    # Step 2: 결측치 처리
    train, test = handle_missing_values(train, test, feature_cols)

    # Step 3: 범주형 인코딩
    cat_input = cfg.categorical_cols if cfg.categorical_cols else None
    train, test, cat_cols = encode_categoricals(train, test, feature_cols, cat_input)

    print(f"[preprocess] 전처리 완료. 최종 피처 수: {len(feature_cols)}")
    print("=" * 55 + "\n")
    return train, test, feature_cols, cat_cols


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import Config
    from src.load_data import load_all

    cfg = Config()
    train, test, sub, target_cols, feature_cols = load_all(cfg)
    train, test, feature_cols, cat_cols = preprocess(train, test, feature_cols, cfg)

    print(f"최종 피처 수 : {len(feature_cols)}")
    print(f"범주형 피처  : {cat_cols}")
    print(f"Train shape  : {train.shape}")
    print(f"Test  shape  : {test.shape}")
