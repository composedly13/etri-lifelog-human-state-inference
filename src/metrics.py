# -*- coding: utf-8 -*-
"""
평가 지표 모듈 (src/metrics.py)
대회 평가 방식이 확정되면 이 파일의 competition_metric 함수만 수정하면
전체 파이프라인에 자동으로 반영됩니다.

현재 기본 지표: RMSE
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# TODO (분류 확장): 분류 metric이 필요하면 아래를 추가하세요
# from sklearn.metrics import (
#     accuracy_score, f1_score, roc_auc_score
# )


# ── 회귀 지표 ─────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (결정 계수)"""
    return float(r2_score(y_true, y_pred))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson Correlation Coefficient"""
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# TODO (분류 확장): 분류 지표 함수를 추가하세요
# def accuracy(y_true, y_pred):
#     return float(accuracy_score(y_true, y_pred))
#
# def macro_f1(y_true, y_pred):
#     return float(f1_score(y_true, y_pred, average="macro"))


# ── 대회 공식 지표 ───────────────────────────────────────────────────────────
# 이 함수만 수정하면 전체 파이프라인에 반영됩니다.

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    대회 공식 평가 지표.

    현재: RMSE (낮을수록 좋음)
    대회 평가 방식 확정 후 이 함수를 수정하세요.

    대회 평가 기준에 따라 교체 예시:
      - MAE 기반: return mae(y_true, y_pred)
      - Pearson: return -pearson_correlation(y_true, y_pred)  # 높을수록 좋도록 -부호
      - 복합: return 0.5 * rmse(...) + 0.5 * mae(...)
    """
    return rmse(y_true, y_pred)


# ── 멀티 타겟 평균 점수 ──────────────────────────────────────────────────────

def multi_target_score(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    target_cols: List[str],
    metric_fn=None,
) -> Dict[str, float]:
    """
    여러 타겟 컬럼에 대한 점수를 계산하고, 평균 점수를 반환합니다.

    Args:
        y_true_df   : 실제값 DataFrame
        y_pred_df   : 예측값 DataFrame
        target_cols : 평가할 타겟 컬럼 리스트
        metric_fn   : 사용할 지표 함수 (None이면 competition_metric 사용)

    Returns:
        {target_col: score, ..., "mean": mean_score}
    """
    metric_fn = metric_fn or competition_metric
    scores: Dict[str, float] = {}

    for col in target_cols:
        if col not in y_true_df.columns or col not in y_pred_df.columns:
            print(f"[metrics] ⚠ '{col}' 컬럼이 없어 스킵됩니다.")
            continue
        y_true = y_true_df[col].values
        y_pred = y_pred_df[col].values

        # NaN이 있는 행 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            print(f"[metrics] ⚠ '{col}': 유효한 샘플이 없습니다.")
            continue

        score = metric_fn(y_true[mask], y_pred[mask])
        scores[col] = score

    if scores:
        scores["mean"] = float(np.mean(list(scores.values())))

    return scores


def print_scores(scores: Dict[str, float], metric_name: str = "Score") -> None:
    """점수 딕셔너리를 정렬하여 출력합니다."""
    print(f"\n{'─'*40}")
    print(f"  {metric_name} Summary")
    print(f"{'─'*40}")
    for k, v in scores.items():
        marker = " ◀ MEAN" if k == "mean" else ""
        print(f"  {k:30s}: {v:.6f}{marker}")
    print(f"{'─'*40}\n")


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_true = np.array([3.0, 2.5, 4.0, 5.0, 1.5])
    y_pred = np.array([2.8, 2.6, 4.2, 4.7, 1.9])

    print(f"RMSE : {rmse(y_true, y_pred):.6f}")
    print(f"MAE  : {mae(y_true, y_pred):.6f}")
    print(f"R²   : {r2(y_true, y_pred):.6f}")
    print(f"Corr : {pearson_correlation(y_true, y_pred):.6f}")
    print(f"Comp : {competition_metric(y_true, y_pred):.6f}")
