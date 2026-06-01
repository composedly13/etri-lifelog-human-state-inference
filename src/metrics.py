# -*- coding: utf-8 -*-
"""
평가 지표 모듈 (src/metrics.py)
대회 평가 방식이 확정되면 이 파일의 competition_metric 함수만 수정하면
전체 파이프라인에 자동으로 반영됩니다.

현재 대회 공식 지표: Average Log-Loss (7개 이진 지표 평균, 낮을수록 좋음)
참고용으로 회귀 지표(RMSE/MAE)와 분류 지표(F1/AUC/Accuracy)도 함께 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss as sk_log_loss,
    f1_score,
    roc_auc_score,
    accuracy_score,
)


# ── 분류 지표 (대회 공식) ─────────────────────────────────────────────────────

def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary Log-Loss (낮을수록 좋음). 대회 공식 평가 지표.

    y_prob는 클래스 1의 확률(0~1). 확신 오답에 무한 벌점이 가지 않도록
    [eps, 1-eps]로 클리핑합니다. y_true에 한 클래스만 있어도 동작하도록
    labels=[0, 1]을 명시합니다.
    """
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(sk_log_loss(y_true, y_prob, labels=[0, 1]))


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """확률을 threshold로 이진화한 뒤 macro F1 (참고용, 높을수록 좋음)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC (참고용, 높을수록 좋음). 한 클래스만 있으면 NaN 반환."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy (참고용, 높을수록 좋음)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


# ── 회귀 지표 (참고용) ────────────────────────────────────────────────────────

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


# ── 대회 공식 지표 ───────────────────────────────────────────────────────────
# 이 함수만 수정하면 전체 파이프라인에 반영됩니다.
# 대회 평가: 7개 이진 지표 각각의 Log-Loss를 평균 (낮을수록 좋음).
# y_pred는 클래스 1의 확률이어야 합니다.

GREATER_IS_BETTER = False   # Log-Loss는 낮을수록 좋음


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """대회 공식 평가 지표: Binary Log-Loss (낮을수록 좋음)."""
    return log_loss(y_true, y_pred)


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
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    # 무정보(0.5) vs 정답에 가까운 예측 비교
    p_half = np.full(200, 0.5)
    p_good = np.clip(y_true * 0.7 + 0.15 + rng.normal(0, 0.1, 200), 0, 1)

    print(f"[0.5 상수]   LogLoss : {competition_metric(y_true, p_half):.6f}  (= ln2 ≈ 0.6931)")
    print(f"[좋은 예측]  LogLoss : {competition_metric(y_true, p_good):.6f}")
    print(f"[좋은 예측]  AUC     : {auc(y_true, p_good):.6f}")
    print(f"[좋은 예측]  macroF1 : {macro_f1(y_true, p_good):.6f}")
