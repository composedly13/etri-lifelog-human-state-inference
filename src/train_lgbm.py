# -*- coding: utf-8 -*-
"""
LightGBM 학습 모듈 (src/train_lgbm.py)
KFold / GroupKFold 기반 cross-validation으로 모든 target 컬럼을 학습합니다.

저장 결과:
  - outputs/logs/     : 학습 로그, feature importance CSV, 실험 요약 JSON
  - outputs/models/   : 각 fold의 모델 pickle
  - outputs/submissions/ : submission CSV

재현성:
  - random seed 고정 (configs/base.yaml의 model.seed)
  - Private 리더보드 재현을 위해 seed를 변경하지 마세요.
"""

import json
import pickle
import logging
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from src.config import Config
from src.metrics import competition_metric, print_scores, multi_target_score

# ── LightGBM 임포트 (미설치 시 친절한 오류 메시지) ──────────────────────────
try:
    import lightgbm as lgb
    print(f"[train_lgbm] LightGBM 버전: {lgb.__version__}")
except ImportError:
    raise ImportError(
        "\n" + "=" * 55 + "\n"
        "[오류] LightGBM이 설치되어 있지 않습니다.\n"
        "아래 명령어로 설치하세요:\n\n"
        "  pip install lightgbm\n\n"
        "또는 프로젝트 requirements 전체 설치:\n\n"
        "  pip install -r requirements.txt\n"
        + "=" * 55
    )

from sklearn.model_selection import KFold, GroupKFold
# 평가 지표는 src/metrics.py에서 관리합니다. 대회 metric 변경 시 metrics.py만 수정하세요.


# ── 로거 설정 ─────────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, exp_name: str) -> logging.Logger:
    """파일 + 콘솔 로거를 설정합니다."""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{exp_name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── 단일 타겟 학습 ────────────────────────────────────────────────────────────

def train_single_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cfg: Config,
    logger: logging.Logger,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    단일 target 컬럼에 대해 KFold cross-validation으로 LightGBM을 학습합니다.

    Returns:
        oof_preds   : OOF 예측값 배열 (len = len(train))
        test_preds  : Test 예측값 배열 (len = len(test))  — fold 평균
        result_dict : 학습 결과 정보 dict
    """
    logger.info("\n" + "=" * 55)
    logger.info(f"Target: {target_col}")
    logger.info(f"피처 수: {len(feature_cols)}")

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]

    oof_preds          = np.zeros(len(train))
    test_preds         = np.zeros(len(test))
    fold_scores: List[float]  = []
    feature_importances = pd.DataFrame()
    models: List      = []

    # ── KFold / GroupKFold 선택 ───────────────────────────────────────────
    if cfg.use_group_kfold and cfg.group_col and cfg.group_col in train.columns:
        logger.info(f"GroupKFold 사용: group_col='{cfg.group_col}'")
        kf     = GroupKFold(n_splits=cfg.n_folds)
        groups = train[cfg.group_col]
        splits = list(kf.split(X_train, y_train, groups=groups))
    else:
        logger.info(f"KFold 사용: n_folds={cfg.n_folds}, seed={cfg.seed}")
        kf     = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        splits = list(kf.split(X_train, y_train))

    # ── 범주형 컬럼 설정 ──────────────────────────────────────────────────
    lgbm_cat = [c for c in (cat_cols or []) if c in feature_cols] or "auto"

    # ── n_estimators를 lgb.train 인수로 분리 ─────────────────────────────
    params = cfg.lgbm_params.copy()
    num_boost_round = params.pop("n_estimators", 1000)

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        logger.info(f"\n── Fold {fold_idx + 1}/{cfg.n_folds} ──")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        dtrain = lgb.Dataset(
            X_tr, label=y_tr,
            categorical_feature=lgbm_cat,
            free_raw_data=False
        )
        dval = lgb.Dataset(
            X_val, label=y_val,
            reference=dtrain,
            free_raw_data=False
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # OOF 예측
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # Test 예측 (fold 평균)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / cfg.n_folds

        fold_score = competition_metric(y_val, val_pred)
        fold_scores.append(fold_score)
        logger.info(f"Fold {fold_idx + 1} {cfg.eval_metric.upper()}: {fold_score:.6f}")

        # Feature importance
        fi = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
            "fold":       fold_idx + 1,
        })
        feature_importances = pd.concat([feature_importances, fi], ignore_index=True)
        models.append(model)

    # ── 전체 OOF 스코어 ────────────────────────────────────────────────────
    oof_score = competition_metric(y_train, oof_preds)
    logger.info("\n" + "=" * 55)
    logger.info(f"[{target_col}] OOF Score    : {oof_score:.6f}")
    logger.info(f"[{target_col}] Fold Scores  : {[f'{s:.4f}' for s in fold_scores]}")
    logger.info(f"[{target_col}] Mean ± Std   : {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")

    fi_mean = (
        feature_importances
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    result_dict = {
        "target_col":         target_col,
        "oof_score":          oof_score,
        "fold_scores":        fold_scores,
        "mean_score":         float(np.mean(fold_scores)),
        "std_score":          float(np.std(fold_scores)),
        "feature_importance": fi_mean,
        "models":             models,
        "best_iterations":    [m.best_iteration for m in models],
    }
    return oof_preds, test_preds, result_dict


# ── 전체 타겟 학습 파이프라인 ─────────────────────────────────────────────────

def train_all_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    cfg: Config,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    모든 target 컬럼에 대해 순차적으로 학습을 수행합니다.

    Returns:
        oof_df       : OOF 예측값 DataFrame
        test_pred_df : Test 예측값 DataFrame
        all_results  : 각 타겟별 결과 dict 리스트
    """
    logger = setup_logger(cfg.paths["log_dir"], cfg.exp_name)
    logger.info(f"실험명    : {cfg.exp_name}  v{cfg.exp_version}")
    logger.info(f"설명      : {cfg.exp_description}")
    logger.info(f"Target    : {target_cols}")
    logger.info(f"피처 수   : {len(feature_cols)}")
    logger.info(f"Seed      : {cfg.seed}")

    oof_dict       = {}
    test_pred_dict = {}
    all_results    = []

    for target_col in target_cols:
        oof_preds, test_preds, result = train_single_target(
            train, test, feature_cols, target_col, cfg, logger, cat_cols
        )
        oof_dict[target_col]       = oof_preds
        test_pred_dict[target_col] = test_preds
        all_results.append(result)

        # Feature importance 저장
        fi_path = cfg.paths["log_dir"] / f"{cfg.exp_name}_{target_col}_feature_importance.csv"
        result["feature_importance"].to_csv(fi_path, index=False, encoding="utf-8")
        logger.info(f"Feature importance 저장: {fi_path}")

        # 모델 저장
        for fold_idx, model in enumerate(result["models"]):
            model_path = cfg.paths["model_dir"] / f"{cfg.exp_name}_{target_col}_fold{fold_idx + 1}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        logger.info(f"모델 저장: {cfg.paths['model_dir']}")

    # 전체 결과 요약 JSON 저장
    summary = {
        "experiment":  cfg.exp_name,
        "version":     cfg.exp_version,
        "description": cfg.exp_description,
        "seed":        cfg.seed,
        "n_folds":     cfg.n_folds,
        "targets":     {},
    }
    for result in all_results:
        summary["targets"][result["target_col"]] = {
            "oof_score":       round(result["oof_score"],   6),
            "mean_fold_score": round(result["mean_score"],  6),
            "std_fold_score":  round(result["std_score"],   6),
            "best_iterations": result["best_iterations"],
        }
    summary_path = cfg.paths["log_dir"] / f"{cfg.exp_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n실험 요약 저장: {summary_path}")

    # 전체 평균 OOF 출력
    all_oof = [r["oof_score"] for r in all_results]
    logger.info(f"\n{'='*55}")
    logger.info(f"전체 평균 OOF Score: {np.mean(all_oof):.6f}")

    # ── 결과 DataFrame 구성 ─────────────────────────────────────────────
    oof_df = pd.DataFrame(oof_dict)
    oof_df.insert(0, cfg.id_col, train[cfg.id_col].values)

    test_pred_df = pd.DataFrame(test_pred_dict)
    test_pred_df.insert(0, cfg.id_col, test[cfg.id_col].values)

    return oof_df, test_pred_df, all_results


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 커맨드라인 인자 파싱 ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="ETRI 휴먼이해 AI 논문경진대회 - LightGBM 학습 스크립트"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="설정 파일 경로 (기본: configs/base.yaml)",
    )
    args = parser.parse_args()

    from src.load_data import load_all
    from src.preprocess import preprocess
    from src.features import build_features
    from src.make_submission import make_submission

    # ── 설정 로드 ────────────────────────────────────────────────────────────
    cfg = Config(args.config)
    print(cfg)

    # ── 시드 고정 (Private LB 재현성) ────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"[main] Seed 고정: {cfg.seed}")

    # ── 파이프라인 실행 ──────────────────────────────────────────────────────
    train, test, sub, target_cols, feature_cols = load_all(cfg)
    train, test, feature_cols, cat_cols         = preprocess(train, test, feature_cols, cfg)
    train, test, feature_cols                   = build_features(train, test, feature_cols, cfg)

    oof_df, test_pred_df, all_results = train_all_targets(
        train, test, feature_cols, target_cols, cfg, cat_cols
    )

    # ── 멀티 타겟 최종 스코어 출력 ───────────────────────────────────────────
    oof_true = train[target_cols].reset_index(drop=True)
    oof_pred = oof_df[target_cols].reset_index(drop=True)
    scores   = multi_target_score(oof_true, oof_pred, target_cols)
    print_scores(scores, metric_name="Final OOF Score")

    make_submission(test_pred_df, sub, cfg)
    print("\n[main] 학습 및 제출 파일 생성 완료!")
