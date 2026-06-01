# -*- coding: utf-8 -*-
"""
베이스라인 학습/제출 (src/baseline.py)

대회: 제5회 ETRI 휴먼이해 AI 논문경진대회 (7개 이진 지표, 평가 = Average Log-Loss)

설계 메모:
  - train/test는 동일한 10명(id01~id10)의 서로 다른 날짜이다 ("처음 보는 사람" 아님).
    → 개인 단위 GroupKFold가 아니라 StratifiedKFold(랜덤)가 테스트 조건에 부합한다.
    → subject_id를 범주형 피처로 포함 (Q1~Q3가 개인 평균 기준이라 개인 식별이 유효).

모드:
  python -m src.baseline --mode mean    # 타깃별 train 평균확률 제출 (sanity floor)
  python -m src.baseline --mode lgbm    # 지표별 LightGBM (기본)
  python -m src.baseline --mode both    # 둘 다 생성
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from src.build_dataset import build_dataset, KEY_COLS, TARGET_COLS
from src.metrics import competition_metric, auc, macro_f1

SEED = 42
N_FOLDS = 5
SUB_DIR = Path("outputs/submissions")
LOG_DIR = Path("outputs/logs")

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "seed":             SEED,
    "verbose":          -1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING  = 100


def _prep_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols):
    """subject_id를 범주형 피처로 추가 (train/test 공통 카테고리)."""
    cats = sorted(set(train["subject_id"]) | set(test["subject_id"]))
    train = train.copy(); test = test.copy()
    train["subject_id"] = pd.Categorical(train["subject_id"], categories=cats)
    test["subject_id"]  = pd.Categorical(test["subject_id"],  categories=cats)
    feats = ["subject_id"] + list(feature_cols)
    return train, test, feats, ["subject_id"]


def _write_submission(sub_sample: pd.DataFrame, preds: dict, tag: str) -> Path:
    """제출 샘플 컬럼 순서를 유지하며 타깃 확률을 채워 저장."""
    out = sub_sample.copy()
    for t in TARGET_COLS:
        out[t] = preds[t]
    out = out[sub_sample.columns]
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUB_DIR / f"submission_{tag}_{ts}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"  → 제출 저장: {path}  shape={out.shape}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1) 평균확률 베이스라인 (sanity floor)
# ──────────────────────────────────────────────────────────────────────────────

def run_mean(train, test, sub_sample):
    print("\n=== [mean] 타깃별 train 평균확률 제출 ===")
    preds = {}
    for t in TARGET_COLS:
        p = float(train[t].mean())
        preds[t] = np.full(len(test), p)
        # 자기 자신에 대한 logloss (상수 예측의 이론적 하한 근처)
        ll = competition_metric(train[t].values, np.full(len(train), p))
        print(f"  {t}: p={p:.3f}  train-LogLoss={ll:.4f}")
    avg = np.mean([competition_metric(train[t].values, np.full(len(train), train[t].mean()))
                   for t in TARGET_COLS])
    print(f"  평균 train-LogLoss(상수): {avg:.4f}")
    _write_submission(sub_sample, preds, tag="mean")


# ──────────────────────────────────────────────────────────────────────────────
# 2) 지표별 LightGBM 베이스라인
# ──────────────────────────────────────────────────────────────────────────────

def train_target(train, test, feats, cat_cols, target):
    X, y = train[feats], train[target].values
    Xt = test[feats]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_iters = []
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        model = lgb.train(
            LGBM_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        oof[va]    = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        test_pred += model.predict(Xt, num_iteration=model.best_iteration) / N_FOLDS
        best_iters.append(model.best_iteration)
    return oof, test_pred, best_iters


def run_lgbm(train, test, sub_sample, feature_cols):
    print("\n=== [lgbm] 지표별 LightGBM (StratifiedKFold, OOF Log-Loss) ===")
    train, test, feats, cat_cols = _prep_features(train, test, feature_cols)

    preds, rows = {}, []
    for t in TARGET_COLS:
        oof, tpred, iters = train_target(train, test, feats, cat_cols, t)
        preds[t] = tpred
        ll = competition_metric(train[t].values, oof)
        a  = auc(train[t].values, oof)
        f1 = macro_f1(train[t].values, oof)
        base = competition_metric(train[t].values, np.full(len(train), train[t].mean()))
        rows.append({"target": t, "oof_logloss": ll, "base_logloss": base,
                     "gain": base - ll, "auc": a, "macro_f1": f1,
                     "mean_best_iter": int(np.mean(iters))})
        print(f"  {t}: LogLoss={ll:.4f} (base {base:.4f}, gain {base-ll:+.4f}) "
              f"AUC={a:.3f} F1={f1:.3f}")

    res = pd.DataFrame(rows)
    mean_ll   = res["oof_logloss"].mean()
    mean_base = res["base_logloss"].mean()
    print(f"\n  ── 평균 OOF Log-Loss: {mean_ll:.4f}  "
          f"(평균확률 base {mean_base:.4f}, gain {mean_base-mean_ll:+.4f}) ──")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "baseline_lgbm", "seed": SEED, "n_folds": N_FOLDS,
               "mean_oof_logloss": round(float(mean_ll), 5),
               "mean_base_logloss": round(float(mean_base), 5),
               "per_target": res.round(5).to_dict(orient="records")}
    with open(LOG_DIR / "baseline_lgbm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  요약 저장: {LOG_DIR / 'baseline_lgbm_summary.json'}")

    _write_submission(sub_sample, preds, tag="baseline_lgbm")
    return res


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mean", "lgbm", "both"], default="lgbm")
    ap.add_argument("--no-cache", action="store_true", help="일별 피처 캐시 무시하고 재집계")
    args = ap.parse_args()

    np.random.seed(SEED)
    train, test, sub_sample, _, feature_cols = build_dataset(use_cache=not args.no_cache)

    if args.mode in ("mean", "both"):
        run_mean(train, test, sub_sample)
    if args.mode in ("lgbm", "both"):
        run_lgbm(train, test, sub_sample, feature_cols)


if __name__ == "__main__":
    main()
