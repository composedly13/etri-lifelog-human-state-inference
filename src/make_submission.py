# -*- coding: utf-8 -*-
"""
제출 파일 생성 모듈 (src/make_submission.py)
test 예측값을 sample_submission 형식에 맞게 정리하여 CSV로 저장합니다.

저장 경로: configs/base.yaml의 paths.submission_dir
파일명 형식: submission_{exp_name}_{timestamp}.csv
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.config import Config


def make_submission(
    test_pred_df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    cfg: Config,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    test 예측값으로 submission 파일을 생성하여 저장합니다.

    Args:
        test_pred_df     : test 예측값 DataFrame (id_col + target_cols 포함)
        sample_submission: sample_submission DataFrame (컬럼 순서 기준)
        cfg              : Config 객체
        suffix           : 파일명 suffix (None이면 타임스탬프 자동 생성)

    Returns:
        생성된 submission DataFrame
    """
    id_col = cfg.id_col
    submission = sample_submission.copy()

    # ── 예측값 병합 ─────────────────────────────────────────────────────────
    if id_col in test_pred_df.columns:
        # id_col 기준 merge (순서 불일치 안전 처리)
        target_cols = [c for c in sample_submission.columns if c != id_col]
        pred_cols   = [id_col] + [c for c in target_cols if c in test_pred_df.columns]
        submission  = submission[[id_col]].merge(
            test_pred_df[pred_cols], on=id_col, how="left"
        )
    else:
        # id_col 없이 순서 보장 방식 (행 수 일치 필요)
        if len(test_pred_df) != len(submission):
            raise ValueError(
                f"\n[오류] test_pred_df 행 수({len(test_pred_df)})와 "
                f"sample_submission 행 수({len(submission)})가 다릅니다.\n"
                f"  - id_col('{id_col}')이 test_pred_df에 포함되어 있는지 확인하세요."
            )
        target_cols = [c for c in sample_submission.columns if c != id_col]
        for col in target_cols:
            if col in test_pred_df.columns:
                submission[col] = test_pred_df[col].values

    # sample_submission의 컬럼 순서 유지
    submission = submission[sample_submission.columns]

    # ── 결측치 확인 ──────────────────────────────────────────────────────────
    missing = submission.isna().sum().sum()
    if missing > 0:
        print(f"[make_submission] ⚠ submission에 결측치 {missing}개 있음 — 확인 필요")

    # ── 파일 저장 ────────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_sfx  = suffix if suffix else ts
    save_path = cfg.paths["submission_dir"] / f"submission_{cfg.exp_name}_{file_sfx}.csv"

    submission.to_csv(save_path, index=False, encoding="utf-8")

    print(f"[make_submission] 제출 파일 저장: {save_path}")
    print(f"[make_submission] Shape: {submission.shape}")
    print(submission.head(3).to_string(index=False))

    return submission


# ── 독립 실행 안내 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        "[make_submission] 이 모듈은 train_lgbm.py에서 호출됩니다.\n"
        "직접 실행하려면 다음과 같이 사용하세요:\n\n"
        "  from src.config import Config\n"
        "  from src.load_data import load_sample_submission\n"
        "  from src.make_submission import make_submission\n\n"
        "  cfg = Config()\n"
        "  sub = load_sample_submission(cfg)\n"
        "  make_submission(test_pred_df, sub, cfg)"
    )
