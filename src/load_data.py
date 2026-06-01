# -*- coding: utf-8 -*-
"""
데이터 로드 모듈 (src/load_data.py)

주요 기능:
  - train / test / sample_submission CSV 로드
  - id_col 자동 추론 (sample_submission 첫 번째 컬럼)
  - target_cols 자동 추론 (sample_submission에서 id_col 제외)
  - train/test 컬럼 불일치 자동 처리
  - UTF-8 인코딩 실패 시 cp949 자동 재시도
  - 데이터 summary 출력 (shape, dtype, missing ratio)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

from src.config import Config


# ── 인코딩 자동 시도 CSV 로더 ────────────────────────────────────────────────

def _read_csv_safe(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    CSV를 읽습니다. 지정 인코딩 실패 시 cp949를 자동으로 시도합니다.
    대회 데이터가 cp949(EUC-KR)로 저장된 경우를 대비합니다.
    """
    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        fallback = "cp949" if encoding.lower() == "utf-8" else "utf-8"
        print(f"[load_data] ⚠ {encoding} 디코딩 실패 → {fallback}로 재시도: {path.name}")
        return pd.read_csv(path, encoding=fallback)


def _not_found_msg(path: Path, file_role: str, cfg: Config) -> str:
    """FileNotFoundError 친절 안내 메시지를 생성합니다."""
    return (
        f"\n{'='*55}\n"
        f"[오류] {file_role} 파일을 찾을 수 없습니다.\n\n"
        f"  찾은 경로: {path.resolve()}\n\n"
        f"  해결 방법:\n"
        f"  1. 대회 데이터를 다음 위치에 넣으세요:\n"
        f"       {cfg.paths['data_dir'].resolve()}/\n"
        f"  2. 파일명이 다르면 configs/base.yaml을 수정하세요:\n"
        f"       paths.{file_role.lower().replace(' ', '_')}_file: \"실제파일명.csv\"\n"
        f"  3. 절대 경로를 사용하려면:\n"
        f"       paths.data_dir: \"/data\"  # 대회 서버 기준\n"
        f"{'='*55}"
    )


# ── 개별 로더 ────────────────────────────────────────────────────────────────

def load_train(cfg: Config) -> pd.DataFrame:
    """train 데이터를 로드합니다."""
    path = cfg.paths["train_file"]
    if not path.exists():
        raise FileNotFoundError(_not_found_msg(path, "Train", cfg))
    df = _read_csv_safe(path, cfg.encoding)
    print(f"[load_data] Train  로드 완료: shape={df.shape}")
    return df


def load_test(cfg: Config) -> pd.DataFrame:
    """test 데이터를 로드합니다."""
    path = cfg.paths["test_file"]
    if not path.exists():
        raise FileNotFoundError(_not_found_msg(path, "Test", cfg))
    df = _read_csv_safe(path, cfg.encoding)
    print(f"[load_data] Test   로드 완료: shape={df.shape}")
    return df


def load_sample_submission(cfg: Config) -> pd.DataFrame:
    """sample_submission 파일을 로드합니다."""
    path = cfg.paths["sample_submission_file"]
    if not path.exists():
        raise FileNotFoundError(_not_found_msg(path, "Sample Submission", cfg))
    df = _read_csv_safe(path, cfg.encoding)
    print(f"[load_data] Sample sub 로드: shape={df.shape}")
    return df


# ── 컬럼 추론 ────────────────────────────────────────────────────────────────

def infer_id_col(sample_submission: pd.DataFrame) -> str:
    """
    sample_submission의 첫 번째 컬럼을 id_col로 추론합니다.
    """
    id_col = sample_submission.columns[0]
    print(f"[load_data] id_col 자동 추론   : '{id_col}' (sample_submission 첫 번째 컬럼)")
    return id_col


def infer_target_cols(sample_submission: pd.DataFrame, id_col: str) -> List[str]:
    """
    sample_submission에서 id_col을 제외한 컬럼을 target_cols로 추론합니다.
    """
    target_cols = [c for c in sample_submission.columns if c != id_col]
    if not target_cols:
        raise ValueError(
            f"\n[오류] target 컬럼을 추론할 수 없습니다.\n"
            f"  - id_col='{id_col}' 가 올바른지 확인하세요.\n"
            f"  - configs/base.yaml의 data.id_col 또는 data.target_cols를 직접 지정하세요.\n"
            f"  - sample_submission 컬럼: {list(sample_submission.columns)}"
        )
    print(f"[load_data] target_cols 자동 추론: {target_cols}")
    return target_cols


def get_feature_cols(
    train: pd.DataFrame,
    target_cols: List[str],
    id_col: str,
    drop_cols: Optional[List[str]] = None,
) -> List[str]:
    """target, id, drop_cols를 제외한 나머지를 feature_cols로 반환합니다."""
    exclude = set(target_cols) | {id_col} | set(drop_cols or [])
    feature_cols = [c for c in train.columns if c not in exclude]
    print(f"[load_data] feature_cols 수    : {len(feature_cols)}")
    return feature_cols


# ── 데이터 요약 출력 ──────────────────────────────────────────────────────────

def print_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    데이터프레임의 shape, dtype, 결측치 비율 요약을 출력합니다.
    EDA 초기 단계와 노트북에서 활용합니다.
    """
    print(f"\n{'─'*55}")
    print(f"  {name} Summary")
    print(f"{'─'*55}")
    print(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} cols")

    # dtype 분포
    dtype_counts = df.dtypes.value_counts().to_dict()
    print(f"  Dtypes  : {dtype_counts}")

    # 결측치 비율 상위 10개
    missing = df.isna().mean().sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]
    if len(missing_nonzero) == 0:
        print(f"  Missing : 없음 ✓")
    else:
        print(f"  Missing : {len(missing_nonzero)}개 컬럼에 결측치 존재")
        for col, ratio in missing_nonzero.head(10).items():
            print(f"            {col:30s}: {ratio*100:.1f}%")
        if len(missing_nonzero) > 10:
            print(f"            ... 외 {len(missing_nonzero)-10}개 컬럼")

    # 날짜형 컬럼 후보
    date_candidates = []
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "time", "dt", "timestamp", "날짜", "일자"]):
            date_candidates.append(col)
    if date_candidates:
        print(f"  Date?   : {date_candidates}")

    # 개인 식별 컬럼 후보
    id_candidates = []
    for col in df.columns:
        if any(kw in col.lower() for kw in ["user", "subject", "person", "id", "uid"]):
            id_candidates.append(col)
    if id_candidates:
        print(f"  GroupBy?: {id_candidates}")

    print(f"{'─'*55}\n")


# ── 통합 로더 ────────────────────────────────────────────────────────────────

def load_all(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    train / test / sample_submission을 한 번에 로드하고
    id_col, target_cols, feature_cols를 결정하여 반환합니다.

    Returns:
        train, test, sample_submission, target_cols, feature_cols
    """
    print("\n" + "=" * 55)
    print("[load_data] 데이터 로드 시작")
    print("=" * 55)

    train = load_train(cfg)
    test  = load_test(cfg)
    sub   = load_sample_submission(cfg)

    # ── id_col 결정 ────────────────────────────────────────────────────────
    if cfg.id_col:
        id_col = cfg.id_col
        print(f"[load_data] id_col (config)    : '{id_col}'")
    else:
        id_col = infer_id_col(sub)
        cfg.id_col = id_col   # config에 역주입하여 이후 모듈에서 참조 가능

    # ── target_cols 결정 ─────────────────────────────────────────────────
    if cfg.target_cols:
        target_cols = cfg.target_cols
        print(f"[load_data] target_cols (config): {target_cols}")
    else:
        target_cols = infer_target_cols(sub, id_col)
        cfg.target_cols = target_cols   # 역주입

    # train에 target 컬럼이 실제로 있는지 검증
    missing_targets = [c for c in target_cols if c not in train.columns]
    if missing_targets:
        raise ValueError(
            f"\n[오류] Train 데이터에 target 컬럼이 없습니다: {missing_targets}\n"
            f"  - configs/base.yaml의 data.target_cols를 확인하세요.\n"
            f"  - Train 컬럼 목록: {list(train.columns)}"
        )

    # ── feature_cols 결정 ─────────────────────────────────────────────────
    feature_cols = get_feature_cols(train, target_cols, id_col, cfg.drop_cols)

    # ── train/test 컬럼 불일치 처리 ──────────────────────────────────────
    train_feat_set = set(feature_cols)
    test_cols_set  = set(test.columns) - {id_col}

    only_in_train = train_feat_set - test_cols_set
    only_in_test  = test_cols_set  - train_feat_set

    if only_in_train:
        print(f"[load_data] ⚠ Train에만 있는 피처 → 제거: {sorted(only_in_train)}")
        feature_cols = [c for c in feature_cols if c not in only_in_train]
    if only_in_test:
        print(f"[load_data] ⚠ Test에만 있는 컬럼  → 무시: {sorted(only_in_test)}")

    print(f"\n[load_data] 최종 feature_cols   : {len(feature_cols)}개")
    print(f"[load_data] target_cols         : {target_cols}")
    print("=" * 55 + "\n")

    return train, test, sub, target_cols, feature_cols


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    train, test, sub, target_cols, feature_cols = load_all(cfg)

    print_data_summary(train, "Train")
    print_data_summary(test,  "Test")

    print(f"feature_cols (처음 10개): {feature_cols[:10]}")
    print(f"target_cols: {target_cols}")
