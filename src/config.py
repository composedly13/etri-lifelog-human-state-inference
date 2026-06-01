# -*- coding: utf-8 -*-
"""
설정 관리 모듈 (src/config.py)
configs/base.yaml을 읽어 전체 파이프라인에서 사용하는 Config 객체를 생성합니다.

사용 예:
    cfg = Config()                           # configs/base.yaml (기본)
    cfg = Config("configs/base.yaml")
    cfg = Config("configs/exp_proposed1.yaml")
"""

import yaml
from pathlib import Path
from typing import Optional, List


def load_yaml(config_path: str = "configs/base.yaml") -> dict:
    """YAML 설정 파일을 로드하여 dict로 반환합니다."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(
            f"\n[오류] 설정 파일을 찾을 수 없습니다: {p.resolve()}\n"
            f"  - 프로젝트 루트 디렉토리에서 실행하고 있는지 확인하세요.\n"
            f"  - 실행 예: python -m src.train_lgbm\n"
            f"         또는: python src/train_lgbm.py --config configs/base.yaml"
        )
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    """
    전체 파이프라인 설정을 관리하는 클래스.
    configs/base.yaml을 읽어 모든 설정을 속성으로 노출합니다.
    """

    def __init__(self, config_path: str = "configs/base.yaml"):
        self.raw          = load_yaml(config_path)
        self._config_path = config_path

        # ── 경로 설정 ──────────────────────────────────────────────────────
        p = self.raw.get("paths", {})
        data_dir = Path(p.get("data_dir", "data/raw"))
        self.paths = {
            "data_dir":               data_dir,
            "train_file":             data_dir / p.get("train_file", "train.csv"),
            "test_file":              data_dir / p.get("test_file", "test.csv"),
            "sample_submission_file": data_dir / p.get("sample_submission_file", "sample_submission.csv"),
            "output_dir":             Path(p.get("output_dir",     "outputs")),
            "submission_dir":         Path(p.get("submission_dir", "outputs/submissions")),
            "model_dir":              Path(p.get("model_dir",      "outputs/models")),
            "log_dir":                Path(p.get("log_dir",        "outputs/logs")),
            "figure_dir":             Path(p.get("figure_dir",     "outputs/figures")),
        }
        # 출력 디렉토리 자동 생성
        for key in ["output_dir", "submission_dir", "model_dir", "log_dir", "figure_dir"]:
            self.paths[key].mkdir(parents=True, exist_ok=True)

        # ── 데이터 설정 ────────────────────────────────────────────────────
        d = self.raw.get("data", {})
        # id_col: null이면 sample_submission 첫 컬럼으로 load_data에서 추론
        self.id_col:      Optional[str]       = d.get("id_col", None)
        self.target_cols: Optional[List[str]] = d.get("target_cols", None)
        self.encoding:    str                 = d.get("encoding", "utf-8")

        # ── 피처 설정 ──────────────────────────────────────────────────────
        f = self.raw.get("features", {})
        self.use_date_features:     bool           = f.get("use_date_features", True)
        self.date_cols:             List[str]      = f.get("date_cols", []) or []
        self.categorical_cols:      List[str]      = f.get("categorical_cols", []) or []
        self.drop_cols:             List[str]      = f.get("drop_cols", []) or []

        # TODO (Proposed 1): 시간 맥락 피처
        self.use_lag_features:      bool           = f.get("use_lag_features", False)
        self.lag_windows:           List[int]      = f.get("lag_windows", [1, 2, 3, 7])
        self.rolling_windows:       List[int]      = f.get("rolling_windows", [3, 7, 14])

        # TODO (Proposed 2): 개인별 기준선 피처
        self.use_personal_baseline: bool           = f.get("use_personal_baseline", False)
        self.feature_group_col:     Optional[str]  = f.get("group_col", None)

        # ── 모델 설정 ──────────────────────────────────────────────────────
        m = self.raw.get("model", {})
        self.model_name:   str  = m.get("name", "lgbm")
        self.seed:         int  = m.get("seed", 42)
        self.lgbm_params:  dict = m.get("lgbm_params", {})

        # ── 학습 설정 ──────────────────────────────────────────────────────
        t = self.raw.get("training", {})
        self.n_folds:               int           = t.get("n_folds", 5)
        self.use_group_kfold:       bool          = t.get("use_group_kfold", False)
        self.group_col:             Optional[str] = t.get("group_col", None)
        self.early_stopping_rounds: int           = t.get("early_stopping_rounds", 100)
        self.eval_metric:           str           = t.get("eval_metric", "rmse")

        # ── 실험 메타데이터 ────────────────────────────────────────────────
        e = self.raw.get("experiment", {})
        self.exp_name:        str = e.get("name",        "baseline_v1")
        self.exp_version:     str = e.get("version",     "1.0.0")
        self.exp_description: str = e.get("description", "")

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  config_path   = {self._config_path}\n"
            f"  exp_name      = {self.exp_name}  v{self.exp_version}\n"
            f"  description   = {self.exp_description}\n"
            f"  data_dir      = {self.paths['data_dir']}\n"
            f"  model         = {self.model_name},  seed={self.seed}\n"
            f"  n_folds       = {self.n_folds},  group_kfold={self.use_group_kfold}\n"
            f"  id_col        = {self.id_col}  (None → auto-infer)\n"
            f"  target_cols   = {self.target_cols}  (None → auto-infer)\n"
            f"  date_features = {self.use_date_features}\n"
            f"  lag_features  = {self.use_lag_features}   [TODO Proposed 1]\n"
            f"  personal_feat = {self.use_personal_baseline}   [TODO Proposed 2]\n"
            f")"
        )


# ── 독립 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    print(cfg)
    print("\n경로 확인:")
    for k, v in cfg.paths.items():
        exists = "✓" if v.exists() else "✗ (없음)"
        print(f"  {k:25s}: {v}  {exists}")
