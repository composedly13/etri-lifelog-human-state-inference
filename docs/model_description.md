# 모델 설명서 (Model Description)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회  
> 작성 기준: 논문 채택 시 코드 제출 규정 준수  
> 마지막 업데이트: 2026-05-26

---

## Overview

| 항목 | 내용 |
|------|------|
| **모델명** | Context-Aware LightGBM Ensemble |
| **버전** | v1.0 (baseline_v1) |
| **목적** | 멀티모달 라이프로그 데이터로 인간 상태(수면·피로·스트레스 등) 예측 |
| **핵심 접근법** | 시간적 맥락 피처 + 개인별 기준선 피처 + LightGBM KFold 앙상블 |
| **평가 지표** | RMSE (대회 공식 metric 확정 후 `src/metrics.py` 업데이트) |

---

## Data Input / Output

### 입력 데이터

| 파일 | 경로 | 설명 |
|------|------|------|
| `train.csv` | `data/raw/train.csv` 또는 `/data/train.csv` | 학습 데이터 (피처 + target 포함) |
| `test.csv` | `data/raw/test.csv` 또는 `/data/test.csv` | 추론 데이터 (피처만 포함) |
| `sample_submission.csv` | `data/raw/sample_submission.csv` 또는 `/data/sample_submission.csv` | 제출 형식 기준 파일 |

**경로 설정**: `configs/base.yaml`의 `paths.data_dir` 변경으로 전환

### 출력 데이터

| 파일 | 경로 | 설명 |
|------|------|------|
| `submission_{exp_name}_{ts}.csv` | `outputs/submissions/` | test 예측값 제출 파일 |
| `{exp_name}_summary.json` | `outputs/logs/` | 실험 요약 (OOF 스코어 등) |
| `{exp_name}_{target}_feature_importance.csv` | `outputs/logs/` | 피처 중요도 |
| `{exp_name}_{target}_fold{n}.pkl` | `outputs/models/` | fold별 학습 모델 |

---

## Preprocessing

### 1. 인코딩 안전 로딩
- 기본: UTF-8로 CSV 로드
- 실패 시: cp949(EUC-KR) 자동 재시도
- 구현: `src/load_data.py` → `_read_csv_safe()`

### 2. 컬럼 자동 추론
- **id_col**: `sample_submission.csv`의 첫 번째 컬럼을 자동 추론 (또는 config에서 지정)
- **target_cols**: `sample_submission.csv`에서 id_col 제외 컬럼으로 자동 추론
- **feature_cols**: train에서 id_col, target_cols, drop_cols 제외 나머지

### 3. train/test 컬럼 불일치 처리
- train에만 있는 피처: 학습에서 제거 (경고 출력)
- test에만 있는 컬럼: 무시

### 4. 결측치 처리
- **수치형 컬럼**: train 기준 median으로 채움 (test에도 동일 값 적용, 데이터 누수 방지)
- **범주형 컬럼**: `"missing"` 문자열로 채움
- 구현: `src/preprocess.py` → `handle_missing_values()`

### 5. 범주형 인코딩
- pandas `Categorical` dtype으로 변환
- train + test 전체 기준으로 카테고리 정의 (unseen 카테고리 방지)
- LightGBM은 `Categorical` dtype을 직접 처리 (별도 레이블 인코딩 불필요)
- 구현: `src/preprocess.py` → `encode_categoricals()`

---

## Feature Engineering

### Baseline 1 — 원본 피처
- 원본 수치형/범주형 피처 그대로 사용
- 결측치 처리 후 LightGBM에 직접 투입

### Baseline 2 — 날짜/맥락 파생 피처 (use_date_features: true)

날짜 컬럼에서 다음 파생 피처를 생성합니다:

| 파생 피처 | 설명 | 인간 상태와의 관련성 |
|-----------|------|---------------------|
| `{col}_year` | 연도 | 계절적 추세 |
| `{col}_month` | 월 | 월별 패턴 |
| `{col}_day` | 일 | 월중 패턴 |
| `{col}_dayofweek` | 요일 (0=월, 6=일) | 주중/주말 생활 패턴 |
| `{col}_is_weekend` | 주말 여부 (0/1) | 휴식 vs 근무 패턴 |
| `{col}_hour` | 시간대 | 일주기 리듬 |
| `{col}_quarter` | 분기 | 계절적 영향 |

- 자동 감지: 컬럼명에 `date/time/dt/timestamp/날짜/일자` 포함 시 시도
- 구현: `src/preprocess.py` → `parse_date_features()`

### Proposed 1 — 시간 맥락 피처 (TODO)

```yaml
# configs/base.yaml
features:
  use_lag_features: true
  lag_windows: [1, 2, 3, 7]
  rolling_windows: [3, 7, 14]
```

- Lag 피처: 전일/2일/3일/7일 전 값
- Rolling 통계: 3일/7일/14일 이동평균, 이동표준편차
- Diff 피처: 전일 대비 변화량
- 구현 예정: `src/features.py` → `add_lag_features()`, `add_rolling_features()`

### Proposed 2 — 개인별 기준선 피처 (TODO)

```yaml
# configs/base.yaml
features:
  use_personal_baseline: true
  group_col: "user_id"  # 또는 "subject_id"
```

각 수치형 피처에 대해 다음을 생성:
- `{col}_personal_mean`: 개인별 평균 (train 기준)
- `{col}_personal_std`: 개인별 표준편차
- `{col}_deviation`: 개인 평균 대비 편차
- `{col}_zscore`: 개인 기준 z-score

**핵심 아이디어**: 동일한 수면 시간이라도 개인마다 기준선이 다름.  
절대값보다 개인 평균 대비 편차가 인간 상태와 더 강한 상관관계를 가질 것으로 가설.

구현 예정: `src/features.py` → `add_personal_baseline_features()`

---

## Model Architecture

### 기본 모델: LightGBM

| 하이퍼파라미터 | 값 | 설명 |
|----------------|-----|------|
| `n_estimators` | 1000 | 최대 트리 수 (early stopping으로 조기 종료) |
| `learning_rate` | 0.05 | 학습률 |
| `num_leaves` | 31 | 최대 리프 노드 수 |
| `max_depth` | -1 | 제한 없음 |
| `min_child_samples` | 20 | 최소 샘플 수 |
| `subsample` | 0.8 | 행 샘플링 비율 |
| `colsample_bytree` | 0.8 | 열 샘플링 비율 |
| `reg_alpha` | 0.1 | L1 정규화 |
| `reg_lambda` | 0.1 | L2 정규화 |
| `random_state` | 42 | 재현성 고정 |

**Multi-target 처리**: target 컬럼별로 독립적으로 모델 학습  
(target 간 상관관계 활용은 Proposed 3+ 에서 검토)

### 앙상블 계획 (Proposed 3 — TODO)

| 모델 | 역할 |
|------|------|
| LightGBM | 기본 모델 (빠르고 강력) |
| CatBoost | 범주형 피처 처리 강점 |
| XGBoost | 다양성 확보 |

가중 평균 또는 스태킹 방식으로 앙상블 예정

---

## Training Procedure

### 학습 파이프라인

```
1. Config 로드              (src/config.py)
2. 시드 고정 (seed=42)
3. 데이터 로드              (src/load_data.py)
   - train / test / sample_submission 로드
   - id_col, target_cols, feature_cols 자동 추론
4. 전처리                   (src/preprocess.py)
   - 날짜 피처 파싱 (use_date_features=true 시)
   - 결측치 처리
   - 범주형 인코딩
5. 피처 엔지니어링          (src/features.py)
   - lag/rolling 피처 (use_lag_features=true 시)
   - 개인별 기준선 피처 (use_personal_baseline=true 시)
6. 모델 학습                (src/train_lgbm.py)
   - KFold (또는 GroupKFold) 5-fold CV
   - target별 독립 학습
   - Early stopping (100 rounds)
7. 결과 저장
   - OOF 예측값, Feature importance, 모델 pickle
8. 제출 파일 생성           (src/make_submission.py)
```

### 실행 명령어

```bash
python src/train_lgbm.py --config configs/base.yaml
```

---

## Validation Strategy

### KFold (기본)

```yaml
training:
  n_folds: 5
  seed: 42
  use_group_kfold: false
```

- 5-Fold Stratified KFold (shuffle=True, random_state=42)
- OOF(Out-of-Fold) 예측으로 전체 train에 대한 validation 점수 산출

### GroupKFold (개인별 데이터 누수 방지 — 권장)

```yaml
training:
  n_folds: 5
  use_group_kfold: true
  group_col: "user_id"  # 데이터에 user_id/subject_id가 있는 경우
```

- 동일 사용자 데이터가 train/val에 분리되어 데이터 누수 방지
- 개인별 일반화 성능 측정 가능

---

## Inference Procedure

```
1. 학습된 fold 모델 5개 로드
2. test 데이터에 동일한 전처리/피처 엔지니어링 적용
3. fold별 예측값 평균 → 최종 test 예측
4. sample_submission 형식에 맞게 제출 파일 생성
5. outputs/submissions/ 에 저장
```

---

## Reproducibility

Private 리더보드 점수 재현 절차:

1. 동일 환경 설치: `pip install -r requirements.txt`
2. 대회 데이터를 `data/raw/` 또는 `/data/`에 배치
3. 실행: `python src/train_lgbm.py --config configs/base.yaml`
4. `outputs/submissions/` 에서 최신 제출 파일 확인

**재현성 보장 요소**:
- Python random seed: `random.seed(42)`
- NumPy seed: `numpy.random.seed(42)`
- LightGBM `random_state: 42`
- KFold `random_state: 42`
- 모든 설정: `configs/base.yaml` (Git으로 버전 관리)

---

## Environment

| 항목 | 버전 |
|------|------|
| Python | 3.10 이상 |
| pandas | 2.0.0 이상 |
| numpy | 1.24.0 이상 |
| scikit-learn | 1.3.0 이상 |
| lightgbm | 4.0.0 이상 |
| pyyaml | 6.0.0 이상 |
| OS | Windows 11 / Linux (대회 서버) |

전체 패키지 목록: `requirements.txt` 참조

---

## External Data / Pretrained Model Usage

> **현재 상태 (v1.0 baseline): 외부 데이터 및 사전학습 모델 미사용**

- 대회 제공 데이터만 사용
- 사전학습된 임베딩, 외부 통계 데이터 등 미사용
- 향후 확장 시 이 섹션에 명시 필요

---

## Limitations

1. **데이터 의존성**: 대회 데이터 구조(컬럼명, target)가 확정되어야 config를 최적화할 수 있음
2. **시계열 가정**: 현재 KFold는 시간 순서를 보장하지 않음. TimeSeriesSplit으로 전환 검토 가능
3. **개인 식별 필요**: Proposed 2 실험은 user_id/subject_id 컬럼이 데이터에 있어야 함
4. **단일 모달리티 피처**: 현재는 tabular 피처만 사용. 시계열/이미지 등 멀티모달 확장은 미구현
5. **target 독립성 가정**: 현재 multi-target을 독립적으로 학습. target 간 상관관계는 미반영

---

## Changelog

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0 | 2026-05-26 | 초기 구조 설정 (baseline pipeline) |
