# 실험 기록 (Experiment Log)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회  
> 마지막 업데이트: 2026-05-26  
> 하루 제출 제한: **3회** — 반드시 로컬 OOF 점수 확인 후 제출할 것

---

## 실험 목록

| # | 실험명 | 날짜 | Branch | OOF Score | Public LB | Private LB | 피처 전략 | 비고 |
|---|--------|------|--------|-----------|-----------|-----------|-----------|------|
| Exp001 | baseline_v1 | - | exp/lgbm-baseline | - | - | - | 원본 피처 + 결측치 처리 | 기준 실험 |
| Exp002 | baseline_v2 | - | exp/context-features | - | - | - | + 날짜/요일/주말 피처 | Baseline 2 |
| Exp003 | proposed_v1 | - | exp/context-features | - | - | - | + lag/rolling 시간 맥락 | Proposed 1 |
| Exp004 | proposed_v2 | - | exp/personal-baseline | - | - | - | + 개인별 편차 피처 | Proposed 2 |
| Exp005 | ensemble_v1 | - | dev | - | - | - | LGB+CB+XGB 앙상블 | Proposed 3 |

---

## 실험 템플릿

새 실험을 시작할 때 아래 템플릿을 복사하여 작성하세요.

```markdown
### Exp___ — [실험명]

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp___ |
| **Date** | YYYY-MM-DD |
| **Branch** | exp/... |
| **Commit Hash** | `git rev-parse --short HEAD` |
| **Purpose** | 이 실험의 목적 (한 줄 요약) |
| **Data Version** | train.csv v? (row수, col수) |
| **Config** | `configs/base.yaml` |

**가설**: 이 피처 전략이 성능을 개선할 것이다. 왜냐하면 ...

**Features**:
- feature 1
- feature 2

**Model**: LightGBM (5-Fold KFold, seed=42)

**Validation Strategy**: 5-Fold KFold / GroupKFold (group_col=user_id)

| Target | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | OOF | vs 이전 |
|--------|-------|-------|-------|-------|-------|-----|--------|
| target1 | | | | | | | |
| target2 | | | | | | | |
| **Mean** | | | | | | | |

| | Score |
|---|-------|
| **Local OOF Score** | 0.XXXXX |
| **Public LB Score** | 0.XXXXX |
| **Private LB Score** | - (대회 종료 후) |

**Submission File**: `outputs/submissions/submission_XXXXX.csv`  
**Model File**: `outputs/models/XXXXX_fold{1-5}.pkl`

**Result Summary**:
- 

**Observations**:
- 

**Next Action**:
- 
```

---

## 실험 상세

---

### Exp001 — baseline_v1

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp001 |
| **Date** | - |
| **Branch** | `exp/lgbm-baseline` |
| **Commit Hash** | - |
| **Purpose** | LightGBM 기본 파이프라인 동작 확인 및 기준 성능 측정 |
| **Data Version** | 데이터 미확보 (대회 오픈 후 기입) |
| **Config** | `configs/base.yaml` |

**가설**: 원본 피처만으로도 의미 있는 예측이 가능하며, 이후 피처 추가의 기준점이 된다.

**Settings**:
```yaml
experiment:
  name: "baseline_v1"
  description: "Baseline 1: 원본 피처 + 기본 결측치 처리"
features:
  use_date_features: false
  use_lag_features: false
  use_personal_baseline: false
training:
  n_folds: 5
  use_group_kfold: false
```

**Features**: 원본 수치형/범주형 피처 전체 (id/target 제외)

**Model**: LightGBM, 5-Fold KFold, seed=42

**Validation Strategy**: KFold (n_splits=5, shuffle=True, random_state=42)

| Target | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | OOF | vs 이전 |
|--------|-------|-------|-------|-------|-------|-----|--------|
| (target1) | - | - | - | - | - | - | 기준 |
| (target2) | - | - | - | - | - | - | 기준 |
| **Mean** | | | | | | - | 기준 |

| | Score |
|---|-------|
| **Local OOF Score** | - |
| **Public LB Score** | - |
| **Private LB Score** | - |

**Submission File**: `outputs/submissions/submission_baseline_v1_YYYYMMDD.csv`  
**Model File**: `outputs/models/baseline_v1_{target}_fold{1-5}.pkl`

**Result Summary**:
- 아직 실험 전

**Next Action**:
- 데이터 확보 후 실행
- feature importance 분석 → 불필요 피처 제거 검토
- baseline_v2 (날짜 피처) 실험 계획

---

### Exp002 — baseline_v2

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp002 |
| **Date** | - |
| **Branch** | `exp/context-features` |
| **Commit Hash** | - |
| **Purpose** | 날짜/요일/주말/시간대 파생 피처가 예측 성능에 기여하는지 검증 |
| **Config** | `configs/base.yaml` (use_date_features: true) |

**가설**: 수면/피로/스트레스는 요일 및 시간대의 영향을 받으므로, 날짜 파생 피처가 성능을 향상시킬 것이다.

**Settings**:
```yaml
experiment:
  name: "baseline_v2"
  description: "Baseline 2: 날짜/요일/주말 피처 추가"
features:
  use_date_features: true
  date_cols: []  # 자동 감지 또는 명시
```

**Features**: 원본 피처 + `{date_col}_year/month/day/dayofweek/is_weekend/hour/quarter`

| Target | OOF | vs Exp001 |
|--------|-----|---------|
| (target1) | - | - |
| **Mean** | - | - |

**Next Action**:
- lag/rolling 피처 추가 (Exp003 = Proposed 1)

---

### Exp003 — proposed_v1 (TODO)

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp003 |
| **Branch** | `exp/context-features` |
| **Purpose** | lag/rolling 시간 맥락 피처로 단기 시계열 패턴 반영 |

**가설**: 전일/3일/7일 이동평균이 당일 상태 예측에 유의미한 정보를 제공한다.

**Settings**:
```yaml
features:
  use_lag_features: true
  lag_windows: [1, 2, 3, 7]
  rolling_windows: [3, 7, 14]
```

**TODO**: `src/features.py`의 `add_lag_features()`, `add_rolling_features()` 구현 필요

---

### Exp004 — proposed_v2 (TODO)

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp004 |
| **Branch** | `exp/personal-baseline` |
| **Purpose** | 개인별 평균 대비 deviation 피처로 개인차 보정 |

**가설**: 동일한 수면 시간이라도 개인마다 기준선이 다르므로, 개인 평균 대비 편차가 더 의미 있는 피처다.

**Settings**:
```yaml
features:
  use_personal_baseline: true
  group_col: "user_id"  # 또는 "subject_id"
training:
  use_group_kfold: true
  group_col: "user_id"
```

**TODO**: `src/features.py`의 `add_personal_baseline_features()` 구현 필요

---

### Exp005 — ensemble_v1 (TODO)

| 항목 | 내용 |
|------|------|
| **Experiment ID** | Exp005 |
| **Branch** | `dev` |
| **Purpose** | LightGBM + CatBoost + XGBoost weighted ensemble |

**TODO**: Exp001-004 완료 후 진행

---

## Feature Importance 메모

> 실험 후 `outputs/logs/{exp_name}_{target}_feature_importance.csv` 참고

| 실험 | Target | Top-5 Feature | Bottom-3 Feature |
|------|--------|---------------|------------------|
| Exp001 | - | - | - |

---

## Hyperparameter Tuning 기록

| 파라미터 | 기본값 | 시도값 | 결과 | 채택 |
|----------|--------|--------|------|------|
| num_leaves | 31 | 63 | - | - |
| learning_rate | 0.05 | 0.01 | - | - |
| n_estimators | 1000 | 2000 | - | - |
| min_child_samples | 20 | 10 | - | - |

---

## 에러 & 트러블슈팅

| 날짜 | 증상 | 원인 | 해결 |
|------|------|------|------|
| - | - | - | - |

---

## 제출 이력 (하루 최대 3회)

| 날짜 | 회차 | 파일명 | Public LB | 비고 |
|------|------|--------|-----------|------|
| - | 1 | - | - | |

---

## 실험 설계 노트

### 실험 전 체크리스트

- [ ] 가설이 명확한가?
- [ ] config 변경 사항이 기록되었는가?
- [ ] 이전 실험 결과를 알고 있는가?
- [ ] 오늘 제출 횟수를 확인했는가? (최대 3회)

### 실험 후 체크리스트

- [ ] OOF 스코어가 기록되었는가?
- [ ] Feature importance를 확인했는가?
- [ ] 다음 실험 아이디어를 기록했는가?
- [ ] 제출 파일 경로가 기록되었는가?
