# 실험 기록 (Experiment Log)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회  
> 마지막 업데이트: 2026-06-02  
> 평가: **Average Log-Loss** (7개 이진 지표 평균, 낮을수록 좋음) · 하루 제출 3회

---

## 실험 목록

| # | 실험명 | 날짜 | Branch | OOF Score | Public LB | Private LB | 피처 전략 | 비고 |
|---|--------|------|--------|-----------|-----------|-----------|-----------|------|
| Exp000 | mean_prob | 2026-06-02 | main | 0.6641 | - | - | 타깃별 train 평균확률(상수) | sanity floor |
| Exp001 | baseline_lgbm | 2026-06-02 | main | **0.6215** | - | - | 일별 센서 집계 52피처 + subject_id | 기준 실험 |
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
| **Date** | 2026-06-02 |
| **Branch** | `main` |
| **Purpose** | 일별 센서 집계 피처로 7개 지표 분류 기준 성능 측정 |
| **Data Version** | 일별 피처 700행 × 52피처 (`data/processed/daily_features.parquet`) |
| **Run** | `python -m src.baseline --mode both` |

**가설**: 라이프로그를 (subject, 하루) 단위로 집계한 통계 피처만으로도 평균확률(상수) 대비 의미 있는 예측이 가능하며, 이후 피처 추가의 기준점이 된다.

**Data 구성**:
- 12종 parquet → (subject_id, date) 일별 집계 → (subject_id, lifelog_date)로 라벨 join
- train 450행 / test 250행, 라벨행 피처 매칭 100% (전부결측 0%)

**Features (53)**: `subject_id`(범주형) + 52개 일별 집계
- wHr: hr_mean/std/min/max/records · wPedo: step·distance·calorie 합계, speed/freq 평균·최대
- m_charging/m_screen_use: 비율·합계 · m_light/w_light: mean/std/min/max
- m_activity: count/nunique/mean/std · m_gps: speed, 좌표 std(이동범위), point 수
- m_ambience: 태그수·최고확률 · m_ble/m_wifi: 기기수·rssi · m_usage_stats: 사용시간·앱수

**Model**: LightGBM `objective=binary`, 지표별 독립 학습, predict_proba

**Validation**: StratifiedKFold(5, shuffle, seed=42) — train/test가 동일 10명이라 GroupKFold(subject) 대신 채택

| Target | OOF LogLoss | base(평균확률) | gain | AUC | macroF1 |
|--------|------------|---------------|------|-----|---------|
| Q1 | 0.6711 | 0.6931 | +0.0220 | 0.615 | 0.598 |
| Q2 | 0.6447 | 0.6854 | +0.0407 | 0.655 | 0.573 |
| Q3 | 0.6458 | 0.6730 | +0.0272 | 0.605 | 0.526 |
| S1 | 0.5771 | 0.6252 | +0.0481 | 0.690 | 0.530 |
| S2 | 0.5970 | 0.6468 | +0.0497 | 0.691 | 0.564 |
| S3 | **0.5641** | 0.6396 | +0.0754 | 0.730 | 0.598 |
| S4 | 0.6504 | 0.6859 | +0.0355 | 0.646 | 0.597 |
| **Mean** | **0.6215** | 0.6641 | +0.0427 | 0.662 | 0.569 |

| | Score |
|---|-------|
| **Local OOF Log-Loss** | **0.6215** |
| **Public LB Score** | **0.6319** (2026-06-02, 제출ID 1460086) · OOF↔LB 갭 +0.0104 |

**Submission File**: `outputs/submissions/submission_baseline_lgbm_*.csv` (+ `submission_mean_*.csv`)  
**Summary**: `outputs/logs/baseline_lgbm_summary.json`

**Result Summary**:
- 평균확률 floor 0.6641 → LGBM 0.6215 (-0.0427). 리더보드 상위권 0.549와의 격차 확인.
- 수면센서 지표(S1~S3)가 설문지표(Q1~Q3)보다 예측 잘 됨. 특히 **S3(SOL) AUC 0.730**으로 최고.
- **Q1(전반 수면질) 최약** (AUC 0.615) — 개인 주관 평균 기준이라 개인별 보정 필요.

**Observations**:
- 일별 단순 집계만으로 전 지표 base 초과. 신호는 있으나 마진이 작아 calibration 중요.
- 야간 시간대(수면 구간) 한정 피처가 S1~S4에 더 직접적일 것.

**Next Action**:
- Exp002: 날짜/요일/주말 + 시간대(야간) 집계 피처
- Exp004: 개인별 평균 대비 deviation/z-score (특히 Q1~Q3 겨냥)
- 첫 LB 제출로 OOF↔LB 갭 확인

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

> 상세 제출 로그: [notebooks/submission_log.md](../notebooks/submission_log.md)

| 날짜 | 회차 | 파일명 | Public LB | 비고 |
|------|------|--------|-----------|------|
| 2026-06-01 | 1 | ch2026_submission_sample.csv | 21.1270 | 샘플(전부 0) — 형식 확인용 |
| 2026-06-02 | 1 | submission_baseline_lgbm_20260602_000207.csv | **0.6319** | Exp001 베이스라인 (OOF 0.6215) |

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
