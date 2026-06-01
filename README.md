# Context-Aware Human State Inference Using Multimodal Lifelog Data

> **제5회 ETRI 휴먼이해 인공지능 논문경진대회** 참가 코드  
> Python 3.10+ | UTF-8 인코딩 | 재현 가능한 실험 관리 기반

---

## Repository Purpose

이 레포지토리는 ETRI 휴먼이해 AI 논문경진대회 제출용 코드 및 실험 기록을 관리합니다.  
- 대회 공식 데이터를 직접 포함하지 않습니다 (`.gitignore`로 제외)
- 모든 실험은 재현 가능하도록 seed와 config를 고정합니다
- 논문 채택 시 코드와 모델 설명서를 제출할 수 있는 구조를 유지합니다

---

## Competition Context

| 항목 | 내용 |
|------|------|
| 대회명 | 제5회 ETRI 휴먼이해 인공지능 논문경진대회 |
| 목표 | 멀티모달 라이프로그(전날) → 수면/상태 **7개 지표 이진 분류** |
| 태스크 | Multi-label binary classification (7개 타깃, 독립 예측) |
| 대상자 / 기간 | id01~id10 (10명) / 2024-06-03 ~ 2024-11-19 |
| 데이터 | 라이프로그 12종(700일분) + 라벨 7지표(450일분) |
| 제출 형식 | 논문 + 코드 + 모델 설명서 |
| 코드 기준 | Private LB 스코어 재현 가능해야 함 |
| 인코딩 | UTF-8 |

---

## Task & Labels

전날 라이프로그(`lifelog_date`)로 다음 날 수면/상태 7개 지표(`sleep_date` 기준)를 예측합니다.
모든 지표는 **0/1 이진값**이며, 각 지표를 독립적으로 분류합니다.

| 지표 | 설명 | 의미 (1 = 긍정) |
|------|------|------|
| **Q1** | 기상 직후 주관적 전반 수면의 질 | 1: 개인 평균 이상 / 0: 평균 미만 |
| **Q2** | 취침 직전 신체 피로도 | 1: 낮음 / 0: 높음 |
| **Q3** | 취침 직전 스트레스 수준 | 1: 낮음 / 0: 높음 |
| **S1** | 총 수면시간(TST) 가이드라인 준수 | 1: 권장 / 0: 미권장 |
| **S2** | 수면 효율(SE) 가이드라인 준수 | 1: 적정 / 0: 부적정 |
| **S3** | 수면 잠복기(SOL) 가이드라인 준수 | 1: 적정 / 0: 부적정 |
| **S4** | 입면 후 각성(WASO) 가이드라인 준수 | 1: 적정 / 0: 부적정 |

- **Q1~Q3**: 사전/사후 설문(5점 척도)을 개인 평균 대비 이진화 → 개인별 기준선 보정이 핵심
- **S1~S4**: Withings 수면센서 측정값을 NSF(미국수면재단) 가이드라인으로 이진화
- **결합 키**: `subject_id` + `sleep_date`(예측 기준일) + `lifelog_date`(센서 매칭용 전날)
- 평가지표는 `ch2026_metrics_description.pdf` 및 대회 공지 기준(전년도와 동일 포맷: 지표별 F1 계열) 사용

> 출처: `data/raw/ch2026_metrics_description.pdf` (Table 1)

---

## Research Direction

**연구 제목 후보**: *Context-Aware Human State Inference Using Multimodal Lifelog Data*

### 핵심 아이디어

> 단순 센서 수치 예측이 아닌, **시간적 맥락**과 **개인별 기준선**을 반영하여  
> 수면·피로·스트레스 등 인간 상태를 더 정확하게 추론하는 것을 목표로 합니다.  
> 특히 Q1~Q3는 라벨 자체가 *개인 평균 대비* 이진화되어 있어, 개인별 기준선 피처가 직접적인 효과를 가집니다.

### 연구 로드맵

| 단계 | 전략 | 브랜치 | 예상 기여 |
|------|------|--------|----------|
| **Baseline 1** | 일별 센서 집계 피처 + 지표별 LightGBM 분류 | `exp/lgbm-baseline` | 기준 성능 측정 |
| **Baseline 2** | 날짜/요일/주말/시간 파생 피처 | `exp/context-features` | 일주기 패턴 반영 |
| **Proposed 1** | lag/rolling/diff 시간 맥락 피처 | `exp/context-features` | 단기 시계열 맥락 |
| **Proposed 2** | 개인별 평균 대비 deviation 피처 | `exp/personal-baseline` | 개인차 보정 |
| **Proposed 3** | LGB + CatBoost + XGB 앙상블 | `dev` | 최종 성능 극대화 |

---

## Branch Strategy

> 상세 내용: [docs/branch_strategy.md](docs/branch_strategy.md)

```
main          ← 제출 가능한 안정 버전 (태그 관리)
 └── dev      ← 통합 개발 브랜치
      ├── exp/lgbm-baseline       ← LightGBM baseline 실험
      ├── exp/context-features    ← 날짜/요일/시간 맥락 피처
      ├── exp/personal-baseline   ← 개인별 기준선 피처
      └── docs/paper-draft        ← 논문 초안 및 문서
```

**규칙**:
- 실험 브랜치는 목적이 명확해야 하며, 결과는 반드시 `docs/experiment_log.md`에 기록
- 성능 개선 or 재현 가능한 코드만 `dev` → `main`으로 merge
- 대회 데이터, 모델 파일, 제출 파일은 Git에 올리지 않음

---

## Directory Structure

```
etri-lifelog-human-state-inference/
│
├── configs/
│   └── base.yaml                # 전체 파이프라인 설정 (경로/모델/피처/학습)
│
├── data/                        # ⛔ Git 미포함 (.gitignore)
│   ├── .gitkeep
│   └── raw/                     # 대회 원본 데이터 위치 (아래 Data Placement 참고)
│       ├── ch2025_data_items/   # 라이프로그 12종 parquet (700일분)
│       ├── ch2026_metrics_train.csv         # 라벨 (450일분, 7지표)
│       ├── ch2026_submission_sample.csv     # 제출 샘플 (250일분)
│       └── ch2026_metrics_description.pdf   # 지표 정의 문서
│
├── docs/
│   ├── branch_strategy.md       # 브랜치 전략
│   ├── daily_log.md             # 일일 작업 기록
│   ├── experiment_log.md        # 실험별 결과 기록
│   ├── model_description.md     # 모델 설명서 (논문 제출용)
│   └── paper_outline.md         # 논문 목차 초안
│
├── notebooks/
│   ├── 00_data_check.ipynb      # 데이터 구조 점검 (최초 실행)
│   └── eda.ipynb                # 탐색적 데이터 분석
│
├── outputs/                     # ⛔ Git 미포함
│   ├── .gitkeep
│   ├── logs/                    # 학습 로그, feature importance
│   ├── models/                  # 모델 pickle 파일
│   ├── figures/                 # 시각화 결과
│   └── submissions/             # 제출 CSV 파일
│
├── scripts/
│   └── run_baseline.ps1         # Windows PowerShell 실행 스크립트
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Config 클래스 (YAML → Python 객체)
│   ├── load_data.py             # 데이터 로드 + id/target 자동 추론
│   ├── preprocess.py            # 결측치 처리 + 날짜 피처 생성
│   ├── features.py              # 피처 엔지니어링 (lag/rolling/personal 포함 TODO)
│   ├── train_lgbm.py            # 메인 학습 스크립트
│   ├── metrics.py               # 평가 지표 (competition_metric)
│   └── make_submission.py       # 제출 파일 생성
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data Placement

대회 데이터를 다음 구조로 `data/raw/` 아래에 배치하세요.
라이프로그는 [ETRI 데이터 포털](https://epretx.etri.re.kr/dataDetail?lang=ko&id=459)에서 `ch2025_data_items.zip`(122MB) 다운로드 후 압축 해제합니다.

```
data/raw/
├── ch2025_data_items/                 # 라이프로그 12종 (subject_id, timestamp 공통)
│   ├── ch2025_mACStatus.parquet       # 충전 상태        (m_charging)
│   ├── ch2025_mActivity.parquet       # 활동 인식        (m_activity)
│   ├── ch2025_mAmbience.parquet       # 주변 음향 태그    (m_ambience)
│   ├── ch2025_mBle.parquet            # BLE 주변 기기     (m_ble)
│   ├── ch2025_mGps.parquet            # GPS 위치          (m_gps)
│   ├── ch2025_mLight.parquet          # 폰 조도           (m_light)
│   ├── ch2025_mScreenStatus.parquet   # 화면 사용         (m_screen_use)
│   ├── ch2025_mUsageStats.parquet     # 앱 사용 통계      (m_usage_stats)
│   ├── ch2025_mWifi.parquet           # WiFi 스캔         (m_wifi)
│   ├── ch2025_wHr.parquet             # 워치 심박         (heart_rate)
│   ├── ch2025_wLight.parquet          # 워치 조도         (w_light)
│   └── ch2025_wPedo.parquet           # 워치 보행(step/distance/speed/calories 등)
│
├── ch2026_metrics_train.csv           # 라벨 450행 (subject_id, sleep_date, lifelog_date, Q1~Q3, S1~S4)
├── ch2026_submission_sample.csv       # 제출 샘플 250행 (동일 스키마, 타깃은 0으로 채워짐)
└── ch2026_metrics_description.pdf     # 7지표 정의
```

| 구분 | 행 수 | 비고 |
|------|-------|------|
| 라이프로그 | 항목별 21K ~ 961K행 | `m*`=폰, `w*`=워치, `mGps`가 최대(~72MB) |
| 학습 라벨 | 450행 | id01~id10, lifelog_date 2024-06-03 ~ 11-14, 결측 0 |
| 제출 대상 | 250행 | lifelog_date 2024-07-06 ~ 11-19 |

> ⚠️ `m_gps · m_ambience · m_ble · m_wifi · m_usage_stats`는 중첩/문자열 형태이므로 피처화 시 파싱이 필요합니다.

### 경로 전환 방법

`configs/base.yaml`에서 `data_dir`만 변경하면 전체 파이프라인에 반영됩니다:

```yaml
paths:
  data_dir: "data/raw"   # 로컬 실험
  # data_dir: "/data"    # 대회 서버 제출 시
```

> ⚠️ **원본 대회 데이터는 절대 GitHub에 업로드하지 마세요.**  
> `data/` 디렉토리는 `.gitignore`로 자동 제외됩니다.

> 📌 현재 `configs/base.yaml` · `src/load_data.py`는 단일 `train.csv`/`test.csv`(회귀) 기준 스캐폴딩 상태입니다.
> parquet 병합·일별 집계·7지표 분류 파이프라인으로 갱신이 필요합니다 (별도 작업).

---

## Environment Setup

### 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.10 이상 |
| pandas | 2.0.0 이상 |
| numpy | 1.24.0 이상 |
| scikit-learn | 1.3.0 이상 |
| lightgbm | 4.0.0 이상 |
| pyyaml | 6.0.0 이상 |

### 설치

```bash
# 가상환경 생성 (권장)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

---

## Baseline Execution

### 방법 1: 직접 실행

```bash
python src/train_lgbm.py --config configs/base.yaml
```

### 방법 2: PowerShell 스크립트 (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
```

### 실행 결과 위치

| 출력물 | 경로 |
|--------|------|
| 학습 로그 | `outputs/logs/{exp_name}_{timestamp}.log` |
| 실험 요약 | `outputs/logs/{exp_name}_summary.json` |
| Feature Importance | `outputs/logs/{exp_name}_{target}_feature_importance.csv` |
| 모델 파일 | `outputs/models/{exp_name}_{target}_fold{n}.pkl` |
| 제출 파일 | `outputs/submissions/submission_{exp_name}_{timestamp}.csv` |

---

## Reproducibility

Private 리더보드 점수를 재현하려면:

```bash
# 1. 동일한 config로 실행
python src/train_lgbm.py --config configs/base.yaml

# 2. seed 확인 (configs/base.yaml)
#    model.seed: 42
#    training.seed: 42

# 3. 동일한 제출 파일 생성 확인
#    outputs/submissions/ 에서 최신 파일 확인
```

**재현성 보장 항목**:
- `random.seed(42)`, `numpy.random.seed(42)` 고정
- LightGBM `random_state: 42`
- KFold `shuffle=True, random_state=42` (라벨 누수 방지를 위해 `subject_id` 기준 GroupKFold 권장)
- `configs/base.yaml` 변경 이력 Git으로 추적

---

## Experiment Tracking

모든 실험은 `docs/experiment_log.md`에 기록합니다.

```markdown
### Exp001 — baseline_v1
- Date: 2026-05-26
- Branch: exp/lgbm-baseline
- Config: configs/base.yaml (use_date_features: false)
- OOF F1 (지표별 Q1~S4): 0.XXX / ... / 평균 0.XXX
- Public LB: 0.XXXXX
- 관찰: ...
```

**하루 제출 3회 제한** 대응 전략:
1. 로컬 OOF 스코어로 먼저 실험 검증
2. OOF 개선 확인 후에만 공식 제출
3. 제출 전 `docs/experiment_log.md` 기록 업데이트

---

## Notebooks

| 노트북 | 목적 | 실행 순서 |
|--------|------|----------|
| `notebooks/00_data_check.ipynb` | 데이터 구조 점검, target/id 추론 | **1번** (데이터 넣은 후 최초 실행) |
| `notebooks/eda.ipynb` | 탐색적 데이터 분석, 시각화 | 2번 |

---

## Paper / Model Documentation

| 문서 | 경로 | 용도 |
|------|------|------|
| 논문 목차 초안 | `docs/paper_outline.md` | 논문 구조 계획 |
| 모델 설명서 | `docs/model_description.md` | 대회 제출용 모델 문서 |
| 실험 로그 | `docs/experiment_log.md` | 실험 결과 추적 |
| 브랜치 전략 | `docs/branch_strategy.md` | 개발 워크플로우 |
| 일일 로그 | `docs/daily_log.md` | 작업 진행 기록 |

---

## Quick Reference

```bash
# 설정 확인
python src/config.py

# 데이터 로드 테스트 (데이터 없으면 친절한 안내 출력)
python src/load_data.py

# baseline 학습
python src/train_lgbm.py --config configs/base.yaml

# 노트북 서버 실행
jupyter notebook notebooks/00_data_check.ipynb
```

---

## Notes

- 이 레포지토리는 대회 코드 공개 기준을 준수합니다
- 모든 주석과 문서는 UTF-8로 작성합니다
- 외부 데이터 및 사전학습 모델은 초기 단계에서 사용하지 않습니다
- 사용 시 `docs/model_description.md`에 명시합니다
