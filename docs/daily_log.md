# 일일 작업 로그 (Daily Work Log)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회  
> 기록 원칙: 매일 작업 종료 후 무엇을 했고, 무엇을 배웠으며, 내일 무엇을 할지 기록한다.

---

## 로그 템플릿

```markdown
### YYYY-MM-DD (Day N)

**Branch**: 현재 브랜치  
**오늘 목표**: 

**완료한 작업**:
- 

**학습/발견**:
- 

**막힌 지점 / 이슈**:
- 

**내일 계획**:
- 

**제출 현황** (해당 시): 오늘 X/3회 제출
```

---

## 작업 기록

---

### 2026-05-26 (Day 1)

**Branch**: `main` (초기 구조 설정)  
**오늘 목표**: 레포지토리 초기 구조 구축 및 전체 파이프라인 코드 작성

**완료한 작업**:
- GitHub 레포지토리 생성 (`etri-lifelog-human-state-inference`)
- 전체 프로젝트 디렉토리 구조 생성
- `configs/base.yaml`: 전체 파이프라인 설정 파일 작성
  - 경로, 데이터, 피처, 모델, 학습, 실험 메타데이터 포함
- `src/config.py`: Config 클래스 (YAML → Python 객체 변환)
- `src/load_data.py`: 데이터 로더 (친절한 에러 처리, 인코딩 자동 시도, id/target 자동 추론)
- `src/preprocess.py`: 전처리 파이프라인 (결측치 처리, 범주형 인코딩, 날짜 피처 파싱)
- `src/features.py`: 피처 엔지니어링 (lag/rolling/개인기준선 TODO 포함)
- `src/train_lgbm.py`: LightGBM 학습 스크립트 (KFold/GroupKFold, multi-target)
- `src/metrics.py`: 평가 지표 모듈 (RMSE/MAE, multi-target 평균)
- `src/make_submission.py`: 제출 파일 생성 모듈
- `docs/branch_strategy.md`: 브랜치 전략 문서
- `docs/experiment_log.md`: 실험 기록 템플릿 및 Exp001-005 계획
- `docs/model_description.md`: 논문 제출용 모델 설명서 초안
- `docs/paper_outline.md`: 논문 목차 초안
- `docs/daily_log.md`: 일일 작업 로그 (이 파일)
- `README.md`: 전체 레포지토리 설명서
- `notebooks/00_data_check.ipynb`: 데이터 점검 노트북
- `scripts/run_baseline.ps1`: Windows PowerShell 실행 스크립트
- `.gitignore`, `requirements.txt`: 환경 설정
- `data/raw/.gitkeep`: 데이터 디렉토리 구조 준비

**학습/발견**:
- 대회 규정: 코드에 `/data` 절대경로 포함 필요 → config에서 `data/raw`와 `/data` 전환 가능하게 설계
- 하루 제출 3회 제한 → 로컬 OOF 기반 실험 관리가 핵심
- UTF-8 인코딩 규정 → 모든 파일 헤더에 `# -*- coding: utf-8 -*-` 명시

**막힌 지점 / 이슈**:
- 실제 데이터가 없어 target 컬럼명, id 컬럼명 미확정
  → config에서 자동 추론 + 명시적 지정 양쪽 지원으로 해결
- `notebooks/eda.ipynb` 파일이 빈 상태로 존재 → 00_data_check.ipynb 신규 생성으로 대응

**내일 계획**:
- 대회 데이터 다운로드 후 `data/raw/`에 배치
- `notebooks/00_data_check.ipynb` 실행하여 데이터 구조 파악
- target 컬럼, id 컬럼 확인 후 `configs/base.yaml` 업데이트
- `exp/lgbm-baseline` 브랜치 생성 후 첫 학습 실행
- Exp001 결과 `docs/experiment_log.md`에 기록

**제출 현황**: 0/3회 (데이터 미확보, 아직 미제출)

---

## 실험-날짜 연결표

| 날짜 | 주요 작업 | 실험 | 제출 |
|------|----------|------|------|
| 2026-05-26 | 초기 구조 설정 | - | 0/3 |

---

## 주간 요약 템플릿

```markdown
## Week N (YYYY-MM-DD ~ YYYY-MM-DD)

**이번 주 목표**: 

**완료**:
- 

**개선된 OOF Score**: X.XXXXX → X.XXXXX

**다음 주 계획**:
- 
```
