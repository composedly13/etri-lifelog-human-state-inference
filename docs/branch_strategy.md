# 브랜치 전략 (Branch Strategy)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회  
> 마지막 업데이트: 2026-05-26

---

## 브랜치 구조 개요

```
main
 └── dev
      ├── exp/lgbm-baseline          ← LightGBM baseline 실험
      ├── exp/context-features       ← 날짜/요일/주말/시간 맥락 피처
      ├── exp/personal-baseline      ← 개인별 기준선 피처
      └── docs/paper-draft           ← 논문 초안 및 문서 정리
```

---

## 브랜치 역할 정의

### `main` — 안정 버전 (제출 기준)

- **목적**: 논문 제출/코드 제출 시점의 검증된 안정 버전만 유지
- **병합 조건**: `dev` 브랜치에서 재현성이 확인된 코드만 merge
- **태그 규칙**: 제출 시점마다 `v1.0`, `v1.1` 형식으로 태그 부여
- **직접 커밋 금지**: 모든 변경은 `dev` → `main` PR/merge 방식으로만 반영
- **재현성 기준**: `main`의 코드로 Private 리더보드 점수를 재현할 수 있어야 함

```bash
# main 최신화 예시
git checkout main
git merge dev
git tag v1.0
git push origin main --tags
```

---

### `dev` — 통합 개발 브랜치

- **목적**: 실험 브랜치 → 통합 → main으로 올리기 전 최종 검증
- **병합 조건**:
  - 실험 결과가 `docs/experiment_log.md`에 기록되어 있을 것
  - 로컬 validation 점수가 이전 dev 버전보다 동등하거나 개선될 것
  - 코드가 실행 가능한 상태일 것 (`python src/train_lgbm.py --config configs/base.yaml` 성공)
- **직접 커밋**: 설정 파일 수정, 문서 업데이트 등 소규모 변경은 직접 커밋 가능

```bash
# dev 최신화 예시
git checkout dev
git merge exp/lgbm-baseline
```

---

### `exp/lgbm-baseline` — LightGBM 기본 모델 실험

- **목적**: 원본 피처 + 기본 결측치 처리 + LightGBM KFold 학습
- **실험 내용**:
  - `src/train_lgbm.py` 동작 검증
  - 기본 하이퍼파라미터 성능 측정
  - feature importance 분석
- **완료 기준**: OOF 스코어 측정 완료, `docs/experiment_log.md` Exp001 기록 완료

```bash
git checkout -b exp/lgbm-baseline dev
```

---

### `exp/context-features` — 시간 맥락 피처 실험

- **목적**: 날짜/요일/주말/시간대 파생 피처가 성능에 미치는 영향 검증
- **실험 내용**:
  - `configs/base.yaml`: `use_date_features: true`
  - `src/preprocess.py`: `parse_date_features()` 검증
  - lag/rolling 피처 (`src/features.py`) 구현 후 추가 실험
- **완료 기준**: baseline 대비 점수 변화 측정, `docs/experiment_log.md` Exp002/003 기록

```bash
git checkout -b exp/context-features dev
```

---

### `exp/personal-baseline` — 개인별 기준선 피처 실험

- **목적**: 개인별 평균 대비 deviation/z-score 피처가 예측 성능에 기여하는지 검증
- **실험 내용**:
  - `src/features.py`: `add_personal_baseline_features()` 구현
  - GroupKFold 적용 (user_id/subject_id 단위 fold 분할)
  - `configs/base.yaml`: `use_personal_baseline: true`, `use_group_kfold: true`
- **완료 기준**: 개인별 편차 피처 구현 완료, `docs/experiment_log.md` Exp004 기록

```bash
git checkout -b exp/personal-baseline dev
```

---

### `docs/paper-draft` — 논문 초안 및 문서 정리

- **목적**: 논문 작성, 실험 요약 정리, 그림/표 생성
- **포함 파일**:
  - `docs/paper_outline.md` — 논문 구조 초안
  - `docs/model_description.md` — 모델 설명서
  - `docs/experiment_log.md` — 실험 결과 요약
  - `notebooks/` — EDA 및 결과 시각화 노트북
- **주의**: 이 브랜치에는 코드 변경을 최소화하고, 문서와 분석 결과 위주로 관리

```bash
git checkout -b docs/paper-draft dev
```

---

## 실험 브랜치 운영 규칙

### ✅ 반드시 지켜야 할 사항

1. **실험 목적 명확화**  
   브랜치 생성 전 `docs/experiment_log.md`에 실험 목적과 가설을 먼저 작성

2. **재현 가능한 코드만 merge**  
   성능이 개선되거나 동등하고, `python src/train_lgbm.py --config configs/base.yaml`로 재현 가능한 코드만 `dev`로 merge

3. **실험 결과 반드시 기록**  
   모든 실험 후 `docs/experiment_log.md`에 OOF 스코어, 피처 전략, 관찰 사항 기록

4. **Git에 올리지 않는 파일**  
   대회 데이터 (`data/`), 모델 파일 (`*.pkl`), 제출 파일 (`outputs/submissions/`)은 `.gitignore`에 의해 자동 제외됨

5. **UTF-8 인코딩 유지**  
   모든 코드 파일과 주석은 UTF-8 인코딩으로 저장 (대회 규정)

### ⛔ 금지 사항

- `main`에 직접 실험용 코드 커밋
- 재현 불가능한 코드를 `dev`에 merge
- 대회 원본 데이터를 Git에 커밋
- 모델 pickle 파일 Git 업로드 (용량/규정 문제)

---

## 커밋 메시지 규칙

```
<type>: <short description>

[optional body]
[optional footer]
```

**타입**:
- `feat`: 새 기능/피처 추가
- `fix`: 버그 수정
- `exp`: 실험 코드 (결과 포함 시 OOF 스코어 기록)
- `docs`: 문서 업데이트
- `refactor`: 코드 구조 개선 (기능 변경 없음)
- `chore`: 설정, 패키지 등 보조 작업

**예시**:
```
exp: add date/weekday features (Baseline 2)

- use_date_features: true 활성화
- parse_date_features()로 year/month/day/dayofweek/is_weekend 파생
- OOF RMSE: 0.XXXXX (Exp001: 0.XXXXX 대비 개선)

Ref: docs/experiment_log.md Exp002
```

---

## 일반적인 워크플로우

```
1. dev에서 실험 브랜치 생성
   git checkout -b exp/context-features dev

2. 실험 목적을 experiment_log.md에 먼저 기록

3. 코드 수정 및 실험 실행
   python src/train_lgbm.py --config configs/base.yaml

4. 결과를 experiment_log.md에 기록

5. 성능 개선 확인 후 dev로 merge
   git checkout dev
   git merge exp/context-features

6. 제출 준비 완료 시 main으로 merge + 태그
   git checkout main
   git merge dev
   git tag v1.1
```

---

## 브랜치 현황 트래킹

| 브랜치 | 상태 | 마지막 OOF Score | 비고 |
|--------|------|-----------------|------|
| main | 🟢 안정 | - | 초기 구조 설정 |
| dev | 🟡 진행중 | - | |
| exp/lgbm-baseline | ⬜ 예정 | - | Exp001 예정 |
| exp/context-features | ⬜ 예정 | - | Exp002/003 예정 |
| exp/personal-baseline | ⬜ 예정 | - | Exp004 예정 |
| docs/paper-draft | ⬜ 예정 | - | 논문 작성 시 |
