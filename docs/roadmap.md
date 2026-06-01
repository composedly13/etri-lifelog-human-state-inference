# 실행 로드맵 (Execution Roadmap)

> 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회 (DACON 236690)
> 작성일: 2026-05-30 · 기준 시점: **D-27 (리더보드/논문 마감 06.26)**
> 목적: 지금부터 최종 제출·시상식까지 전 과정을 단계별 실행 순서로 정리한다.

---

## 0. 시작 전에 반드시 짚을 2가지 (Critical)

대회 공식 페이지를 확인한 결과, 현재 레포지토리 가정과 **두 가지 핵심 불일치**가 있다. 본격 작업 전에 먼저 정리해야 한다.

### (1) 평가 산식은 회귀(RMSE)가 아니라 **분류 / Average Log-Loss**

- 공식 평가 산식: **Average Log-Loss** (대회 태그도 `분류 | Log Loss`)
- 예측 대상:
  - 설문지표 — **Q1**(취침 후 수면의 질), **Q2**(취침 전 피로도), **Q3**(스트레스)
  - 수면지표 — **S1**(총 수면시간), **S2**(수면효율), **S3**(수면 지연시간), **S4**(수면 중 각성시간)
- 현재 `src/metrics.py`, `src/train_lgbm.py`, config는 **RMSE 회귀** 기준으로 작성되어 있음 → **분류(LightGBM `objective=binary`/`multiclass`) + `log_loss` 평가로 전환 필요.** 제출 형식도 클래스 확률(probability)일 가능성이 높으므로 `sample_submission.csv`로 반드시 확인.
- 즉 첫 작업은 데이터 확보 후 **target이 몇 개 클래스인지, 제출이 확률인지 라벨인지** 확인하는 것.

### (2) 리더보드 점수는 최종 순위에 **직접 반영되지 않는다**

> "리더보드 성적은 최종 수상 순위에 직접 반영되지 않습니다."

- 최종 수상(1~5위)은 **논문 채택(ICTC 2026 Workshop) → 재현성 검증 → 종합 평가(참신성·기술 완성도·기여도)** 순으로 결정된다.
- 리더보드는 **필수 참여 요건**이며, **Private 상위 60팀에 논문 작성용 Claude API 크레딧**이 제공된다.
- **전략적 함의**: LB 1등이 목표가 아니다. *재현 가능한 견고한 모델 + 설득력 있는 논문*이 본질이다. 모델링과 논문 작성을 **병행**해야 한다 (둘 다 06.26 마감).

---

## 1. 전체 대회 일정과 단계 매핑

| 날짜 | 공식 마일스톤 | 우리 작업 단계 |
|------|--------------|---------------|
| 04.13 | 참가 접수 시작 | (완료) |
| 04.20 | 대회 시작 / 사전 설명회 | (완료) |
| **~05.30** | — | **레포 구조 구축 완료, 데이터 투입 직전 (현재)** |
| **06.24** | 팀 병합 마감 | 팀 구성 확정 (단독/팀 결정) |
| **06.26** | **리더보드 제출 마감 + 논문 접수 마감** | **모델 최종 제출 + 논문 제출 (최대 고비)** |
| 09.01 | 논문 채택 결과 발표 | 결과 대기 |
| 09.01 | 코드·모델 설명서 제출 마감 | (채택 시) 재현 패키지 제출 |
| 09.30 | 코드 검증 | (채택 시) 재현성 대응 |
| 10.15 | 포스터 발표 및 시상식 | (수상 시) 발표 준비 |

> **핵심**: 06.26까지 약 4주. 모델과 논문을 **동시에** 끝내야 한다. 09월 이후는 채택된 팀만 해당된다.

---

## 2. 단계별 실행 과정 (Phase 0 → Phase 6)

### Phase 0 — 셋업 정리 (~D-26, 05/30~05/31)

- [ ] 대회 데이터 다운로드 → `data/raw/`에 배치 (train/test/sample_submission)
- [ ] `sample_submission.csv` 구조 확인: **타깃 개수, 클래스 수, 제출값 형식(확률 vs 라벨)**
- [ ] **평가 산식 전환**: `src/metrics.py`에 `log_loss` 추가, `train_lgbm.py` objective를 분류로 변경, `configs/base.yaml`의 `task: regression → classification`
- [ ] `python src/load_data.py`로 id/target 컬럼 자동 추론 결과 검증 후 config에 명시

### Phase 1 — 데이터 이해 & EDA (D-25~D-23, 06/01~06/03)

- [ ] `notebooks/00_data_check.ipynb` 실행 — row/col 수, 결측, dtype, target 분포
- [ ] `notebooks/eda.ipynb` 작성 — 모달리티별 분포, target별 클래스 불균형, 사용자(user) 단위 구조 파악
- [ ] train/test 분할 기준 확인 (날짜 기준? 사용자 기준?) → **GroupKFold 필요 여부 결정**
- [ ] 논문 Dataset 섹션용 기초 통계·그림 저장 (`outputs/figures/`)

### Phase 2 — Baseline 확립 (D-22~D-20, 06/04~06/06)

- [ ] `exp/lgbm-baseline` 브랜치 — **Exp001**: 원본 피처 + LightGBM 분류 + KFold, OOF Log-Loss 측정
- [ ] 제출 파이프라인 검증: `make_submission.py`로 확률 제출 → **첫 LB 제출 1회** (형식 검증 목적)
- [ ] `docs/experiment_log.md`에 Exp001 결과 기록 → 이후 모든 실험의 기준선

### Phase 3 — 핵심 피처 엔지니어링 (D-19~D-12, 06/07~06/14)

연구의 핵심 기여. 각 단계는 **OOF로 먼저 검증 후** 개선 시에만 제출.

- [ ] **Exp002** (`exp/context-features`): 날짜/요일/주말/시간대 파생 피처
- [ ] **Exp003**: lag/rolling/diff 시간 맥락 피처 — `src/features.py`의 `add_lag_features()`, `add_rolling_features()` 구현
- [ ] **Exp004** (`exp/personal-baseline`): 개인별 평균 대비 deviation/z-score 피처 — `add_personal_baseline_features()` 구현, GroupKFold 도입
- [ ] 각 실험마다 feature importance 분석 → 논문 Ablation 표 채우기

### Phase 4 — 모델 고도화 & 앙상블 (D-11~D-7, 06/15~06/19)

- [ ] 하이퍼파라미터 튜닝 (num_leaves, learning_rate, min_child_samples 등)
- [ ] **Exp005** (`dev`): LightGBM + CatBoost + XGBoost 가중 앙상블
- [ ] target별(7개) 최적 전략 점검 — 어떤 타깃은 단순 모델이 더 나을 수 있음
- [ ] 최종 후보 모델 2~3개 선정, OOF 기준 비교

### Phase 5 — 최종 제출 & 논문 작성 (D-6~D-0, 06/20~06/26) ⚠️ 최대 고비

모델과 논문을 **병행**. 논문은 Phase 2부터 초안을 점진적으로 채워두면 이 구간이 수월해진다.

- [ ] 최종 모델 확정 → 재현 가능하도록 seed/config 고정, 제출 파일 생성
- [ ] **하루 3회 제출 제한** 고려한 마지막 제출 일정 관리 (마감일 당일 몰리지 않게)
- [ ] 논문 작성 (`docs/paper_outline.md` 구조 기반) — ICTC 2026 양식·분량 규정 준수
  - Introduction → Related Work → Dataset → Method → Experiments → Results → Conclusion
  - Ablation 결과 표, feature importance 그림 삽입
- [ ] **06.26 마감 전 제출 완료**: ① 리더보드 최종 제출 ② ICTC 2026 Workshop Track 논문 접수
- [ ] `docs/model_description.md` 최신화 (채택 시 재현 패키지의 기반)

### Phase 6 — 채택 이후 (07월~10월, 해당 팀만)

- [ ] 09.01 논문 채택 발표 대기
- [ ] (채택 시) 09.01까지 코드 + 모델 설명서 제출 — 재현성 검증 통과 필수
- [ ] 09.30 코드 검증 대응
- [ ] 10.15 포스터 발표·시상식 참여 (수상 시, 팀 최소 1인 필수)

---

## 3. 상시 운영 규칙 (매 작업 반복)

- **실험 관리**: 새 실험 → 가설 명시 → OOF로 검증 → 개선 시에만 LB 제출 → `experiment_log.md` 기록
- **제출 절약**: 하루 3회 제한. 로컬 OOF Log-Loss가 개선됐을 때만 제출
- **재현성**: seed=42 고정, config 변경은 Git 추적, 데이터/모델/제출물은 커밋 금지
- **기록**: 매일 작업 종료 시 `docs/daily_log.md` 갱신 (한 것 / 배운 것 / 막힌 것 / 내일 계획)
- **브랜치**: 실험은 `exp/*`, 통합은 `dev`, 제출 안정본만 `main`

---

## 4. 지금 당장 할 일 (Next Actions)

1. 대회 데이터 다운로드 후 `data/raw/`에 배치
2. `sample_submission.csv`로 **제출 형식(확률/라벨)과 클래스 수** 확인
3. **평가 산식을 Log-Loss(분류)로 전환** — metrics/train/config 수정
4. `notebooks/00_data_check.ipynb` 실행 → target/id 확정 → `configs/base.yaml` 반영
5. `exp/lgbm-baseline` 브랜치에서 Exp001(baseline) 실행 및 기록

---

## 참고

- 대회 개요: https://dacon.io/competitions/official/236690/overview/description
- 평가 방식: https://dacon.io/competitions/official/236690/overview/evaluation
- ICTC 2026: https://ictc.org/
