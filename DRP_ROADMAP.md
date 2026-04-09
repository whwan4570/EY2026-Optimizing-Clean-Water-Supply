# DRP 개선 로드맵

**현재 유지**: DRP = full_gems + 2stage(TA/EC).  
**다음 단계**: DRP target transform/loss 변경, TA/EC 보조 강도 약화 실험.  
**그 다음**: 아래 1→2→3→4 순위로 구현.

---

## 1순위: Baseline + Event Excess (regime-switch DRP)

**문제와의 정합성**: DRP·nutrient export가 event-driven이고, catchment 상태에 따라 농도-유량 반응이 달라진다는 논문 결과와 맞음. 현재 residual+2stage(TA/EC)는 TA/EC가 주는 지역 패턴까지 끌어와 unseen region에서 무너질 수 있음. “평상시 수준”과 “storm mobilization”을 분리하면 더 robust해질 가능성 큼.

**구조**:
- **Head A (baseline DRP)**: soil, elevation, landcover, seasonality, weak GEMS 등 **느리게 변하는 피처** → “평상시 DRP 수준” 예측.
- **Head B (event excess)**: 최근 3/7/14/30일 강수, API(antecedent precipitation), NDMI/MNDWI **변화량**, PET, rainfall intensity proxy → **폭우/flush로 생기는 추가 DRP**만 예측.
- **최종**:  
  `DRP = baseline + p(event) × excess` 또는 `DRP = baseline + ReLU(excess)`.

**구현 포인트**:
- baseline용 피처: soil_*, elevation_m, lc_*, month/sin_doy/cos_doy, gems_DRP(또는 weak GEMS).
- event용 피처: pr 3/7/14/30d 요약, API, NDMI/MNDWI delta, pet, intensity proxy (submission-safe 범위에서 선택).
- p(event)는 이진 분류기 또는 excess가 0이 아닐 확률 추정으로 두고, 최종은 baseline + p(event)*excess 또는 baseline + ReLU(excess).

---

## 2순위: DRP만 Region-Cluster Group DRO / Worst-Group Weighting

**이유**: 이미 region cluster holdout을 쓰고 있으므로, 그 **cluster를 group label**로 사용. DRP loss를 “평균” 대신 **가장 못 맞히는 cluster에 더 큰 가중치**를 주면, distribution shift에서 spurious feature에 덜 의존하고 환경별 invariant feature 쪽으로 유도할 수 있음 (IRM/group-robust 계열 논문).

**구현 (단순 버전)**:
- Fold 내부에서 **cluster별 DRP RMSE** 계산.
- **상위 1~2개 worst cluster** 샘플에 **1.5~3배** sample weight 부여.
- XGBoost `sample_weight` (또는 동일 가중 합 loss)로 재학습.
- 복잡한 DRO 목적함수 없이 “worst cluster 가중”만으로도 아이디어 반영 가능.

---

## 3순위: DRP만 Sequence (30–90일 daily) + Static, LSTM/TCN

**이유**: 이미 daily weather가 있으면, DRP는 **시퀀스**가 event timing·antecedent condition을 학습하는 쪽이 설득력 있음. ungauged-basin 쪽에서 LSTM이 regionalization보다 지역 일반화가 강한 사례 많음. 현재 rain_sum, lag, cum_anom 같은 **요약 통계**는 event spike 정보 손실이 클 수 있음.

**구조**:
- **입력**: 최근 **60일** (또는 30/90일) sequence: precipitation, temp, pet, soil moisture proxy, NDMI/MNDWI 등 (가능한 일별 시퀀스).
- **정적 입력**: soil, elevation, lc_*, weak GEMS.
- **출력**: log1p(DRP).
- **모델**: DRP 전용 **small LSTM 또는 TCN** (TA/EC는 기존 유지).

TA/EC까지 바꾸는 것이 아니라 **DRP head만 sequence model**로 두는 형태.

---

## 4순위: DRP Loss 변경

**이유**: DRP가 양수·spike가 크면, plain squared error보다 **log1p target + Huber**, 또는 **Tweedie/Gamma** 계열이 더 적합할 수 있음.

**XGBoost 옵션**:
- `reg:tweedie` (분산 거동을 power parameter로 조절, Gamma 쪽으로 가능).
- `reg:pseudohubererror`, `reg:quantileerror` 등.

**참고**: 예전 로그에서 Tweedie가 안 좋았던 것은 **당시 피처/모드 조합** 때문일 수 있음. “event-gated 모델”이나 “1-stage DRP”와 **결합**했을 때는 다시 시도할 가치 있음. 1~3순위보다는 후순위.

---

## 적용 순서 요약

| 순위 | 내용 | 비고 | 확인 방법 |
|------|------|------|-----------|
| 유지 | DRP = full_gems + 2stage, compact+ 18~22개 | USE_TA_EC_FOR_DRP=True, DRP_COMPACT_PLUS_CANDIDATES | 아래 “확인 방법” 참고 |
| 전제 | DRP target transform/loss, TA/EC 보조 강도 약화 | 실험 후 1순위로 | |
| 1 | Baseline + event excess (Head A/B, regime-switch) | 논문·도메인과 가장 잘 맞음 | |
| 2 | DRP만 cluster별 Group DRO (worst-group 1.5~3배 가중) | 구현 부담 적음 | |
| 3 | DRP만 60일 sequence + static, LSTM/TCN | 일별 데이터 전제 | |
| 4 | DRP loss: Tweedie/Huber 등 | 1~3과 결합 시 재시도 | |

---

## 확인 방법 (한눈에 보는 표)

| 항목 | 실행/확인 내용 | 결과 파일·기준 | 적용됨 조건 |
|------|----------------|----------------|-------------|
| **유지** (full_gems + 2stage, compact+) | `python run_benchmark_notebook.py` → 로그에 `residual+2stage(TA/EC)`, `drp_compact_plus` 18~22개 | `drp_variant_results.csv`에 full_gems·drp_compact_plus 행, `chosen` 확인 / `results_summary.csv` DRP R2 | 두 CSV에 위 내용 있음 |
| **전제** (transform/loss/TA·EC 약화) | loss·TA/EC 가중치 등 여러 설정 실험 | `drp_variant_results.csv` 또는 동일 형식으로 R2_Test·RMSE 비교 | 실험한 설정 + 선택 설정이 표/로그에 기록됨 |
| **1순위** (Baseline + event excess) | Head A/B 구현 후 동일 CV 실행 | `drp_variant_results.csv`에 regime_switch 또는 event-excess variant 추가 → full_gems 대비 R2_Test(·GEMS-weak) 비교 | event-excess variant 행 + 수치 비교 있음 |
| **2순위** (Group DRO) | cluster별 RMSE → worst 1~2 cluster 1.5~3배 가중 후 재학습 | `drp_variant_results.csv` 또는 별도 CSV에 `dro` 등 variant명 + R2_Test | DRO 버전 R2_Test(·worst-cluster) 비교 표/로그 있음 |
| **3순위** (Sequence LSTM/TCN) | 60일 시퀀스+정적으로 DRP LSTM/TCN 학습 후 동일 holdout 평가 | `drp_variant_results.csv`에 `drp_lstm` 등 추가 또는 `drp_sequence_results.csv`에 R2_Test | sequence 모델 R2_Test가 다른 variant와 비교 가능 |
| **4순위** (loss: Tweedie/Huber) | XGBoost reg:tweedie 등으로 DRP 학습 | 동일 피처·CV로 기본 MSE vs Tweedie/Huber R2_Test를 표 또는 `drp_variant_results.csv`에 기록 | loss별 R2_Test 비교가 1회 이상 표/로그에 있음 |

**공통**: 새 variant마다 `drp_variant_results.csv`에 **variant, R2_Test, n_features**를 넣어 두면 위 확인을 한 표로 관리할 수 있음.
