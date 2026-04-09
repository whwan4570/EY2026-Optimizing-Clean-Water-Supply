# benchmark_model / benchmark_model1 vs run_benchmark_notebook 비교

## 요약

| 항목 | run_benchmark_notebook (현재) | benchmark_model1 (오리지널) | benchmark_model (확장) |
|------|------------------------------|----------------------------|-------------------------|
| **모델** | 단일 XGBoost (또는 RF) | XGB + LGB + CatBoost + HistGBM + ExtraTrees 앙상블 | 동일 + Markowitz/Regional/통합모델 옵션 |
| **공간 CV** | GroupKFold + KMeans cluster(6) holdout | GroupKFold + Spatial block(0.5°) | 동일 + per-target block deg |
| **TA/EC** | full_gems 1회 학습, 5-fold CV | fold별 TA/EC 재학습 + OOF | + Markowitz GEMS blend (w≈0.15), Regional standardization |
| **DRP** | 20a_bare 등 6 variant, residual+2stage(TA/EC), OOF TA/EC | residual + 2stage, OOF TA/EC, fold-safe top-K | + distance-weighted prior (gems_distance_km, decay_km), TA/EC uncertainty |
| **피처** | run_benchmark_experiments 기반, UNSAFE/VAL_NEAR_CONSTANT 제외 | EXTENDED_* + DROP_FEATURES | + VAL_UNSAFE_FEATURES (val 100%/90% 결측 제외) |
| **Impute** | 전역 median fillna (또는 fold 내부) | **fold별 train median**으로 val impute (누수 방지) | 동일 |
| **제출** | R2 최고 variant 1개로 DRP 제출 | GBDT 앙상블 1개 | + prior_w * gems_DRP + residual (거리 가중) |

---

## run_benchmark_notebook에 도입 시 유망한 것 (우선순위)

### 1. **TA/EC·DRP 앙상블 (XGB + LightGBM 등)**  
- **효과**: 단일 XGB 대비 분산 감소, R2 소폭 상승 가능.  
- **부담**: LightGBM/CatBoost 등 의존성 추가, 학습 시간 증가.  
- **도입**: `train_model()`을 benchmark_model1처럼 `EnsembleRegressor` 래핑 옵션으로 두고, 플래그 하나로 단일/앙상블 전환.

### 2. **DRP 거리 가중 prior (distance-weighted prior)**  
- **효과**: validation이 GEMS 약한 지역이면 `prior_w = exp(-distance_km / decay_km)`로 gems_DRP 신뢰도를 낮춤 → LB 대비 보수적 예측.  
- **부담**: `gems_distance_km` 컬럼 필요, 제출 시 공식 한 줄 추가.  
- **도입**: `run_pipeline_drp_cv`에서 `gems_distance_km` 있으면 최종 예측을 `prior_w * gems_DRP + (1 - prior_w) * residual_pred` 형태로 혼합 (또는 benchmark와 동일한 공식 적용).

### 3. **TA/EC Markowitz GEMS blend**  
- **효과**: TA/EC를 “모델 예측 + GEMS prior” 블렌드(예: 0.85*model + 0.15*gems)로 제출. GEMS가 강한 지역에서 안정화.  
- **부담**: OOF 한 번 더 계산, 가중치 계산(또는 고정 0.15) 필요.  
- **도입**: TA/EC 최종 예측만 블렌드하고, DRP는 현재처럼 2stage(TA/EC 예측값) 유지.

### 4. **Fold 내부 median impute**  
- **효과**: val에 train 분포 누수 방지 → CV가 LB에 더 가깝게.  
- **부담**: 거의 없음.  
- **도입**: `run_pipeline_drp_cv`·`run_pipeline_cv`에서 이미 fold별로 나누므로, **fold train median으로 fold test만 impute**하도록 명시 (전역 median 대신).

### 5. **Regional standardization (TA/EC)**  
- **효과**: 타깃을 `y - mean(y)`로 학습하고 제출 시 `pred + mean(y)`. 지역 평균 제거로 다른 지역 extrapolation 완화.  
- **부담**: TA/EC 파이프라인만 수정하면 됨.  
- **도입**: TA/EC 학습 타깃을 `y - y.mean()`으로 바꾸고, 제출 시 TA/EC 예측에 `y.mean()` 가산.

### 6. **Early stopping (XGB/LGB)**  
- **효과**: 과적합 완화, 학습 시간 단축.  
- **부담**: validation set 또는 fold 내부 val 비율 필요.  
- **도입**: `train_model()`에 `eval_set`·`early_stopping_rounds` 추가 (benchmark_model1과 동일).

### 7. **DRP fold-safe top-K 피처**  
- **효과**: fold마다 DRP 피처 상위 K개만 사용 → 피처 수 많을 때 과적합 완화.  
- **부담**: importance 계산·정렬·유지할 컬럼 관리 필요.  
- **도입**: notebook은 이미 20a_bare 등 **피처 수 적은 variant** 위주라 우선순위 낮음. full_gems 쓸 때만 고려.

---

## 새 방식 vs 오리지널 (한 줄 요약)

- **run_benchmark_notebook**: 단순·가벼움, **DRP 20a_bare 계열로 LB 0.336 달성**, TA/EC는 full_gems 고정.  
- **benchmark_model / benchmark_model1**: 앙상블·거리 가중 prior·Markowitz·Regional·fold-safe impute 등 **안정화·LB 정렬**에 초점.

**실전 추천**:  
- **즉시 적용해볼 만한 것**: (4) fold 내부 median impute, (2) DRP 거리 가중 prior(데이터에 `gems_distance_km` 있을 때).  
- **다음 단계**: (1) 앙상블, (3) TA/EC Markowitz blend, (5) Regional standardization.  
- **선택**: (6) Early stopping, (7) DRP top-K(피처 많아질 때).

이 파일은 세 스크립트 간 차이와 notebook 쪽 도입 후보를 정리한 참고용입니다.  
실제 코드 반영은 `run_benchmark_notebook.py`에서 위 순서대로 옵션/플래그로 넣는 것을 권장합니다.
