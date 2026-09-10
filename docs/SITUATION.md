# EY 2026 Water Quality – 현재 상황 요약

## 프로젝트
- **대회**: EY AI & Data Challenge 2026
- **타겟**: Total Alkalinity (TA), Electrical Conductance (EC), Dissolved Reactive Phosphorus (DRP)
- **데이터**: ~9k training, Landsat + TerraClimate + GEMS(enriched)

## 현재 Baseline (LB 0.2889 복원)

| 구분 | 설정 |
|------|------|
| **모델** | XGBoost (RandomizedSearchCV 튜닝) |
| **TA/EC** | GEMS 포함, GroupKFold(5) |
| **DRP** | 2단계(TA·EC 예측값 피처), residual 모드, blend OFF |
| **CV groups** | Location 기준 (Lat, Lon round 6) |
| **피처** | Landsat(swir22, NDMI, MNDWI 등), pet, seasonality, wet_index, water_stress, GEMS |

## 이슈 / 리스크

1. **Prior 과의존** – gems_DRP가 DRP 예측에 과도하게 영향 → LB보다 CV가 과대 추정 가능
2. **Spatial leakage** – Location 단위 CV라 인접 지점이 train/val에 분리되어 들어갈 수 있음
3. **모델 차이 미미** – XGB/LGB/HistGradientBoosting 비교 시 성능 거의 동일

## 구현된 것

- **Spatial Block CV** – 0.25°/0.5°/1.0° 비교 (현재 `USE_SPATIAL_BLOCK_CV = False` → baseline에서는 OFF)
- **실험 모드 Step 1/2/3** – pure/blend/catchment (현재 `EXPERIMENT_STEP = 0` → 기본 모드)
- **prior regularization** – reg_alpha/lambda 상향 → **적용 중** (MODEL_CONFIG_DRP)

## 시도해볼 전략

1. **Spatial Block CV** – `USE_SPATIAL_BLOCK_CV = True` → LB와 CV 정렬
2. **거리 가중 prior** – gems_distance_km로 prior 신뢰도 조절
3. **prior regularize 강화** – reg_alpha/lambda 추가 상향, max_depth 축소
