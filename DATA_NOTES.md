# 데이터 요약 (DRP, rainfall, residual, sample size)

## 파이프라인 점검 (DRP R² 급락 시)
- **gems_DRP 필수**: DRP는 prior(gems_DRP) + residual 예측. `water_quality_training_dataset_enriched.csv`가 없으면 prior=0 → R² 급락/음수.
- **진단**: 실행 시 로그에서 `use_gems=`, `gems_DRP in wq_data=`, `gems_DRP non-null=` 확인. combine/merge 후 `rows=` 변경되면 merge 키 중복 가능성.
- **외부 피처**: `USE_EXTERNAL_FEATURES=False` 또는 `--no-external`로 끄고 재실행해 성능 비교. external merge 시 행 수가 늘어나면 ext_df 키 중복 → 코드에서 dedupe 적용됨.

## Training sample size
- **9,319** rows (water_quality_training_dataset_enriched.csv 기준, Landsat·TerraClimate merge 후 동일 행 수)

## DRP (Dissolved Reactive Phosphorus) distribution
| 통계 | 값 |
|------|-----|
| count | 9319 |
| mean  | 43.5 |
| std   | 51.0 |
| min   | 5    |
| 25%   | 10   |
| 50%   | 20   |
| 75%   | 48   |
| 95%   | 166  |
| max   | 195  |
| **Skew** | **1.65** (우측 꼬리) |

- 0 값 없음 (최소 5).
- 우편향: 소수 고농도 샘플이 평균을 끌어올림 → log/잔차 모드가 유리할 수 있음.

## Rainfall feature
- **없음.**  
- TerraClimate 피처는 **`pet`(potential evapotranspiration)** 만 사용 중 (`terraclimate_features_training.csv`: Latitude, Longitude, Sample Date, **pet**).
- 강수(ppt/precipitation)가 원본 TerraClimate나 다른 소스에 있다면 merge 후 `BASE_FEATURES` / 파생 피처에 추가하면 DRP·EC 등에 도움될 수 있음.

## Residual spatial cluster
- **코드 상 residual의 공간 클러스터 분석/시각화는 없음.**  
- “Residual”은 **DRP 잔차 타깃** (target = DRP − gems_DRP, final = gems_DRP + pred) 의미로만 사용됨.
- 잔차가 위도/경도나 공간 블록별로 클러스터를 보이는지 확인하려면, 학습/검증 후 (y_true − y_pred)를 Lat/Lon 또는 `get_spatial_block_groups` 그룹별로 집계·시각화하는 스크립트를 따로 추가해야 함.
