# EY 2026 Optimizing Clean Water Supply — 실행 순서

아래 순서대로 스크립트를 실행하면 됩니다. **필수** 단계만 해도 벤치마크는 돌리 수 있고, **선택** 단계는 해당 데이터가 있을 때만 실행하면 됩니다.

---

## 0. 사전 준비 (챌린지/제공 파일)

다음 파일이 프로젝트 폴더에 있어야 합니다.

| 파일 | 용도 |
|------|------|
| `water_quality_training_dataset.csv` | 수질 학습 타깃 (필수) |
| `landsat_features_training.csv` | Landsat 학습 피처 (필수) |
| `terraclimate_features_training.csv` | TerraClimate 학습 피처 (필수) |
| `submission_template.csv` | 제출 템플릿 (필수) |
| `landsat_features_validation.csv` | Landsat 검증 피처 (필수) |
| `terraclimate_features_validation.csv` | TerraClimate 검증 피처 (필수) |
| `samples_with_coordinates.csv` | GEMS 지점 데이터 (enriched·GEMS 사용 시 필요) |

---

## 1. GEMS 보강 (GEMS 피처 쓰는 경우)

**파일:** `create_enriched_dataset.py`

**입력:**  
`water_quality_training_dataset.csv`, `samples_with_coordinates.csv`, `submission_template.csv`

**출력:**  
`water_quality_training_dataset_enriched.csv`, `gems_features_validation.csv`

**실행:**
```bash
python create_enriched_dataset.py
```

- GEMS를 쓰지 않으면 이 단계를 건너뛰고, 2단계에서 `water_quality_training_dataset.csv`만 사용하면 됨.
- GEMS를 쓰려면 **반드시 1번을 먼저** 실행한 뒤, 4번 벤치마크에서 enriched CSV를 쓰도록 설정.

---

## 2. 강수(pr) 데이터 (선택)

`precipitation_training.csv`가 있으면 벤치마크에서 pr(및 선택 시 era5_rh, era5_sm)를 merge 합니다.  
**2a**와 **2b** 중 하나만 쓰면 됩니다.

### 2a. GRIB → 강수 (PptData에 `.grib` 있을 때)

**파일:** `build_precipitation_training_from_grib.py`

**입력:**  
`water_quality_training_dataset_enriched.csv` (또는 1번 생략 시 `water_quality_training_dataset.csv`),  
`PptData/` 폴더 내 `*_YYYYMM_*.grib` 파일

**출력:**  
`precipitation_training.csv`

**실행:**
```bash
python build_precipitation_training_from_grib.py
# 또는
python build_precipitation_training_from_grib.py PptData
```

### 2b. ERA5 NetCDF → 강수/검증 (NetCDF 있을 때)

**파일:** `era5_netcdf_to_training.py`

**입력:**  
`water_quality_training_dataset_enriched.csv`(또는 `water_quality_training_dataset.csv`),  
선택 시 `water_quality_validation_dataset.csv`,  
그리고 NetCDF 파일(스크립트가 `PptData*.nc`, `era5_ecv*.nc` 등 탐색)

**출력:**  
`precipitation_training.csv`, `precipitation_validation.csv`

**실행:**
```bash
python era5_netcdf_to_training.py
# 또는 NetCDF 경로 지정
python era5_netcdf_to_training.py path/to/era5_file.nc
```

---

## 3. 강수 이상치·수문 피처 (선택)

**파일:** `build_precip_anomaly_features.py`

**입력:**  
`water_quality_training_dataset_enriched.csv`,  
`PptData/` 내 월별 **anomaly** GRIB 파일(`*_YYYYMM_*`),  
필요 시 `terraclimate_features_training.csv`(pet merge용)

**출력:**  
`water_quality_with_precip_anomaly.csv`

**실행:**
```bash
python build_precip_anomaly_features.py
```

- 벤치마크는 이 파일이 있으면 precip anomaly·수문 파생 피처를 merge 합니다.
- **1번(create_enriched_dataset)을 먼저** 실행한 뒤 3번을 돌리는 것이 안전합니다.

---

## 4. 벤치마크 학습·제출

**파일:** `benchmark_model.py`

**입력 (자동으로 같은 폴더에서 로드):**

| 파일 | 필수 여부 |
|------|-----------|
| `water_quality_training_dataset.csv` 또는 `water_quality_training_dataset_enriched.csv` | 필수 |
| `landsat_features_training.csv` | 필수 |
| `terraclimate_features_training.csv` | 필수 |
| `submission_template.csv` | 필수 |
| `landsat_features_validation.csv` | 필수 |
| `terraclimate_features_validation.csv` | 필수 |
| `precipitation_training.csv` | 선택 (있으면 pr merge) |
| `precipitation_validation.csv` | 선택 (있으면 검증에 pr 등 merge) |
| `water_quality_with_precip_anomaly.csv` | 선택 (있으면 강수 이상·수문 피처 merge) |
| `gems_features_validation.csv` | 선택 (GEMS 사용 시) |

**출력:**  
`submission.csv`, `benchmark_results_latest.csv`

**실행:**
```bash
python benchmark_model.py
```

---

## 요약: 최소/권장 순서

**최소 (GEMS·강수 없이):**  
- 0 준비 → **4. benchmark_model.py**

**GEMS 사용:**  
- 0 준비 → **1. create_enriched_dataset.py** → **4. benchmark_model.py**

**GEMS + 강수(pr):**  
- 0 준비 → **1. create_enriched_dataset.py** → **2a 또는 2b** → **4. benchmark_model.py**

**GEMS + 강수 + 강수 이상치 피처:**  
- 0 준비 → **1. create_enriched_dataset.py** → **2a 또는 2b** → **3. build_precip_anomaly_features.py** → **4. benchmark_model.py**
