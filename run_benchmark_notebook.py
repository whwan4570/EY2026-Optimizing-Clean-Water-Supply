"""
Benchmark Model Notebook 기본 버전 → 단일 파이썬 스크립트.
XGBoost로 TA/EC/DRP 학습. train/test 공간 분할(블록 홀드아웃) 적용.
실행: python run_benchmark_notebook.py
데이터: 스크립트와 같은 폴더의 data/ 디렉터리 사용.

DRP 개선 로드맵: DRP_ROADMAP.md 참고 (1순위 baseline+event excess → 2 DRO → 3 sequence → 4 loss).
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

LANDSAT_COLS = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False

try:
    from run_benchmark_experiments import load_baseline_data, get_experiment_features
    HAS_EXPERIMENT_LOADER = True
except Exception:
    HAS_EXPERIMENT_LOADER = False

# 데이터 디렉터리 (스크립트 위치 기준 data/ 폴더)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 공간 분할: 블록 단위 홀드아웃 (동일 블록은 train 또는 test에만)
SPATIAL_BLOCK_DEG = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 추가 데이터 사용 (benchmark_model.py 스타일, 파일 있으면 병합)
USE_GEMS = True          # water_quality_training_dataset_enriched.csv + gems_features_validation.csv
USE_ERA5_PRECIP = True   # precipitation_training.csv
USE_PRECIP_ANOMALY = True  # water_quality_with_precip_anomaly.csv
USE_EXTRA_FEATURES = True  # train_with_hyriv_era5_events.csv (HydroRIVERS+ERA5 등)
USE_SOILGRIDS = False     # SoilGrids API 500 오류로 비활성화 → train_with_hyriv의 soil_clay_pct, soil_organic_carbon, soil_ph 사용

# 실험 스크립트와 동일한 데이터·피처·5-fold 공간 CV 사용 (결과를 실험 수준으로 맞춤)
USE_EXPERIMENT_STYLE = True
N_FOLDS = 5

ACTUAL_LB_R2 = 0.3439  # 제출 후 갱신

KEY_COLS = ["Latitude", "Longitude", "Sample Date"]
BASE_FEATURES = ["Latitude", "Longitude", "swir22", "NDMI", "MNDWI", "pet", "month", "dayofyear", "sin_doy", "cos_doy"]
# GEMS/추가 피처: 존재하는 컬럼만 사용
GEMS_TA = ["gems_Alk_Tot", "gems_Ca_Dis", "gems_Mg_Dis", "gems_Si_Dis", "gems_pH"]
GEMS_EC = ["gems_EC", "gems_Cl_Dis", "gems_SO4_Dis", "gems_Na_Dis", "gems_Ca_Dis", "gems_Mg_Dis", "gems_Sal", "gems_pH"]
GEMS_DRP = ["wet_index", "water_stress", "gems_DRP", "gems_DRP_log", "gems_partial_P", "gems_NOxN", "gems_NH4N", "gems_pH"]  # gems_TP/gems_TP_log: validation 90.5% 결측으로 제외

# 시계열 피처: validation 결측 확인 후 결측률이 낮은 컬럼만 허용 (main()에서 설정)
TIMESERIES_MISSING_THRESHOLD = 0.95   # 결측률 < 95% 이면 허용 후보
TIMESERIES_ZERO_THRESHOLD = 0.99      # (비결측 중) 0 비율 < 99% 이면 허용 후보
ALLOWED_TIMESERIES_FEATURES = []      # main()에서 설정
# True: validation 시계열이 전부 결측이어도 train으로 채우고 시계열 피처 사용
USE_TIMESERIES_IMPUTE_FROM_TRAIN = True
# "median" = 전역 train 중앙값, "spatial" = 가까운 train K-NN 중앙값 (위치·계절 반영)
TIMESERIES_IMPUTE_MODE = "spatial"
TIMESERIES_IMPUTE_K_NEIGHBORS = 15   # spatial일 때 사용할 이웃 수
TIMESERIES_PATTERNS = [
    "rain_sum_", "rain_max_", "sm_mean_", "sm_lag_",
    "wetness_rain_sm", "dilution_proxy", "ionic_flush_proxy", "storm_cnt_",
]
# train에서도 전부 0/결측이라 정보 없음 → 시계열 허용 목록에서 제외 (노이즈 제거)
TIMESERIES_EXCLUDE_PATTERNS = [
    "sm_mean_", "sm_lag_",
    "wetness_rain_sm", "dilution_proxy", "ionic_flush_proxy", "storm_cnt_",
]

# Submission-safe: validation에서 100% 결측 또는 전부 0으로 죽는 피처 제외 (CV도 동일 세트로 해서 LB proxy 신뢰)
UNSAFE_FEATURE_PATTERNS = [
    "storm_cnt", "wetness_rain_sm", "dilution_proxy", "ionic_flush_proxy",
    "rain_sum_", "rain_max_", "sm_mean_", "sm_lag_",
    "te_tot_spatial", "te_ele_spatial", "te_dis_spatial",
    "lag_tot_prev", "lag_ele_prev", "lag_dis_prev",
    "lag_tot_days", "lag_ele_days", "lag_dis_days",
    "population_density", "log_pop_density",  # validation 100% missing → 전 target 제외
    "gems_TP",  # validation 90.5% 결측 → 모델 입력에서 제외 (gems_TP_log 등 파생도 제외하려면 패턴 유지)
]
# Validation에서 "살아 있는데 정보량 거의 없음" (거의 상수: 0/0/0 등) → 제거
VAL_NEAR_CONSTANT_PATTERNS = [
    "cumulative_anom", "precip_anom_lag", "cum_anom_3m", "cumulative_wetness",
]
# DRP 추가 제외 (TA/EC는 UNSAFE에서 이미 제외된 것만 사용)
DRP_EXTRA_DROP = []
# Static 피처: 날짜 없이 Lat/Lon만으로 merge (Sample Date 넣으면 validation에서 전부 miss 가능)
STATIC_COL_PATTERNS = ["soil_clay", "soil_organic", "soil_ph", "elevation_m", "lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_urban", "lc_bare", "lc_water"]

# CV splitting mode: "spatial" = GroupKFold cluster holdout, "random" = KFold (mimics LB GEMS-rich conditions)
CV_MODE = "spatial"  # "spatial" or "random"
USE_CLUSTER_HOLDOUT = True
N_CLUSTERS = 8

# ---------- 이스턴케이프(제출/LB 지역) 위주 학습 (남아공 Eastern Cape) ----------
# 제출 = Eastern Cape만 → train 8.6%만 EC, 91% 비EC. 단, EC만 필터 시 LB 급락(0.17대).
#   이유: train EC 고유 좌표 14개, 제출 24개인데 겹치는 좌표 0개 → EC만 쓰면 데이터 적고 새 위치에 일반화 불가.
EASTERN_CAPE_LAT = (-34.2, -30.5)   # 위도 범위
EASTERN_CAPE_LON = (22.5, 30.0)     # 경도 범위
USE_EC_FOCUS_TRAINING = "weight"    # "filter": EC만 학습( LB 0.17 급락, 비권장), "weight": EC 가중만, False: 비활성
EC_SAMPLE_WEIGHT = 2.0             # USE_EC_FOCUS_TRAINING=="weight"일 때 EC 행에 곱할 가중치 (1.0=동일)
USE_EC_REGION_FEATURE = True       # True: is_eastern_cape 피처 추가 (LB 지역 구분)
USE_EC_CENTERED_CV = True          # True: CV에서 EC를 한 fold로 묶어 EC holdout R² 직접 확인 (group 0=EC, 1..K=비EC)

# TA/EC: GEMS 피처 + 3-seed 평균. DRP: 2-stage(pred_TA/EC) + log residual + stacking + train 퍼센타일 클리핑.

# DRP 구조 플래그
USE_TA_EC_FOR_DRP = True        # False: pred_TA/EC 미사용 (1-stage DRP)
DRP_RESIDUAL_MODE = True        # False: log1p(DRP) 직접 예측 (GEMS residual 미사용)
DRP_LOG_RESIDUAL = True         # residual_mode=True 시: log1p 차이 vs 단순 차이

# DRP prior 유무 분리 시 LB 하락 → False (단일 모델 유지)
USE_DRP_PRIOR_SPLIT = False

# DRP: 단일 파이프라인 (residual + 2-stage TA/EC + stacking만 사용)
DRP_BEST_MODE = True   # True: one single pipeline; regime/DRO/prior 분기 무시

# DRP 구조 개선 플래그 (TA/EC는 건드리지 않음)
USE_DRP_HYDRO_FEATURES = True    # HydroRIVERS + spectral interaction + DRP 파생 피처 추가
USE_DRP_STACKING = True           # XGB shallow/deep + RF + LGB(옵션) → Ridge meta
USE_DRP_HIGH_WEIGHT = False
USE_DRP_REGION_CALIBRATION = False  # cluster별 bias correction
USE_DRP_OBJECTIVE_SEARCH = True   # CV에서 best config 선택 후 DRP 5-seed 평균
USE_DRP_EARLY_STOPPING = True    # Overfitting 완화: fold 내 85% fit / 15% eval로 early stop
EARLY_STOPPING_ROUNDS_DRP = 25
DRP_VAL_FRAC = 0.15
USE_DRP_CLUSTER_FEATURE = True   # cluster_id를 DRP 피처로 추가 (지역별 패턴 학습)
USE_DRP_REGIME_SWITCH = False   # True: Head A(baseline) + Head B(event excess) [DRP_BEST_MODE=True면 무시]
USE_DRP_GROUP_DRO = True        # True: worst-cluster 가중치 (DRP 병목 개선용)
DRP_DRO_WORST_WEIGHT = 1.5      # worst cluster sample weight (1.5~2.5)
DRP_DRO_N_WORST = 2             # worst cluster 개수

# DRP feature variant: drp_compact_plus_20a_bare, drp_compact_plus_20a_bare_partial_p, _month_add, _water, _shrub
DRP_FIXED_VARIANT = "drp_compact_plus_20a_bare_partial_p"
TA_EC_BLEND_DRP = 1.0           # pred_TA/EC 가중치 (0.8~1.0: TA/EC 의존 약화)
DRP_RATIO_TARGET = False        # True: target = y / max(gems_DRP, eps) instead of log residual
DRP_RATIO_EPS = 1.0             # epsilon for ratio denominator
USE_LIGHTGBM_DRP = True         # True: DRP stack에 LightGBM base 추가 (lightgbm 패키지 필요)
USE_CATBOOST_DRP = True         # True: DRP stack에 CatBoost base 추가 (catboost 패키지 필요)
DRP_GEMS_HYBRID = False         # True: split features into gems-aware vs gems-free paths
USE_DRP_SEQUENCE_PROXY = True   # True: pr/pet/NDMI/MNDWI+seasonality를 시퀀스 대리로 DRP에 추가 (rain_sum 등 val 100%결측으로 미사용)

# TA/EC 모델 개선 플래그
TA_EC_N_ESTIMATORS = 300         # 200→300 (과적합 방지 위해 regularization과 함께)
TA_EC_MAX_DEPTH = 6
TA_EC_LEARNING_RATE = 0.08      # 0.1→0.08 (더 안정적 수렴)
TA_EC_SUBSAMPLE = 0.75           # 0.8→0.75 (일반화)
TA_EC_COLSAMPLE = 0.75           # 0.8→0.75
TA_EC_REG_ALPHA = 0.1            # 0→0.1 (L1 정규화)
TA_EC_REG_LAMBDA = 1.5           # 1.0→1.5 (L2 정규화)
TA_EC_GAMMA = 0.1                # 0→0.1 (분할 최소 손실)
TA_EC_MULTI_SEED = True         # TA/EC 최종 모델 3-seed 평균 (stacking 미사용 시)
TA_EC_N_SEEDS = 5               # 3→5 (앙상블 다양성)
USE_TA_EC_STACKING = True       # True: TA/EC도 XGB×2+RF+LGB+CatBoost→Ridge stacking (DRP와 동일)
USE_LIGHTGBM_TA_EC = True       # stacking 시 LGB base 추가
USE_TA_DERIVED_FEATURES = True  # random CV +0.005 TA
EC_XGB_OVERRIDE = None  # 지정 시 LB 하락
# 전략 반영 (기본 OFF: 켜도 0.35대 개선 없음 → 실험 시에만 True)
USE_TA_EC_CLUSTER_FEATURE = False  # True: cluster_id를 TA/EC에 추가 (지질 맥락). 기본 OFF(개선 없음)
USE_EC_LOG_TARGET = False          # True: EC log1p 학습 → expm1 복구 (이상치 완화)
USE_EC_HUBER_LOSS = True          # True: EC reg:pseudohubererror (이상치 완화)
USE_R2_EARLY_STOPPING = False      # True: early stop 기준 val R²
USE_CATBOOST_TA = True             # True: TA XGB+CatBoost 평균 (stacking 미사용 시)
USE_CATBOOST_EC = True             # True: EC XGB+CatBoost 평균 (stacking 미사용 시)

# 모델+GEMS 블렌딩 자동화: gems_DRP 있을 때 pred = alpha*pred + (1-alpha)*gems. CV로 best alpha 선택
USE_GEMS_BLEND_AUTO = True       # True: CV로 GEMS 블렌딩 alpha 자동 선택
GEMS_BLEND_ALPHA_CANDIDATES = [0.7, 0.8, 0.9, 1.0]  # 1.0=모델만, 0.7=70%모델+30%GEMS
GEMS_BLEND_ALPHA = 0.9          # USE_GEMS_BLEND_AUTO=False일 때 사용 (기본 90% 모델)

# 아래 옵션 True 시 LB 하락: USE_MARKOWITZ_BLEND, USE_GEMS_CALIBRATION, USE_DRP_PRIOR_SPLIT, EC_XGB_OVERRIDE 지정
USE_FOLD_MEDIAN_IMPUTE = True
USE_DRP_DISTANCE_PRIOR = True
PRIOR_DECAY_KM = 15.0           # GEMS distance decay (10~25: 작을수록 GEMS 의존 강함)
USE_MARKOWITZ_BLEND = False
MARKOWITZ_W = 0.15
USE_GEMS_CALIBRATION = False
GEMS_CALIB_TA_RATIO = 0.5    # (USE_GEMS_CALIBRATION=True일 때만 사용)
GEMS_CALIB_EC_RATIO = 0.5
GEMS_CALIB_DRP_RATIO = 1.0
GEMS_CALIB_DIST_THRESH = 1.0
# DRP 제출 단위: LB는 mg/L (학습과 동일). 1.0 유지.
DRP_SUBMISSION_SCALE = 1.0
# 제출 전 예측값을 train 분포로 winsorize (EC/TA 극단치·DRP 과도한 0 수축 방지)
USE_TRAIN_PERCENTILE_CLIP = True
SUBMISSION_CLIP_PERCENTILE = (1, 99)   # (p_lo, p_hi): 예측을 train의 이 백분위 구간으로 클리핑
# DRP: 0~1e4 대신 train p01~p99로 클리핑해 분산 확보 (관측치 하나만 넣은 느낌 방지)
DRP_CLIP_TO_TRAIN_RANGE = True
USE_REGIONAL_STANDARDIZATION = False
USE_EARLY_STOPPING = True
EARLY_STOPPING_ROUNDS = 50

# Submission profile (환경변수 SUBMISSION_PROFILE 로 선택)
# - stable_anchor: 보수적 단일 DRP
# - stable_stack_cat: DRP stack(XGB/RF/CatBoost)
# - stable_stack_lgb: DRP stack(XGB/RF/LightGBM)
ANCHOR_PROFILE = "stable_stack_cat"
SUBMISSION_PROFILE = os.getenv("SUBMISSION_PROFILE", ANCHOR_PROFILE).strip().lower()
if SUBMISSION_PROFILE not in {"stable_anchor", "stable_stack_cat", "stable_stack_lgb"}:
    SUBMISSION_PROFILE = ANCHOR_PROFILE

if SUBMISSION_PROFILE == "stable_anchor":
    CV_MODE = "spatial"
    USE_MARKOWITZ_BLEND = False
    USE_DRP_DISTANCE_PRIOR = True
    PRIOR_DECAY_KM = 15.0
    USE_DRP_EARLY_STOPPING = False
    USE_DRP_STACKING = False
    USE_DRP_OBJECTIVE_SEARCH = True
    USE_DRP_GROUP_DRO = False
elif SUBMISSION_PROFILE == "stable_stack_cat":
    CV_MODE = "spatial"
    USE_MARKOWITZ_BLEND = False
    USE_DRP_DISTANCE_PRIOR = True
    PRIOR_DECAY_KM = 15.0
    USE_DRP_EARLY_STOPPING = False
    USE_DRP_STACKING = True
    USE_LIGHTGBM_DRP = False
    USE_CATBOOST_DRP = True
    USE_DRP_OBJECTIVE_SEARCH = True
    USE_DRP_GROUP_DRO = False
elif SUBMISSION_PROFILE == "stable_stack_lgb":
    CV_MODE = "spatial"
    USE_MARKOWITZ_BLEND = False
    USE_DRP_DISTANCE_PRIOR = True
    PRIOR_DECAY_KM = 15.0
    USE_DRP_EARLY_STOPPING = False
    USE_DRP_STACKING = True
    USE_LIGHTGBM_DRP = True
    USE_CATBOOST_DRP = False
    USE_DRP_OBJECTIVE_SEARCH = True
    USE_DRP_GROUP_DRO = False

# 토양: train_with_hyriv_era5_events.csv의 soil_clay_pct, soil_organic_carbon, soil_ph 사용 (SoilGrids API 500 오류로 비활성화)

# DRP simple: GEMS 없이 weather+soil+landcover 중심
DRP_SIMPLE_CANDIDATES = [
    "Latitude", "Longitude", "swir22", "NDMI", "MNDWI", "pet", "pr",
    "month", "dayofyear", "sin_doy", "cos_doy", "wet_index", "water_stress",
    "soil_clay_pct", "soil_organic_carbon", "soil_ph", "elevation_m",
    "lc_tree_pct", "lc_shrub_pct", "lc_grassland_pct", "lc_cropland_pct",
    "lc_urban_pct", "lc_bare_pct", "lc_water_pct",
]

# DRP compact (15개급): GEMS 핵심 + pr/pet + NDMI/MNDWI + soil + lc 일부
DRP_COMPACT_CANDIDATES = [
    "Latitude", "Longitude",
    "gems_DRP", "gems_DRP_log", "gems_NOxN", "gems_pH",
    "pr", "pet", "NDMI", "MNDWI",
    "soil_ph", "soil_organic_carbon", "soil_clay_pct", "elevation_m",
    "lc_cropland_pct", "lc_water_pct",
]

# DRP compact+ (18~22개): compact + seasonality + wetness + lc 2~3개 → 더 robust
DRP_COMPACT_PLUS_CANDIDATES = [
    "Latitude", "Longitude",
    "gems_DRP", "gems_DRP_log", "gems_NOxN", "gems_pH",
    "pr", "pet", "NDMI", "MNDWI",
    "month", "dayofyear", "sin_doy", "cos_doy", "wet_index", "water_stress",
    "soil_ph", "soil_organic_carbon", "soil_clay_pct", "elevation_m",
    "lc_cropland_pct", "lc_water_pct", "lc_tree_pct", "lc_grassland_pct",
]

# DRP 기본: 20a_bare (20a + lc_bare_pct). 나머지 variant는 20a_bare 기준 미세 변형만 비교.
DRP_COMPACT_PLUS_20A = [
    "gems_DRP", "gems_DRP_log", "gems_NOxN", "gems_NH4N", "gems_pH", "gems_Sal",
    "pr", "pet", "storm_pet_ratio", "wet_index",
    "NDMI", "MNDWI",
    "soil_ph", "soil_organic_carbon", "soil_clay_pct", "elevation_m",
    "lc_cropland_pct", "lc_urban_pct",
    "sin_doy", "cos_doy",
]
DRP_COMPACT_PLUS_20A_BARE = DRP_COMPACT_PLUS_20A + ["lc_bare_pct"]

# 20a_bare 주변 비교 (5개)
DRP_COMPACT_PLUS_20A_BARE_PARTIAL_P = DRP_COMPACT_PLUS_20A_BARE + ["gems_partial_P"]
DRP_COMPACT_PLUS_20A_BARE_MONTH_ADD = DRP_COMPACT_PLUS_20A_BARE + ["month"]
# 20a_bare - sin/cos + month
DRP_COMPACT_PLUS_20A_BARE_MONTH = [c for c in DRP_COMPACT_PLUS_20A_BARE if c not in ("sin_doy", "cos_doy")] + ["month"]
DRP_COMPACT_PLUS_20A_BARE_SHRUB = DRP_COMPACT_PLUS_20A_BARE + ["lc_shrub_pct"]
DRP_COMPACT_PLUS_20A_BARE_WATER = DRP_COMPACT_PLUS_20A_BARE + ["lc_water_pct"]

# Prior split 시 브랜치별 전용 피처 (입력 구조 차별화)
# prior = GEMS/prior 보정용, no-prior = 환경/순수 예측용
PRIOR_ONLY_FEATURES = [
    "gems_DRP", "gems_DRP_log", "gems_partial_P", "gems_NOxN", "gems_NH4N", "gems_pH",
    "gems_distance_km", "gems_within_limit", "prior_weight",
    "sin_doy", "cos_doy", "Latitude", "Longitude",
]
NO_PRIOR_FEATURES = [
    "pr", "pet", "NDMI", "MNDWI", "wet_index", "water_stress",
    "soil_ph", "soil_organic_carbon", "soil_clay_pct", "elevation_m",
    "lc_cropland_pct", "lc_water_pct", "lc_bare_pct",
    "sin_doy", "cos_doy", "Latitude", "Longitude",
]

# DRP HydroRIVERS + spectral + derived (train/val 모두 존재, 결측 0%)
DRP_HYDRO_FEATURES = [
    "hyriv_log_q", "hyriv_flow_order", "hyriv_log_upcells",
    "river_dist_m", "hyriv_dist_bin", "hyriv_q_over_up", "hyriv_order_x_up",
]
DRP_SPECTRAL_FEATURES = [
    "WRI", "NDMI_sq", "MNDWI_sq", "NDMI_MNDWI", "NDMI_anom", "MNDWI_anom",
]
DRP_CLIMATE_FEATURES = [
    "log1p_pr", "pr_pet_ratio", "log1p_pet", "pet_norm", "pet_seasonal_anom",
    "dry_index",
]
DRP_DERIVED_FEATURE_DEFS = {
    "wet_DRP": ("wet_index", "gems_DRP"),
    "pet_DRP": ("pet", "gems_DRP"),
    "water_stress_DRP": ("water_stress", "gems_DRP"),
    "gems_DRP_season": ("gems_DRP", "sin_doy"),
    "pr_DRP": ("pr", "gems_DRP"),
    "NDMI_DRP": ("NDMI", "gems_DRP"),
    "MNDWI_DRP": ("MNDWI", "gems_DRP"),
    "pr_pet_DRP": ("pr_pet_ratio", "gems_DRP"),
}

# 1순위 regime-switch: Head A(baseline) vs Head B(event excess)
DRP_BASELINE_PATTERNS = ["soil_", "elevation_m", "lc_", "month", "dayofyear", "sin_doy", "cos_doy", "gems_DRP", "gems_pH", "Latitude", "Longitude"]
DRP_EVENT_PATTERNS = ["pr", "pet", "NDMI", "MNDWI", "wet_index", "water_stress"]

# 3순위 sequence proxy: rolling 없으면 pr, pet, NDMI, MNDWI + seasonality + static (일별 시퀀스 대신)
DRP_SEQUENCE_PROXY_CANDIDATES = [
    "pr", "pet", "NDMI", "MNDWI", "wet_index", "water_stress",
    "month", "sin_doy", "cos_doy",
    "soil_ph", "soil_organic_carbon", "soil_clay_pct", "elevation_m",
    "lc_cropland_pct", "lc_water_pct", "gems_DRP", "gems_pH",
]


def check_validation_timeseries_quality(
    data_dir, missing_threshold=None, zero_threshold=None
) -> list:
    """
    validation(val_with_hyriv_era5_events)에서 시계열 관련 컬럼의 결측률·0비율을 계산하고,
    기준 이하인 컬럼만 반환. 반환된 컬럼은 ALLOWED_TIMESERIES_FEATURES로 넣어 UNSAFE에서 제외.
    """
    data_dir = Path(data_dir)
    val_path = data_dir / "val_with_hyriv_era5_events.csv"
    if not val_path.exists():
        return []
    missing_threshold = missing_threshold if missing_threshold is not None else TIMESERIES_MISSING_THRESHOLD
    zero_threshold = zero_threshold if zero_threshold is not None else TIMESERIES_ZERO_THRESHOLD

    patterns = [
        "rain_sum_", "rain_max_", "sm_mean_", "sm_lag_",
        "wetness_rain_sm", "dilution_proxy", "ionic_flush_proxy", "storm_cnt_",
    ]
    val = pd.read_csv(val_path, nrows=10000)
    candidates = [c for c in val.columns if any(p in c for p in patterns)]
    if not candidates:
        return []

    results = []
    allowed = []
    for c in candidates:
        s = val[c]
        missing = s.isna().mean()
        non_miss = s.dropna()
        zero_frac = (non_miss == 0).mean() if len(non_miss) else 1.0
        ok_miss = missing < missing_threshold
        ok_zero = zero_frac < zero_threshold
        pass_ = ok_miss and ok_zero
        results.append((c, missing, zero_frac, pass_))
        if pass_:
            allowed.append(c)

    print("\n  --- Validation 시계열 피처 품질 (결측 확인 후 허용 대상) ---")
    print(f"  기준: 결측률 < {missing_threshold:.0%}, (비결측 중) 0 비율 < {zero_threshold:.0%}")
    for c, miss, zero, pass_ in sorted(results, key=lambda x: (not x[3], x[1])):
        status = "OK → 허용" if pass_ else "제외"
        print(f"    {c}: 결측={miss:.1%}  0비율={zero:.1%}  {status}")
    print(f"  → 허용 컬럼 수: {len(allowed)} (0이면 시계열 피처는 계속 UNSAFE로 제외)")
    print("  -------------------------------------------------------")
    return allowed


def impute_val_timeseries_from_train(val_data: pd.DataFrame, wq_data: pd.DataFrame, cols: list, k: int = 15) -> pd.DataFrame:
    """
    Validation 시계열 결측을 train의 공간 K-NN 중앙값으로 채움.
    (Lat, Lon) 기준으로 가까운 train k개 행의 해당 컬럼 중앙값 사용.
    """
    if not cols or not HAS_SCIPY:
        return val_data
    need = ["Latitude", "Longitude"]
    if not all(c in val_data.columns for c in need) or not all(c in wq_data.columns for c in need):
        return val_data
    cols = [c for c in cols if c in wq_data.columns and c in val_data.columns and pd.api.types.is_numeric_dtype(wq_data[c])]
    if not cols:
        return val_data

    def build_pts(df):
        lat = df["Latitude"].values.astype(np.float64)
        lon = df["Longitude"].values.astype(np.float64)
        phi, lam = np.radians(lat), np.radians(lon)
        return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

    train_pts = build_pts(wq_data)
    val_pts = build_pts(val_data)
    tree = cKDTree(train_pts)
    k_use = min(k, len(wq_data))
    dists, idx = tree.query(val_pts, k=k_use)

    result = val_data.copy()
    for c in cols:
        train_col = wq_data[c].values.astype(np.float64)
        vals = np.zeros(len(val_data), dtype=np.float64)
        for i in range(len(val_data)):
            neighbors = train_col[idx[i]]
            neighbors = neighbors[~np.isnan(neighbors)]
            vals[i] = np.median(neighbors) if len(neighbors) > 0 else np.nanmedian(train_col)
        result[c] = vals
    return result


def is_unsafe_feature(col: str, for_drp: bool = False) -> bool:
    if col in ALLOWED_TIMESERIES_FEATURES:
        return False
    if any(p in col for p in UNSAFE_FEATURE_PATTERNS):
        return True
    if any(p in col for p in VAL_NEAR_CONSTANT_PATTERNS):
        return True
    if for_drp and col in DRP_EXTRA_DROP:
        return True
    return False


def get_drp_feature_variants(wq_data: pd.DataFrame, target_cols: list):
    """
    DRP: 20a_bare 기본 + 20a_bare 주변 5개만 비교 (submission-safe 적용).
    """
    def _filter(cols):
        return [c for c in cols if c in wq_data.columns and c not in target_cols and not is_unsafe_feature(c, for_drp=True)]

    bare = _filter(DRP_COMPACT_PLUS_20A_BARE)
    bare_partial_p = _filter(DRP_COMPACT_PLUS_20A_BARE_PARTIAL_P)
    bare_month_add = _filter(DRP_COMPACT_PLUS_20A_BARE_MONTH_ADD)
    bare_month = _filter(DRP_COMPACT_PLUS_20A_BARE_MONTH)
    bare_shrub = _filter(DRP_COMPACT_PLUS_20A_BARE_SHRUB)
    bare_water = _filter(DRP_COMPACT_PLUS_20A_BARE_WATER)
    return {
        "drp_compact_plus_20a_bare": bare if bare else None,
        "drp_compact_plus_20a_bare_partial_p": bare_partial_p if bare_partial_p else None,
        "drp_compact_plus_20a_bare_month_add": bare_month_add if bare_month_add else None,
        "drp_compact_plus_20a_bare_month": bare_month if bare_month else None,
        "drp_compact_plus_20a_bare_shrub": bare_shrub if bare_shrub else None,
        "drp_compact_plus_20a_bare_water": bare_water if bare_water else None,
    }


def add_seasonality_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:
    """month, dayofyear, sin_doy, cos_doy 추가."""
    result = df.copy()
    if date_col not in result.columns:
        return result
    dates = pd.to_datetime(result[date_col], format="mixed", dayfirst=True, errors="coerce")
    result["month"] = dates.dt.month
    result["dayofyear"] = dates.dt.dayofyear
    result["sin_doy"] = np.sin(2 * np.pi * result["dayofyear"] / 365.25)
    result["cos_doy"] = np.cos(2 * np.pi * result["dayofyear"] / 365.25)
    return result


def add_wetness_features(df: pd.DataFrame) -> pd.DataFrame:
    """wet_index, water_stress 추가 (NDMI, pet 필요)."""
    result = df.copy()
    if "NDMI" in result.columns and "pet" in result.columns:
        result["wet_index"] = result["NDMI"] * result["pet"]
        result["water_stress"] = result["NDMI"] / (result["pet"] + 1e-6)
    return result


def add_ta_ec_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """TA/EC interaction features. Only added to TA feat list (EC unchanged)."""
    result = df.copy()
    def _mul(a, b, name):
        if a in result.columns and b in result.columns:
            result[name] = result[a].fillna(0) * result[b].fillna(0)
    def _ratio(a, b, name, eps=1e-6):
        if a in result.columns and b in result.columns:
            result[name] = result[a].fillna(0) / (result[b].fillna(0).abs() + eps)
    def _log1p_feat(a, name):
        if a in result.columns:
            result[name] = np.log1p(result[a].fillna(0).clip(lower=0))

    _mul("gems_Alk_Tot", "gems_pH", "alk_ph_interaction")
    _mul("gems_Ca_Dis", "gems_pH", "ca_ph_interaction")
    _mul("gems_EC", "gems_pH", "ec_ph_interaction")
    _ratio("gems_Alk_Tot", "gems_EC", "alk_ec_ratio")
    _mul("gems_Alk_Tot", "NDMI", "alk_NDMI")
    _mul("gems_EC", "NDMI", "ec_NDMI")
    _mul("gems_Ca_Dis", "gems_Mg_Dis", "ca_mg_product")
    _ratio("gems_Na_Dis", "gems_EC", "na_fraction_ec")
    _mul("soil_ph", "gems_pH", "soil_gems_ph")
    _mul("elevation_m", "NDMI", "elev_ndmi")
    _mul("pr", "NDMI", "pr_ndmi")
    if "gems_Alk_Tot" in result.columns and "sin_doy" in result.columns:
        result["alk_seasonal"] = result["gems_Alk_Tot"].fillna(0) * result["sin_doy"].fillna(0)
    if "gems_EC" in result.columns and "sin_doy" in result.columns:
        result["ec_seasonal"] = result["gems_EC"].fillna(0) * result["sin_doy"].fillna(0)
    _log1p_feat("gems_Alk_Tot", "log1p_alk")
    _log1p_feat("gems_EC", "log1p_ec")
    if "Latitude" in result.columns:
        result["lat_abs"] = result["Latitude"].abs()
    _mul("gems_Cl_Dis", "gems_Na_Dis", "cl_na_product")
    _mul("gems_SO4_Dis", "gems_Ca_Dis", "so4_ca_product")
    _ratio("gems_Cl_Dis", "gems_SO4_Dis", "cl_so4_ratio")
    if "gems_EC" in result.columns and "cos_doy" in result.columns:
        result["ec_cos_seasonal"] = result["gems_EC"].fillna(0) * result["cos_doy"].fillna(0)
    _mul("gems_EC", "pet", "ec_pet")
    _mul("gems_Alk_Tot", "pet", "alk_pet")
    # 건조 지수: EC는 증발 많을 때 농축 → pet/(pr+eps). pr≈0이면 폭발하므로 상한 클리핑 후 log1p로 train/val 스케일 통일
    if "pet" in result.columns and "pr" in result.columns:
        raw = result["pet"].fillna(0) / (result["pr"].fillna(0).abs() + 1e-6)
        result["dry_index"] = np.log1p(np.clip(raw, 0.0, 1e8))
    return result


TA_DERIVED_FEATURES = [
    "alk_ph_interaction", "ca_ph_interaction", "ec_ph_interaction",
    "alk_ec_ratio", "alk_NDMI", "ec_NDMI", "ca_mg_product",
    "na_fraction_ec", "soil_gems_ph", "elev_ndmi", "pr_ndmi",
    "alk_seasonal", "ec_seasonal", "log1p_alk", "log1p_ec", "lat_abs",
    "cl_na_product", "so4_ca_product", "cl_so4_ratio",
    "ec_cos_seasonal", "ec_pet", "alk_pet",
]

EC_DERIVED_FEATURES = [
    "ec_ph_interaction", "ec_NDMI", "ec_seasonal", "ec_cos_seasonal",
    "log1p_ec", "na_fraction_ec", "ca_mg_product",
    "cl_na_product", "so4_ca_product", "cl_so4_ratio",
    "soil_gems_ph", "elev_ndmi", "pr_ndmi", "lat_abs", "ec_pet",
]


def add_drp_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """DRP 전용 파생 피처: val에만 있는 것을 train에서도 재현."""
    result = df.copy()
    for feat_name, (col_a, col_b) in DRP_DERIVED_FEATURE_DEFS.items():
        if feat_name not in result.columns and col_a in result.columns and col_b in result.columns:
            a = result[col_a].fillna(0).values.astype(np.float64)
            b = result[col_b].fillna(0).values.astype(np.float64)
            result[feat_name] = a * b
    if "DRP_prior_center" not in result.columns and "gems_DRP" in result.columns:
        g = result["gems_DRP"].fillna(0).values.astype(np.float64)
        if "Latitude" in result.columns and "Longitude" in result.columns:
            coords = result[["Latitude", "Longitude"]].round(1).astype(str).agg("_".join, axis=1)
            group_med = result.groupby(coords)["gems_DRP"].transform("median")
            result["DRP_prior_center"] = group_med.fillna(0)
            group_std = result.groupby(coords)["gems_DRP"].transform("std")
            result["DRP_prior_spread"] = group_std.fillna(0)
        else:
            result["DRP_prior_center"] = g
            result["DRP_prior_spread"] = 0.0
    if "drp_extreme" not in result.columns and "gems_DRP" in result.columns:
        g = result["gems_DRP"].fillna(0).values.astype(np.float64)
        result["drp_extreme"] = (g > np.percentile(g[g > 0], 90) if (g > 0).sum() > 10 else g > 0).astype(np.float64)
    return result


def get_drp_expanded_features(feat_base: list, df: pd.DataFrame) -> list:
    """기존 DRP 피처에 HydroRIVERS + spectral + climate + derived [+ sequence proxy] 추가 (존재하는 것만)."""
    if not USE_DRP_HYDRO_FEATURES:
        return feat_base
    extra = DRP_HYDRO_FEATURES + DRP_SPECTRAL_FEATURES + DRP_CLIMATE_FEATURES
    extra += list(DRP_DERIVED_FEATURE_DEFS.keys()) + ["DRP_prior_center", "DRP_prior_spread", "drp_extreme"]
    if USE_DRP_SEQUENCE_PROXY:
        extra += [c for c in DRP_SEQUENCE_PROXY_CANDIDATES if c not in extra]
    extra_exist = [c for c in extra if c in df.columns and c not in feat_base and not is_unsafe_feature(c, True)]
    return feat_base + extra_exist


def impute_landsat_spatial(val_df: pd.DataFrame, train_df: pd.DataFrame, landsat_cols: list = None):
    """
    Landsat 결측을 공간 기반 보간: 동일 좌표 median → 근처 val → 근처 train → fallback median.
    scipy 없으면 fillna(median)만 수행.
    """
    if landsat_cols is None:
        landsat_cols = [c for c in LANDSAT_COLS if c in val_df.columns]
    result = val_df.copy()
    for col in landsat_cols:
        if col not in result.columns:
            continue
        missing_mask = result[col].isna()
        if not missing_mask.any():
            continue
        fallback = result[col].median()
        if pd.isna(fallback) and col in train_df.columns:
            fallback = train_df[col].median()
        if pd.isna(fallback):
            fallback = 0
        if HAS_SCIPY:
            for idx in np.where(missing_mask)[0]:
                lat, lon = result.loc[idx, "Latitude"], result.loc[idx, "Longitude"]
                same_loc = val_df[(val_df["Latitude"] == lat) & (val_df["Longitude"] == lon)]
                valid_vals = same_loc[col].dropna()
                if len(valid_vals) > 0:
                    result.loc[idx, col] = valid_vals.median()
                    missing_mask = result[col].isna()
            still_missing = result[col].isna()
            if still_missing.any():
                has_val = val_df[~val_df[col].isna()]
                has_train = train_df[~train_df[col].isna()] if col in train_df.columns else pd.DataFrame()

                def build_pts(df):
                    lat = df["Latitude"].values
                    lon = df["Longitude"].values
                    phi, lam = np.radians(lat), np.radians(lon)
                    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

                tree_val = cKDTree(build_pts(has_val)) if len(has_val) > 0 else None
                tree_train = cKDTree(build_pts(has_train)) if len(has_train) > 0 else None
                miss_idx = np.where(still_missing)[0]
                pts_miss = build_pts(result.loc[miss_idx])
                if tree_val is not None:
                    d, i = tree_val.query(pts_miss, k=1)
                    for j, orig in enumerate(miss_idx):
                        if d.flat[j] < 0.015:
                            result.loc[orig, col] = has_val[col].iloc[i.flat[j]]
                still_missing = result[col].isna()
                if still_missing.any() and tree_train is not None:
                    miss_idx2 = np.where(still_missing)[0]
                    pts2 = build_pts(result.loc[miss_idx2])
                    _, i2 = tree_train.query(pts2, k=1)
                    for j, orig in enumerate(miss_idx2):
                        result.loc[orig, col] = has_train[col].iloc[i2.flat[j]]
        result[col] = result[col].fillna(fallback)
    return result


def get_spatial_block_groups(df: pd.DataFrame, block_deg: float) -> np.ndarray:
    """좌표를 block_deg grid로 묶어 블록 ID. 블록 단위 분할 시 LB(다른 지역)에 가깝게."""
    lat = df["Latitude"].values
    lon = df["Longitude"].values
    lat_min, lon_min = lat.min(), lon.min()
    block = (
        np.floor((lat - lat_min) / block_deg).astype(int) * 10_000
        + np.floor((lon - lon_min) / block_deg).astype(int)
    )
    groups, _ = pd.factorize(block)
    return groups


def get_gems_quality_split(df: pd.DataFrame, min_test: int = 20):
    """GEMS 품질 기반 split: train=within_limit True, val=within_limit False (또는 distance 긴 쪽)."""
    if "gems_within_limit" in df.columns:
        within = df["gems_within_limit"].fillna(False)
        train_mask = (within == True).values
        test_mask = ~train_mask
        if train_mask.sum() < min_test or test_mask.sum() < min_test:
            return None
        return np.where(train_mask)[0], np.where(test_mask)[0]
    if "gems_distance_km" in df.columns:
        dist = df["gems_distance_km"].fillna(np.nanmedian(df["gems_distance_km"]))
        thresh = np.nanmedian(dist)
        train_mask = (dist <= thresh).values
        test_mask = ~train_mask
        if train_mask.sum() < min_test or test_mask.sum() < min_test:
            return None
        return np.where(train_mask)[0], np.where(test_mask)[0]
    return None


def get_has_prior(df: pd.DataFrame) -> np.ndarray:
    """Prior 유무: gems_within_limit True 이거나, gems_DRP 유효(비결측·>0). has_prior=1 → residual 모델, 0 → 순수 예측."""
    if "gems_within_limit" in df.columns:
        return np.asarray(df["gems_within_limit"].fillna(False).astype(bool))
    if "gems_DRP" in df.columns:
        g = df["gems_DRP"].values
        return (~pd.isna(g)) & (np.asarray(g, dtype=float) > 0)
    return np.zeros(len(df), dtype=bool)


def get_cluster_groups(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42, return_model: bool = False):
    """Lat/Lon KMeans로 region cluster 생성. cluster 하나 통째로 val → 같은 국가/유역/기후대 패턴 단위 holdout."""
    lat = df["Latitude"].values.reshape(-1, 1)
    lon = df["Longitude"].values.reshape(-1, 1)
    X = np.hstack([lat, lon])
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    if return_model:
        return labels, km
    return labels


def is_eastern_cape_mask(df: pd.DataFrame) -> np.ndarray:
    """Lat/Lon이 Eastern Cape 범위 안이면 True."""
    lat = np.asarray(df["Latitude"], dtype=np.float64)
    lon = np.asarray(df["Longitude"], dtype=np.float64)
    return (
        (lat >= EASTERN_CAPE_LAT[0]) & (lat <= EASTERN_CAPE_LAT[1])
        & (lon >= EASTERN_CAPE_LON[0]) & (lon <= EASTERN_CAPE_LON[1])
    )


def add_ec_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_eastern_cape (0/1) 컬럼 추가. 제출 지역(EC) 구분용."""
    out = df.copy()
    mask = is_eastern_cape_mask(out)
    out["is_eastern_cape"] = np.where(mask, 1.0, 0.0).astype(np.float32)
    return out


def get_cluster_groups_ec_centered(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42):
    """EC를 group 0으로 고정, 비EC는 KMeans로 n_clusters개 그룹(1..n_clusters). → EC 전용 holdout fold 확보.
    전부 EC면(필터 후) KMeans로 EC 내부만 n_clusters 분할."""
    ec = is_eastern_cape_mask(df)
    groups = np.zeros(len(df), dtype=np.int32)
    if ec.sum() == 0:
        return get_cluster_groups(df, n_clusters=n_clusters, random_state=random_state)
    non_ec_ix = np.where(~ec)[0]
    if len(non_ec_ix) < n_clusters:
        if ec.sum() == len(df):
            k = min(n_clusters, max(2, len(df) // 20))
            return get_cluster_groups(df, n_clusters=k, random_state=random_state)
        groups[ec] = 0
        groups[~ec] = 1
        return groups
    lat = df["Latitude"].values[non_ec_ix].reshape(-1, 1)
    lon = df["Longitude"].values[non_ec_ix].reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    non_ec_labels = km.fit_predict(np.hstack([lat, lon]))
    groups[ec] = 0
    groups[non_ec_ix] = non_ec_labels + 1
    return groups


def combine_two_datasets(dataset1, dataset2, dataset3):
    """세 데이터셋을 컬럼 기준으로 결합 (중복 컬럼 제거)."""
    data = pd.concat([dataset1, dataset2, dataset3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


class StackedDRPModel:
    """DRP stacking: XGB(shallow) + XGB(deep) + RF [+ LGB] [+ CatBoost] → Ridge meta."""
    def __init__(self, models, meta, scalers):
        self.models = models
        self.meta = meta
        self.scalers = scalers
        if models and hasattr(models[0], "feature_importances_"):
            self.feature_importances_ = models[0].feature_importances_

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        base_preds = np.column_stack([m.predict(X) for m in self.models])
        return self.meta.predict(base_preds)

    @staticmethod
    def train_stacked(X_tr, y_tr, X_groups, n_splits=5, use_lgb=False, use_catboost=False, xgb_kw_override=None):
        """fold-safe OOF로 meta 학습. use_lgb/use_catboost: LGB/CatBoost base 추가.
        xgb_kw_override: DRP config에서 전달된 objective/n_estimators 등 → XGB base에 적용."""
        if CV_MODE == "random":
            _stk_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            _stk_splitter = GroupKFold(n_splits=n_splits)
        n = len(y_tr)
        # override: objective, reg 등만 적용. n_estimators/max_depth는 shallow/deep 다양성 유지
        override = xgb_kw_override or {}
        override = {k: v for k, v in override.items() if k in ("objective", "reg_alpha", "reg_lambda", "gamma", "subsample", "colsample_bytree", "learning_rate", "tweedie_variance_power")}
        configs = [
            dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, random_state=42, verbosity=0),
            dict(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, random_state=43, verbosity=0),
        ]
        for c in configs:
            c.update(override)
        n_base = len(configs) + 1
        lgb_ok = False
        if use_lgb:
            try:
                import lightgbm as lgb
                lgb_ok = True
                n_base += 1
            except ImportError:
                pass
        catboost_ok = False
        if use_catboost:
            try:
                import catboost as cb
                catboost_ok = True
                n_base += 1
            except ImportError:
                pass
        oof = np.zeros((n, n_base), dtype=np.float64)
        _stk_args = (X_tr, y_tr, X_groups) if CV_MODE == "spatial" else (X_tr, y_tr)
        cboff = len(configs) + (1 if lgb_ok else 0)
        for fold, (tr_ix, te_ix) in enumerate(_stk_splitter.split(*_stk_args)):
            for j, kw in enumerate(configs):
                m = xgb.XGBRegressor(**kw)
                m.fit(X_tr[tr_ix], y_tr[tr_ix])
                oof[te_ix, j] = m.predict(X_tr[te_ix])
            rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=44, n_jobs=-1)
            rf.fit(X_tr[tr_ix], y_tr[tr_ix])
            oof[te_ix, len(configs)] = rf.predict(X_tr[te_ix])
            if lgb_ok:
                lgb_m = lgb.LGBMRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.75,
                                           colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.0, random_state=45, verbose=-1)
                lgb_m.fit(X_tr[tr_ix], y_tr[tr_ix])
                oof[te_ix, len(configs) + 1] = lgb_m.predict(X_tr[te_ix])
            if catboost_ok:
                cb_m = cb.CatBoostRegressor(iterations=250, depth=6, learning_rate=0.06, random_state=46, verbose=0)
                cb_m.fit(X_tr[tr_ix], y_tr[tr_ix])
                oof[te_ix, cboff] = cb_m.predict(X_tr[te_ix])
        meta = Ridge(alpha=1.0)
        meta.fit(oof, y_tr)
        models = []
        for kw in configs:
            m = xgb.XGBRegressor(**kw)
            m.fit(X_tr, y_tr)
            models.append(m)
        rf_full = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=44, n_jobs=-1)
        rf_full.fit(X_tr, y_tr)
        models.append(rf_full)
        if lgb_ok:
            lgb_full = lgb.LGBMRegressor(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.75,
                                          colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.0, random_state=45, verbose=-1)
            lgb_full.fit(X_tr, y_tr)
            models.append(lgb_full)
        if catboost_ok:
            cb_full = cb.CatBoostRegressor(iterations=250, depth=6, learning_rate=0.06, random_state=46, verbose=0)
            cb_full.fit(X_tr, y_tr)
            models.append(cb_full)
        return StackedDRPModel(models, meta, None)


def compute_cluster_bias(y_true, y_pred, groups):
    """cluster별 평균 오차(bias) 계산 → dict {cluster_id: mean_error}."""
    bias = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 0:
            bias[g] = float(np.mean(y_true[mask] - y_pred[mask]))
    return bias


def apply_cluster_bias(pred, groups, bias_dict):
    """validation 예측에 cluster bias 보정 적용."""
    corrected = pred.copy()
    for g in np.unique(groups):
        mask = groups == g
        if g in bias_dict:
            corrected[mask] += bias_dict[g]
    return corrected


def split_data(X, y, groups=None, test_size=0.2, random_state=42):
    """groups가 있으면 공간 분할(GroupShuffleSplit), 없으면 랜덤 분할."""
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train, eval_set=None, early_stopping_rounds=0):
    """eval_set=None이면 early stopping 미사용. 단일 XGB (또는 RF fallback)."""
    X_train_scaled = np.nan_to_num(np.asarray(X_train_scaled, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    use_es = USE_EARLY_STOPPING and early_stopping_rounds > 0 and eval_set is not None
    es_rounds = EARLY_STOPPING_ROUNDS if use_es else 0

    if USE_XGB:
        kwargs = dict(
            n_estimators=TA_EC_N_ESTIMATORS, max_depth=TA_EC_MAX_DEPTH,
            learning_rate=TA_EC_LEARNING_RATE,
            subsample=TA_EC_SUBSAMPLE, colsample_bytree=TA_EC_COLSAMPLE,
            reg_alpha=TA_EC_REG_ALPHA, reg_lambda=TA_EC_REG_LAMBDA,
            gamma=TA_EC_GAMMA,
            random_state=42, verbosity=0,
        )
        if use_es:
            kwargs["early_stopping_rounds"] = es_rounds
        model = xgb.XGBRegressor(**kwargs)
        if use_es:
            model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train_scaled, y_train)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
    return model


def train_model_drp(X_train_scaled, y_train, sample_weight=None, objective="reg:squarederror"):
    """DRP 전용: sample_weight(DRO), objective(Tweedie/Huber) 지원. 단일 XGB."""
    X_train_scaled = np.nan_to_num(np.asarray(X_train_scaled, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if USE_XGB:
        kwargs = dict(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        if objective != "reg:squarederror":
            kwargs["objective"] = objective
            if objective == "reg:tweedie":
                kwargs["tweedie_variance_power"] = 1.5
        model = xgb.XGBRegressor(**kwargs)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(model, X_scaled, y_true, dataset_name="Test"):
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return y_pred, r2, rmse


def run_pipeline(X, y, param_name="Parameter", groups=None):
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = split_data(X, y, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_train_scaled, y_train, "Train")
    evaluate_model(model, X_test_scaled, y_test, "Test")

    results = {
        "Parameter": param_name,
        "R2_Train": r2_score(y_train, model.predict(X_train_scaled)),
        "RMSE_Train": np.sqrt(mean_squared_error(y_train, model.predict(X_train_scaled))),
        "R2_Test": r2_score(y_test, model.predict(X_test_scaled)),
        "RMSE_Test": np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled))),
    }
    return model, scaler, pd.DataFrame([results])


def run_pipeline_cv(X, y, param_name="Parameter", groups=None, n_splits=5, xgb_override=None, log_target=False, sample_weight=None):
    """CV_MODE-aware pipeline: spatial GroupKFold or random KFold, fold별 R²/RMSE 평균 보고.
    xgb_override: dict of XGB params to override global TA_EC_* settings for this target.
    log_target: if True, fit on log1p(y), report R2/rmse on expm1(pred) vs y, final model predicts in log space (제출 시 expm1 필요).
    sample_weight: optional per-sample weight (e.g. EC 위주 학습)."""
    if groups is None:
        groups = np.zeros(len(X), dtype=int)
    if CV_MODE == "random":
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv_splitter = GroupKFold(n_splits=n_splits)
    r2_train_list, rmse_train_list = [], []
    r2_test_list, rmse_test_list = [], []
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    # EC 로그 타겟: 학습은 log1p(y), 평가/제출은 원단위
    y_orig = np.asarray(y_arr, dtype=np.float64)
    if log_target:
        y_arr = np.log1p(np.maximum(y_orig, 0))
    _ov = xgb_override or {}
    if param_name == "Electrical Conductance" and USE_EC_HUBER_LOSS and "objective" not in _ov:
        _ov = dict(_ov, objective="reg:pseudohubererror")
    _n_est = _ov.get("n_estimators", TA_EC_N_ESTIMATORS)
    _md = _ov.get("max_depth", TA_EC_MAX_DEPTH)
    _lr = _ov.get("learning_rate", TA_EC_LEARNING_RATE)
    _ss = _ov.get("subsample", TA_EC_SUBSAMPLE)
    _cs = _ov.get("colsample_bytree", TA_EC_COLSAMPLE)
    _ra = _ov.get("reg_alpha", TA_EC_REG_ALPHA)
    _rl = _ov.get("reg_lambda", TA_EC_REG_LAMBDA)
    _gm = _ov.get("gamma", TA_EC_GAMMA)

    def _local_train(X_s, y_s, eval_set=None, es_rounds=0, use_r2_es=False, sw=None):
        X_s = np.nan_to_num(np.asarray(X_s, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        use_es = USE_EARLY_STOPPING and es_rounds > 0 and eval_set is not None
        if USE_XGB:
            kw = dict(n_estimators=_n_est, max_depth=_md, learning_rate=_lr,
                      subsample=_ss, colsample_bytree=_cs, reg_alpha=_ra, reg_lambda=_rl,
                      gamma=_gm, random_state=42, verbosity=0)
            if use_es:
                kw["early_stopping_rounds"] = es_rounds or EARLY_STOPPING_ROUNDS
            if use_es and use_r2_es:
                kw["eval_metric"] = _eval_metric_r2
            if _ov.get("objective") is not None:
                kw["objective"] = _ov["objective"]
            m = xgb.XGBRegressor(**kw)
            if use_es:
                m.fit(X_s, y_s, eval_set=eval_set, verbose=False, sample_weight=sw)
            else:
                m.fit(X_s, y_s, sample_weight=sw)
        else:
            m = RandomForestRegressor(n_estimators=100, random_state=42)
            m.fit(X_s, y_s, sample_weight=sw)
        return m

    y_mean = float(y.mean()) if USE_REGIONAL_STANDARDIZATION and not log_target else 0.0
    y_for_fit = (y - y_mean) if USE_REGIONAL_STANDARDIZATION and not log_target else (pd.Series(y_arr, index=y.index) if log_target and hasattr(y, "index") else (y_arr if log_target else y))
    if log_target and hasattr(y_for_fit, "iloc"):
        y_for_fit = pd.Series(y_arr, index=y.index) if hasattr(y, "index") else y_arr
    X_df = X if hasattr(X, "iloc") else pd.DataFrame(X)

    def _eval_metric_r2(preds, dtrain):
        """XGB custom metric: R² (XGB minimizes, so return -R²). sklearn API일 때 dtrain이 DMatrix 또는 array."""
        labels = dtrain.get_label() if hasattr(dtrain, "get_label") else np.asarray(dtrain)
        r2 = r2_score(labels, preds)
        return "r2", -r2

    _split_args = (X, y, groups) if CV_MODE == "spatial" else (X, y)
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(*_split_args)):
        if USE_FOLD_MEDIAN_IMPUTE:
            med_tr = X_df.iloc[train_idx].median()
            X_tr = X_df.iloc[train_idx].copy().fillna(med_tr)
            X_te = X_df.iloc[test_idx].copy().fillna(med_tr)
        else:
            X_tr = X_df.iloc[train_idx].copy().fillna(X_df.median())
            X_te = X_df.iloc[test_idx].copy().fillna(X_df.median())
        y_tr = y_for_fit.iloc[train_idx] if hasattr(y_for_fit, "iloc") else (y_for_fit[train_idx] if hasattr(y_for_fit, "__getitem__") else np.asarray(y_for_fit)[train_idx])
        y_te_orig = y.iloc[test_idx] if hasattr(y, "iloc") else y_orig[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        eval_set = None
        use_ta_ec_stk = USE_TA_EC_STACKING and param_name in ("Total Alkalinity", "Electrical Conductance") and USE_XGB
        use_cb = (param_name == "Total Alkalinity" and USE_CATBOOST_TA) or (param_name == "Electrical Conductance" and USE_CATBOOST_EC)
        ec_ov = {"objective": "reg:pseudohubererror"} if param_name == "Electrical Conductance" and USE_EC_HUBER_LOSS else None
        if use_ta_ec_stk:
            fold_grp = groups[train_idx] if groups is not None else np.zeros(len(train_idx), dtype=int)
            n_stk = min(n_splits, len(np.unique(fold_grp)) if len(np.unique(fold_grp)) > 1 else n_splits)
            model = StackedDRPModel.train_stacked(X_tr_s, np.asarray(y_tr), fold_grp, n_splits=n_stk,
                                                 use_lgb=USE_LIGHTGBM_TA_EC, use_catboost=use_cb,
                                                 xgb_kw_override=ec_ov)
        elif USE_EARLY_STOPPING and EARLY_STOPPING_ROUNDS > 0 and len(train_idx) > 20:
            n_ev = max(1, int(0.2 * len(train_idx)))
            fit_idx = train_idx[:-n_ev]
            ev_idx = train_idx[-n_ev:]
            med_fit = X_df.iloc[fit_idx].median()
            X_fit = X_df.iloc[fit_idx].copy().fillna(med_fit)
            X_ev = X_df.iloc[ev_idx].copy().fillna(med_fit)
            scaler_es = StandardScaler()
            X_fit_s = scaler_es.fit_transform(X_fit)
            X_ev_s = scaler_es.transform(X_ev)
            y_fit_fold = y_for_fit.iloc[fit_idx] if hasattr(y_for_fit, "iloc") else np.asarray(y_for_fit)[fit_idx]
            y_ev_fold = y_for_fit.iloc[ev_idx] if hasattr(y_for_fit, "iloc") else np.asarray(y_for_fit)[ev_idx]
            eval_set = [(X_ev_s, y_ev_fold)]
            sw_fit = sample_weight[fit_idx] if sample_weight is not None else None
            model = _local_train(X_fit_s, y_fit_fold, eval_set=eval_set, es_rounds=EARLY_STOPPING_ROUNDS, use_r2_es=USE_R2_EARLY_STOPPING, sw=sw_fit)
            X_tr_s = scaler_es.transform(X_tr)
            X_te_s = scaler_es.transform(X_te)
        else:
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None
            model = _local_train(X_tr_s, y_tr, use_r2_es=False, sw=sw_tr)
        pred_tr = model.predict(X_tr_s)
        pred_te = model.predict(X_te_s)
        if log_target:
            pred_tr = np.expm1(pred_tr)
            pred_te = np.expm1(pred_te)
        else:
            pred_te = pred_te + (y_mean if USE_REGIONAL_STANDARDIZATION else 0)
        y_tr_orig = y.iloc[train_idx] if hasattr(y, "iloc") else y_orig[train_idx]
        r2_fold = r2_score(y_te_orig, pred_te)
        r2_train_list.append(r2_score(y_tr_orig, pred_tr))
        rmse_train_list.append(np.sqrt(mean_squared_error(y_tr_orig, pred_tr)))
        r2_test_list.append(r2_fold)
        rmse_test_list.append(np.sqrt(mean_squared_error(y_te_orig, pred_te)))
        if USE_EC_CENTERED_CV and groups is not None and test_idx is not None and len(test_idx) <= len(groups):
            te_grp = np.unique(groups[test_idx])
            if len(te_grp) == 1 and te_grp[0] == 0:
                if not hasattr(run_pipeline_cv, "_ec_fold_r2"):
                    run_pipeline_cv._ec_fold_r2 = []
                run_pipeline_cv._ec_fold_r2.append((param_name, r2_fold))

    X_full = X_df.copy().fillna(X_df.median())
    scaler_final = StandardScaler()
    X_full_s = scaler_final.fit_transform(X_full)
    y_fit_final = np.asarray(y_for_fit) if not hasattr(y_for_fit, "values") else (y_for_fit.values if hasattr(y_for_fit, "values") else y_for_fit)

    use_cb_ta = param_name == "Total Alkalinity" and USE_CATBOOST_TA
    use_cb_ec = param_name == "Electrical Conductance" and USE_CATBOOST_EC
    use_catboost_blend = use_cb_ta or use_cb_ec
    use_ta_ec_stk_final = USE_TA_EC_STACKING and param_name in ("Total Alkalinity", "Electrical Conductance") and USE_XGB
    ec_ov_final = {"objective": "reg:pseudohubererror"} if param_name == "Electrical Conductance" and USE_EC_HUBER_LOSS else None

    if use_ta_ec_stk_final:
        grp_full = groups if groups is not None else np.zeros(len(X_full_s), dtype=int)
        n_stk_full = min(n_splits, len(np.unique(grp_full)) if len(np.unique(grp_full)) > 1 else n_splits)
        model_final = StackedDRPModel.train_stacked(X_full_s, y_fit_final, grp_full, n_splits=n_stk_full,
                                                    use_lgb=USE_LIGHTGBM_TA_EC, use_catboost=use_catboost_blend,
                                                    xgb_kw_override=ec_ov_final)
    elif (TA_EC_MULTI_SEED and USE_XGB) or use_catboost_blend:
        seed_models = []
        if TA_EC_MULTI_SEED and USE_XGB:
            for si in range(TA_EC_N_SEEDS):
                kw = dict(
                    n_estimators=_n_est, max_depth=_md, learning_rate=_lr,
                    subsample=_ss, colsample_bytree=_cs,
                    reg_alpha=_ra, reg_lambda=_rl, gamma=_gm,
                    random_state=42 + si * 7, verbosity=0,
                )
                if _ov.get("objective") is not None:
                    kw["objective"] = _ov["objective"]
                m_i = xgb.XGBRegressor(**kw)
                m_i.fit(X_full_s, y_fit_final, sample_weight=sample_weight)
                seed_models.append(m_i)
        elif USE_XGB and use_catboost_blend:
            kw = dict(n_estimators=_n_est, max_depth=_md, learning_rate=_lr,
                      subsample=_ss, colsample_bytree=_cs, reg_alpha=_ra, reg_lambda=_rl, gamma=_gm,
                      random_state=42, verbosity=0)
            if _ov.get("objective") is not None:
                kw["objective"] = _ov["objective"]
            m_xgb = xgb.XGBRegressor(**kw)
            m_xgb.fit(X_full_s, y_fit_final, sample_weight=sample_weight)
            seed_models.append(m_xgb)

        if use_catboost_blend:
            try:
                import catboost as cb
                cb_m = cb.CatBoostRegressor(iterations=_n_est, depth=_md, learning_rate=_lr, random_state=47, verbose=0)
                cb_m.fit(X_full_s, y_fit_final, sample_weight=sample_weight)
                seed_models.append(cb_m)
            except ImportError:
                pass

        if seed_models:
            class _MultiSeedTA_EC:
                def __init__(self, models):
                    self.models = models
                    if models and hasattr(models[0], "feature_importances_"):
                        self.feature_importances_ = models[0].feature_importances_
                def predict(self, X):
                    return np.mean([m.predict(X) for m in self.models], axis=0)
            model_final = _MultiSeedTA_EC(seed_models)
        else:
            model_final = _local_train(X_full_s, y_fit_final, sw=sample_weight)
    else:
        model_final = _local_train(X_full_s, y_fit_final, sw=sample_weight)

    r2_test_avg = np.mean(r2_test_list)
    results = {
        "Parameter": param_name,
        "R2_Train": np.mean(r2_train_list),
        "RMSE_Train": np.mean(rmse_train_list),
        "R2_Test": r2_test_avg,
        "RMSE_Test": np.mean(rmse_test_list),
    }
    n_seed_str = " (stacking)" if use_ta_ec_stk_final else (f" ({TA_EC_N_SEEDS}-seed)" if TA_EC_MULTI_SEED else "")
    print(f"  {param_name}: {n_splits}-fold avg R2_Test={r2_test_avg:.4f}{n_seed_str}")
    out_mean = y_mean if USE_REGIONAL_STANDARDIZATION else None
    return model_final, scaler_final, pd.DataFrame([results]), out_mean, log_target


def run_gems_weak_holdout(wq_data, feat_TA, feat_EC, feat_DRP, y_TA, y_EC, y_DRP, residual_mode=True, log_residual=True, use_ta_ec=True):
    """
    Train=GEMS 좋은 지역, Val=GEMS 약한 지역으로 1회 학습·평가.
    use_ta_ec=False면 DRP는 pred_TA/EC 없이 단독 학습.
    """
    split = get_gems_quality_split(wq_data, min_test=20)
    if split is None:
        return None
    train_idx, test_idx = split
    X_TA = wq_data[feat_TA].fillna(wq_data[feat_TA].median())
    X_EC = wq_data[feat_EC].fillna(wq_data[feat_EC].median())
    X_DRP = wq_data[feat_DRP].fillna(wq_data[feat_DRP].median())
    y_DRP_raw = y_DRP.values
    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(len(X_DRP))
    gems_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    if residual_mode and log_residual:
        y_DRP_res = np.log1p(np.maximum(y_DRP_raw, 0)) - np.log1p(np.maximum(gems_safe, 0))
    elif residual_mode:
        y_DRP_res = y_DRP_raw - gems_safe
    else:
        y_DRP_res = np.log1p(np.maximum(y_DRP_raw, 0)) if log_residual else y_DRP_raw

    scaler_ta = StandardScaler()
    scaler_ec = StandardScaler()
    scaler_drp = StandardScaler()
    X_ta_tr = X_TA.iloc[train_idx]; X_ta_te = X_TA.iloc[test_idx]
    X_ec_tr = X_EC.iloc[train_idx]; X_ec_te = X_EC.iloc[test_idx]
    m_ta = train_model(scaler_ta.fit_transform(X_ta_tr), y_TA.iloc[train_idx])
    m_ec = train_model(scaler_ec.fit_transform(X_ec_tr), y_EC.iloc[train_idx])
    r2_ta = r2_score(y_TA.iloc[test_idx], m_ta.predict(scaler_ta.transform(X_ta_te)))
    r2_ec = r2_score(y_EC.iloc[test_idx], m_ec.predict(scaler_ec.transform(X_ec_te)))

    X_drp_tr = X_DRP.iloc[train_idx].copy()
    X_drp_te = X_DRP.iloc[test_idx].copy()
    if use_ta_ec:
        X_drp_tr["pred_TA"] = m_ta.predict(scaler_ta.transform(X_ta_tr))
        X_drp_tr["pred_EC"] = m_ec.predict(scaler_ec.transform(X_ec_tr))
        X_drp_te["pred_TA"] = m_ta.predict(scaler_ta.transform(X_ta_te))
        X_drp_te["pred_EC"] = m_ec.predict(scaler_ec.transform(X_ec_te))
    y_drp_tr = y_DRP_res[train_idx]
    y_drp_te_orig = y_DRP_raw[test_idx]
    m_drp = train_model(scaler_drp.fit_transform(X_drp_tr), y_drp_tr)
    raw = m_drp.predict(scaler_drp.transform(X_drp_te))
    if residual_mode:
        g_te = gems_safe[test_idx]
        if log_residual:
            pred_drp = np.maximum(np.expm1(np.log1p(np.maximum(g_te, 0)) + raw), 0)
        else:
            pred_drp = np.maximum(g_te + raw, 0)
    else:
        pred_drp = np.maximum(np.expm1(raw) if log_residual else raw, 0)
    r2_drp = r2_score(y_drp_te_orig, pred_drp)
    return (r2_ta, r2_ec, r2_drp)


def _oof_predictions_same_splits(X, y, groups, n_splits):
    """OOF 예측 (각 행이 정확히 한 fold의 test에만 속함)."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    if CV_MODE == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        splitter = GroupKFold(n_splits=n_splits)
    X_df = X if hasattr(X, "iloc") else pd.DataFrame(X)
    _sp_args = (X_df, y, groups) if CV_MODE == "spatial" else (X_df, y)
    for train_idx, test_idx in splitter.split(*_sp_args):
        if USE_FOLD_MEDIAN_IMPUTE:
            med_tr = X_df.iloc[train_idx].median()
            X_tr = X_df.iloc[train_idx].copy().fillna(med_tr)
            X_te = X_df.iloc[test_idx].copy().fillna(med_tr)
        else:
            X_tr = X_df.iloc[train_idx].copy().fillna(X_df.median())
            X_te = X_df.iloc[test_idx].copy().fillna(X_df.median())
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = train_model(X_tr_s, y_tr)
        oof[test_idx] = model.predict(X_te_s)
    return oof


def run_pipeline_drp_cv(
    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
    model_TA, model_EC, scaler_TA, scaler_EC,
    n_splits=5, residual_mode=True, log_residual=True, use_ta_ec=True,
    sample_weight=None, xgb_objective="reg:squarederror", ta_ec_blend=1.0,
    distance_km=None, decay_km=None, has_prior=None,
):
    """
    DRP: use_ta_ec, residual. USE_DRP_PRIOR_SPLIT & has_prior 있으면 prior/noprior 완전 분리(8-tuple), 아니면 단일(6-tuple).
    """
    if groups is None:
        groups = np.zeros(len(y_DRP), dtype=int)
    if CV_MODE == "random":
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv_splitter = GroupKFold(n_splits=n_splits)
    n = len(y_DRP)
    use_drp_model = sample_weight is not None or xgb_objective != "reg:squarederror"
    MIN_PRIOR_SPLIT = 15
    use_prior_split = (
        USE_DRP_PRIOR_SPLIT and has_prior is not None
        and has_prior.sum() >= MIN_PRIOR_SPLIT and (n - has_prior.sum()) >= MIN_PRIOR_SPLIT
    )
    if use_prior_split:
        prior_ix = np.where(has_prior)[0]
        noprior_ix = np.where(~has_prior)[0]
    if use_ta_ec:
        oof_pred_TA = _oof_predictions_same_splits(X_TA, y_TA, groups, n_splits)
        oof_pred_EC = _oof_predictions_same_splits(X_EC, y_EC, groups, n_splits)
        if ta_ec_blend != 1.0:
            oof_pred_TA = oof_pred_TA * ta_ec_blend
            oof_pred_EC = oof_pred_EC * ta_ec_blend

    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(n)
    gems_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    _ratio_target = DRP_RATIO_TARGET and residual_mode
    if _ratio_target:
        y_DRP_res = y_DRP.values / np.maximum(gems_safe, DRP_RATIO_EPS)
    elif residual_mode:
        if log_residual:
            y_DRP_res = np.log1p(np.maximum(y_DRP.values, 0)) - np.log1p(np.maximum(gems_safe, 0))
        else:
            y_DRP_res = y_DRP.values - gems_safe
    else:
        y_DRP_res = np.log1p(np.maximum(y_DRP.values, 0)) if log_residual else y_DRP.values
    res_q01, res_q99 = np.percentile(y_DRP_res, 1), np.percentile(y_DRP_res, 99)
    y_DRP_res = np.clip(y_DRP_res, res_q01, res_q99)
    y_DRP_res = pd.Series(y_DRP_res, index=y_DRP.index)
    y_DRP_log = pd.Series(np.log1p(np.maximum(y_DRP.values, 0)), index=y_DRP.index)

    if use_prior_split:
        prior_cols_base = [c for c in PRIOR_ONLY_FEATURES if c in X_DRP.columns]
        noprior_cols_base = [c for c in NO_PRIOR_FEATURES if c in X_DRP.columns]
        if not prior_cols_base:
            prior_cols_base = [c for c in X_DRP.columns if c not in ("pred_TA", "pred_EC")]
        if not noprior_cols_base:
            noprior_cols_base = [c for c in X_DRP.columns if c not in ("pred_TA", "pred_EC")]
        add_prior_weight = distance_km is not None and decay_km is not None
        prior_cols = prior_cols_base + (["prior_weight"] if add_prior_weight else [])
        noprior_cols = list(noprior_cols_base)

        def _prior_df(ix, med, dist_km=None, dec_km=None):
            out = X_DRP.reindex(columns=prior_cols_base).iloc[ix].copy().fillna(med.reindex(prior_cols_base).fillna(0))
            if add_prior_weight and dist_km is not None and dec_km is not None:
                d = dist_km.iloc[ix].values if hasattr(dist_km, "iloc") else dist_km[ix]
                out["prior_weight"] = np.exp(-np.nan_to_num(d, nan=999.0) / dec_km)
            return out

        r2_list, rmse_list = [], []
        r2_train_list, rmse_train_list = [], []
        _drp_split_args = (X_DRP, y_DRP, groups) if CV_MODE == "spatial" else (X_DRP, y_DRP)
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(*_drp_split_args)):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            te_prior = has_prior[test_idx]
            te_noprior = ~te_prior
            tr_prior = has_prior[train_idx]
            tr_noprior = ~tr_prior
            train_prior_ix, train_noprior_ix = train_idx[tr_prior], train_idx[tr_noprior]
            test_prior_ix, test_noprior_ix = test_idx[te_prior], test_idx[te_noprior]
            med_tr = X_DRP.iloc[train_idx].median()
            pred_fold = np.full(n, np.nan, dtype=np.float64)

            if use_ta_ec:
                X_ta_tr = X_TA.iloc[train_idx].fillna(X_TA.iloc[train_idx].median())
                X_ta_te = X_TA.iloc[test_idx].fillna(X_TA.iloc[train_idx].median())
                X_ec_tr = X_EC.iloc[train_idx].fillna(X_EC.iloc[train_idx].median())
                X_ec_te = X_EC.iloc[test_idx].fillna(X_EC.iloc[train_idx].median())
                scaler_ta = StandardScaler()
                scaler_ec = StandardScaler()
                m_ta = train_model(scaler_ta.fit_transform(X_ta_tr), y_TA.iloc[train_idx])
                m_ec = train_model(scaler_ec.fit_transform(X_ec_tr), y_EC.iloc[train_idx])
                pred_TA_te = m_ta.predict(scaler_ta.transform(X_ta_te)) * ta_ec_blend
                pred_EC_te = m_ec.predict(scaler_ec.transform(X_ec_te)) * ta_ec_blend
            else:
                pred_TA_te = pred_EC_te = np.zeros(len(test_idx))

            if len(train_prior_ix) >= 5 and len(test_prior_ix) > 0:
                X_tr_p = _prior_df(train_prior_ix, med_tr, distance_km, decay_km)
                X_te_p = _prior_df(test_prior_ix, med_tr, distance_km, decay_km)
                if use_ta_ec:
                    X_tr_p["pred_TA"] = oof_pred_TA[train_prior_ix]
                    X_tr_p["pred_EC"] = oof_pred_EC[train_prior_ix]
                    X_te_p["pred_TA"] = pred_TA_te[te_prior]
                    X_te_p["pred_EC"] = pred_EC_te[te_prior]
                scaler_p = StandardScaler()
                m_p = train_model(scaler_p.fit_transform(X_tr_p), y_DRP_res.iloc[train_prior_ix])
                raw_p = m_p.predict(scaler_p.transform(X_te_p))
                g_te = gems_safe[test_prior_ix]
                pred_fold[test_prior_ix] = np.maximum(np.expm1(np.log1p(np.maximum(g_te, 0)) + raw_p), 0)
            if len(train_noprior_ix) >= 5 and len(test_noprior_ix) > 0:
                X_tr_n = X_DRP.reindex(columns=noprior_cols).iloc[train_noprior_ix].copy().fillna(med_tr.reindex(noprior_cols).fillna(0))
                X_te_n = X_DRP.reindex(columns=noprior_cols).iloc[test_noprior_ix].copy().fillna(med_tr.reindex(noprior_cols).fillna(0))
                if use_ta_ec:
                    X_tr_n["pred_TA"] = oof_pred_TA[train_noprior_ix]
                    X_tr_n["pred_EC"] = oof_pred_EC[train_noprior_ix]
                    X_te_n["pred_TA"] = pred_TA_te[te_noprior]
                    X_te_n["pred_EC"] = pred_EC_te[te_noprior]
                scaler_n = StandardScaler()
                m_n = train_model_drp(scaler_n.fit_transform(X_tr_n), y_DRP_log.iloc[train_noprior_ix], objective="reg:pseudohubererror")
                raw_n = m_n.predict(scaler_n.transform(X_te_n))
                raw_n = np.clip(raw_n, -15.0, 15.0)
                pred_fold[test_noprior_ix] = np.maximum(np.expm1(raw_n), 0)

            # Train metrics for prior_split (models exist when we have both train and test for each branch)
            pred_train = np.full(n, np.nan, dtype=np.float64)
            if len(train_prior_ix) >= 5 and len(test_prior_ix) > 0:
                raw_p_tr = m_p.predict(scaler_p.transform(X_tr_p))
                g_tr = gems_safe[train_prior_ix]
                pred_train[train_prior_ix] = np.maximum(np.expm1(np.log1p(np.maximum(g_tr, 0)) + raw_p_tr), 0)
            if len(train_noprior_ix) >= 5 and len(test_noprior_ix) > 0:
                raw_n_tr = m_n.predict(scaler_n.transform(X_tr_n))
                raw_n_tr = np.clip(raw_n_tr, -15.0, 15.0)
                pred_train[train_noprior_ix] = np.maximum(np.expm1(raw_n_tr), 0)
            train_valid = ~np.isnan(pred_train[train_idx])
            if train_valid.sum() > 0:
                r2_train_list.append(r2_score(y_DRP.iloc[train_idx].values[train_valid], pred_train[train_idx][train_valid]))
                rmse_train_list.append(np.sqrt(mean_squared_error(y_DRP.iloc[train_idx].values[train_valid], pred_train[train_idx][train_valid])))
            else:
                r2_train_list.append(0.0)
                rmse_train_list.append(0.0)

            valid = ~np.isnan(pred_fold[test_idx])
            if valid.sum() > 0:
                r2_list.append(r2_score(y_DRP.iloc[test_idx].values[valid], pred_fold[test_idx][valid]))
                rmse_list.append(np.sqrt(mean_squared_error(y_DRP.iloc[test_idx].values[valid], pred_fold[test_idx][valid])))
            else:
                r2_list.append(0.0)
                rmse_list.append(0.0)

        r2_avg = np.mean(r2_list) if r2_list else 0.0
        print(f"  Dissolved Reactive Phosphorus (DRP prior분리+2stage): {n_splits}-fold avg R2_Test={r2_avg:.4f}  (prior n={prior_ix.size}, noprior n={noprior_ix.size}) prior피처={len(prior_cols)} noPrior피처={len(noprior_cols)}")
        X_full = X_DRP.copy().fillna(X_DRP.median())
        if use_ta_ec:
            X_full["pred_TA"] = model_TA.predict(scaler_TA.transform(X_TA.fillna(X_TA.median()))) * ta_ec_blend
            X_full["pred_EC"] = model_EC.predict(scaler_EC.transform(X_EC.fillna(X_EC.median()))) * ta_ec_blend
        X_full_prior = _prior_df(prior_ix, X_full.median(), distance_km, decay_km)
        if use_ta_ec:
            X_full_prior["pred_TA"] = model_TA.predict(scaler_TA.transform(X_TA.fillna(X_TA.median()))) * ta_ec_blend
            X_full_prior["pred_EC"] = model_EC.predict(scaler_EC.transform(X_EC.fillna(X_EC.median()))) * ta_ec_blend
        X_full_noprior = X_full.reindex(columns=noprior_cols).iloc[noprior_ix].copy().fillna(X_full.median().reindex(noprior_cols).fillna(0))
        if use_ta_ec:
            X_full_noprior["pred_TA"] = model_TA.predict(scaler_TA.transform(X_TA.fillna(X_TA.median()))) * ta_ec_blend
            X_full_noprior["pred_EC"] = model_EC.predict(scaler_EC.transform(X_EC.fillna(X_EC.median()))) * ta_ec_blend
        scaler_prior = StandardScaler()
        scaler_noprior = StandardScaler()
        model_prior = train_model(scaler_prior.fit_transform(X_full_prior), y_DRP_res.iloc[prior_ix])
        model_noprior = train_model_drp(scaler_noprior.fit_transform(X_full_noprior), y_DRP_log.iloc[noprior_ix], objective="reg:pseudohubererror")
        r2_train_avg = np.mean(r2_train_list) if r2_train_list else np.nan
        rmse_train_avg = np.mean(rmse_train_list) if rmse_train_list else np.nan
        results = {"Parameter": "Dissolved Reactive Phosphorus", "R2_Train": r2_train_avg, "RMSE_Train": rmse_train_avg, "R2_Test": r2_avg, "RMSE_Test": np.mean(rmse_list) if rmse_list else np.nan}
        _decay_km = (decay_km if decay_km is not None else PRIOR_DECAY_KM) if USE_DRP_DISTANCE_PRIOR else None
        return (model_prior, model_noprior, scaler_prior, scaler_noprior, pd.DataFrame([results]), True, True, _decay_km, prior_cols, noprior_cols)

    # LB results: orig_500_d5+Markowitz→0.316, reg_450_d5→0.377, reg_400_d4→0.3709
    # DRP 전용: loss/하이퍼파라미터 다양화 (skewed target → Tweedie/Huber, event-driven → depth/reg)
    DRP_CONFIGS = [
        {"label": "reg_450_d5", "objective": "reg:squarederror",
         "n_estimators": 450, "max_depth": 5, "learning_rate": 0.04,
         "subsample": 0.72, "colsample_bytree": 0.68, "reg_alpha": 0.3, "reg_lambda": 1.5,
         "gamma": 0.8},
        {"label": "reg_500_d5_huber", "objective": "reg:pseudohubererror",
         "n_estimators": 500, "max_depth": 5, "learning_rate": 0.035,
         "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 0.4, "reg_lambda": 1.8,
         "gamma": 1.0},
        {"label": "reg_500_d6", "objective": "reg:squarederror",
         "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
         "subsample": 0.68, "colsample_bytree": 0.65, "reg_alpha": 0.5, "reg_lambda": 2.0,
         "gamma": 1.2},
        {"label": "reg_400_d4", "objective": "reg:squarederror",
         "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.25, "reg_lambda": 1.2,
         "gamma": 0.6},
        {"label": "reg_450_d5_tweedie", "objective": "reg:tweedie",
         "n_estimators": 450, "max_depth": 5, "learning_rate": 0.04,
         "subsample": 0.72, "colsample_bytree": 0.68, "reg_alpha": 0.3, "reg_lambda": 1.5,
         "gamma": 0.8, "tweedie_variance_power": 1.5},
    ]
    if not (USE_DRP_OBJECTIVE_SEARCH and USE_XGB):
        DRP_CONFIGS = [DRP_CONFIGS[0]]

    def _cv_one_config(cfg, cv_obj):
        r2s, rmses = [], []
        r2_train_list, rmse_train_list = [], []
        r2_by_alpha = {}
        obj = cfg["objective"]
        xgb_kw = {k: cfg[k] for k in cfg if k not in ("label", "objective")}
        xgb_kw["random_state"] = 42
        xgb_kw["verbosity"] = 0
        if obj != "reg:squarederror":
            xgb_kw["objective"] = obj
        _cfg_split_args = (X_DRP, y_DRP, groups) if CV_MODE == "spatial" else (X_DRP, y_DRP)
        for fold, (train_idx, test_idx) in enumerate(cv_obj.split(*_cfg_split_args)):
            X_drp_tr = X_DRP.iloc[train_idx].copy().fillna(X_DRP.iloc[train_idx].median())
            X_drp_te = X_DRP.iloc[test_idx].copy().fillna(X_DRP.iloc[train_idx].median())
            y_drp_tr = y_DRP_res.iloc[train_idx]
            y_drp_te_orig = y_DRP.iloc[test_idx]
            if use_ta_ec:
                X_ta_tr, X_ta_te = X_TA.iloc[train_idx], X_TA.iloc[test_idx]
                X_ec_tr, X_ec_te = X_EC.iloc[train_idx], X_EC.iloc[test_idx]
                scaler_ta = StandardScaler()
                scaler_ec = StandardScaler()
                m_ta = train_model(scaler_ta.fit_transform(X_ta_tr.fillna(X_ta_tr.median())), y_TA.iloc[train_idx])
                m_ec = train_model(scaler_ec.fit_transform(X_ec_tr.fillna(X_ec_tr.median())), y_EC.iloc[train_idx])
                pred_TA_te = m_ta.predict(scaler_ta.transform(X_ta_te.fillna(X_ta_tr.median())))
                pred_EC_te = m_ec.predict(scaler_ec.transform(X_ec_te.fillna(X_ec_tr.median())))
                X_drp_tr = X_drp_tr.copy()
                X_drp_te = X_drp_te.copy()
                X_drp_tr["pred_TA"] = oof_pred_TA[train_idx]
                X_drp_tr["pred_EC"] = oof_pred_EC[train_idx]
                X_drp_te["pred_TA"] = pred_TA_te * ta_ec_blend
                X_drp_te["pred_EC"] = pred_EC_te * ta_ec_blend
            scaler_drp = StandardScaler()
            X_tr_s = scaler_drp.fit_transform(X_drp_tr)
            X_te_s = scaler_drp.transform(X_drp_te)
            sw_fold = sample_weight[train_idx] if sample_weight is not None else None
            if USE_DRP_STACKING and USE_XGB:
                fold_groups = groups[train_idx] if groups is not None else np.zeros(len(train_idx), dtype=int)
                xgb_override = {k: v for k, v in xgb_kw.items() if k not in ("random_state", "verbosity")}
                m_drp = StackedDRPModel.train_stacked(X_tr_s, y_drp_tr.values if hasattr(y_drp_tr, "values") else y_drp_tr, fold_groups, n_splits=min(n_splits, len(np.unique(fold_groups))), use_lgb=USE_LIGHTGBM_DRP, use_catboost=USE_CATBOOST_DRP, xgb_kw_override=xgb_override)
            else:
                use_es_drp = USE_DRP_EARLY_STOPPING and len(train_idx) > 300 and EARLY_STOPPING_ROUNDS_DRP > 0
                xgb_kw_fold = dict(xgb_kw)
                if use_es_drp and groups is not None:
                    xgb_kw_fold["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS_DRP
                m_drp = xgb.XGBRegressor(**xgb_kw_fold)
                if use_es_drp and groups is not None:
                    gss = GroupShuffleSplit(n_splits=1, test_size=DRP_VAL_FRAC, random_state=42)
                    tr_inner, val_inner = next(gss.split(X_drp_tr, y_drp_tr, groups[train_idx]))
                    X_tr_inner_s = X_tr_s[tr_inner]
                    y_tr_inner = y_drp_tr.iloc[tr_inner] if hasattr(y_drp_tr, "iloc") else np.asarray(y_drp_tr)[tr_inner]
                    X_val_s = X_tr_s[val_inner]
                    y_val_s = y_drp_tr.iloc[val_inner].values if hasattr(y_drp_tr, "iloc") else np.asarray(y_drp_tr)[val_inner]
                    sw_inner = sw_fold[tr_inner] if sw_fold is not None else None
                    m_drp.fit(X_tr_inner_s, y_tr_inner, sample_weight=sw_inner,
                              eval_set=[(X_val_s, y_val_s)], verbose=False)
                else:
                    m_drp.fit(X_tr_s, y_drp_tr, sample_weight=sw_fold)
            raw = m_drp.predict(X_te_s)
            if obj == "reg:pseudohubererror":
                raw = np.clip(raw, -15.0, 15.0)
            if _ratio_target:
                g_te = gems_safe[test_idx]
                pred_drp = np.maximum(raw * np.maximum(g_te, DRP_RATIO_EPS), 0)
            elif residual_mode:
                g_te = gems_safe[test_idx]
                if log_residual:
                    pred_drp = np.maximum(np.expm1(np.log1p(np.maximum(g_te, 0)) + raw), 0)
                else:
                    pred_drp = np.maximum(g_te + raw, 0)
            else:
                pred_drp = np.maximum(np.expm1(raw) if log_residual else raw, 0)
            if USE_DRP_DISTANCE_PRIOR and distance_km is not None and decay_km is not None:
                dist_te = np.nan_to_num(distance_km.iloc[test_idx].values if hasattr(distance_km, "iloc") else distance_km[test_idx], nan=999.0)
                prior_w = np.exp(-dist_te / decay_km)
                g_te = gems_safe[test_idx]
                pred_drp = pred_drp + (prior_w - 1.0) * g_te
                pred_drp = np.maximum(pred_drp, 0)
            pred_drp = np.clip(pred_drp, 0.0, 1e4)
            g_te = gems_safe[test_idx]
            r2s.append(r2_score(y_drp_te_orig, pred_drp))
            rmses.append(np.sqrt(mean_squared_error(y_drp_te_orig, pred_drp)))
            if USE_GEMS_BLEND_AUTO and residual_mode and (g_te > 0).any():
                for _a in GEMS_BLEND_ALPHA_CANDIDATES:
                    _pb = np.where(g_te > 0, _a * pred_drp + (1 - _a) * g_te, pred_drp)
                    _pb = np.clip(_pb, 0.0, 1e4)
                    _r2_blend = r2_score(y_drp_te_orig, _pb)
                    r2_by_alpha.setdefault(_a, []).append(_r2_blend)

            raw_train = m_drp.predict(X_tr_s)
            if obj == "reg:pseudohubererror":
                raw_train = np.clip(raw_train, -15.0, 15.0)
            if _ratio_target:
                g_tr = gems_safe[train_idx]
                pred_drp_train = np.maximum(raw_train * np.maximum(g_tr, DRP_RATIO_EPS), 0)
            elif residual_mode:
                g_tr = gems_safe[train_idx]
                if log_residual:
                    pred_drp_train = np.maximum(np.expm1(np.log1p(np.maximum(g_tr, 0)) + raw_train), 0)
                else:
                    pred_drp_train = np.maximum(g_tr + raw_train, 0)
            else:
                pred_drp_train = np.maximum(np.expm1(raw_train) if log_residual else raw_train, 0)
            pred_drp_train = np.clip(pred_drp_train, 0.0, 1e4)
            y_drp_tr_orig = y_DRP.iloc[train_idx]
            r2_train_list.append(r2_score(y_drp_tr_orig, pred_drp_train))
            rmse_train_list.append(np.sqrt(mean_squared_error(y_drp_tr_orig, pred_drp_train)))
        return np.mean(r2s), np.mean(rmses), cfg, r2s, r2_train_list, rmse_train_list, r2_by_alpha

    config_results = []
    for cfg in DRP_CONFIGS:
        r2_c, rmse_c, _, fold_r2s, r2_train_folds, rmse_train_folds, r2_by_alpha = _cv_one_config(cfg, cv_splitter)
        config_results.append((r2_c, rmse_c, cfg, fold_r2s, r2_train_folds, rmse_train_folds, r2_by_alpha))
        fold_str = " ".join([f"f{i}={v:.4f}" for i, v in enumerate(fold_r2s)])
        print(f"    DRP [{cfg['label']}] depth={cfg.get('max_depth',6)}: R2={r2_c:.4f}  ({fold_str})")
    config_results.sort(key=lambda x: -x[0])
    best_r2, best_rmse, best_cfg, best_fold_r2s, best_r2_train_folds, best_rmse_train_folds, best_r2_by_alpha = config_results[0]
    if len(DRP_CONFIGS) > 1:
        print(f"    >>> BEST: [{best_cfg['label']}] R2={best_r2:.4f}")

    best_alpha = GEMS_BLEND_ALPHA if not USE_GEMS_BLEND_AUTO else 1.0
    if USE_GEMS_BLEND_AUTO and best_r2_by_alpha:
        mean_r2_by_alpha = {a: np.mean(v) for a, v in best_r2_by_alpha.items()}
        best_alpha = max(mean_r2_by_alpha, key=mean_r2_by_alpha.get)
        print(f"  GEMS 블렌딩 alpha 자동 선택: {best_alpha} (R2={mean_r2_by_alpha[best_alpha]:.4f})")

    r2_list = [best_r2]
    rmse_list = [best_rmse]
    best_xgb_objective = best_cfg["objective"]
    best_xgb_kw = {k: best_cfg[k] for k in best_cfg if k not in ("label", "objective")}
    best_xgb_kw["random_state"] = 42
    best_xgb_kw["verbosity"] = 0
    if best_xgb_objective != "reg:squarederror":
        best_xgb_kw["objective"] = best_xgb_objective
    xgb_objective = best_xgb_objective

    r2_avg = best_r2
    stage = "residual+2stage(TA/EC)" if use_ta_ec else "단독(TA/EC 미사용)"
    print(f"  Dissolved Reactive Phosphorus (DRP {stage}): best R2_Test={r2_avg:.4f}")

    X_full_drp = X_DRP.copy().fillna(X_DRP.median())
    if use_ta_ec:
        pred_TA_full = model_TA.predict(scaler_TA.transform(X_TA.fillna(X_TA.median()))) * ta_ec_blend
        pred_EC_full = model_EC.predict(scaler_EC.transform(X_EC.fillna(X_EC.median()))) * ta_ec_blend
        X_full_drp["pred_TA"] = pred_TA_full
        X_full_drp["pred_EC"] = pred_EC_full
    scaler_DRP_final = StandardScaler()
    X_full_s = scaler_DRP_final.fit_transform(X_full_drp)
    sw_full = sample_weight if sample_weight is not None else None
    if USE_DRP_STACKING and USE_XGB:
        groups_full = groups if groups is not None else np.zeros(len(y_DRP_res), dtype=int)
        xgb_final_override = {k: v for k, v in best_xgb_kw.items() if k not in ("random_state", "verbosity")}
        model_DRP_final = StackedDRPModel.train_stacked(X_full_s, np.asarray(y_DRP_res), groups_full, n_splits=n_splits, use_lgb=USE_LIGHTGBM_DRP, use_catboost=USE_CATBOOST_DRP, xgb_kw_override=xgb_final_override)
        _bases = ["XGB×2", "RF"] + (["LGB"] if USE_LIGHTGBM_DRP else []) + (["CatBoost"] if USE_CATBOOST_DRP else [])
        print(f"  DRP stacking: {len(_bases)} base models ({', '.join(_bases)}) + Ridge meta")
    elif USE_DRP_OBJECTIVE_SEARCH and USE_XGB:
        N_SEEDS = 5
        seed_models = []
        for si in range(N_SEEDS):
            kw_i = dict(best_xgb_kw)
            kw_i["random_state"] = 42 + si * 7
            m_i = xgb.XGBRegressor(**kw_i)
            m_i.fit(X_full_s, y_DRP_res, sample_weight=sw_full)
            seed_models.append(m_i)

        class _MultiSeedModel:
            def __init__(self, models, alpha=1.0):
                self.models = models
                self.alpha = alpha
                if models and hasattr(models[0], "feature_importances_"):
                    self.feature_importances_ = models[0].feature_importances_
            def predict(self, X):
                return np.mean([m.predict(X) for m in self.models], axis=0) * self.alpha

        model_DRP_final = _MultiSeedModel(seed_models, alpha=best_alpha)
        print(f"  DRP final model: [{best_cfg['label']}] obj={best_xgb_objective}, {N_SEEDS}-seed, alpha={best_alpha:.2f}")
    elif use_drp_model:
        model_DRP_final = train_model_drp(X_full_s, y_DRP_res, sample_weight=sw_full, objective=xgb_objective)
    elif USE_XGB:
        # stacking/objective_search 꺼져 있어도 최종 모델은 CV에서 쓴 reg_450_d5 사용 (train_model은 TA/EC용 200/d6 → LB 괴리 원인)
        m_drp = xgb.XGBRegressor(**best_xgb_kw)
        m_drp.fit(X_full_s, y_DRP_res, sample_weight=sw_full)
        model_DRP_final = m_drp
        print(f"  DRP final model: [{best_cfg['label']}] single-seed (제출용)")
    else:
        model_DRP_final = train_model(X_full_s, y_DRP_res)

    r2_train_avg = np.mean(best_r2_train_folds) if best_r2_train_folds else np.nan
    rmse_train_avg = np.mean(best_rmse_train_folds) if best_rmse_train_folds else np.nan
    results = {
        "Parameter": "Dissolved Reactive Phosphorus",
        "R2_Train": r2_train_avg,
        "RMSE_Train": rmse_train_avg,
        "R2_Test": r2_avg,
        "RMSE_Test": np.mean(rmse_list),
    }
    _decay_km = (decay_km if decay_km is not None else PRIOR_DECAY_KM) if USE_DRP_DISTANCE_PRIOR else None
    return model_DRP_final, scaler_DRP_final, pd.DataFrame([results]), residual_mode, log_residual, _decay_km, best_alpha


def run_pipeline_drp_dro_cv(
    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
    model_TA, model_EC, scaler_TA, scaler_EC,
    n_splits=5, residual_mode=True, log_residual=True, use_ta_ec=True,
    worst_cluster_weight=2.0, n_worst=2,
    distance_km=None, decay_km=None, has_prior=None, ta_ec_blend=1.0,
    base_sample_weight=None,
):
    """2순위: cluster별 RMSE 계산 후 worst 1~2 cluster에 가중치 부여."""
    if groups is None:
        groups = np.zeros(len(y_DRP), dtype=int)
    n = len(y_DRP)
    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(n)
    gems_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    y_DRP_res = np.log1p(np.maximum(y_DRP.values, 0)) - np.log1p(np.maximum(gems_safe, 0)) if residual_mode else np.log1p(np.maximum(y_DRP.values, 0))
    if CV_MODE == "random":
        _dro_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        _dro_splitter = GroupKFold(n_splits=n_splits)
    cluster_rmse = {}
    _dro_split_args = (X_DRP, y_DRP, groups) if CV_MODE == "spatial" else (X_DRP, y_DRP)
    for fold, (train_idx, test_idx) in enumerate(_dro_splitter.split(*_dro_split_args)):
        X_tr = X_DRP.iloc[train_idx].fillna(X_DRP.median())
        X_te = X_DRP.iloc[test_idx].fillna(X_DRP.median())
        if use_ta_ec:
            oof_ta = _oof_predictions_same_splits(X_TA, y_TA, groups, n_splits)
            oof_ec = _oof_predictions_same_splits(X_EC, y_EC, groups, n_splits)
            X_tr = X_tr.copy()
            X_tr["pred_TA"] = oof_ta[train_idx]
            X_tr["pred_EC"] = oof_ec[train_idx]
            X_ta_te = X_TA.iloc[test_idx].fillna(X_TA.median())
            X_ec_te = X_EC.iloc[test_idx].fillna(X_EC.median())
            scaler_ta = StandardScaler()
            scaler_ec = StandardScaler()
            m_ta = train_model(scaler_ta.fit_transform(X_TA.iloc[train_idx].fillna(X_TA.median())), y_TA.iloc[train_idx])
            m_ec = train_model(scaler_ec.fit_transform(X_EC.iloc[train_idx].fillna(X_EC.median())), y_EC.iloc[train_idx])
            X_te = X_te.copy()
            X_te["pred_TA"] = m_ta.predict(scaler_ta.transform(X_ta_te))
            X_te["pred_EC"] = m_ec.predict(scaler_ec.transform(X_ec_te))
        scaler_d = StandardScaler()
        m = train_model(scaler_d.fit_transform(X_tr), pd.Series(y_DRP_res).iloc[train_idx])
        pred_te = m.predict(scaler_d.transform(X_te))
        if residual_mode:
            pred_drp = np.maximum(np.expm1(np.log1p(np.maximum(gems_safe[test_idx], 0)) + pred_te), 0)
        else:
            pred_drp = np.maximum(np.expm1(pred_te), 0)
        for i, idx in enumerate(test_idx):
            g = groups[idx]
            if g not in cluster_rmse:
                cluster_rmse[g] = []
            cluster_rmse[g].append((y_DRP.values[idx] - pred_drp[i]) ** 2)
    mean_rmse_per_cluster = {g: np.sqrt(np.mean(sq)) for g, sq in cluster_rmse.items()}
    worst_clusters = sorted(mean_rmse_per_cluster.keys(), key=lambda g: mean_rmse_per_cluster[g], reverse=True)[:n_worst]
    dro_weight = np.ones(n, dtype=np.float64)
    for i in range(n):
        if groups[i] in worst_clusters:
            dro_weight[i] = worst_cluster_weight
    sample_weight = dro_weight * base_sample_weight if base_sample_weight is not None else dro_weight
    return run_pipeline_drp_cv(
        X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
        model_TA, model_EC, scaler_TA, scaler_EC,
        n_splits=n_splits, residual_mode=residual_mode, log_residual=log_residual, use_ta_ec=use_ta_ec,
        sample_weight=sample_weight,
        distance_km=distance_km, decay_km=decay_km, has_prior=has_prior, ta_ec_blend=ta_ec_blend,
    )


def _split_baseline_event_cols(cols, baseline_patterns, event_patterns):
    baseline_cols = [c for c in cols if any(p in c for p in baseline_patterns)]
    event_cols = [c for c in cols if any(p in c for p in event_patterns)]
    return baseline_cols, event_cols


def run_pipeline_drp_regime_switch_cv(
    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
    model_TA, model_EC, scaler_TA, scaler_EC,
    baseline_cols, event_cols,
    n_splits=5, use_ta_ec=True,
):
    """1순위: Head A(baseline) + Head B(event excess), 최종 baseline + ReLU(excess)."""
    if not baseline_cols or not event_cols:
        return None
    baseline_cols = [c for c in baseline_cols if c in X_DRP.columns]
    event_cols = [c for c in event_cols if c in X_DRP.columns]
    if not baseline_cols or not event_cols:
        return None
    if groups is None:
        groups = np.zeros(len(y_DRP), dtype=int)
    if CV_MODE == "random":
        _rs_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        _rs_splitter = GroupKFold(n_splits=n_splits)
    n = len(y_DRP)
    y_raw = np.maximum(y_DRP.values.astype(np.float64), 0)
    if use_ta_ec:
        oof_ta = _oof_predictions_same_splits(X_TA, y_TA, groups, n_splits)
        oof_ec = _oof_predictions_same_splits(X_EC, y_EC, groups, n_splits)
    r2_list = []
    _rs_split_args = (X_DRP, y_DRP, groups) if CV_MODE == "spatial" else (X_DRP, y_DRP)
    for fold, (train_idx, test_idx) in enumerate(_rs_splitter.split(*_rs_split_args)):
        X_b_tr = X_DRP[baseline_cols].iloc[train_idx].fillna(X_DRP[baseline_cols].iloc[train_idx].median())
        X_b_te = X_DRP[baseline_cols].iloc[test_idx].fillna(X_DRP[baseline_cols].iloc[train_idx].median())
        X_e_tr = X_DRP[event_cols].iloc[train_idx].fillna(X_DRP[event_cols].iloc[train_idx].median())
        X_e_te = X_DRP[event_cols].iloc[test_idx].fillna(X_DRP[event_cols].iloc[train_idx].median())
        if use_ta_ec:
            X_b_tr = X_b_tr.copy()
            X_b_te = X_b_te.copy()
            X_e_tr = X_e_tr.copy()
            X_e_te = X_e_te.copy()
            X_b_tr["pred_TA"] = oof_ta[train_idx]
            X_b_tr["pred_EC"] = oof_ec[train_idx]
            X_e_tr["pred_TA"] = oof_ta[train_idx]
            X_e_tr["pred_EC"] = oof_ec[train_idx]
            X_ta_tr = X_TA.iloc[train_idx].fillna(X_TA.median())
            X_ta_te = X_TA.iloc[test_idx].fillna(X_TA.median())
            X_ec_tr = X_EC.iloc[train_idx].fillna(X_EC.median())
            X_ec_te = X_EC.iloc[test_idx].fillna(X_EC.median())
            scaler_ta = StandardScaler()
            scaler_ec = StandardScaler()
            m_ta = train_model(scaler_ta.fit_transform(X_ta_tr), y_TA.iloc[train_idx])
            m_ec = train_model(scaler_ec.fit_transform(X_ec_tr), y_EC.iloc[train_idx])
            pred_ta_te = m_ta.predict(scaler_ta.transform(X_ta_te))
            pred_ec_te = m_ec.predict(scaler_ec.transform(X_ec_te))
            X_b_te["pred_TA"] = pred_ta_te
            X_b_te["pred_EC"] = pred_ec_te
            X_e_te["pred_TA"] = pred_ta_te
            X_e_te["pred_EC"] = pred_ec_te
        scaler_b = StandardScaler()
        scaler_e = StandardScaler()
        m_b = train_model(scaler_b.fit_transform(X_b_tr), np.log1p(y_raw[train_idx]))
        pred_b_tr = np.expm1(m_b.predict(scaler_b.transform(X_b_tr)))
        excess_tr = np.maximum(y_raw[train_idx] - pred_b_tr, 0.0)  # Head B: 양의 excess만 학습 (안정화)
        m_e = train_model(scaler_e.fit_transform(X_e_tr), excess_tr)
        pred_b_te = np.clip(np.expm1(m_b.predict(scaler_b.transform(X_b_te))), 0, None)
        pred_e_te = np.clip(m_e.predict(scaler_e.transform(X_e_te)), 0, np.percentile(y_raw, 99) + 1e-6)
        pred_drp = np.maximum(pred_b_te + pred_e_te, 0)
        r2_list.append(r2_score(y_DRP.iloc[test_idx], pred_drp))
    r2_avg = np.mean(r2_list)
    X_b_full = X_DRP[baseline_cols].fillna(X_DRP[baseline_cols].median())
    X_e_full = X_DRP[event_cols].fillna(X_DRP[event_cols].median())
    if use_ta_ec:
        X_b_full = X_b_full.copy()
        X_e_full = X_e_full.copy()
        X_b_full["pred_TA"] = model_TA.predict(scaler_TA.transform(X_TA.fillna(X_TA.median())))
        X_b_full["pred_EC"] = model_EC.predict(scaler_EC.transform(X_EC.fillna(X_EC.median())))
        X_e_full["pred_TA"] = X_b_full["pred_TA"].values
        X_e_full["pred_EC"] = X_b_full["pred_EC"].values
    scaler_b = StandardScaler()
    scaler_e = StandardScaler()
    m_b = train_model(scaler_b.fit_transform(X_b_full), np.log1p(y_raw))
    pred_b_full = np.expm1(m_b.predict(scaler_b.transform(X_b_full)))
    excess_full = np.maximum(y_raw - pred_b_full, 0.0)
    m_e = train_model(scaler_e.fit_transform(X_e_full), excess_full)
    results = pd.DataFrame([{"Parameter": "Dissolved Reactive Phosphorus", "R2_Train": np.nan, "RMSE_Train": np.nan, "R2_Test": r2_avg, "RMSE_Test": np.nan}])
    regime_tuple = (m_b, m_e, scaler_b, scaler_e, baseline_cols, event_cols)
    return (regime_tuple, None, results, False, True)


def print_feature_importance_per_target(model_TA, model_EC, model_DRP, feature_names):
    """TA/EC/DRP 모델별 피처 중요도 출력 (어떤 피처를 각 모델에 쓸지 참고)."""
    if not hasattr(model_TA, "feature_importances_"):
        return
    names = list(feature_names)
    imp_TA = model_TA.feature_importances_
    imp_EC = model_EC.feature_importances_
    imp_DRP = model_DRP.feature_importances_
    df = pd.DataFrame({
        "feature": names,
        "TA_importance": imp_TA,
        "EC_importance": imp_EC,
        "DRP_importance": imp_DRP,
    })
    df = df.sort_values("TA_importance", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 60)
    print("Feature importance (current models, TA/EC/DRP)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\n  TA  top features:", ", ".join(df.nlargest(3, "TA_importance")["feature"].tolist()))
    print("  EC  top features:", ", ".join(df.nlargest(3, "EC_importance")["feature"].tolist()))
    print("  DRP top features:", ", ".join(df.nlargest(3, "DRP_importance")["feature"].tolist()))
    print("=" * 60)


def print_feature_correlation_per_target(wq_data, target_cols):
    """데이터 내 모든 숫자형 피처 후보와 TA/EC/DRP 상관관계 (추가할 피처 참고)."""
    numeric = wq_data.select_dtypes(include=[np.number])
    candidate_cols = [c for c in numeric.columns if c not in target_cols]
    if not candidate_cols:
        return
    print("\n" + "=" * 60)
    print("Feature vs target correlation (candidates to add per model)")
    print("=" * 60)
    for target in target_cols:
        if target not in wq_data.columns:
            continue
        corr = wq_data[candidate_cols + [target]].corr()[target].drop(target, errors="ignore")
        order = corr.abs().sort_values(ascending=False).index
        corr = corr.reindex(order)
        short = target.replace("Total Alkalinity", "TA").replace("Electrical Conductance", "EC").replace("Dissolved Reactive Phosphorus", "DRP")
        print(f"\n  [{short}] |corr| high -> low:")
        for f in corr.index[:10]:
            print(f"    {f}: {corr[f]:+.3f}")
    print("=" * 60)


def main():
    global ALLOWED_TIMESERIES_FEATURES
    print("=" * 60)
    print("Benchmark (Notebook 기본 버전)")
    print("  모델:", "XGBoost" if USE_XGB else "RandomForest (xgboost 미설치)")
    print("=" * 60)

    # Validation 시계열: 결측이어도 train 중앙값으로 채울지 여부
    if USE_EXTRA_FEATURES and (DATA_DIR / "val_with_hyriv_era5_events.csv").exists():
        if USE_TIMESERIES_IMPUTE_FROM_TRAIN:
            val_head = pd.read_csv(DATA_DIR / "val_with_hyriv_era5_events.csv", nrows=1)
            ALLOWED_TIMESERIES_FEATURES = [
                c for c in val_head.columns
                if any(p in c for p in TIMESERIES_PATTERNS)
                and not any(ex in c for ex in TIMESERIES_EXCLUDE_PATTERNS)
            ]
            mode_str = "spatial K-NN" if TIMESERIES_IMPUTE_MODE == "spatial" else "전역 train 중앙값"
            print(f"\n  시계열 피처: validation 전부 결측 → {mode_str}으로 impute (정보없는 시계열 제외, 허용 {len(ALLOWED_TIMESERIES_FEATURES)}개)")
        else:
            ALLOWED_TIMESERIES_FEATURES = check_validation_timeseries_quality(DATA_DIR)
    else:
        ALLOWED_TIMESERIES_FEATURES = []

    target_cols = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    use_cv = USE_EXPERIMENT_STYLE and HAS_EXPERIMENT_LOADER

    if use_cv:
        # TA/EC = full_gems 유지. DRP = 별도 트랙 → full_gems / weak_gems / no_gems / simple 4종 비교
        wq_data = load_baseline_data()
        if USE_SOILGRIDS and (DATA_DIR / "external_soilgrids.csv").exists():
            soil = pd.read_csv(DATA_DIR / "external_soilgrids.csv")
            add_s = [c for c in soil.columns if c not in ("Latitude", "Longitude") and c not in wq_data.columns]
            if add_s:
                wq_data = wq_data.merge(
                    soil[["Latitude", "Longitude"] + add_s].drop_duplicates(subset=["Latitude", "Longitude"], keep="first"),
                    on=["Latitude", "Longitude"], how="left",
                )
                print(f"  SoilGrids: external_soilgrids merged ({len(add_s)} cols)")
        else:
            if USE_SOILGRIDS and not (DATA_DIR / "external_soilgrids.csv").exists():
                print("  SoilGrids: external_soilgrids.csv 없음 → 스킵 (fetch_external_data.py 실행 필요)")
        wq_data = add_ta_ec_derived_features(wq_data)
        wq_data = add_ec_region_features(wq_data)
        if USE_EC_FOCUS_TRAINING == "filter":
            ec_mask = is_eastern_cape_mask(wq_data)
            wq_data = wq_data.loc[ec_mask].reset_index(drop=True)
            print(f"  이스턴케이프만 학습: {len(wq_data)}행 (EC 필터)")
        feat_TA, feat_EC, _ = get_experiment_features(wq_data, "full_gems")
        feat_TA = [c for c in feat_TA if c in wq_data.columns and c not in target_cols and not is_unsafe_feature(c)]
        feat_EC = [c for c in feat_EC if c in wq_data.columns and c not in target_cols and not is_unsafe_feature(c)]
        if USE_TA_DERIVED_FEATURES:
            _ta_extra = [c for c in TA_DERIVED_FEATURES if c in wq_data.columns and c not in feat_TA]
            feat_TA = feat_TA + _ta_extra
        if USE_TA_EC_CLUSTER_FEATURE and "cluster_id" in wq_data.columns:
            if "cluster_id" not in feat_TA:
                feat_TA = list(feat_TA) + ["cluster_id"]
            if "cluster_id" not in feat_EC:
                feat_EC = list(feat_EC) + ["cluster_id"]
        if USE_EC_REGION_FEATURE and "is_eastern_cape" in wq_data.columns:
            if "is_eastern_cape" not in feat_TA:
                feat_TA = list(feat_TA) + ["is_eastern_cape"]
            if "is_eastern_cape" not in feat_EC:
                feat_EC = list(feat_EC) + ["is_eastern_cape"]
        _ec_derived = [c for c in EC_DERIVED_FEATURES if c in wq_data.columns and c not in feat_EC]
        if _ec_derived:
            feat_EC = list(feat_EC) + _ec_derived
        if USE_DRP_HYDRO_FEATURES:
            wq_data = add_drp_derived_features(wq_data)
        drp_variants = get_drp_feature_variants(wq_data, target_cols)
        feat_DRP = (drp_variants.get("drp_compact_plus_20a_bare") or []) if drp_variants else []
        feature_cols = list(dict.fromkeys(feat_TA + feat_EC + feat_DRP))
        print("  토양: train_with_hyriv의 soil_clay_pct, soil_organic_carbon, soil_ph 사용")
        print("  데이터: load_baseline_data (파생 피처 포함)")
        print("  Submission-safe: missing + validation 거의 상수 + population_density 제외")
        print(f"  TA/EC: full_gems 유지 (TA={len(feat_TA)}, EC={len(feat_EC)}). DRP: 20a_bare + 5 variants 비교.")
        n_splits_cv = N_FOLDS
        drp_km_model = None
        if USE_CLUSTER_HOLDOUT:
            if USE_EC_CENTERED_CV and CV_MODE == "spatial":
                groups = get_cluster_groups_ec_centered(wq_data, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
                wq_data = wq_data.copy()
                wq_data["cluster_id"] = groups.astype(np.float32)
                n_grp = len(np.unique(groups))
                n_splits_cv = min(N_FOLDS, n_grp)
                n_ec = (groups == 0).sum()
                print(f"  CV mode: EC-centered (group 0=Eastern Cape, n_EC={n_ec}, groups={n_grp}), GroupKFold(n_splits={n_splits_cv})")
            else:
                if USE_DRP_CLUSTER_FEATURE:
                    groups, drp_km_model = get_cluster_groups(wq_data, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, return_model=True)
                    wq_data = wq_data.copy()
                    wq_data["cluster_id"] = groups.astype(np.float32)
                else:
                    groups = get_cluster_groups(wq_data, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
                n_grp = len(np.unique(groups))
                n_splits_cv = min(N_FOLDS, n_grp)
                if CV_MODE == "random":
                    print(f"  CV mode: RANDOM KFold(n_splits={n_splits_cv}, shuffle=True)")
                else:
                    print(f"  CV mode: SPATIAL cluster holdout (KMeans n_clusters={N_CLUSTERS}), GroupKFold(n_splits={n_splits_cv})")
        else:
            groups = get_spatial_block_groups(wq_data, SPATIAL_BLOCK_DEG)
            if CV_MODE == "random":
                print(f"  CV mode: RANDOM KFold(n_splits={n_splits_cv}, shuffle=True)")
            else:
                print(f"  CV mode: SPATIAL block holdout, blocks={len(np.unique(groups))}, GroupKFold(n_splits={n_splits_cv})")
        wq_full_for_corr = wq_data.select_dtypes(include=[np.number]).copy()
    else:
        # 기존: 수동 병합 + BASE+GEMS만
        wq_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
        if USE_GEMS and wq_path.exists():
            water_quality = pd.read_csv(wq_path)
            print("  Water Quality: enriched (GEMS 포함)")
        else:
            water_quality = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
            print("  Water Quality: base (enriched 없음 또는 USE_GEMS=False)")
        landsat_train = pd.read_csv(DATA_DIR / "landsat_features_training.csv")
        terra_train = pd.read_csv(DATA_DIR / "terraclimate_features_training.csv")
        wq_data = combine_two_datasets(water_quality, landsat_train, terra_train)

        if USE_ERA5_PRECIP and (DATA_DIR / "precipitation_training.csv").exists():
            pr_df = pd.read_csv(DATA_DIR / "precipitation_training.csv")
            if all(k in wq_data.columns and k in pr_df.columns for k in KEY_COLS):
                add_pr = [c for c in pr_df.columns if c not in KEY_COLS and c not in wq_data.columns]
                if add_pr:
                    wq_data = wq_data.merge(pr_df[KEY_COLS + add_pr], on=KEY_COLS, how="left")
                    print("  ERA5 precip: merged")
        if USE_PRECIP_ANOMALY and (DATA_DIR / "water_quality_with_precip_anomaly.csv").exists():
            pa_df = pd.read_csv(DATA_DIR / "water_quality_with_precip_anomaly.csv")
            extra = [c for c in pa_df.columns if c not in wq_data.columns and c not in KEY_COLS]
            if extra and all(k in wq_data.columns and k in pa_df.columns for k in KEY_COLS):
                wq_data = wq_data.merge(pa_df[KEY_COLS + extra], on=KEY_COLS, how="left")
                print("  Precip anomaly: merged")
        if USE_EXTRA_FEATURES and (DATA_DIR / "train_with_hyriv_era5_events.csv").exists():
            tw = pd.read_csv(DATA_DIR / "train_with_hyriv_era5_events.csv")
            skip = set(KEY_COLS) | set(target_cols)
            to_add = [c for c in tw.columns if c not in wq_data.columns and c not in skip]
            if to_add and all(k in wq_data.columns and k in tw.columns for k in KEY_COLS):
                tw_sub = tw[KEY_COLS + to_add].drop_duplicates(subset=KEY_COLS, keep="first")
                wq_data = wq_data.merge(tw_sub, on=KEY_COLS, how="left")
                print(f"  Extra (HydroRIVERS+ERA5): {len(to_add)} cols merged")
        if USE_SOILGRIDS and (DATA_DIR / "external_soilgrids.csv").exists():
            soil = pd.read_csv(DATA_DIR / "external_soilgrids.csv")
            add_s = [c for c in soil.columns if c not in ("Latitude", "Longitude") and c not in wq_data.columns]
            if add_s:
                wq_data = wq_data.merge(
                    soil[["Latitude", "Longitude"] + add_s].drop_duplicates(subset=["Latitude", "Longitude"], keep="first"),
                    on=["Latitude", "Longitude"], how="left",
                )
                print(f"  SoilGrids: external_soilgrids merged ({len(add_s)} cols)")
        else:
            if USE_SOILGRIDS and not (DATA_DIR / "external_soilgrids.csv").exists():
                print("  SoilGrids: external_soilgrids.csv 없음 → 스킵")
        wq_data = add_seasonality_features(wq_data)
        wq_data = add_wetness_features(wq_data)
        if "gems_TP" in wq_data.columns and "gems_DRP" in wq_data.columns and "gems_partial_P" not in wq_data.columns:
            wq_data["gems_partial_P"] = np.maximum(wq_data["gems_TP"].fillna(0).values - wq_data["gems_DRP"].fillna(0).values, 0)
        wq_data = wq_data.fillna(wq_data.median(numeric_only=True))
        wq_full_for_corr = wq_data.select_dtypes(include=[np.number]).copy()

        all_candidates = list(BASE_FEATURES) + list(GEMS_TA) + list(GEMS_EC) + list(GEMS_DRP)
        feature_cols = [c for c in all_candidates if c in wq_data.columns]
        feature_cols = list(dict.fromkeys(feature_cols))
        for k in ["Latitude", "Longitude"]:
            if k in wq_data.columns and k not in feature_cols:
                feature_cols.append(k)
        keep = [c for c in (feature_cols + target_cols) if c in wq_data.columns]
        wq_data = wq_data[keep]
        feature_cols = [c for c in feature_cols if c in wq_data.columns and c not in target_cols]
        print(f"  토양: train_with_hyriv의 soil_clay_pct, soil_organic_carbon, soil_ph 사용")
        print(f"  Feature 수: {len(feature_cols)} (BASE+seasonality+wetness+GEMS/extra)")

        if "Latitude" in wq_data.columns and "Longitude" in wq_data.columns:
            groups = get_spatial_block_groups(wq_data, SPATIAL_BLOCK_DEG)
            n_blocks = len(np.unique(groups))
            print(f"  공간 분할: block_deg={SPATIAL_BLOCK_DEG}, 블록 수={n_blocks}, test_size={TEST_SIZE}")
        else:
            groups = None
            print("  공간 분할: Lat/Lon 없음 -> 랜덤 분할")
        feat_TA = feat_EC = feat_DRP = feature_cols

    # --- 모델 학습 ---
    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]
    EC_log_target = False  # use_cv이고 USE_EC_LOG_TARGET일 때만 True로 갱신

    if use_cv:
        ec_sample_weight = None
        if USE_EC_FOCUS_TRAINING == "weight" and "is_eastern_cape" in wq_data.columns:
            ec_mask = is_eastern_cape_mask(wq_data)
            ec_sample_weight = np.where(ec_mask, float(EC_SAMPLE_WEIGHT), 1.0).astype(np.float64)
            print(f"  이스턴케이프 가중치: EC 샘플에 {EC_SAMPLE_WEIGHT}x (n_EC={ec_mask.sum()})")
        print("\n  CV (region cluster holdout). TA/EC: full_gems 1회. DRP: drp_compact_plus_20a_bare_partial_p 고정.")
        X_TA = wq_data[feat_TA].copy()
        X_EC = wq_data[feat_EC].copy()
        _out_TA = run_pipeline_cv(X_TA, y_TA, "Total Alkalinity", groups=groups, n_splits=n_splits_cv, sample_weight=ec_sample_weight)
        _out_EC = run_pipeline_cv(X_EC, y_EC, "Electrical Conductance", groups=groups, n_splits=n_splits_cv, xgb_override=EC_XGB_OVERRIDE, log_target=USE_EC_LOG_TARGET, sample_weight=ec_sample_weight)
        if USE_EC_CENTERED_CV and getattr(run_pipeline_cv, "_ec_fold_r2", None):
            print("  [진단] EC holdout fold R² (LB와 유사):", run_pipeline_cv._ec_fold_r2)
            del run_pipeline_cv._ec_fold_r2
        model_TA, scaler_TA, results_TA = _out_TA[0], _out_TA[1], _out_TA[2]
        model_EC, scaler_EC, results_EC = _out_EC[0], _out_EC[1], _out_EC[2]
        y_TA_mean_cv = _out_TA[3] if len(_out_TA) > 3 and _out_TA[3] is not None else 0.0
        y_EC_mean_cv = _out_EC[3] if len(_out_EC) > 3 and _out_EC[3] is not None else 0.0
        EC_log_target = _out_EC[4] if len(_out_EC) > 4 else False

        _distance_km = wq_data["gems_distance_km"] if "gems_distance_km" in wq_data.columns else None
        _decay_km = PRIOR_DECAY_KM if USE_DRP_DISTANCE_PRIOR else None
        if USE_DRP_PRIOR_SPLIT:
            feat_DRP = [c for c in list(dict.fromkeys(PRIOR_ONLY_FEATURES + NO_PRIOR_FEATURES)) if c in wq_data.columns and not is_unsafe_feature(c, True)]
            if not feat_DRP and drp_variants:
                feat_DRP = drp_variants.get(DRP_FIXED_VARIANT) or next((v for v in (drp_variants or {}).values() if v), [])
        else:
            feat_DRP = (drp_variants.get(DRP_FIXED_VARIANT) or []) if drp_variants else []
            if not feat_DRP and drp_variants:
                feat_DRP = next((v for v in drp_variants.values() if v), [])
        if USE_DRP_HYDRO_FEATURES:
            feat_DRP = get_drp_expanded_features(feat_DRP, wq_data)
            print(f"  DRP 확장 피처: {len(feat_DRP)}개 (HydroRIVERS+spectral+derived 추가)")
        if USE_DRP_CLUSTER_FEATURE and "cluster_id" in wq_data.columns:
            feat_DRP = list(feat_DRP) + ["cluster_id"]
        if USE_EC_REGION_FEATURE and "is_eastern_cape" in wq_data.columns and "is_eastern_cape" not in feat_DRP:
            feat_DRP = list(feat_DRP) + ["is_eastern_cape"]
        X_DRP = wq_data[feat_DRP].copy() if feat_DRP else wq_data[feat_TA].copy()
        _has_prior = get_has_prior(wq_data) if (USE_DRP_PRIOR_SPLIT and not DRP_BEST_MODE) else None
        _drp_sample_weight = None
        if USE_DRP_HIGH_WEIGHT:
            y_drp_vals = np.asarray(y_DRP, dtype=np.float64)
            pos = y_drp_vals > 0
            q90 = np.percentile(y_drp_vals[pos], 90) if pos.sum() > 0 else 1.0
            _drp_sample_weight = 1.0 + np.clip(y_drp_vals / max(q90, 1e-6), 0, 1)
        if ec_sample_weight is not None and _drp_sample_weight is not None:
            _drp_sample_weight = _drp_sample_weight * ec_sample_weight
        elif ec_sample_weight is not None:
            _drp_sample_weight = ec_sample_weight
        _use_regime_switch = USE_DRP_REGIME_SWITCH and not DRP_BEST_MODE
        _use_group_dro = USE_DRP_GROUP_DRO

        if _use_regime_switch:
            baseline_cols, event_cols = _split_baseline_event_cols(feat_DRP, DRP_BASELINE_PATTERNS, DRP_EVENT_PATTERNS)
            _drp_out = run_pipeline_drp_regime_switch_cv(
                X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                model_TA, model_EC, scaler_TA, scaler_EC,
                baseline_cols, event_cols, n_splits=n_splits_cv, use_ta_ec=USE_TA_EC_FOR_DRP,
            )
            if _drp_out is not None:
                regime_tuple, _, results_DRP, drp_residual_mode, drp_log_residual = _drp_out
                model_DRP = regime_tuple
                scaler_DRP = regime_tuple[2]  # scaler_b for compatibility
                drp_decay_km = None
                drp_prior_split = False
                drp_use_ta_ec = USE_TA_EC_FOR_DRP
                drp_ta_ec_blend = TA_EC_BLEND_DRP
                print(f"\n  DRP regime-switch (Head A/B): R2_Test={results_DRP.loc[0, 'R2_Test']:.4f}")
            else:
                _use_regime_switch = False  # fallback to standard
                _drp_out = run_pipeline_drp_cv(
                    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                    model_TA, model_EC, scaler_TA, scaler_EC,
                    n_splits=n_splits_cv, residual_mode=DRP_RESIDUAL_MODE, log_residual=DRP_LOG_RESIDUAL, use_ta_ec=USE_TA_EC_FOR_DRP,
                    distance_km=_distance_km, decay_km=_decay_km, has_prior=_has_prior,
                    ta_ec_blend=TA_EC_BLEND_DRP, sample_weight=_drp_sample_weight,
                )
                drp_prior_split = len(_drp_out) == 10
                drp_gems_blend_alpha = 1.0
                if drp_prior_split:
                    model_DRP = (_drp_out[0], _drp_out[1], _drp_out[2], _drp_out[3], _drp_out[8], _drp_out[9])
                    scaler_DRP = _drp_out[2]
                    results_DRP = _drp_out[4]
                    drp_residual_mode, drp_log_residual = _drp_out[5], _drp_out[6]
                    drp_decay_km = _drp_out[7] if len(_drp_out) > 7 else None
                else:
                    model_DRP, scaler_DRP, results_DRP = _drp_out[0], _drp_out[1], _drp_out[2]
                    drp_residual_mode, drp_log_residual = _drp_out[3], _drp_out[4]
                    drp_decay_km = _drp_out[5] if len(_drp_out) > 5 else None
                    drp_gems_blend_alpha = _drp_out[6] if len(_drp_out) > 6 else 1.0
                drp_use_ta_ec = USE_TA_EC_FOR_DRP
                drp_ta_ec_blend = TA_EC_BLEND_DRP
        else:
            if _use_group_dro:
                _drp_out = run_pipeline_drp_dro_cv(
                    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                    model_TA, model_EC, scaler_TA, scaler_EC,
                    n_splits=n_splits_cv, residual_mode=DRP_RESIDUAL_MODE, log_residual=DRP_LOG_RESIDUAL, use_ta_ec=USE_TA_EC_FOR_DRP,
                    worst_cluster_weight=DRP_DRO_WORST_WEIGHT, n_worst=DRP_DRO_N_WORST,
                    distance_km=_distance_km, decay_km=_decay_km, has_prior=_has_prior, ta_ec_blend=TA_EC_BLEND_DRP,
                    base_sample_weight=_drp_sample_weight,
                )
            else:
                _drp_out = run_pipeline_drp_cv(
                    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                    model_TA, model_EC, scaler_TA, scaler_EC,
                    n_splits=n_splits_cv, residual_mode=DRP_RESIDUAL_MODE, log_residual=DRP_LOG_RESIDUAL, use_ta_ec=USE_TA_EC_FOR_DRP,
                    distance_km=_distance_km, decay_km=_decay_km, has_prior=_has_prior,
                    ta_ec_blend=TA_EC_BLEND_DRP, sample_weight=_drp_sample_weight,
                )
            drp_prior_split = len(_drp_out) == 10
            drp_gems_blend_alpha = 1.0
            if drp_prior_split:
                model_DRP = (_drp_out[0], _drp_out[1], _drp_out[2], _drp_out[3], _drp_out[8], _drp_out[9])
                scaler_DRP = _drp_out[2]
                results_DRP = _drp_out[4]
                drp_residual_mode, drp_log_residual = _drp_out[5], _drp_out[6]
                drp_decay_km = _drp_out[7] if len(_drp_out) > 7 else None
            else:
                model_DRP, scaler_DRP, results_DRP = _drp_out[0], _drp_out[1], _drp_out[2]
                drp_residual_mode, drp_log_residual = _drp_out[3], _drp_out[4]
                drp_decay_km = _drp_out[5] if len(_drp_out) > 5 else None
                drp_gems_blend_alpha = _drp_out[6] if len(_drp_out) > 6 else 1.0
            drp_use_ta_ec = USE_TA_EC_FOR_DRP
        drp_ta_ec_blend = TA_EC_BLEND_DRP
        drp_regime_switch = (not drp_prior_split and isinstance(model_DRP, tuple) and len(model_DRP) == 6
                            and isinstance(model_DRP[4], (list, tuple)))
        extras = []
        if DRP_BEST_MODE:
            extras.append("single-pipeline")
        if drp_regime_switch:
            extras.append("regime-switch")
        if _use_group_dro and not drp_regime_switch:
            extras.append("group-dro")
        if USE_DRP_HYDRO_FEATURES:
            extras.append("hydro+spectral피처")
        if USE_DRP_STACKING:
            extras.append("stacking")
        if USE_DRP_REGION_CALIBRATION:
            extras.append("region-cal")
        extra_str = " [" + "+".join(extras) + "]" if extras else ""
        if not drp_regime_switch:
            print(f"\n  DRP {DRP_FIXED_VARIANT}: R2_Test={results_DRP.loc[0, 'R2_Test']:.4f}{extra_str}" + (" (prior분리)" if drp_prior_split else ""))

        # Region calibration: OOF DRP 예측으로 cluster bias 계산 (제출 시 적용)
        drp_cluster_bias = None
        if USE_DRP_REGION_CALIBRATION and not drp_prior_split and not drp_regime_switch and groups is not None:
            X_DRP_full = X_DRP.copy().fillna(X_DRP.median())
            if drp_use_ta_ec:
                X_DRP_full["pred_TA"] = model_TA.predict(scaler_TA.transform(wq_data[feat_TA].fillna(wq_data[feat_TA].median())))
                X_DRP_full["pred_EC"] = model_EC.predict(scaler_EC.transform(wq_data[feat_EC].fillna(wq_data[feat_EC].median())))
            X_DRP_full_s = scaler_DRP.transform(X_DRP_full)
            oof_drp_raw = model_DRP.predict(X_DRP_full_s)
            gems_DRP_vals = wq_data["gems_DRP"].fillna(0).values if "gems_DRP" in wq_data.columns else np.zeros(len(wq_data))
            gems_safe_vals = np.maximum(np.nan_to_num(gems_DRP_vals, nan=0.0), 0)
            _oof_ratio = DRP_RATIO_TARGET and drp_residual_mode
            if _oof_ratio:
                oof_drp_pred = np.maximum(oof_drp_raw * np.maximum(gems_safe_vals, DRP_RATIO_EPS), 0)
            elif drp_residual_mode and drp_log_residual:
                oof_drp_pred = np.maximum(np.expm1(np.log1p(np.maximum(gems_safe_vals, 0)) + oof_drp_raw), 0)
            elif drp_residual_mode:
                oof_drp_pred = np.maximum(gems_safe_vals + oof_drp_raw, 0)
            else:
                oof_drp_pred = np.maximum(np.expm1(oof_drp_raw) if drp_log_residual else oof_drp_raw, 0)
            drp_cluster_bias = compute_cluster_bias(y_DRP.values, oof_drp_pred, groups)
            bias_vals = list(drp_cluster_bias.values())
            print(f"  DRP region calibration: {len(drp_cluster_bias)} clusters, bias range=[{min(bias_vals):.4f}, {max(bias_vals):.4f}]")
    else:
        X = wq_data[feature_cols].copy()
        model_TA, scaler_TA, results_TA = run_pipeline(X, y_TA, "Total Alkalinity", groups=groups)
        model_EC, scaler_EC, results_EC = run_pipeline(X, y_EC, "Electrical Conductance", groups=groups)
        model_DRP, scaler_DRP, results_DRP = run_pipeline(X, y_DRP, "Dissolved Reactive Phosphorus", groups=groups)
        drp_residual_mode, drp_log_residual = False, False
        drp_use_ta_ec = False
        drp_ta_ec_blend = 1.0
        drp_prior_split = False
        drp_cluster_bias = None
        drp_gems_blend_alpha = 1.0
        y_TA_mean_cv = y_EC_mean_cv = 0.0
        drp_decay_km = None
        drp_km_model = None

    results_summary = pd.concat([results_TA, results_EC, results_DRP], ignore_index=True)
    r2_ta = results_summary.loc[0, "R2_Test"]
    r2_ec = results_summary.loc[1, "R2_Test"]
    r2_drp = results_summary.loc[2, "R2_Test"]
    avg_r2_lb_proxy = (r2_ta + r2_ec + r2_drp) / 3.0

    if use_cv:
        print("\n" + "=" * 60)
        print("결과 (harsh spatial split = region holdout)")
        print("=" * 60)
    print("\nResults Summary:")
    print(results_summary)
    out_summary = BASE_DIR / "results_summary.csv"
    results_summary.to_csv(out_summary, index=False)
    print(f"  (저장: {out_summary})")
    if use_cv:
        print("\n  --- LB proxy (standard) ---")
        if CV_MODE == "random":
            print("  Split: KFold random (LB-aligned; GEMS preserved in folds).")
        else:
            print("  Split: GroupKFold region/cluster holdout (동일 cluster 통째로 val).")
        print(f"  R2_TA={r2_ta:.4f}  R2_EC={r2_ec:.4f}  R2_DRP={r2_drp:.4f}  → Avg_R2(standard proxy)={avg_r2_lb_proxy:.6f}")
        print("  ---------------------------------")
    else:
        print(f"\n  Avg_R2(LB proxy): {avg_r2_lb_proxy:.6f}  (R2_TA + R2_EC + R2_DRP) / 3")
    print(f"  Actual LB R²:     {ACTUAL_LB_R2:.4f}  (리더보드 실제 점수, proxy와 차이 큼)")
    if use_cv:
        print("  (submission-safe 피처)")
        print("=" * 60)

    # GEMS-weak holdout: train=within_limit True 또는 distance≤median, val=그 반대 → 보수적 LB proxy
    if use_cv and ("gems_within_limit" in wq_data.columns or "gems_distance_km" in wq_data.columns):
        split = get_gems_quality_split(wq_data, min_test=20)
        if split is not None:
            train_idx, test_idx = split
            n_weak = len(test_idx)
            n_total = len(wq_data)
            ratio_weak = n_weak / n_total if n_total else 0
            if "gems_within_limit" in wq_data.columns:
                split_desc = "train = gems_within_limit True, val = gems_within_limit False"
            else:
                split_desc = "train = gems_distance_km ≤ median, val = gems_distance_km > median"
            gems_weak = run_gems_weak_holdout(wq_data, feat_TA, feat_EC, feat_DRP, y_TA, y_EC, y_DRP, residual_mode=drp_residual_mode, log_residual=drp_log_residual, use_ta_ec=drp_use_ta_ec)
            if gems_weak is not None:
                r2_ta_g, r2_ec_g, r2_drp_g = gems_weak
                proxy_weak = (r2_ta_g + r2_ec_g + r2_drp_g) / 3.0
                print("\n  --- LB proxy (GEMS-weak val) ---")
                print(f"  Split: {split_desc} (1회 학습·평가).")
                print(f"  GEMS-weak subset: n={n_weak} / {n_total}  ({ratio_weak:.1%})")
                print(f"  R2_TA={r2_ta_g:.4f}  R2_EC={r2_ec_g:.4f}  R2_DRP={r2_drp_g:.4f}  → Avg_R2(weak proxy)={proxy_weak:.4f}")
                print("  (실제 LB가 GEMS 약한 환경이면 위 수치가 더 가깝습니다.)")
                print("  ---------------------------------")

    # --- TA/EC/DRP별 어떤 피처를 쓸지 확인 (출력 비활성화) ---
    # print_feature_importance_per_target(model_TA, model_EC, model_DRP, feature_cols)
    # print_feature_correlation_per_target(wq_full_for_corr, target_cols)

    # --- 제출용 validation: val_with_hyriv merge + Landsat 공간 보간 + target별 피처 정렬 ---
    test_file = pd.read_csv(DATA_DIR / "submission_template.csv")
    landsat_val = pd.read_csv(DATA_DIR / "landsat_features_validation.csv")
    terra_val = pd.read_csv(DATA_DIR / "terraclimate_features_validation.csv")

    val_data = test_file[KEY_COLS].copy()
    val_data = val_data.merge(
        landsat_val[KEY_COLS + [c for c in LANDSAT_COLS if c in landsat_val.columns]],
        on=KEY_COLS, how="left"
    )
    val_data = val_data.merge(terra_val[KEY_COLS + ["pet"]], on=KEY_COLS, how="left")

    if USE_GEMS and (DATA_DIR / "gems_features_validation.csv").exists():
        gems_val = pd.read_csv(DATA_DIR / "gems_features_validation.csv")
        merge_on = [k for k in KEY_COLS if k in val_data.columns and k in gems_val.columns]
        if merge_on:
            extra = [c for c in gems_val.columns if c not in KEY_COLS and c not in val_data.columns]
            if extra:
                val_data = val_data.merge(gems_val[merge_on + extra].drop_duplicates(subset=merge_on, keep="first"), on=merge_on, how="left")
            gems_merge_method = "키 기준 merge (" + ", ".join(merge_on) + ")"
        else:
            for col in gems_val.columns:
                if col not in KEY_COLS and col not in val_data.columns:
                    val_data[col] = gems_val[col].values
            gems_merge_method = "행 순서로 결합 (키 없음)"
        if "gems_TP" in val_data.columns and "gems_DRP" in val_data.columns and "gems_partial_P" not in val_data.columns:
            val_data["gems_partial_P"] = np.maximum(val_data["gems_TP"].fillna(0).values - val_data["gems_DRP"].fillna(0).values, 0)
        print("  Validation: GEMS merged")
    else:
        gems_merge_method = "미사용"

    # Landsat 결측: 공간 기반 보간 (train 좌표·값 사용)
    landsat_cols_used = [c for c in LANDSAT_COLS if c in val_data.columns and c in wq_data.columns]
    if HAS_SCIPY and landsat_cols_used:
        train_landsat = wq_data[KEY_COLS + landsat_cols_used].copy()
        val_data = impute_landsat_spatial(val_data, train_landsat, landsat_cols_used)
        landsat_method = "spatial imputation (동일좌표→근처 val→근처 train→median)"
    else:
        if landsat_cols_used:
            val_data = val_data.fillna(val_data.median(numeric_only=True))
        landsat_method = "단순 fill (median 등)" if landsat_cols_used else "해당 없음"
    if landsat_cols_used:
        print(f"  Validation Landsat 결측: {landsat_method}")

    val_data = add_seasonality_features(val_data)
    val_data = add_wetness_features(val_data)
    if USE_DRP_HYDRO_FEATURES:
        val_data = add_drp_derived_features(val_data)

    need_cols_val = list(dict.fromkeys(feat_TA + feat_EC + feat_DRP))
    if USE_EXTRA_FEATURES and (DATA_DIR / "val_with_hyriv_era5_events.csv").exists():
        vw = pd.read_csv(DATA_DIR / "val_with_hyriv_era5_events.csv")
        need = [c for c in need_cols_val if c not in val_data.columns and c in vw.columns]
        merge_on = [k for k in KEY_COLS if k in val_data.columns and k in vw.columns]
        if need and merge_on:
            vw_sub = vw[merge_on + need].drop_duplicates(subset=merge_on, keep="first")
            val_data = val_data.merge(vw_sub, on=merge_on, how="left")
        print("  Validation: val_with_hyriv_era5_events merged (키: Lat, Lon, Sample Date)")

        # Static 피처: Sample Date 없이 Lat/Lon만으로 보충 (날짜 키면 validation 전부 miss 가능)
        static_cols = [c for c in vw.columns if c not in KEY_COLS and any(p in c for p in STATIC_COL_PATTERNS)]
        if static_cols and "Latitude" in vw.columns and "Longitude" in vw.columns:
            vw_static = vw[["Latitude", "Longitude"] + static_cols].drop_duplicates(subset=["Latitude", "Longitude"], keep="first")
            val_data = val_data.merge(vw_static, on=["Latitude", "Longitude"], how="left", suffixes=("", "_st"))
            for c in static_cols:
                if c + "_st" not in val_data.columns:
                    continue
                if c in val_data.columns:
                    val_data[c] = val_data[c].fillna(val_data[c + "_st"])
                else:
                    val_data[c] = val_data[c + "_st"]
                val_data.drop(columns=[c + "_st"], inplace=True)
            print("  Validation: static 피처 Lat/Lon 기준 보충 merge (soil, elevation, lc_* 등)")
    if USE_SOILGRIDS and (DATA_DIR / "external_soilgrids.csv").exists():
        soil = pd.read_csv(DATA_DIR / "external_soilgrids.csv")
        add_s = [c for c in soil.columns if c not in ("Latitude", "Longitude") and c not in val_data.columns]
        if add_s:
            val_data = val_data.merge(
                soil[["Latitude", "Longitude"] + add_s].drop_duplicates(subset=["Latitude", "Longitude"], keep="first"),
                on=["Latitude", "Longitude"], how="left",
            )
            print(f"  Validation: SoilGrids merged ({len(add_s)} cols)")

    val_data = add_ta_ec_derived_features(val_data)
    val_data = add_ec_region_features(val_data)

    val_cols = [c for c in need_cols_val if c in val_data.columns]
    val_missing = val_data[val_cols].isna().mean()
    val_missing_sorted = val_missing.sort_values(ascending=False)
    n_has_missing = (val_missing_sorted > 0).sum()
    lag_like = [c for c in need_cols_val if any(x in c.lower() for x in ["lag", "rolling", "cum", "mean_", "sum_", "prev", "spatial"])]
    val_missing_lag = val_data[[c for c in lag_like if c in val_data.columns]].isna().mean() if lag_like else pd.Series()

    print("\n  --- Validation 피처 결측률 (fillna 전) ---")
    if n_has_missing == 0:
        print("    결측 없음.")
    else:
        top = val_missing_sorted[val_missing_sorted > 0].head(25)
        for c in top.index:
            print(f"    {c}: {top[c]:.1%}")
        print(f"    (결측 있는 피처 수: {n_has_missing} / {len(val_cols)})")
    print("  -----------------------------------------")

    if USE_EC_CENTERED_CV and "cluster_id" in need_cols_val:
        val_data = val_data.copy()
        val_data["cluster_id"] = 0.0
    elif USE_DRP_CLUSTER_FEATURE and drp_km_model is not None and "cluster_id" in need_cols_val and "Latitude" in val_data.columns and "Longitude" in val_data.columns:
        val_data = val_data.copy()
        val_xy = val_data[["Latitude", "Longitude"]].values
        val_data["cluster_id"] = drp_km_model.predict(val_xy).astype(np.float32)
    # validation 시계열 전부 결측 → train으로 채움 (median=전역 중앙값, spatial=가까운 train K-NN 중앙값)
    if USE_TIMESERIES_IMPUTE_FROM_TRAIN and ALLOWED_TIMESERIES_FEATURES:
        if TIMESERIES_IMPUTE_MODE == "spatial" and HAS_SCIPY:
            val_data = impute_val_timeseries_from_train(
                val_data, wq_data,
                [c for c in ALLOWED_TIMESERIES_FEATURES if c in val_data.columns and c in wq_data.columns],
                k=TIMESERIES_IMPUTE_K_NEIGHBORS,
            )
            print("  Validation 시계열: spatial K-NN impute (train 이웃 중앙값) 적용")
        else:
            for c in ALLOWED_TIMESERIES_FEATURES:
                if c in val_data.columns and c in wq_data.columns and pd.api.types.is_numeric_dtype(wq_data[c]):
                    val_data[c] = val_data[c].fillna(wq_data[c].median())
    val_data = val_data.fillna(val_data.median(numeric_only=True))
    for c in need_cols_val:
        if c not in val_data.columns:
            val_data[c] = 0

    # Train vs Validation 분포 비교 (숫자형 피처)
    common_num = [c for c in need_cols_val if c in wq_data.columns and c in val_data.columns and pd.api.types.is_numeric_dtype(wq_data[c]) and pd.api.types.is_numeric_dtype(val_data[c])]
    if common_num:
        train_mean = wq_data[common_num].mean()
        val_mean = val_data[common_num].mean()
        train_std = wq_data[common_num].std().replace(0, np.nan)
        diff = (val_mean - train_mean).abs()
        z_diff = (diff / train_std).fillna(0)
        large_diff = z_diff[z_diff > 2].sort_values(ascending=False).head(15)
        print("\n  --- Train vs Validation 분포 (mean 차이 큰 피처, |z|>2) ---")
        if len(large_diff) == 0:
            print("    없음 (전체 유사).")
        else:
            for c in large_diff.index:
                print(f"    {c}: train_mean={train_mean[c]:.4g} val_mean={val_mean[c]:.4g} z_diff={large_diff[c]:.2f}")
        print("  -------------------------------------------------------")

    # GEMS distance / within_limit 분포 비교
    gems_diag = []
    if "gems_distance_km" in wq_data.columns and "gems_distance_km" in val_data.columns:
        t_d = wq_data["gems_distance_km"].dropna()
        v_d = val_data["gems_distance_km"].dropna()
        if len(t_d) and len(v_d):
            gems_diag.append(f"  gems_distance_km: train mean={t_d.mean():.2f} std={t_d.std():.2f} | val mean={v_d.mean():.2f} std={v_d.std():.2f} (val n={len(v_d)})")
        else:
            gems_diag.append(f"  gems_distance_km: train mean={t_d.mean():.2f} (n={len(t_d)}) | val n={len(v_d)} (전부 0/결측 시 요약 생략)")
    if "gems_within_limit" in wq_data.columns and "gems_within_limit" in val_data.columns:
        t_w = wq_data["gems_within_limit"].value_counts().to_dict()
        v_w = val_data["gems_within_limit"].value_counts().to_dict()
        gems_diag.append(f"  gems_within_limit: train {t_w} | val {v_w}")
    if gems_diag:
        print("\n  --- GEMS distance / within_limit 분포 ---")
        for line in gems_diag:
            print(line)
        print("  -----------------------------------------")

    # Offline val이 "GEMS 완벽 환경"이면 실제 LB는 더 어려울 수 있음 → 경고
    val_gems_too_perfect = False
    if "gems_within_limit" in val_data.columns and "gems_distance_km" in val_data.columns:
        v_within = val_data["gems_within_limit"].dropna()
        v_dist = val_data["gems_distance_km"].replace(np.nan, 0).dropna()
        if len(v_within) and v_within.isin([True, 1]).all() and len(v_dist) and (v_dist <= 0.01).all():
            val_gems_too_perfect = True
            print("\n  *** 경고: Offline validation이 GEMS 완벽 환경입니다 (distance~=0, within_limit 전부 True).")
            print("      실제 hidden LB는 GEMS 약한 지역이 포함될 수 있어 proxy 과대평가 가능 → 아래 GEMS-weak holdout 참고.")

    # Lag/rolling 피처 정상값 여부 (validation)
    if lag_like:
        common_lag = [c for c in lag_like if c in wq_data.columns and c in val_data.columns]
        print("\n  --- Validation lag/rolling 피처 확인 ---")
        if common_lag:
            for c in common_lag[:20]:
                tr = wq_data[c]
                vv = val_data[c]
                v_min, v_max, v_mean = vv.min(), vv.max(), vv.mean()
                t_min, t_max = tr.min(), tr.max()
                ok = "OK" if (t_min <= v_mean <= t_max or abs(v_mean) < 1e6) else "CHECK"
                miss = val_missing_lag.get(c, 0)
                print(f"    {c}: val min/mean/max={v_min:.4g}/{v_mean:.4g}/{v_max:.4g} (train range [{t_min:.4g},{t_max:.4g}]) 결측률(전)={miss:.1%} {ok}")
            if len(common_lag) > 20:
                print(f"    ... 외 {len(common_lag) - 20}개")
        else:
            print("    (need_cols_val 내 lag/rolling 없음)")
        print("  ---------------------------------------")

    # Target별 feature set train/validation 동일 여부
    print("\n  --- Target별 feature set train/val 동일 여부 ---")
    for name, flist in [("TA", feat_TA), ("EC", feat_EC), ("DRP", feat_DRP)]:
        in_train = set(flist) & set(wq_data.columns)
        in_val = set(flist) & set(val_data.columns)
        miss_train = set(flist) - set(wq_data.columns)
        miss_val = set(flist) - set(val_data.columns)
        if not miss_train and not miss_val:
            suffix = " (DRP 고정 variant 20a_bare_partial_p)" if name == "DRP" else ""
            print(f"    {name}: 동일 (train {len(in_train)}개, val {len(in_val)}개){suffix}")
        else:
            if miss_train:
                print(f"    {name}: train에 없음 {list(miss_train)[:5]}{'...' if len(miss_train)>5 else ''}")
            if miss_val:
                print(f"    {name}: val에 없음 {list(miss_val)[:5]}{'...' if len(miss_val)>5 else ''}")
    print("  -----------------------------------------------")

    # 설정 요약 (네 가지 확인용)
    drp_method = "residual+2단계(TA/EC)" if (drp_residual_mode and drp_use_ta_ec) else ("residual만(GEMS)" if drp_residual_mode else "1단계 단독")
    if not drp_use_ta_ec:
        drp_method += " (DRP는 TA/EC 미사용)"
    feature_method = f"target별 feature set (TA={len(feat_TA)}, EC={len(feat_EC)}, DRP={len(feat_DRP)})" if use_cv else f"union 공통 ({len(feature_cols)}개)"
    print("\n  --- Validation & 모델 설정 요약 ---")
    print(f"  1) GEMS validation: {gems_merge_method}")
    print(f"  2) Landsat 결측:   {landsat_method}")
    print(f"  3) DRP:            {drp_method}")
    print(f"  4) 피처:           {feature_method}")
    print("  ---------------------------------")

    def _align_to_scaler(val_df, feat_list, scaler):
        if hasattr(scaler, "feature_names_in_"):
            order = list(scaler.feature_names_in_)
            return val_df.reindex(columns=order).fillna(0)
        return val_df.reindex(columns=feat_list).fillna(0)

    submission_val_data_TA = _align_to_scaler(val_data, feat_TA, scaler_TA)
    submission_val_data_EC = _align_to_scaler(val_data, feat_EC, scaler_EC)
    X_sub_TA = scaler_TA.transform(submission_val_data_TA)
    X_sub_EC = scaler_EC.transform(submission_val_data_EC)
    pred_TA = model_TA.predict(X_sub_TA)
    pred_EC = model_EC.predict(X_sub_EC)
    if EC_log_target:
        pred_EC = np.expm1(pred_EC)
    if USE_REGIONAL_STANDARDIZATION and (y_TA_mean_cv != 0 or y_EC_mean_cv != 0):
        pred_TA = pred_TA + y_TA_mean_cv
        pred_EC = pred_EC + y_EC_mean_cv
    if USE_MARKOWITZ_BLEND:
        if "gems_Alk_Tot" in val_data.columns:
            g_ta = np.maximum(np.nan_to_num(val_data["gems_Alk_Tot"].values, nan=0.0), 0)
            pred_TA = (1 - MARKOWITZ_W) * pred_TA + MARKOWITZ_W * g_ta
        if "gems_EC" in val_data.columns:
            g_ec = np.maximum(np.nan_to_num(val_data["gems_EC"].values, nan=0.0), 0)
            pred_EC = (1 - MARKOWITZ_W) * pred_EC + MARKOWITZ_W * g_ec

    # DRP: prior분리(6-tuple: prior_cols / noprior_cols) / regime_switch(6-tuple, 4번째가 list) / 단일 모델
    if drp_prior_split and isinstance(model_DRP, tuple) and len(model_DRP) >= 6:
        m_prior, m_noprior, scaler_prior, scaler_noprior, prior_cols, noprior_cols = model_DRP[0], model_DRP[1], model_DRP[2], model_DRP[3], model_DRP[4], model_DRP[5]
        val_has_prior = get_has_prior(val_data)
        pred_DRP = np.zeros(len(val_data), dtype=np.float64)
        if val_has_prior.any():
            sub_p = val_data.reindex(columns=[c for c in prior_cols if c in val_data.columns]).fillna(0).copy()
            if "prior_weight" in prior_cols and drp_decay_km is not None and "gems_distance_km" in val_data.columns:
                sub_p["prior_weight"] = np.exp(-np.nan_to_num(val_data["gems_distance_km"].values, nan=999.0) / drp_decay_km)
            if drp_use_ta_ec:
                sub_p["pred_TA"], sub_p["pred_EC"] = pred_TA * drp_ta_ec_blend, pred_EC * drp_ta_ec_blend
            feat_prior_sub = list(prior_cols) + (["pred_TA", "pred_EC"] if drp_use_ta_ec else [])
            X_p = _align_to_scaler(sub_p.loc[val_has_prior], feat_prior_sub, scaler_prior)
            raw_p = m_prior.predict(scaler_prior.transform(X_p))
            g_p = np.maximum(np.nan_to_num(val_data.loc[val_has_prior, "gems_DRP"].values, nan=0.0), 0) if "gems_DRP" in val_data.columns else np.zeros(val_has_prior.sum())
            pred_DRP[val_has_prior] = np.maximum(np.expm1(np.log1p(np.maximum(g_p, 0)) + raw_p), 0)
        if (~val_has_prior).any():
            sub_n = val_data.reindex(columns=[c for c in noprior_cols if c in val_data.columns]).fillna(0).copy()
            if drp_use_ta_ec:
                sub_n["pred_TA"], sub_n["pred_EC"] = pred_TA * drp_ta_ec_blend, pred_EC * drp_ta_ec_blend
            feat_noprior_sub = list(noprior_cols) + (["pred_TA", "pred_EC"] if drp_use_ta_ec else [])
            X_n = _align_to_scaler(sub_n.loc[~val_has_prior], feat_noprior_sub, scaler_noprior)
            raw_n = m_noprior.predict(scaler_noprior.transform(X_n))
            raw_n = np.clip(raw_n, -15.0, 15.0)
            pred_DRP[~val_has_prior] = np.maximum(np.expm1(raw_n), 0)
    elif isinstance(model_DRP, tuple) and len(model_DRP) == 6 and isinstance(model_DRP[4], (list, tuple)):
        m_b, m_e, scaler_b, scaler_e, baseline_cols, event_cols = model_DRP
        feat_b = list(baseline_cols) + (["pred_TA", "pred_EC"] if drp_use_ta_ec else [])
        feat_e = list(event_cols) + (["pred_TA", "pred_EC"] if drp_use_ta_ec else [])
        sub_b = val_data.reindex(columns=baseline_cols).fillna(0).copy()
        sub_e = val_data.reindex(columns=event_cols).fillna(0).copy()
        if drp_use_ta_ec:
            sub_b["pred_TA"], sub_b["pred_EC"] = pred_TA * drp_ta_ec_blend, pred_EC * drp_ta_ec_blend
            sub_e["pred_TA"], sub_e["pred_EC"] = pred_TA * drp_ta_ec_blend, pred_EC * drp_ta_ec_blend
        sub_b = _align_to_scaler(sub_b, feat_b, scaler_b)
        sub_e = _align_to_scaler(sub_e, feat_e, scaler_e)
        pred_b = np.expm1(m_b.predict(scaler_b.transform(sub_b)))
        pred_e = np.maximum(m_e.predict(scaler_e.transform(sub_e)), 0)
        pred_DRP = np.maximum(pred_b + pred_e, 0)
    else:
        feat_DRP_sub = list(feat_DRP) + (["pred_TA", "pred_EC"] if drp_use_ta_ec else [])
        submission_val_data_DRP = val_data.reindex(columns=feat_DRP).fillna(0).copy()
        if drp_use_ta_ec:
            submission_val_data_DRP["pred_TA"] = pred_TA * drp_ta_ec_blend
            submission_val_data_DRP["pred_EC"] = pred_EC * drp_ta_ec_blend
        submission_val_data_DRP = _align_to_scaler(submission_val_data_DRP, feat_DRP_sub, scaler_DRP)
        X_sub_DRP = scaler_DRP.transform(submission_val_data_DRP)
        pred_residual = model_DRP.predict(X_sub_DRP)
        _sub_ratio = DRP_RATIO_TARGET and drp_residual_mode
        if _sub_ratio:
            g_val = np.maximum(np.nan_to_num(val_data["gems_DRP"].values, nan=0.0), 0) if "gems_DRP" in val_data.columns else np.zeros(len(val_data))
            pred_DRP = np.maximum(pred_residual * np.maximum(g_val, DRP_RATIO_EPS), 0)
        elif drp_residual_mode:
            g_val = np.maximum(np.nan_to_num(val_data["gems_DRP"].values, nan=0.0), 0) if "gems_DRP" in val_data.columns else np.zeros(len(val_data))
            if drp_log_residual:
                pred_DRP = np.maximum(np.expm1(np.log1p(np.maximum(g_val, 0)) + pred_residual), 0)
            else:
                pred_DRP = np.maximum(g_val + pred_residual, 0)
        else:
            pred_DRP = np.maximum(np.expm1(pred_residual) if drp_log_residual else pred_residual, 0)
    if USE_DRP_DISTANCE_PRIOR and drp_decay_km is not None and "gems_distance_km" in val_data.columns and "gems_DRP" in val_data.columns:
        dist_val = np.nan_to_num(val_data["gems_distance_km"].values, nan=999.0)
        prior_w = np.exp(-dist_val / drp_decay_km)
        g_val = np.maximum(np.nan_to_num(val_data["gems_DRP"].values, nan=0.0), 0)
        pred_DRP = pred_DRP + (prior_w - 1.0) * g_val
        pred_DRP = np.maximum(pred_DRP, 0)
    if drp_gems_blend_alpha < 1.0 and "gems_DRP" in val_data.columns and drp_residual_mode:
        g_val = np.maximum(np.nan_to_num(val_data["gems_DRP"].values, nan=0.0), 0)
        mask = g_val > 0
        pred_DRP = np.where(mask, drp_gems_blend_alpha * pred_DRP + (1 - drp_gems_blend_alpha) * g_val, pred_DRP)
        pred_DRP = np.maximum(pred_DRP, 0)

    if not DRP_CLIP_TO_TRAIN_RANGE:
        pred_DRP = np.clip(pred_DRP, 0.0, 1e4)
    # DRP_CLIP_TO_TRAIN_RANGE=True면 제출 직전 train p01~p99로 클리핑 (분산 확보)

    # Region calibration: val 좌표에 가장 가까운 train cluster → bias 보정
    if USE_DRP_REGION_CALIBRATION and drp_cluster_bias is not None and "Latitude" in val_data.columns and "Longitude" in val_data.columns:
        val_groups = get_cluster_groups(
            val_data[["Latitude", "Longitude"]].assign(Latitude=val_data["Latitude"], Longitude=val_data["Longitude"]),
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
        )
        pred_DRP = apply_cluster_bias(pred_DRP, val_groups, drp_cluster_bias)
        pred_DRP = np.maximum(pred_DRP, 0)
        if not DRP_CLIP_TO_TRAIN_RANGE:
            pred_DRP = np.clip(pred_DRP, 0.0, 1e4)
        print(f"  DRP region calibration applied: {len(np.unique(val_groups))} val clusters")

    if USE_GEMS_CALIBRATION:
        _dist = np.nan_to_num(val_data["gems_distance_km"].values, nan=999.0) if "gems_distance_km" in val_data.columns else np.full(len(val_data), 999.0)
        _close = _dist < GEMS_CALIB_DIST_THRESH
        _n_close = _close.sum()
        if _n_close > 0:
            if "gems_Alk_Tot" in val_data.columns:
                g_ta = np.nan_to_num(val_data["gems_Alk_Tot"].values, nan=0.0)
                _lo = g_ta * (1 - GEMS_CALIB_TA_RATIO)
                _hi = g_ta * (1 + GEMS_CALIB_TA_RATIO)
                _lo = np.where(g_ta > 0, _lo, pred_TA)
                _hi = np.where(g_ta > 0, _hi, pred_TA)
                pred_TA = np.where(_close, np.clip(pred_TA, _lo, _hi), pred_TA)
            if "gems_EC" in val_data.columns:
                g_ec = np.nan_to_num(val_data["gems_EC"].values, nan=0.0)
                _lo = g_ec * (1 - GEMS_CALIB_EC_RATIO)
                _hi = g_ec * (1 + GEMS_CALIB_EC_RATIO)
                _lo = np.where(g_ec > 0, _lo, pred_EC)
                _hi = np.where(g_ec > 0, _hi, pred_EC)
                pred_EC = np.where(_close, np.clip(pred_EC, _lo, _hi), pred_EC)
            if "gems_DRP" in val_data.columns:
                g_drp = np.nan_to_num(val_data["gems_DRP"].values, nan=0.0)
                _lo_drp = np.maximum(g_drp * (1 - GEMS_CALIB_DRP_RATIO), 0)
                _hi_drp = g_drp * (1 + GEMS_CALIB_DRP_RATIO)
                _lo_drp = np.where(g_drp > 0, _lo_drp, pred_DRP)
                _hi_drp = np.where(g_drp > 0, _hi_drp, pred_DRP)
                pred_DRP = np.where(_close, np.clip(pred_DRP, _lo_drp, _hi_drp), pred_DRP)
            print(f"  GEMS calibration applied: {_n_close}/{len(val_data)} samples within {GEMS_CALIB_DIST_THRESH}km")

    # 제출 전 winsorize: TA/EC 극단치·DRP 과도한 수축 방지 (train 1~99 percentile로 클리핑)
    if USE_TRAIN_PERCENTILE_CLIP and SUBMISSION_CLIP_PERCENTILE:
        p_lo, p_hi = SUBMISSION_CLIP_PERCENTILE
        t_ta = wq_data["Total Alkalinity"].dropna()
        t_ec = wq_data["Electrical Conductance"].dropna()
        t_drp = wq_data["Dissolved Reactive Phosphorus"].dropna()
        p01_ta, p99_ta = np.percentile(t_ta, [p_lo, p_hi])
        p01_ec, p99_ec = np.percentile(t_ec, [p_lo, p_hi])
        p01_drp, p99_drp = np.percentile(t_drp, [p_lo, p_hi])
        pred_TA = np.clip(pred_TA, p01_ta, p99_ta)
        pred_EC = np.clip(pred_EC, p01_ec, p99_ec)
        pred_DRP = np.clip(pred_DRP, p01_drp, p99_drp)
        print(f"  Submission clipped to train {p_lo}~{p_hi}pct: TA[{p01_ta:.1f},{p99_ta:.1f}] EC[{p01_ec:.1f},{p99_ec:.1f}] DRP[{p01_drp:.4f},{p99_drp:.4f}]")

    submission_df = pd.DataFrame({
        "Latitude": test_file["Latitude"].values,
        "Longitude": test_file["Longitude"].values,
        "Sample Date": test_file["Sample Date"].values,
        "Total Alkalinity": pred_TA,
        "Electrical Conductance": pred_EC,
        "Dissolved Reactive Phosphorus": pred_DRP * DRP_SUBMISSION_SCALE,
    })

    out_name = f"submission_{SUBMISSION_PROFILE}.csv"
    out_path = BASE_DIR / out_name
    submission_df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")


if __name__ == "__main__":
    main()
