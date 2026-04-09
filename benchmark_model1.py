"""
Water Quality Prediction: Benchmark Model

EY AI & Data Challenge 2026 - 물질 수질 예측 모델
Landsat, TerraClimate 피처를 활용한 Total Alkalinity, Electrical Conductance,
Dissolved Reactive Phosphorus 예측

로컬 실행: 데이터 파일 경로를 BASE_DIR 기준으로 사용

--- 데이터 누수 방지 정책 (5가지) ---
1) 공간 누수 차단: Spatial Block CV (0.5° 등) + GroupKFold. 동일/인접 지역이 train/val에 동시에 들어가지 않도록.
2) 전체 데이터로 미리 만든 통계/인코딩 금지: 위치별·월별 집계(NDMI_anom, pet_norm 등)는 FoldPreprocessor에서 fold별 train만으로 계산 후 val에 적용.
   add_derived_features(..., skip_aggregations=True)로 로우 단위 안전 피처만 선계산.
3) DRP에서 TA/EC 정답값 미사용: CV에서는 TA/EC 실제값 대신 OOF 예측값 사용 (DRP_USE_TA_EC_TRUE_IN_CV=False).
4) 선택/튜닝은 fold 내부만: DRP Top-K, 하이퍼파라미터 튜닝은 train fold 내부에서만 수행.
5) 외부/보조 데이터 병합은 타깃을 쓰지 않는 방식만: 날짜·위치 키로 조인되는 설명변수만 사용. 타깃 기반 통계 피처는 금지/폴드화.
"""

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Feature preprocessing and data splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    USE_XGBOOST = False
    xgb = None
    print("XGBoost 미사용 (Mac: brew install libomp). RandomForest 사용.")
else:
    print("XGBoost 사용.")

try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    USE_LIGHTGBM = True
except ImportError:
    USE_LIGHTGBM = False
    lgb = None

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor  # type: ignore
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False
    _CatBoostRegressor = None

USE_ENSEMBLE = True  # XGBoost + LightGBM (+ CatBoost) ensemble
USE_HIST_GBM = True   # sklearn HistGradientBoostingRegressor 추가 (다양성, 의존성 없음)
USE_EXTRA_TREES = True  # sklearn ExtraTreesRegressor 추가 (다양성, 의존성 없음)


class EnsembleRegressor:
    """여러 모델의 predict를 평균."""
    def __init__(self, models):
        self.models = models
        for m in models:
            if hasattr(m, 'feature_importances_'):
                self.feature_importances_ = m.feature_importances_
                break
        for m in models:
            if hasattr(m, 'best_iteration') and getattr(m, 'best_iteration', 0) > 0:
                self.best_iteration = m.best_iteration
                break

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Optional


# Base directory (스크립트 위치 기준). 데이터 파일은 data/ 폴더 사용
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ============ 실험 모드 (Step 1/2/3) ============
# Step 1: gems_* 전부 제거 → pure satellite+climate. CV↓지만 LB↑면 prior가 원인.
# Step 2: gems를 약하게만 사용 → final = 0.85*model + 0.15*gems
# Step 3: Hard CV → catchment(공간 그리드) 단위 GroupKFold. CV가 현실에 가까워짐.
EXPERIMENT_STEP = 0  # 0=기본(DRP 3-blend) | 1 | 2 | 3
CATCHMENT_GRID_DEG = 0.25  # Step 3용: ~28km 그리드 (catchment 프록시)

# Spatial Block CV (권장: LB와 비슷한 현실적 CV)
USE_SPATIAL_BLOCK_CV = True   # LB-CV 정렬을 위해 블록 단위 GroupKFold
SPATIAL_BLOCK_COMPARE = False  # True면 0.25/0.5/1.0 비교 후 선택, False면 0.5° 고정 (로그 형식)
SPATIAL_BLOCK_PER_TARGET = False
SPATIAL_BLOCK_DEG = 0.5       # ~56km 격자
SPATIAL_BLOCK_CANDIDATES = [0.25, 0.5, 1.0]  # 후보 (각각 ~28km, ~56km, ~111km)
# 블록 크기 가이드: 0.25°=측정소 많을 때 세밀분리 | 0.5°=보통 균형 | 1.0°=지역 집중·fold적을 때

# Landsat 피처 컬럼
LANDSAT_COLS = ['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']

# 확장 피처 (ERA5, 강수 이상치, 외부, HydroRIVERS) — 해당 CSV 있으면 병합
USE_ERA5_PRECIP = True        # precipitation_training.csv의 pr 병합
USE_PRECIP_ANOMALY = True     # water_quality_with_precip_anomaly.csv 병합
USE_EXTERNAL_FEATURES = True  # soil, elevation, population, land cover 등
USE_HYRIV_ERA5_EVENTS = True  # train_with_hyriv_era5_events.csv로 HydroRIVERS+ERA5 이벤트 병합
USE_FOLD_SAFE_DRP_TOP_K = True  # DRP fold 내 top-K 피처 선택 (CV 누수 방지)
DRP_TOP_K = 50


def add_seasonality_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:
    """Add month, dayofyear, sin_doy, cos_doy from Sample Date."""
    result = df.copy()
    dates = pd.to_datetime(result[date_col], format="mixed", dayfirst=True)
    result["month"] = dates.dt.month
    result["dayofyear"] = dates.dt.dayofyear
    result["sin_doy"] = np.sin(2 * np.pi * result["dayofyear"] / 365)
    result["cos_doy"] = np.cos(2 * np.pi * result["dayofyear"] / 365)
    return result


def add_wetness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wet_index = NDMI * pet, water_stress = NDMI / (pet + 1e-6)."""
    result = df.copy()
    if "NDMI" in result.columns and "pet" in result.columns:
        result["wet_index"] = result["NDMI"] * result["pet"]
        result["water_stress"] = result["NDMI"] / (result["pet"] + 1e-6)
    return result


def add_derived_features(
    df: pd.DataFrame,
    loc_stats: Optional[object] = None,
    add_water_index: bool = True,
    add_band_ratios: bool = True,
    add_ndmi_mndwi_variants: bool = True,
    add_pet_derived: bool = True,
    add_season: bool = True,
    add_ta_specific: bool = True,
    add_ec_specific: bool = True,
    add_drp_specific: bool = True,
    skip_aggregations: bool = False,
) -> pd.DataFrame:
    """
    위성·기후·GEMS 파생 피처 추가.
    loc_stats: (Lat,Lon)별 NDMI_mean, MNDWI_mean (및 pet_mean 등) train에서 계산 후 val에 전달.
    skip_aggregations=True: 위치/월별 집계(NDMI_anom, MNDWI_anom, pet_norm, pet_seasonal_anom)는 추가하지 않음.
    → FoldPreprocessor에서 fold별 train만으로 계산해 사용 (누수 방지).
    """
    result = df.copy()
    eps = 1e-6

    # --- 공통: Water index / band ratios ---
    if add_water_index and all(c in result.columns for c in ["green", "swir16", "nir", "swir22"]):
        denom = result["nir"] + result["swir22"] + eps
        result["WRI"] = (result["green"] + result["swir16"]) / denom
    if add_band_ratios:
        if "green" in result.columns and "nir" in result.columns:
            result["green_nir_ratio"] = result["green"] / (result["nir"] + eps)
        if "green" in result.columns and "swir22" in result.columns:
            result["green_swir22_ratio"] = result["green"] / (result["swir22"] + eps)
        if "nir" in result.columns and "swir16" in result.columns:
            result["nir_swir16_ratio"] = result["nir"] / (result["swir16"] + eps)
        if "nir" in result.columns and "swir22" in result.columns:
            result["nir_swir22_ratio"] = result["nir"] / (result["swir22"] + eps)

    # --- NDMI, MNDWI 변형 ---
    if add_ndmi_mndwi_variants:
        if "NDMI" in result.columns:
            result["NDMI_sq"] = result["NDMI"] ** 2
        if "MNDWI" in result.columns:
            result["MNDWI_sq"] = result["MNDWI"] ** 2
        if "NDMI" in result.columns and "MNDWI" in result.columns:
            result["NDMI_MNDWI"] = result["NDMI"] * result["MNDWI"]

    # --- NDMI/MNDWI anomaly (위치별 연평균 대비) ---
    if not skip_aggregations:
        if loc_stats is not None and isinstance(loc_stats, pd.DataFrame):
            result = result.merge(loc_stats, on=["Latitude", "Longitude"], how="left")
            if "NDMI" in result.columns and "NDMI_mean" in result.columns:
                result["NDMI_anom"] = result["NDMI"] - result["NDMI_mean"].fillna(result["NDMI"].median())
            if "MNDWI" in result.columns and "MNDWI_mean" in result.columns:
                result["MNDWI_anom"] = result["MNDWI"] - result["MNDWI_mean"].fillna(result["MNDWI"].median())
            result.drop(columns=[c for c in ["NDMI_mean", "MNDWI_mean"] if c in result.columns], inplace=True)
        else:
            if "NDMI" in result.columns and "Latitude" in result.columns:
                ndmi_loc = result.groupby(["Latitude", "Longitude"])["NDMI"].transform("mean")
                result["NDMI_anom"] = result["NDMI"] - ndmi_loc
            if "MNDWI" in result.columns and "Latitude" in result.columns:
                mndwi_loc = result.groupby(["Latitude", "Longitude"])["MNDWI"].transform("mean")
                result["MNDWI_anom"] = result["MNDWI"] - mndwi_loc

    # --- pet 파생 ---
    if add_pet_derived and "pet" in result.columns:
        result["log1p_pet"] = np.log1p(np.maximum(result["pet"], 0))
        if not skip_aggregations:
            if "Latitude" in result.columns:
                pet_loc = result.groupby(["Latitude", "Longitude"])["pet"].transform("mean")
                result["pet_norm"] = result["pet"] / (pet_loc + eps)
            if "month" in result.columns:
                pet_month = result.groupby("month")["pet"].transform("mean")
                result["pet_seasonal_anom"] = result["pet"] - pet_month
        if "sin_doy" in result.columns:
            result["pet_sin_doy"] = result["pet"] * result["sin_doy"]
        if "cos_doy" in result.columns:
            result["pet_cos_doy"] = result["pet"] * result["cos_doy"]
        if "Latitude" in result.columns:
            result["pet_lat"] = result["pet"] * np.abs(result["Latitude"])

    # --- season (1~4) ---
    if add_season and "month" in result.columns:
        m = result["month"]
        result["season"] = np.where(m.isin([12, 1, 2]), 1, np.where(m.isin([3, 4, 5]), 2, np.where(m.isin([6, 7, 8]), 3, 4)))
        result["growing_season"] = (m >= 4) & (m <= 9)  # 남반구 대략 4~9월

    # --- TA 전용: GEMS 탄산계 조합 ---
    if add_ta_specific:
        if all(c in result.columns for c in ["gems_Ca_Dis", "gems_Mg_Dis"]):
            result["hardness_like"] = result["gems_Ca_Dis"].fillna(0) + result["gems_Mg_Dis"].fillna(0)
        if all(c in result.columns for c in ["gems_Ca_Dis", "gems_Mg_Dis", "gems_Na_Dis"]):
            result["cation_balance"] = result["gems_Ca_Dis"].fillna(0) + result["gems_Mg_Dis"].fillna(0) + result["gems_Na_Dis"].fillna(0)
        if all(c in result.columns for c in ["gems_Ca_Dis", "gems_Mg_Dis"]):
            result["Ca_Mg_ratio"] = result["gems_Ca_Dis"] / (result["gems_Mg_Dis"] + eps)

    # --- EC 전용: 이온 조합 ---
    if add_ec_specific:
        ion_cols = ["gems_Na_Dis", "gems_Cl_Dis", "gems_SO4_Dis", "gems_Ca_Dis", "gems_Mg_Dis"]
        if all(c in result.columns for c in ion_cols):
            result["major_ions"] = sum(result[c].fillna(0) for c in ion_cols)
        if all(c in result.columns for c in ["gems_Na_Dis", "gems_Cl_Dis"]):
            result["na_cl_ratio"] = result["gems_Na_Dis"] / (result["gems_Cl_Dis"] + eps)
        if all(c in result.columns for c in ["gems_SO4_Dis", "gems_Cl_Dis"]):
            result["sulphate_chloride_ratio"] = result["gems_SO4_Dis"] / (result["gems_Cl_Dis"] + eps)
        if all(c in result.columns for c in ["gems_Sal", "gems_EC"]):
            result["sal_over_EC"] = result["gems_Sal"] / (result["gems_EC"] + eps)
        if "NDMI" in result.columns and "MNDWI" in result.columns:
            result["dry_index"] = -result["NDMI"] + (1 - result["MNDWI"])

    # --- DRP 전용: P·N 조합 ---
    if add_drp_specific:
        if all(c in result.columns for c in ["gems_NOxN", "gems_NH4N"]):
            result["N_total_like"] = result["gems_NOxN"].fillna(0) + result["gems_NH4N"].fillna(0)
        if all(c in result.columns for c in ["gems_NOxN", "gems_NH4N", "gems_TP"]):
            n = result["gems_NOxN"].fillna(0) + result["gems_NH4N"].fillna(0)
            result["N_to_P_ratio"] = n / (result["gems_TP"] + eps)
        if all(c in result.columns for c in ["gems_DRP", "gems_TP"]):
            result["DRP_to_TP_ratio"] = result["gems_DRP"] / (result["gems_TP"] + eps)
        if all(c in result.columns for c in ["gems_partial_P", "gems_TP"]):
            result["partialP_ratio"] = result["gems_partial_P"] / (result["gems_TP"] + eps)
        if "gems_DRP" in result.columns and "gems_DRP_log" in result.columns:
            drp_log_val = np.expm1(result["gems_DRP_log"].fillna(0))
            result["DRP_prior_center"] = (result["gems_DRP"].fillna(0) + drp_log_val) / 2
            result["DRP_prior_spread"] = np.abs(result["gems_DRP"].fillna(0) - drp_log_val)
        if "wet_index" in result.columns and "gems_TP" in result.columns:
            result["wet_TP"] = result["wet_index"] * result["gems_TP"].fillna(0)
        if "water_stress" in result.columns and "gems_TP" in result.columns:
            result["water_stress_TP"] = result["water_stress"] * result["gems_TP"].fillna(0)
        if "wet_index" in result.columns and "gems_DRP" in result.columns:
            result["wet_DRP"] = result["wet_index"] * result["gems_DRP"].fillna(0)
        if "NDMI" in result.columns and "gems_TP" in result.columns:
            result["NDMI_TP"] = result["NDMI"] * result["gems_TP"].fillna(0)
        if "MNDWI" in result.columns and "gems_TP" in result.columns:
            result["MNDWI_TP"] = result["MNDWI"] * result["gems_TP"].fillna(0)
        if "NDMI" in result.columns and "N_total_like" in result.columns:
            result["NDMI_N"] = result["NDMI"] * result["N_total_like"]

    return result


def impute_landsat_spatial(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    landsat_cols: list = LANDSAT_COLS,
) -> pd.DataFrame:
    """
    Landsat 결측을 공간 기반으로 보간:
    1) 동일 (Lat, Long) 다른 날짜의 median
    2) 가장 가까운 validation 지점(데이터 있음)의 값
    3) 가장 가까운 training 지점의 값
    4) 전체 median (fallback)
    """
    result = val_df.copy()

    for col in landsat_cols:
        if col not in result.columns:
            continue

        missing_mask = result[col].isna()
        if not missing_mask.any():
            continue

        # 4) Fallback: 전체 median
        fallback = result[col].median()
        if pd.isna(fallback):
            fallback = train_df[col].median() if col in train_df.columns else 0

        # 1) 동일 좌표 median (validation 내)
        for idx in np.where(missing_mask)[0]:
            lat, lon = result.loc[idx, 'Latitude'], result.loc[idx, 'Longitude']
            same_loc = val_df[
                (val_df['Latitude'] == lat) & (val_df['Longitude'] == lon)
            ]
            valid_vals = same_loc[col].dropna()
            if len(valid_vals) > 0:
                result.loc[idx, col] = valid_vals.median()
                missing_mask.loc[idx] = False

        # 아직 결측인 행에 대해 2), 3) 적용
        still_missing = result[col].isna()
        if not still_missing.any():
            continue

        # 데이터 있는 행들의 좌표
        has_data_val = val_df[~val_df[col].isna()]
        has_data_train = train_df[~train_df[col].isna()] if col in train_df.columns else pd.DataFrame()

        def latlon_to_rad(lat, lon):
            return np.radians(lat), np.radians(lon)

        def build_tree(df):
            if len(df) == 0:
                return None, df
            lat = df['Latitude'].values
            lon = df['Longitude'].values
            phi = np.radians(lat)
            lam = np.radians(lon)
            x = np.cos(phi) * np.cos(lam)
            y = np.cos(phi) * np.sin(lam)
            z = np.sin(phi)
            return cKDTree(np.column_stack([x, y, z])), df

        tree_val, tree_train = None, None
        if len(has_data_val) > 0:
            tree_val, _ = build_tree(has_data_val)
        if len(has_data_train) > 0:
            tree_train, _ = build_tree(has_data_train)

        missing_indices = np.where(still_missing)[0]
        if len(missing_indices) == 0:
            continue

        lats = result.loc[missing_indices, 'Latitude'].values
        lons = result.loc[missing_indices, 'Longitude'].values
        phi = np.radians(lats)
        lam = np.radians(lons)
        pts = np.column_stack([
            np.cos(phi) * np.cos(lam),
            np.cos(phi) * np.sin(lam),
            np.sin(phi)
        ])

        # 2) Validation 내 가장 가까운 지점 (~60km)
        if tree_val is not None:
            dists, idxs = tree_val.query(pts, k=1)
            idxs = np.atleast_1d(idxs).flatten()
            for j, orig_idx in enumerate(missing_indices):
                if dists.flat[j] < 0.015:  # ~95km (단위구 chord)
                    result.loc[orig_idx, col] = has_data_val[col].iloc[idxs[j]]

        # 3) 여전히 결측: Training에서 가장 가까운 지점 (~200km)
        still_missing = result[col].isna()
        if still_missing.any() and tree_train is not None:
            missing_idx2 = np.where(still_missing)[0]
            lats2 = result.loc[missing_idx2, 'Latitude'].values
            lons2 = result.loc[missing_idx2, 'Longitude'].values
            phi2 = np.radians(lats2)
            lam2 = np.radians(lons2)
            pts2 = np.column_stack([
                np.cos(phi2) * np.cos(lam2),
                np.cos(phi2) * np.sin(lam2),
                np.sin(phi2)
            ])
            _, idxs2 = tree_train.query(pts2, k=1)
            idxs2 = np.atleast_1d(idxs2).flatten()
            for j, orig_idx in enumerate(missing_idx2):
                result.loc[orig_idx, col] = has_data_train[col].iloc[idxs2[j]]

        # 4) 최종 fallback
        result[col] = result[col].fillna(fallback)

    return result


class FoldSafeFeatureBuilder:
    """
    Fold 내부에서 fit(train) 후 transform(train/val) 적용 — CV 누수 방지.
    DRP용: train에서 feature importance로 top-K 선택, 해당 컬럼만 사용.
    """
    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if USE_XGBOOST and len(X.columns) > self.top_k:
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X.fillna(X.median()))
                m = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
                m.fit(Xs, y)
                imp = m.feature_importances_
                idx = np.argsort(imp)[::-1][:self.top_k]
                self.selected_cols_ = [X.columns[i] for i in idx]
            except Exception:
                self.selected_cols_ = X.columns.tolist()
        else:
            self.selected_cols_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_cols_ is None:
            return X
        exist = [c for c in self.selected_cols_ if c in X.columns]
        return X[exist].copy()


# 누수 방지: fold 내부에서만 train으로 위치/월별 통계 계산 후 val에 적용
AGGREGATION_FEATURES = ["NDMI_anom", "MNDWI_anom", "pet_norm", "pet_seasonal_anom"]
USE_FOLD_SAFE_AGGREGATIONS = True


class FoldPreprocessor:
    """
    Fold 내부에서 train만으로 위치별·월별 통계를 계산하고, train/val에 적용.
    add_derived_features(skip_aggregations=True)와 함께 사용해 전체 데이터 집계 누수 방지.
    """
    def __init__(self):
        self.loc_stats_ = None  # (Latitude, Longitude) -> NDMI_mean, MNDWI_mean, pet_mean
        self.month_pet_ = None  # month -> pet mean

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        req = ["Latitude", "Longitude", "NDMI", "MNDWI", "pet", "month"]
        if not all(c in X.columns for c in req):
            return self
        loc = X.groupby(["Latitude", "Longitude"])[["NDMI", "MNDWI", "pet"]].mean().reset_index()
        loc.columns = ["Latitude", "Longitude", "NDMI_mean", "MNDWI_mean", "pet_mean"]
        self.loc_stats_ = loc
        self.month_pet_ = X.groupby("month")["pet"].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        eps = 1e-6
        if self.loc_stats_ is not None:
            out = out.merge(self.loc_stats_, on=["Latitude", "Longitude"], how="left")
            if "NDMI" in out.columns and "NDMI_mean" in out.columns:
                out["NDMI_anom"] = out["NDMI"] - out["NDMI_mean"].fillna(out["NDMI"].median())
            if "MNDWI" in out.columns and "MNDWI_mean" in out.columns:
                out["MNDWI_anom"] = out["MNDWI"] - out["MNDWI_mean"].fillna(out["MNDWI"].median())
            if "pet" in out.columns and "pet_mean" in out.columns:
                denom = out["pet_mean"].fillna(out["pet"].median()) + eps
                out["pet_norm"] = out["pet"] / denom
            out.drop(columns=[c for c in ["NDMI_mean", "MNDWI_mean", "pet_mean"] if c in out.columns], inplace=True)
        if self.month_pet_ is not None and "pet" in out.columns and "month" in out.columns:
            out["pet_seasonal_anom"] = out["pet"] - out["month"].map(self.month_pet_).fillna(out["pet"].median())
        return out


def combine_two_datasets(dataset1, dataset2, dataset3):
    """
    Returns a vertically concatenated dataset.
    Attributes:
        dataset1 - Dataset 1 to be combined
        dataset2 - Dataset 2 to be combined
        dataset3 - Dataset 3 to be combined
    """
    data = pd.concat([dataset1, dataset2, dataset3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def get_location_groups(df: pd.DataFrame) -> np.ndarray:
    """
    (Latitude, Longitude)로 location_id 생성.
    동일 측정소 = 동일 그룹 (Group K-Fold에서 한 지점이 train/val에 동시에 나뉘지 않도록)
    """
    loc_key = (
        df['Latitude'].round(6).astype(str) + '_' + df['Longitude'].round(6).astype(str)
    )
    groups, _ = pd.factorize(loc_key)
    return groups


def get_catchment_groups(df: pd.DataFrame, grid_deg: float = 0.25) -> np.ndarray:
    """
    공간 그리드 기반 catchment 프록시 (Hard CV용).
    동일 그리드 셀 = 동일 catchment → 근접 지역이 train/val에 동시 나뉘지 않음.
    grid_deg=0.25 → ~28km 셀.
    """
    lat_bin = (df['Latitude'] / grid_deg).astype(int).astype(str)
    lon_bin = (df['Longitude'] / grid_deg).astype(int).astype(str)
    catchment_key = lat_bin + '_' + lon_bin
    groups, _ = pd.factorize(catchment_key)
    return groups


def get_spatial_block_groups(df: pd.DataFrame, block_deg: float) -> np.ndarray:
    """
    Spatial Block CV: 좌표계를 격자(block_deg°)로 나누고 블록 단위 그룹 생성.
    LB와 비슷한 현실적 CV. block_deg 후보: 0.25°(~28km), 0.5°(~56km), 1.0°(~111km)
    """
    lat = df['Latitude'].values
    lon = df['Longitude'].values
    lat_min, lon_min = lat.min(), lon.min()
    block = (
        (np.floor((lat - lat_min) / block_deg).astype(int) * 10_000)
        + np.floor((lon - lon_min) / block_deg).astype(int)
    )
    groups, _ = pd.factorize(block)
    return groups


def split_data_spatial(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Group Shuffle Split: 동일 location_id는 train 또는 test에만 속함.
    Random shuffle 사용 안 함 → 공간 일반화 시뮬레이션.
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


EARLY_STOPPING_ROUNDS = 50  # CV fold에서 early stopping (0이면 비활성)

def train_model(X_train_scaled, y_train, model_config=None,
                eval_set=None, early_stopping_rounds=0, ensemble=None):
    """Train regressor. ensemble=None이면 USE_ENSEMBLE 따름. False면 single XGB."""
    # ExtraTrees/sklearn 일부는 NaN 미지원 → 남은 NaN/Inf 제거
    X_train_scaled = np.nan_to_num(np.asarray(X_train_scaled, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if eval_set:
        X_ev, y_ev = eval_set[0]
        X_ev = np.nan_to_num(np.asarray(X_ev, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        eval_set = [(X_ev, y_ev)]

    cfg = model_config or {}
    n_est = cfg.get("n_estimators", 200)
    max_d = cfg.get("max_depth", 10)
    lr = cfg.get("learning_rate", 0.1)
    do_ensemble = USE_ENSEMBLE if ensemble is None else ensemble

    if USE_XGBOOST:
        use_es = eval_set is not None and early_stopping_rounds > 0
        xgb_kw = dict(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            min_child_weight=cfg.get("min_child_weight", 1),
            gamma=cfg.get("gamma", 0),
            reg_alpha=cfg.get("reg_alpha", 0),
            reg_lambda=cfg.get("reg_lambda", 1),
            random_state=42,
            verbosity=0,
        )
        try:
            if use_es:
                xgb_kw["early_stopping_rounds"] = early_stopping_rounds
            xgb_model = xgb.XGBRegressor(**xgb_kw)
            fit_kw = {"eval_set": eval_set, "verbose": False} if use_es else {}
            xgb_model.fit(X_train_scaled, y_train, **fit_kw)
        except TypeError:
            xgb_kw.pop("early_stopping_rounds", None)
            xgb_model = xgb.XGBRegressor(**xgb_kw)
            xgb_model.fit(X_train_scaled, y_train)
    else:
        xgb_model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)

    if not do_ensemble:
        return xgb_model

    models = [xgb_model]

    if USE_LIGHTGBM:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            min_child_weight=cfg.get("min_child_weight", 1),
            min_split_gain=cfg.get("gamma", 0),
            reg_alpha=cfg.get("reg_alpha", 0),
            reg_lambda=cfg.get("reg_lambda", 1),
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        lgb_fit_kw = {}
        if use_es and eval_set:
            lgb_fit_kw["eval_set"] = eval_set
            lgb_fit_kw["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False),
                                       lgb.log_evaluation(period=-1)]
        lgb_model.fit(X_train_scaled, y_train, **lgb_fit_kw)
        models.append(lgb_model)

    if USE_CATBOOST:
        cb_model = _CatBoostRegressor(
            iterations=n_est,
            depth=min(max_d, 10),
            learning_rate=lr,
            subsample=cfg.get("subsample", 0.8) if cfg.get("subsample", 0.8) < 1.0 else 0.99,
            l2_leaf_reg=cfg.get("reg_lambda", 1),
            random_seed=42,
            verbose=0,
            bootstrap_type='Bernoulli',
        )
        cb_fit_kw = {}
        if use_es and eval_set:
            cb_fit_kw["eval_set"] = eval_set
            cb_fit_kw["early_stopping_rounds"] = early_stopping_rounds
        cb_model.fit(X_train_scaled, y_train, **cb_fit_kw)
        models.append(cb_model)

    if USE_HIST_GBM:
        hgb = HistGradientBoostingRegressor(
            max_iter=n_est,
            max_depth=min(max_d, 10),
            learning_rate=lr,
            l2_regularization=cfg.get("reg_lambda", 1),
            min_samples_leaf=cfg.get("min_child_weight", 1),
            random_state=42,
            early_stopping=True,
            n_iter_no_change=early_stopping_rounds if early_stopping_rounds > 0 else 10,
            validation_fraction=0.1,
        )
        hgb.fit(X_train_scaled, y_train)
        models.append(hgb)

    if USE_EXTRA_TREES:
        et = ExtraTreesRegressor(
            n_estimators=min(n_est, 500),
            max_depth=max_d,
            min_samples_leaf=cfg.get("min_child_weight", 1),
            random_state=42,
            n_jobs=1,
        )
        et.fit(X_train_scaled, y_train)
        models.append(et)

    if len(models) == 1:
        return models[0]
    return EnsembleRegressor(models)


def evaluate_model(model, X_scaled, y_true, dataset_name="Test"):
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{dataset_name} Evaluation:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return y_pred, r2, rmse


def run_pipeline(X, y, groups, param_name="Parameter", n_folds=5, model_config=None, aggregation_cols=None, leaky=False):
    """
    Spatial Cross-Validation: Group K-Fold by location_id.
    model_config: 파라미터별 XGBoost/RF 하이퍼파라미터.
    aggregation_cols: fold 내부에서만 train으로 집계 피처(NDMI_anom 등) 계산 시 사용. 누수 방지.
    leaky: True면 fold에서 전역 X 기준 impute/scaler 사용 (CV 과대평가), 비교용. 반환은 (None, None, results, None).
    """
    print(f"\n{'='*60}")
    print(f"Training Model for {param_name} (Group K-Fold CV, n_folds={n_folds})")
    if model_config:
        print(f"  Config: n_est={model_config.get('n_estimators')}, max_d={model_config.get('max_depth')}, lr={model_config.get('learning_rate')}")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    best_iters = []
    all_feature_cols = list(X.columns) + (aggregation_cols or [])

    # leaky 비교용: 전역 X 기준 impute/scaler (aggregation_cols 있으면 전역도 동일 컬럼으로)
    _train_median_leaky = None
    _X_full_imputed_leaky = None
    if leaky:
        if aggregation_cols:
            _fp = FoldPreprocessor()
            _fp.fit(X)
            _X_full = _fp.transform(X.copy())
            _X_full = _X_full[all_feature_cols]
        else:
            _X_full = X
        _train_median_leaky = _X_full.median()
        _X_full_imputed_leaky = _X_full.fillna(_train_median_leaky)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if aggregation_cols:
            fp = FoldPreprocessor()
            fp.fit(X_train)
            X_train = fp.transform(X_train)
            X_test = fp.transform(X_test)
            X_train = X_train[all_feature_cols]
            X_test = X_test[all_feature_cols]

        # Fold 내 train median으로만 impute (val 분포 누수 방지). leaky면 전역 X 기준
        if leaky:
            X_train = X_train.fillna(_train_median_leaky)
            X_test = X_test.fillna(_train_median_leaky)
            scaler = StandardScaler()
            scaler.fit(_X_full_imputed_leaky)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            train_median = X_train.median()
            X_train = X_train.fillna(train_median)
            X_test = X_test.fillna(train_median)
            X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        model = train_model(
            X_train_scaled, y_train, model_config=model_config,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        if hasattr(model, 'best_iteration') and model.best_iteration > 0:
            best_iters.append(model.best_iteration)

        y_test_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        fold_results.append({"fold": fold + 1, "R2": r2, "RMSE": rmse})
        print(f"  Fold {fold + 1}/{n_folds}: R²={r2:.3f}, RMSE={rmse:.3f}")

    r2_mean = np.mean([r["R2"] for r in fold_results])
    r2_std = np.std([r["R2"] for r in fold_results])
    rmse_mean = np.mean([r["RMSE"] for r in fold_results])
    rmse_std = np.std([r["RMSE"] for r in fold_results])
    print(f"\n  CV Mean: R²={r2_mean:.3f} ± {r2_std:.3f}, RMSE={rmse_mean:.3f} ± {rmse_std:.3f}")

    if leaky:
        results = {"Parameter": param_name, "R2_CV_mean": r2_mean, "R2_CV_std": r2_std, "RMSE_CV_mean": rmse_mean, "RMSE_CV_std": rmse_std}
        return None, None, pd.DataFrame([results]), None

    # Final model: 전체 데이터로 학습 (submission용)
    final_config = model_config
    if best_iters:
        avg_best = int(np.mean(best_iters))
        print(f"  Early stopping avg best_iteration={avg_best} (folds: {best_iters})")
        final_config = {**model_config, "n_estimators": avg_best} if model_config else model_config

    fp_full = None
    if aggregation_cols:
        fp_full = FoldPreprocessor()
        fp_full.fit(X)
        X = fp_full.transform(X)
        X = X[all_feature_cols]
    # 최종 모델: 전체 train 기준 median으로 impute (제출용)
    X = X.fillna(X.median())
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    model_full = train_model(X_scaled_full, y, model_config=final_config)

    # Feature importance (XGBoost), importance > 0 만 표시
    if USE_XGBOOST and hasattr(model_full, "feature_importances_"):
        feat_names = all_feature_cols
        imp = model_full.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        imp_df = imp_df[imp_df["importance"] > 0]
        print(f"\n  {param_name} Feature Importance (XGBoost, non-zero only):")
        print(imp_df.to_string(index=False) if len(imp_df) else "  (all zero)")

    results = {
        "Parameter": param_name,
        "R2_CV_mean": r2_mean,
        "R2_CV_std": r2_std,
        "RMSE_CV_mean": rmse_mean,
        "RMSE_CV_std": rmse_std,
    }
    return model_full, scaler_full, pd.DataFrame([results]), fp_full


def oof_predictions_with_groups(X_df, y_series, groups_arr, n_splits=4, model_config=None):
    """
    Out-of-fold predictions using GroupKFold (by location groups).
    model_config: TA 또는 EC용 config 전달.
    """
    gkf_inner = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(X_df), dtype=float)

    for tr_i, va_i in gkf_inner.split(X_df, y_series, groups_arr):
        X_tr, X_va = X_df.iloc[tr_i], X_df.iloc[va_i]
        y_tr = y_series.iloc[tr_i]

        X_tr_s, X_va_s, _ = scale_data(X_tr, X_va)
        m = train_model(X_tr_s, y_tr, model_config=model_config, ensemble=False)
        oof_pred[va_i] = m.predict(X_va_s)

    return oof_pred


def oof_predictions_with_uncertainty(X_df, y_series, groups_arr, n_splits=5, model_config=None):
    """
    OOF 예측 + fold별 모델 예측의 표준편차(uncertainty).
    K개 fold 모델 각각이 전체에 예측 → sample당 K개 예측 → mean, std 반환.
    """
    gkf_inner = GroupKFold(n_splits=n_splits)
    preds = np.zeros((len(X_df), n_splits), dtype=float)

    for fold, (tr_i, va_i) in enumerate(gkf_inner.split(X_df, y_series, groups_arr)):
        X_tr = X_df.iloc[tr_i]
        y_tr = y_series.iloc[tr_i]
        X_tr_s, X_full_s, _ = scale_data(X_tr, X_df)  # fit on train, transform both
        m = train_model(X_tr_s, y_tr, model_config=model_config, ensemble=False)
        preds[:, fold] = m.predict(X_full_s)

    oof_pred = np.mean(preds, axis=1)
    oof_std = np.std(preds, axis=1)
    oof_std = np.maximum(oof_std, 1e-6)  # 0 방지
    return oof_pred, oof_std


def run_pipeline_drp(
    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
    model_TA, model_EC, scaler_TA, scaler_EC,
    n_folds=5,
    log_target=False,
    blend_gems=0.0,
    residual_mode=False,
    log_residual=False,
    use_ta_ec_uncertainty=False,
    use_ta_ec_true_in_cv=False,
    drp_config_override=None,
    config_TA=None,
    config_EC=None,
    distance_km=None,
    decay_km=None,
    use_fold_safe_top_k=False,
    drp_top_k=50,
    aggregation_cols=None,
    leaky=False,
    verbose=True,
):
    """
    DRP 모델: TA·EC 예측값을 피처로 사용 (2단계 파이프라인).
    drp_config_override: 튜닝 시 사용, MODEL_CONFIG_DRP 대체.
    aggregation_cols: Fold-safe 집계 피처(NDMI_anom 등); fold 내부에서 FoldPreprocessor로 추가.
    leaky: True면 fold에서 전역 X_full_drp 기준 impute/scaler (비교용). 반환은 (None, None, results, ...).
    """
    if verbose and not leaky:
        print(f"\n{'='*60}")
        print(f"Training DRP Model (TA·EC as features, Group K-Fold CV, n_folds={n_folds})")
        if residual_mode:
            mode_str = "log residual" if log_residual else "linear residual"
            print(f"  [Residual 모드 ({mode_str}): target=DRP-gems_DRP, final=gems_DRP+residual_pred]")
        if log_target:
            print("  [log(1+DRP) 변환 적용]")
        if use_ta_ec_true_in_cv:
            print("  [CV: TA/EC 실제값 사용 (stacking noise 제거)]")
        else:
            print("  [CV: TA/EC OOF 예측값 사용 (제출과 동일 조건)]")
        if blend_gems > 0 and not residual_mode:
            print(f"  [Blend: w={blend_gems}]")
        print(f"{'='*60}")

    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(len(X_DRP))
    gems_DRP_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    _decay_km = decay_km if decay_km is not None else PRIOR_DECAY_KM

    if residual_mode:
        if log_residual:
            # log space: target = log1p(DRP) - log1p(gems_DRP), 분포 정규화
            y_target = np.log1p(np.maximum(y_DRP.values, 0)) - np.log1p(np.maximum(gems_DRP_safe, 0))
            y_DRP_model = pd.Series(y_target, index=y_DRP.index)
        else:
            y_target = y_DRP.values - gems_DRP_safe
            y_DRP_model = pd.Series(y_target, index=y_DRP.index)
    else:
        y_DRP_model = np.log1p(y_DRP.values) if log_target else y_DRP.values
        y_DRP_model = pd.Series(y_DRP_model, index=y_DRP.index)

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    # leaky 비교용: 전역 X_full_drp (pred_TA/EC 포함)로 impute/scaler
    X_full_drp_leaky = None
    if leaky:
        X_TA_f = X_TA
        X_EC_f = X_EC
        X_DRP_f = X_DRP
        if aggregation_cols:
            _fp = FoldPreprocessor()
            _fp.fit(X_DRP)
            all_ta = list(X_TA.columns) + aggregation_cols
            all_ec = list(X_EC.columns) + aggregation_cols
            all_drp = list(X_DRP.columns) + aggregation_cols
            X_TA_f = _fp.transform(X_TA.copy())[all_ta]
            X_EC_f = _fp.transform(X_EC.copy())[all_ec]
            X_DRP_f = _fp.transform(X_DRP.copy())[all_drp]
        X_TA_f = X_TA_f.fillna(X_TA_f.median())
        X_EC_f = X_EC_f.fillna(X_EC_f.median())
        X_DRP_f = X_DRP_f.fillna(X_DRP_f.median())
        pred_TA_full = model_TA.predict(scaler_TA.transform(X_TA_f))
        pred_EC_full = model_EC.predict(scaler_EC.transform(X_EC_f))
        X_full_drp_leaky = X_DRP_f.copy()
        X_full_drp_leaky["pred_TA"] = pred_TA_full
        X_full_drp_leaky["pred_EC"] = pred_EC_full
        X_full_drp_leaky = X_full_drp_leaky.fillna(X_full_drp_leaky.median())

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_DRP, y_DRP, groups)):
        X_TA_train, X_TA_test = X_TA.iloc[train_idx].copy(), X_TA.iloc[test_idx].copy()
        X_EC_train, X_EC_test = X_EC.iloc[train_idx].copy(), X_EC.iloc[test_idx].copy()
        X_train, X_test = X_DRP.iloc[train_idx].copy(), X_DRP.iloc[test_idx].copy()
        y_TA_train, y_TA_test = y_TA.iloc[train_idx], y_TA.iloc[test_idx]
        y_EC_train, y_EC_test = y_EC.iloc[train_idx], y_EC.iloc[test_idx]
        y_DRP_train = y_DRP_model.iloc[train_idx]
        y_DRP_test_orig = y_DRP.iloc[test_idx]

        if aggregation_cols:
            fp = FoldPreprocessor()
            fp.fit(X_train)
            all_ta = list(X_TA.columns) + aggregation_cols
            all_ec = list(X_EC.columns) + aggregation_cols
            all_drp = list(X_DRP.columns) + aggregation_cols
            X_TA_train = fp.transform(X_TA_train)[all_ta]
            X_TA_test = fp.transform(X_TA_test)[all_ta]
            X_EC_train = fp.transform(X_EC_train)[all_ec]
            X_EC_test = fp.transform(X_EC_test)[all_ec]
            X_train = fp.transform(X_train)[all_drp]
            X_test = fp.transform(X_test)[all_drp]

        # Fold 내 train median으로만 impute (val 분포 누수 방지)
        m_ta = X_TA_train.median()
        X_TA_train = X_TA_train.fillna(m_ta)
        X_TA_test = X_TA_test.fillna(m_ta)
        m_ec = X_EC_train.median()
        X_EC_train = X_EC_train.fillna(m_ec)
        X_EC_test = X_EC_test.fillna(m_ec)
        m_drp = X_train.median()
        X_train = X_train.fillna(m_drp)
        X_test = X_test.fillna(m_drp)

        # Fold별 TA, EC 모델 학습 (각각의 피처 사용)
        _cfg_TA = config_TA or MODEL_CONFIG_TA
        _cfg_EC = config_EC or MODEL_CONFIG_EC
        X_TA_train_s, X_TA_test_s, _ = scale_data(X_TA_train, X_TA_test)
        X_EC_train_s, X_EC_test_s, _ = scale_data(X_EC_train, X_EC_test)
        model_TA_f = train_model(X_TA_train_s, y_TA_train, model_config=_cfg_TA, ensemble=False)
        model_EC_f = train_model(X_EC_train_s, y_EC_train, model_config=_cfg_EC, ensemble=False)
        pred_TA_test = model_TA_f.predict(X_TA_test_s)
        pred_EC_test = model_EC_f.predict(X_EC_test_s)

        # (A) Train에도 "예측값"을 쓰기 위해 OOF preds 생성 (제출과 동일 조건)
        # OOF CV = 메인 DRP CV와 동일 n_splits로 정렬 (stacking 정보 누수 방지)
        groups_train = groups[train_idx]
        if use_ta_ec_uncertainty:
            pred_TA_train_oof, pred_TA_train_std = oof_predictions_with_uncertainty(
                X_TA_train, y_TA_train, groups_train, n_splits=n_folds, model_config=_cfg_TA
            )
            pred_EC_train_oof, pred_EC_train_std = oof_predictions_with_uncertainty(
                X_EC_train, y_EC_train, groups_train, n_splits=n_folds, model_config=_cfg_EC
            )
        else:
            pred_TA_train_oof = oof_predictions_with_groups(
                X_TA_train, y_TA_train, groups_train, n_splits=n_folds, model_config=_cfg_TA
            )
            pred_EC_train_oof = oof_predictions_with_groups(
                X_EC_train, y_EC_train, groups_train, n_splits=n_folds, model_config=_cfg_EC
            )

        # Test용 uncertainty: train OOF std의 평균으로 근사 (test는 1개 모델만 있어 정확한 std 불가)
        if use_ta_ec_uncertainty:
            pred_TA_test_std = np.full(len(test_idx), np.mean(pred_TA_train_std))
            pred_EC_test_std = np.full(len(test_idx), np.mean(pred_EC_train_std))

        # DRP용 피처: [X_DRP, pred_TA, pred_EC] (+ pred_TA_std, pred_EC_std)
        # fold 내 aggregation_cols 추가 시 X_train/X_test 열 수가 X_DRP와 다르므로 실제 columns 사용
        X_train_drp = X_train.copy()
        X_test_drp = X_test.copy()
        if use_ta_ec_true_in_cv:
            X_train_drp["pred_TA"] = y_TA_train.values
            X_train_drp["pred_EC"] = y_EC_train.values
            X_test_drp["pred_TA"] = y_TA_test.values
            X_test_drp["pred_EC"] = y_EC_test.values
        else:
            X_train_drp["pred_TA"] = pred_TA_train_oof
            X_train_drp["pred_EC"] = pred_EC_train_oof
            X_test_drp["pred_TA"] = pred_TA_test
            X_test_drp["pred_EC"] = pred_EC_test
        if use_ta_ec_uncertainty:
            X_train_drp["pred_TA_std"] = pred_TA_train_std
            X_train_drp["pred_EC_std"] = pred_EC_train_std
            X_test_drp["pred_TA_std"] = pred_TA_test_std
            X_test_drp["pred_EC_std"] = pred_EC_test_std

        # Fold 내 DRP top-K 피처 선택 (CV 누수 방지). leaky일 때는 생략
        if use_fold_safe_top_k and not leaky and len(X_train_drp.columns) > drp_top_k:
            fsb = FoldSafeFeatureBuilder(top_k=drp_top_k)
            fsb.fit(X_train_drp.fillna(X_train_drp.median()), y_DRP_train)
            X_train_drp = fsb.transform(X_train_drp)
            X_test_drp = fsb.transform(X_test_drp)

        if leaky:
            med = X_full_drp_leaky.median()
            cols = X_full_drp_leaky.columns
            X_train_drp = X_train_drp.reindex(columns=cols).fillna(med)
            X_test_drp = X_test_drp.reindex(columns=cols).fillna(med)
            scaler_drp_f = StandardScaler()
            scaler_drp_f.fit(X_full_drp_leaky)
            X_train_drp_s = scaler_drp_f.transform(X_train_drp)
            X_test_drp_s = scaler_drp_f.transform(X_test_drp)
        else:
            X_train_drp_s, X_test_drp_s, scaler_drp_f = scale_data(X_train_drp, X_test_drp)
        _drp_cfg = drp_config_override or MODEL_CONFIG_DRP
        model_DRP_f = train_model(
            X_train_drp_s, y_DRP_train, model_config=_drp_cfg,
            eval_set=[(X_test_drp_s, y_DRP_model.iloc[test_idx])],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )

        raw_pred = model_DRP_f.predict(X_test_drp_s)
        if log_target and not residual_mode:
            raw_pred = np.expm1(raw_pred)

        if residual_mode:
            gems_test = X_test_drp["gems_DRP"].values if "gems_DRP" in X_test_drp.columns else np.zeros(len(X_test_drp))
            gems_test_safe = np.maximum(np.nan_to_num(gems_test, nan=0.0), 0)
            # 거리 가중 prior: 멀수록 prior 신뢰도 하락
            if distance_km is not None:
                dist = np.nan_to_num(distance_km.iloc[test_idx].values, nan=999.0)
                prior_w = np.exp(-dist / _decay_km)
                gems_test_safe = gems_test_safe * prior_w
            if log_residual:
                # final = expm1(log1p(prior) + pred_residual)
                g = np.asarray(gems_test_safe, dtype=np.float64)
                y_pred = np.maximum(np.expm1(np.log1p(g) + raw_pred), 0)
            else:
                y_pred = np.maximum(gems_test_safe + raw_pred, 0)
        else:
            y_pred = raw_pred.copy()
            if blend_gems > 0 and "gems_DRP" in X_test_drp.columns:
                gems_val = np.maximum(X_test_drp["gems_DRP"].values, 0)
                y_pred = (1 - blend_gems) * y_pred + blend_gems * gems_val
        # (B) DRP는 0 이상이므로 클리핑
        y_pred = np.maximum(y_pred, 0)
        r2 = r2_score(y_DRP_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_DRP_test_orig, y_pred))
        fold_results.append({"fold": fold + 1, "R2": r2, "RMSE": rmse})
        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}: R²={r2:.3f}, RMSE={rmse:.3f}")

    r2_mean = np.mean([r["R2"] for r in fold_results])
    r2_std = np.std([r["R2"] for r in fold_results])
    rmse_mean = np.mean([r["RMSE"] for r in fold_results])
    rmse_std = np.std([r["RMSE"] for r in fold_results])
    if verbose:
        print(f"\n  CV Mean: R²={r2_mean:.3f} ± {r2_std:.3f}, RMSE={rmse_mean:.3f} ± {rmse_std:.3f}")

    if leaky:
        res = {"Parameter": "Dissolved Reactive Phosphorus", "R2_CV_mean": r2_mean, "R2_CV_std": r2_std, "RMSE_CV_mean": rmse_mean, "RMSE_CV_std": rmse_std}
        return None, None, pd.DataFrame([res]), None, None, None, None, None, None

    # Final: 전체 데이터에서도 pred_TA/pred_EC를 "예측값"으로 맞춤 (제출과 동일)
    X_TA_final = X_TA
    X_EC_final = X_EC
    X_DRP_final = X_DRP
    if aggregation_cols:
        fp_full = FoldPreprocessor()
        fp_full.fit(X_DRP)
        all_ta = list(X_TA.columns) + aggregation_cols
        all_ec = list(X_EC.columns) + aggregation_cols
        all_drp = list(X_DRP.columns) + aggregation_cols
        X_TA_final = fp_full.transform(X_TA.copy())[all_ta]
        X_EC_final = fp_full.transform(X_EC.copy())[all_ec]
        X_DRP_final = fp_full.transform(X_DRP.copy())[all_drp]
    # 최종 모델용: 전체 train median으로 impute
    X_TA_final = X_TA_final.fillna(X_TA_final.median())
    X_EC_final = X_EC_final.fillna(X_EC_final.median())
    X_DRP_final = X_DRP_final.fillna(X_DRP_final.median())
    X_scaled_full_TA = scaler_TA.transform(X_TA_final)
    pred_TA_full = model_TA.predict(X_scaled_full_TA)
    X_scaled_full_EC = scaler_EC.transform(X_EC_final)
    pred_EC_full = model_EC.predict(X_scaled_full_EC)

    X_full_drp = pd.DataFrame(X_DRP_final.values, columns=X_DRP_final.columns, index=X_DRP_final.index).copy()
    X_full_drp['pred_TA'] = pred_TA_full
    X_full_drp['pred_EC'] = pred_EC_full
    if use_ta_ec_uncertainty:
        _, pred_TA_std_full = oof_predictions_with_uncertainty(
            X_TA, y_TA, groups, n_splits=n_folds, model_config=config_TA or MODEL_CONFIG_TA
        )
        _, pred_EC_std_full = oof_predictions_with_uncertainty(
            X_EC, y_EC, groups, n_splits=n_folds, model_config=config_EC or MODEL_CONFIG_EC
        )
        X_full_drp['pred_TA_std'] = pred_TA_std_full
        X_full_drp['pred_EC_std'] = pred_EC_std_full

    scaler_drp_full = StandardScaler()
    X_drp_scaled = scaler_drp_full.fit_transform(X_full_drp)
    _drp_cfg = drp_config_override or MODEL_CONFIG_DRP
    model_DRP_full = train_model(X_drp_scaled, y_DRP_model, model_config=_drp_cfg)

    # DRP feature importance (XGBoost)
    if verbose and USE_XGBOOST and hasattr(model_DRP_full, "feature_importances_"):
        feat_names = X_full_drp.columns.tolist()
        imp = model_DRP_full.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        imp_df = imp_df[imp_df["importance"] > 0]
        print("\n  DRP Feature Importance (XGBoost, non-zero only):")
        print(imp_df.to_string(index=False) if len(imp_df) else "  (all zero)")
        gems_imp = imp_df[imp_df["feature"] == "gems_DRP"]["importance"].values
        if len(gems_imp) > 0 and gems_imp[0] > 0.5:
            print("\n  ⚠️ gems_DRP importance > 50%: model이 prior에 과의존 중.")

    results = {
        "Parameter": "Dissolved Reactive Phosphorus",
        "R2_CV_mean": r2_mean,
        "R2_CV_std": r2_std,
        "RMSE_CV_mean": rmse_mean,
        "RMSE_CV_std": rmse_std,
    }
    if use_ta_ec_uncertainty:
        pred_TA_std_mean, pred_EC_std_mean = np.mean(pred_TA_std_full), np.mean(pred_EC_std_full)
    else:
        pred_TA_std_mean, pred_EC_std_mean = None, None
    return model_DRP_full, scaler_drp_full, pd.DataFrame([results]), log_target, residual_mode, log_residual, _decay_km, pred_TA_std_mean, pred_EC_std_mean


# 피처 옵션
USE_SPATIAL_FEATURES = True   # 위도·경도 추가
USE_TA_EC_FOR_DRP = True     # DRP 모델에 TA·EC 예측값을 피처로 사용
DRP_USE_TA_EC_UNCERTAINTY = False  # pred_TA_std/pred_EC_std OFF (DRP에서 노이즈 증가)
DRP_LOG_TARGET = False       # DRP: log(1+y) 변환 끔
DRP_USE_RESIDUAL = True       # residual 학습: target=DRP-gems_DRP, final=gems_DRP+residual_pred
DRP_USE_LOG_RESIDUAL = False  # log residual OFF (DRP 악화 원인 → linear residual 사용)
PRIOR_DECAY_KM = 25.0         # 거리 가중 prior: weight = exp(-distance_km / PRIOR_DECAY_KM)
TUNE_PRIOR_DECAY = False      # decay 튜닝 OFF (DRP에 거의 영향 없음)
PRIOR_DECAY_CANDIDATES = [15.0, 25.0, 50.0, 75.0]  # km
DRP_BLEND_GEMS = 0.0          # blend OFF (baseline)
DRP_BLEND_WEIGHTS_TUNE = []    # blend OFF
DRP_COMPARE_RESIDUAL = False   # 비교 생략, residual 고정 (baseline 복원)
DRP_USE_TA_EC_TRUE_IN_CV = False  # False=OOF 사용 (제출과 동일 조건, 실전형 CV)

# 파라미터별 모델 아키텍처 & 튜닝 (XGBoost/RF)
MODEL_CONFIG_TA = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
}
MODEL_CONFIG_EC = {
    "n_estimators": 800,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}
MODEL_CONFIG_DRP = {
    "n_estimators": 800,
    "max_depth": 5,   # baseline 복원 (4→5), 피처 수 감소로 과적합 리스크 ↓
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.15,   # baseline 복원 (0.2→0.15)
    "reg_lambda": 1.5,   # baseline 복원 (2.0→1.5)
}

# 하이퍼파라미터 튜닝 (TUNE_HYPERPARAMS=True 시 실행)
TUNE_HYPERPARAMS = True
TUNE_N_ITER = 20       # RandomizedSearchCV 시도 횟수 (TA, EC 각각)
TUNE_DRP_N_ITER = 10   # DRP config 시도 횟수

PARAM_GRID_TA = {
    "n_estimators": [500, 700, 800, 1000, 1200],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
    "subsample": [0.7, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.85, 0.9],
    "min_child_weight": [1, 2, 3, 5],
    "gamma": [0, 0.05, 0.1, 0.2],
    "reg_alpha": [0.01, 0.05, 0.1, 0.2],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}
PARAM_GRID_EC = {
    "n_estimators": [500, 700, 800, 1000, 1200],
    "max_depth": [5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
    "subsample": [0.7, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.85, 0.9],
    "min_child_weight": [2, 3, 5],
    "gamma": [0, 0.05, 0.1, 0.2],
    "reg_alpha": [0.05, 0.1, 0.2],
    "reg_lambda": [1.0, 1.5, 2.0],
}
PARAM_GRID_DRP = {
    "n_estimators": [500, 700, 800, 1000],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.02, 0.03, 0.04, 0.05],
    "subsample": [0.7, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.85, 0.9],
    "min_child_weight": [2, 3, 5],
    "gamma": [0, 0.05, 0.1],
    "reg_alpha": [0.05, 0.1, 0.15, 0.2],
    "reg_lambda": [1.0, 1.5, 2.0],
}


def tune_single_target(X, y, groups, param_name, param_grid, n_iter=24):
    """RandomizedSearchCV with GroupKFold. Returns best config dict."""
    if not USE_XGBOOST:
        print(f"  [Skip tuning for {param_name}: XGBoost 없음]")
        return None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(random_state=42, verbosity=0)),
    ])
    # param_grid keys need "model__" prefix
    search_grid = {f"model__{k}": v for k, v in param_grid.items()}

    gkf = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        pipe,
        search_grid,
        n_iter=n_iter,
        cv=gkf,
        scoring="r2",
        random_state=42,
        n_jobs=1,  # 메모리 안정
        verbose=1,  # 진행률 출력
    )
    search.fit(X, y, groups=groups)

    best_params = search.best_params_
    config = {k.replace("model__", ""): v for k, v in best_params.items()}
    print(f"  Best {param_name}: R²={search.best_score_:.3f}")
    print(f"    {config}")
    return config


def tune_drp_config(X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups, model_TA, model_EC, scaler_TA, scaler_EC, config_TA, config_EC, n_iter=16, distance_km=None):
    """DRP config 그리드 서치 (run_pipeline_drp 호출)."""
    import random
    random.seed(42)
    _agg = AGGREGATION_FEATURES if USE_FOLD_SAFE_AGGREGATIONS else None
    best_r2 = -np.inf
    best_config = MODEL_CONFIG_DRP.copy()

    # Random sampling from param grid
    configs = []
    for _ in range(n_iter):
        cfg = {k: random.choice(v) for k, v in PARAM_GRID_DRP.items()}
        configs.append(cfg)

    for i, cfg in enumerate(configs):
        print(f"    DRP tune {i+1}/{n_iter}...")
        # Temporarily override MODEL_CONFIG_DRP for run_pipeline_drp
        # We need to pass config to run_pipeline_drp - it uses MODEL_CONFIG_DRP
        # So we need to modify run_pipeline_drp to accept an optional drp_config override
        # For now, we'll patch globally - but that's not clean. Let me add drp_config param.
        _, _, res, _, _, _, _, _, _ = run_pipeline_drp(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            log_target=DRP_LOG_TARGET,
            blend_gems=DRP_BLEND_GEMS,
            residual_mode=DRP_USE_RESIDUAL,
            log_residual=DRP_USE_LOG_RESIDUAL,
            use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
            drp_config_override=cfg,
            config_TA=config_TA,
            config_EC=config_EC,
            distance_km=distance_km,
            aggregation_cols=_agg,
            verbose=False,
        )
        r2 = res["R2_CV_mean"].values[0]
        if r2 > best_r2:
            best_r2 = r2
            best_config = cfg.copy()
        print(f"      → R²={r2:.3f} (best={best_r2:.3f})")

    print(f"  Best DRP: R²={best_r2:.3f}")
    print(f"    {best_config}")
    return best_config


# GEMS 피처 사용 여부 (create_enriched_dataset.py 실행 후 True)
USE_GEMS_FEATURES = True
GEMS_FEATURE_COLS = [
    'gems_pH', 'gems_Ca_Dis', 'gems_Mg_Dis', 'gems_Cl_Dis',
    'gems_SO4_Dis', 'gems_Na_Dis', 'gems_Si_Dis',
    'gems_Alk_Tot', 'gems_EC',
    'gems_DRP', 'gems_DRP_log',  # blend용
    'gems_TP', 'gems_TP_log',
    'gems_partial_P',  # DRP 전용 파생
    'gems_H_T', 'gems_Sal', 'gems_NH4N', 'gems_NOxN',
]

# 파라미터별 피처 목록 (존재하는 컬럼만 사용하도록 main에서 필터링)
USE_DERIVED_FEATURES = True   # 위성·기후·GEMS 파생 피처 사용
DROP_FEATURES = []           # importance 낮은 피처 드랍 (예: ["WRI", "green_nir_ratio"])
BASE_FEATURES = [
    'Latitude', 'Longitude', 'swir22', 'NDMI', 'MNDWI', 'pet',
    'month', 'dayofyear', 'sin_doy', 'cos_doy',
]
# 파생 피처 (USE_DERIVED_FEATURES 시 추가, 존재하는 컬럼만 사용)
DERIVED_FEATURES_COMMON = [
    'WRI', 'green_nir_ratio', 'green_swir22_ratio', 'nir_swir16_ratio', 'nir_swir22_ratio',
    'NDMI_sq', 'MNDWI_sq', 'NDMI_MNDWI', 'NDMI_anom', 'MNDWI_anom',
    'log1p_pet', 'pet_norm', 'pet_seasonal_anom', 'pet_sin_doy', 'pet_cos_doy', 'pet_lat',
    'season', 'growing_season',
]
DERIVED_FEATURES_TA = ['hardness_like', 'cation_balance', 'Ca_Mg_ratio']
DERIVED_FEATURES_EC = ['major_ions', 'na_cl_ratio', 'sulphate_chloride_ratio', 'sal_over_EC', 'dry_index']
DERIVED_FEATURES_DRP_COMMON = [
    'NDMI_sq', 'MNDWI_sq', 'NDMI_MNDWI', 'NDMI_anom', 'MNDWI_anom',
    'log1p_pet', 'pet_seasonal_anom', 'season',
]
DERIVED_FEATURES_DRP = [
    'N_total_like', 'N_to_P_ratio', 'DRP_to_TP_ratio',
    'wet_TP', 'wet_DRP', 'NDMI_TP', 'MNDWI_TP',
]

# Step 1용: pure satellite+climate (gems 제거)
FEATURES_TA_PURE = BASE_FEATURES
FEATURES_EC_PURE = BASE_FEATURES
FEATURES_DRP_PURE = BASE_FEATURES + ['wet_index', 'water_stress']

# 기본: GEMS 포함
FEATURES_TA = BASE_FEATURES + [
    'gems_Alk_Tot', 'gems_Ca_Dis', 'gems_Mg_Dis', 'gems_Si_Dis', 'gems_pH', 'gems_H_T',
]
FEATURES_EC = BASE_FEATURES + [
    'gems_EC', 'gems_Cl_Dis', 'gems_SO4_Dis', 'gems_Na_Dis',
    'gems_Ca_Dis', 'gems_Mg_Dis', 'gems_Sal', 'gems_pH',
]
# baseline (0.2889 복원): gems_DRP feature 포함 | residual ON | blend OFF
FEATURES_DRP = BASE_FEATURES + [
    'wet_index', 'water_stress',
    'gems_TP', 'gems_TP_log', 'gems_NOxN', 'gems_NH4N', 'gems_pH',
    'gems_DRP', 'gems_DRP_log', 'gems_partial_P',
]

# 확장 피처 (ERA5, 강수 이상치, 외부, HydroRIVERS) — 해당 컬럼 있으면 자동 포함
EXTENDED_BASE = [
    "pr", "era5_rh", "era5_sm",
    "precip_anom", "cumulative_anom_3m", "precip_anom_lag1", "wetness_index",
    "precip_anom_lag2", "storm_pet_ratio", "cumulative_wetness_3m", "anom_pos", "anom_neg",
]
EXTENDED_EXTERNAL = [
    "soil_clay_pct", "soil_organic_carbon", "soil_ph", "elevation_m",
    "population_density", "lc_tree_pct", "lc_shrub_pct", "lc_grassland_pct",
    "lc_cropland_pct", "lc_urban_pct", "lc_bare_pct", "lc_water_pct",
]
EXTENDED_HYRIV_ERA5 = [
    "river_dist_m", "hyriv_dist_bin", "hyriv_log_upcells", "hyriv_flow_order",
    "hyriv_log_q", "hyriv_q_over_up", "hyriv_order_x_up",
    "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
    "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
    "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
    "sm_mean_1m", "sm_mean_3m", "sm_lag_1m", "sm_lag_3m",
    "wetness_rain_sm_1m", "wetness_rain_sm_3m", "dilution_proxy_1m", "ionic_flush_proxy_3m",
]
# TA/EC/DRP 전용 확장 (te_*, lag_*, clay_pet, elev_pet, soc_NDMI, log_pop_density 등)
EXTENDED_TA = ["te_tot_spatial", "lag_tot_prev", "lag_tot_days", "clay_pet", "elev_pet", "soc_NDMI", "hardness_season", "gems_Alk_Tot_NDMI"]
EXTENDED_EC = ["te_ele_spatial", "lag_ele_prev", "lag_ele_days", "clay_pet", "elev_pet", "log_pop_density", "gems_EC_NDMI", "major_ions_season"]
EXTENDED_DRP = [
    "te_dis_spatial", "lag_dis_prev", "lag_dis_days", "lag_tot_prev", "lag_ele_prev",
    "gems_TP_season", "precip_anom_mm", "precip_anom_TP", "precip_anom_DRP", "precip_anom_N",
    "cum_anom_3m_TP", "cum_anom_3m_DRP", "wetness_TP", "wetness_DRP", "anom_pos_TP", "anom_neg_DRP",
    "storm_pet_TP", "pr_TP", "pr_DRP", "drp_extreme", "precip_anom_extreme", "wetness_extreme", "storm_spike",
    "partialP_ratio", "MNDWI_N", "cum_anom_3m_N", "cropland_TP", "cropland_precip", "urban_pop",
    "clay_pet", "elev_pet", "soc_NDMI", "log_pop_density",
]


def main():
    import os
    use_pure_features = EXPERIMENT_STEP in (1, 2)
    use_catchment_cv = EXPERIMENT_STEP == 3
    use_weak_gems_blend = EXPERIMENT_STEP == 2

    # 로그 형식: Ensemble 활성, USE_GPU
    ensemble_active = []
    if USE_XGBOOST:
        ensemble_active.append("XGBoost")
    else:
        ensemble_active.append("-")
    ensemble_active.append("LightGBM" if USE_LIGHTGBM else "-")
    ensemble_active.append("CatBoost" if USE_CATBOOST else "-")
    ensemble_active.append("HistGBM" if USE_HIST_GBM else "-")
    ensemble_active.append("ExtraTrees" if USE_EXTRA_TREES else "-")
    use_gpu = os.environ.get("USE_GPU", "True")
    gpu_note = " (환경변수 USE_GPU=0 으로 CPU 강제)" if use_gpu != "0" else ""

    print("\n" + "=" * 60)
    print("EY 2026 Water Quality Benchmark 시작")
    print(f"  Ensemble: XGB + LGB + CatBoost + HistGBM + ExtraTrees (활성: {ensemble_active})")
    print(f"  USE_GPU={use_gpu}{gpu_note}")
    print(f"  EXPERIMENT_STEP={EXPERIMENT_STEP}: "
          + ("pure satellite+climate" if use_pure_features else "GEMS 포함")
          + (", catchment CV" if use_catchment_cv else "")
          + (", weak blend 0.85*model+0.15*gems" if use_weak_gems_blend else ""))
    print("=" * 60)

    # Step 2: prediction시 gems blend 필요 → enriched 로드
    need_gems_for_blend = use_weak_gems_blend
    # --- Load training data ---
    print("\n[데이터 로드] ...")
    wq_path = (
        DATA_DIR / "water_quality_training_dataset_enriched.csv"
        if (USE_GEMS_FEATURES or need_gems_for_blend)
        else DATA_DIR / "water_quality_training_dataset.csv"
    )
    if not wq_path.exists() and (USE_GEMS_FEATURES or need_gems_for_blend):
        print("Enriched 데이터 없음. create_enriched_dataset.py 먼저 실행하세요.")
        wq_path = DATA_DIR / "water_quality_training_dataset.csv"
        use_gems = False
    else:
        use_gems = (USE_GEMS_FEATURES or need_gems_for_blend) and wq_path.exists()

    Water_Quality_df = pd.read_csv(wq_path)
    n_wq, c_wq = Water_Quality_df.shape
    print(f"Water Quality Training Data ({wq_path.name}):")
    print(f"  rows={n_wq}, cols={c_wq}")
    if "gems_DRP" in Water_Quality_df.columns:
        nn = Water_Quality_df["gems_DRP"].notna().sum()
        print(f"  gems_DRP: non-null={nn} (DRP prior에 필수)")
    print(Water_Quality_df.head())

    landsat_train_features = pd.read_csv(
        DATA_DIR / "landsat_features_training.csv"
    )
    print("\nLandsat Training Features:")
    print(landsat_train_features.head())

    Terraclimate_df = pd.read_csv(
        DATA_DIR / "terraclimate_features_training.csv"
    )
    print("\nTerraClimate Training Features:")
    print(Terraclimate_df.head())

    # --- Combine datasets ---
    wq_data = combine_two_datasets(
        Water_Quality_df, landsat_train_features, Terraclimate_df
    )
    n_comb, c_comb = wq_data.shape
    print(f"  combine 후: rows={n_comb}, cols={c_comb}")

    # ERA5 pr 병합 (precipitation_training.csv) — wq_data에 Sample Date 있음
    if USE_ERA5_PRECIP and (DATA_DIR / "precipitation_training.csv").exists():
        pr_df = pd.read_csv(DATA_DIR / "precipitation_training.csv")
        key_cols = ["Latitude", "Longitude", "Sample Date"]
        if all(k in wq_data.columns and k in pr_df.columns for k in key_cols):
            add_cols = [c for c in pr_df.columns if c not in key_cols and c not in wq_data.columns]
            if add_cols or "pr" in pr_df.columns:
                merge_cols = key_cols + (add_cols if add_cols else (["pr"] if "pr" in pr_df.columns else []))
                if merge_cols != key_cols:
                    wq_data = wq_data.merge(pr_df[merge_cols], on=key_cols, how="left")
                print("\nTerraClimate + ERA5 (pr): merged")

    # --- Add seasonality and wetness features (before feature selection) ---
    wq_data = add_seasonality_features(wq_data)
    wq_data = add_wetness_features(wq_data)

    # 강수 이상치 피처 병합 (water_quality_with_precip_anomaly.csv)
    precip_anom_path = DATA_DIR / "water_quality_with_precip_anomaly.csv"
    if USE_PRECIP_ANOMALY and precip_anom_path.exists():
        pa_df = pd.read_csv(precip_anom_path)
        key_cols = ["Latitude", "Longitude", "Sample Date"]
        extra = [c for c in pa_df.columns if c not in wq_data.columns and c not in key_cols]
        if extra and all(k in wq_data.columns and k in pa_df.columns for k in key_cols):
            pa_merge = pa_df[key_cols + extra].copy()
            wq_data = wq_data.merge(pa_merge, on=key_cols, how="left")
            print(f"\nPrecip anomaly features merged: {', '.join(extra)}")

    # 파생 피처 (위성·기후·GEMS 조합). 집계 피처는 FoldPreprocessor에서 fold별 계산 (누수 방지)
    if USE_DERIVED_FEATURES:
        wq_data = add_derived_features(wq_data, skip_aggregations=USE_FOLD_SAFE_AGGREGATIONS)

    # 외부 + HydroRIVERS + ERA5 이벤트 병합 — 키 기반 merge만 사용 (행 순서 가정 금지, 누수/미스매치 방지)
    train_with_path = DATA_DIR / "train_with_hyriv_era5_events.csv"
    key_ext = ["Latitude", "Longitude", "Sample Date"]
    if (USE_EXTERNAL_FEATURES or USE_HYRIV_ERA5_EVENTS) and train_with_path.exists():
        tw = pd.read_csv(train_with_path)
        target_cols_here = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
        skip = set(target_cols_here) | set(key_ext)
        to_add = [c for c in tw.columns if c not in wq_data.columns and c not in skip]
        merge_on = [k for k in key_ext if k in wq_data.columns and k in tw.columns]
        if to_add and len(merge_on) == len(key_ext):
            tw_sub = tw[merge_on + to_add].drop_duplicates(subset=merge_on, keep="first")
            wq_data = wq_data.merge(tw_sub, on=merge_on, how="left")
            external_names = [c for c in to_add if any(c.startswith(p) for p in ["soil_", "elevation", "population", "lc_", "log_pop"])]
            hyriv_era5_names = [c for c in to_add if c.startswith("river_") or c.startswith("hyriv_") or c.startswith("rain_") or c.startswith("storm_") or c.startswith("sm_") or "wetness_rain" in c or c.startswith("dilution_") or c.startswith("ionic_")]
            if external_names:
                print(f"\nExternal features merged (key={merge_on}): {', '.join(external_names)}")
            if hyriv_era5_names or to_add:
                print(f"\n  [HydroRIVERS+ERA5] 피처 목록에 추가: {len(to_add)} (key merge)")
            if to_add and not external_names and not hyriv_era5_names:
                print(f"\nExternal features merged (key={merge_on}): {len(to_add)} cols")
        elif to_add and merge_on:
            tw_sub = tw[merge_on + [c for c in to_add if c in tw.columns]].drop_duplicates(subset=merge_on, keep="first")
            wq_data = wq_data.merge(tw_sub, on=merge_on, how="left")
            print(f"\nExternal features merged (key={merge_on}, no Sample Date in file): {len(to_add)} cols")

    # gems_partial_P: DRP 전용 파생 피처 (입자성 인 프록시), enriched에 없으면 여기서 생성
    if "gems_TP" in wq_data.columns and "gems_DRP" in wq_data.columns and "gems_partial_P" not in wq_data.columns:
        wq_data["gems_partial_P"] = np.maximum(
            wq_data["gems_TP"].fillna(0).values - wq_data["gems_DRP"].fillna(0).values, 0
        )

    n_final, c_final = wq_data.shape
    precip_cols = [c for c in ["pr", "precip_anom", "precip_anom_mm", "pr_TP", "pr_DRP"] if c in wq_data.columns]
    if precip_cols:
        print("\nPrecipitation columns present: " + str(precip_cols))
        for pc in precip_cols[:5]:
            nn = wq_data[pc].notna().sum()
            pct = 100.0 * nn / n_final if n_final else 0
            print(f"  {pc}: non-null = {nn} ({pct:.1f}%)")

    print("\nCombined Data:")
    print(f"  rows={n_final}, cols={c_final}")
    print(f"  [DRP 필수] use_gems={use_gems}, gems_DRP in wq_data={'gems_DRP' in wq_data.columns}")
    if "gems_DRP" in wq_data.columns:
        nn_drp = wq_data["gems_DRP"].notna().sum()
        print(f"  gems_DRP non-null: {nn_drp}/{n_final} (prior=0이면 DRP R² 급락)")
    print(wq_data.head())

    # --- Preprocessing (fillna는 CV fold 내 train median으로만 처리 → 누수 방지) ---
    nan_before = wq_data.isna().sum().sum()
    print("\nMissing values (CV 전 global fillna 없음, fold 내 train median으로 impute):")
    print(f"{int(nan_before)} total NaN")

    if USE_FOLD_SAFE_DRP_TOP_K and not use_pure_features:
        print("\n  DRP 피처 선택: fold 내부에서 수행 (top-50, CV 누수 방지)")
    if (USE_EXTERNAL_FEATURES or USE_HYRIV_ERA5_EVENTS) and train_with_path.exists():
        print("\n[HydroRIVERS+ERA5] FoldSafeFeatureBuilder 생성 (fold 내 fit/transform)")

    target_cols = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

    # 파라미터별 피처 (존재하는 컬럼만)
    _feat_ta = FEATURES_TA_PURE if use_pure_features else FEATURES_TA
    _feat_ec = FEATURES_EC_PURE if use_pure_features else FEATURES_EC
    _feat_drp = FEATURES_DRP_PURE if use_pure_features else FEATURES_DRP
    if USE_DERIVED_FEATURES and not use_pure_features:
        _feat_ta = _feat_ta + DERIVED_FEATURES_COMMON + DERIVED_FEATURES_TA
        _feat_ec = _feat_ec + DERIVED_FEATURES_COMMON + DERIVED_FEATURES_EC
        _feat_drp = _feat_drp + DERIVED_FEATURES_DRP_COMMON + DERIVED_FEATURES_DRP
    # 확장 피처 (ERA5, 강수 이상치, 외부, HydroRIVERS) — 컬럼 있으면 추가
    if not use_pure_features:
        ext_common = [c for c in (EXTENDED_BASE + EXTENDED_EXTERNAL + EXTENDED_HYRIV_ERA5) if c in wq_data.columns]
        if ext_common:
            _feat_ta = list(dict.fromkeys(_feat_ta + ext_common + [c for c in EXTENDED_TA if c in wq_data.columns]))
            _feat_ec = list(dict.fromkeys(_feat_ec + ext_common + [c for c in EXTENDED_EC if c in wq_data.columns]))
            _feat_drp = list(dict.fromkeys(_feat_drp + ext_common + [c for c in EXTENDED_DRP if c in wq_data.columns]))
    FEATURES_TA_f = [c for c in _feat_ta if c in wq_data.columns and c not in DROP_FEATURES]
    FEATURES_EC_f = [c for c in _feat_ec if c in wq_data.columns and c not in DROP_FEATURES]
    FEATURES_DRP_f = [c for c in _feat_drp if c in wq_data.columns and c not in DROP_FEATURES]
    if USE_FOLD_SAFE_AGGREGATIONS:
        for a in AGGREGATION_FEATURES:
            if a not in FEATURES_TA_f:
                FEATURES_TA_f.append(a)
            if a not in FEATURES_EC_f:
                FEATURES_EC_f.append(a)
            if a not in FEATURES_DRP_f:
                FEATURES_DRP_f.append(a)

    if not use_gems and not use_pure_features:
        FEATURES_TA_f = [c for c in FEATURES_TA_f if not c.startswith('gems_')]
        FEATURES_EC_f = [c for c in FEATURES_EC_f if not c.startswith('gems_')]
        FEATURES_DRP_f = [c for c in FEATURES_DRP_f if not c.startswith('gems_')]

    cols_to_keep = list(set(FEATURES_TA_f + FEATURES_EC_f + FEATURES_DRP_f + target_cols) & set(wq_data.columns))
    if "gems_distance_km" in wq_data.columns:
        cols_to_keep.append("gems_distance_km")  # 거리 가중 prior용
    wq_data = wq_data[cols_to_keep]

    # Spatial CV: location / catchment / spatial block
    if USE_SPATIAL_BLOCK_CV:
        if SPATIAL_BLOCK_COMPARE:
            print("\n" + "=" * 60)
            print("Spatial Block CV 비교 (0.25°~28km, 0.5°~56km, 1.0°~111km)")
            print("  → 0.25°: 세밀(블록 많음) 0.5°: 균형 1.0°: 거침(블록 적음)")
            print("=" * 60)
            best_block_TA = 0.5
            best_block_EC = 0.5
            best_r2_TA = -np.inf
            best_r2_EC = -np.inf
            X_TA_pre = wq_data[[c for c in FEATURES_TA_f if c in wq_data.columns]].copy()
            X_EC_pre = wq_data[[c for c in FEATURES_EC_f if c in wq_data.columns]].copy()
            y_TA_pre = wq_data['Total Alkalinity']
            y_EC_pre = wq_data['Electrical Conductance']
            for bdeg in SPATIAL_BLOCK_CANDIDATES:
                g = get_spatial_block_groups(wq_data, block_deg=bdeg)
                n_grp = len(np.unique(g))
                _, _, res_ta, _ = run_pipeline(X_TA_pre, y_TA_pre, g, "TA (block compare)", model_config=MODEL_CONFIG_TA, n_folds=5, aggregation_cols=AGGREGATION_FEATURES if USE_FOLD_SAFE_AGGREGATIONS else None)
                r2_ta = res_ta['R2_CV_mean'].values[0]
                print(f"  block={bdeg}°: 그룹 수={n_grp}, TA R²={r2_ta:.3f}", end="")
                if r2_ta > best_r2_TA:
                    best_r2_TA = r2_ta
                    best_block_TA = bdeg
                if SPATIAL_BLOCK_PER_TARGET:
                    _, _, res_ec, _ = run_pipeline(X_EC_pre, y_EC_pre, g, "EC (block compare)", model_config=MODEL_CONFIG_EC, n_folds=5, aggregation_cols=AGGREGATION_FEATURES if USE_FOLD_SAFE_AGGREGATIONS else None)
                    r2_ec = res_ec['R2_CV_mean'].values[0]
                    print(f", EC R²={r2_ec:.3f}", end="")
                    if r2_ec > best_r2_EC:
                        best_r2_EC = r2_ec
                        best_block_EC = bdeg
                print()
            if SPATIAL_BLOCK_PER_TARGET:
                print(f"  → TA 선택: block={best_block_TA}°, EC 선택: block={best_block_EC}°")
                groups_TA = get_spatial_block_groups(wq_data, block_deg=best_block_TA)
                groups_EC = get_spatial_block_groups(wq_data, block_deg=best_block_EC)
                groups = groups_EC  # DRP는 EC 블록 사용 (수질 특성 유사)
                print(f"\n[Spatial Block CV] TA block={best_block_TA}°, EC/DRP block={best_block_EC}°")
            else:
                print(f"  → 선택: block={best_block_TA}° (TA 기준)")
                groups = get_spatial_block_groups(wq_data, block_deg=best_block_TA)
                groups_TA = groups_EC = groups
                print(f"\n[Spatial Block CV] block={best_block_TA}°, 블록 수: {len(np.unique(groups))}")
        else:
            groups = get_spatial_block_groups(wq_data, block_deg=SPATIAL_BLOCK_DEG)
            groups_TA = groups_EC = groups
            print(f"\n[Spatial Block CV] block={SPATIAL_BLOCK_DEG}°, 블록 수: {len(np.unique(groups))}")
    elif use_catchment_cv:
        groups = get_catchment_groups(wq_data, grid_deg=CATCHMENT_GRID_DEG)
        groups_TA = groups_EC = groups
        print(f"\n[Hard CV] catchment 그룹 수: {len(np.unique(groups))} (grid={CATCHMENT_GRID_DEG}°)")
    else:
        groups = get_location_groups(wq_data)
        groups_TA = groups_EC = groups
        print(f"\n고유 측정소 수: {len(np.unique(groups))}")
    print("\nTarget-specific feature lists:")
    print(f"  FEATURES_TA ({len(FEATURES_TA_f)}): {FEATURES_TA_f}")
    print(f"  FEATURES_EC ({len(FEATURES_EC_f)}): {FEATURES_EC_f}")
    print(f"  FEATURES_DRP ({len(FEATURES_DRP_f)}): {FEATURES_DRP_f}")
    # 로그 형식: 확장 버전에서는 여기서 DRP top-50, HydroRIVERS+ERA5, FoldSafeFeatureBuilder 메시지 출력

    X_TA = wq_data[[c for c in FEATURES_TA_f if c in wq_data.columns]].copy()
    X_EC = wq_data[[c for c in FEATURES_EC_f if c in wq_data.columns]].copy()
    X_DRP = wq_data[[c for c in FEATURES_DRP_f if c in wq_data.columns]].copy()
    y_TA = wq_data['Total Alkalinity']
    y_EC = wq_data['Electrical Conductance']
    y_DRP = wq_data['Dissolved Reactive Phosphorus']
    # 거리 가중 prior: gems_distance_km 있으면 전달
    _distance_km = wq_data["gems_distance_km"] if "gems_distance_km" in wq_data.columns else None

    config_TA = MODEL_CONFIG_TA.copy()
    config_EC = MODEL_CONFIG_EC.copy()
    config_DRP = MODEL_CONFIG_DRP.copy()

    # 진행률용 총 스텝 수 (Step 1/2: blend·residual 비교 생략)
    _skip_drp_tuning = use_pure_features
    _n = 2  # TA pipeline, EC pipeline
    if TUNE_HYPERPARAMS:
        _n += 2  # TA tune, EC tune
    if TUNE_HYPERPARAMS and USE_TA_EC_FOR_DRP and not _skip_drp_tuning:
        _n += 1  # DRP tune
    if USE_TA_EC_FOR_DRP and DRP_BLEND_WEIGHTS_TUNE and not _skip_drp_tuning:
        _n += 1  # Blend tune
    if USE_TA_EC_FOR_DRP and DRP_COMPARE_RESIDUAL and not _skip_drp_tuning:
        _n += 1  # Residual vs Blend
    if TUNE_PRIOR_DECAY and USE_TA_EC_FOR_DRP and DRP_USE_RESIDUAL and _distance_km is not None:
        _n += 1  # Prior decay 튜닝
    _n += 1  # Final DRP
    _n += 1  # Validation
    TOTAL_STEPS = _n
    _step = [0]

    def progress(msg):
        _step[0] += 1
        pct = int(100 * _step[0] / TOTAL_STEPS)
        print(f"\n{'='*60}")
        print(f"[{_step[0]}/{TOTAL_STEPS} ({pct}%)] {msg}")
        print("=" * 60)

    # --- 하이퍼파라미터 튜닝 ---
    if TUNE_HYPERPARAMS:
        progress("TA 하이퍼파라미터 튜닝")
        tuned_TA = tune_single_target(X_TA, y_TA, groups_TA, "TA", PARAM_GRID_TA, n_iter=TUNE_N_ITER)
        if tuned_TA:
            config_TA = tuned_TA
        progress("EC 하이퍼파라미터 튜닝")
        tuned_EC = tune_single_target(X_EC, y_EC, groups_EC, "EC", PARAM_GRID_EC, n_iter=TUNE_N_ITER)
        if tuned_EC:
            config_EC = tuned_EC

    # --- Train models (Group K-Fold CV) ---
    _agg = AGGREGATION_FEATURES if USE_FOLD_SAFE_AGGREGATIONS else None
    progress("TA 모델 학습 (Group K-Fold CV)")
    model_TA, scaler_TA, results_TA, fp_TA = run_pipeline(
        X_TA, y_TA, groups_TA, "Total Alkalinity", model_config=config_TA, aggregation_cols=_agg
    )
    progress("EC 모델 학습 (Group K-Fold CV)")
    model_EC, scaler_EC, results_EC, _ = run_pipeline(
        X_EC, y_EC, groups_EC, "Electrical Conductance", model_config=config_EC, aggregation_cols=_agg
    )

    # DRP 튜닝 (TA, EC 모델 필요) - Step 1/2에서는 생략
    if TUNE_HYPERPARAMS and USE_TA_EC_FOR_DRP and not _skip_drp_tuning:
        progress("DRP 하이퍼파라미터 튜닝")
        tuned_DRP = tune_drp_config(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            config_TA, config_EC,
            n_iter=TUNE_DRP_N_ITER,
            distance_km=_distance_km,
        )
        config_DRP = tuned_DRP

    # 1️⃣ Blend weight 튜닝 (residual 대비 비교용) - Step 1/2에서 생략
    best_blend_r2 = -np.inf
    best_blend_w = DRP_BLEND_GEMS
    if use_pure_features:
        best_blend_w = 0.0
        use_residual_final = False
    if USE_TA_EC_FOR_DRP and DRP_BLEND_WEIGHTS_TUNE and not use_pure_features:
        progress(f"Blend weight 튜닝: {DRP_BLEND_WEIGHTS_TUNE}")
        for idx, w in enumerate(DRP_BLEND_WEIGHTS_TUNE):
            print(f"  Blend {idx+1}/{len(DRP_BLEND_WEIGHTS_TUNE)} (w={w})...")
            _, _, res, _, _, _, _, _, _ = run_pipeline_drp(
                X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                model_TA, model_EC, scaler_TA, scaler_EC,
                log_target=DRP_LOG_TARGET,
                blend_gems=w,
                residual_mode=False,
                use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
                drp_config_override=config_DRP,
                config_TA=config_TA,
                config_EC=config_EC,
                distance_km=_distance_km,
                aggregation_cols=_agg,
                verbose=False,
            )
            r2 = res['R2_CV_mean'].values[0]
            print(f"    → R²={r2:.3f}")
            if r2 > best_blend_r2:
                best_blend_r2 = r2
                best_blend_w = w

    # 2️⃣ Residual vs Blend 비교 (DRP_COMPARE_RESIDUAL 시) - Step 1/2에서 이미 설정됨
    if not use_pure_features:
        use_residual_final = DRP_USE_RESIDUAL
    if USE_TA_EC_FOR_DRP and DRP_COMPARE_RESIDUAL and not use_pure_features:
        progress("DRP Residual vs Blend 모드 비교")
        _, _, res_res, _, _, _, _, _, _ = run_pipeline_drp(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            log_target=DRP_LOG_TARGET,
            blend_gems=0.0,
            residual_mode=True,
            log_residual=DRP_USE_LOG_RESIDUAL,
            use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
        drp_config_override=config_DRP,
        config_TA=config_TA,
        config_EC=config_EC,
        distance_km=_distance_km,
        aggregation_cols=_agg,
        verbose=False,
        )
        r2_residual = res_res['R2_CV_mean'].values[0]
        print(f"  Residual 모드: R²={r2_residual:.3f}")

        # Blend R²: 튜닝에서 얻었거나, 지금 한 번 더 실행
        if DRP_BLEND_WEIGHTS_TUNE and best_blend_r2 > -np.inf:
            print(f"  Blend 모드 (best w={best_blend_w}): R²={best_blend_r2:.3f}")
        else:
            _, _, res_blend, _, _, _, _, _, _ = run_pipeline_drp(
                X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                model_TA, model_EC, scaler_TA, scaler_EC,
                log_target=DRP_LOG_TARGET,
                blend_gems=DRP_BLEND_GEMS,
                residual_mode=False,
                use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
                drp_config_override=config_DRP,
                config_TA=config_TA,
                config_EC=config_EC,
                distance_km=_distance_km,
                aggregation_cols=_agg,
                verbose=False,
            )
            best_blend_r2 = res_blend['R2_CV_mean'].values[0]
            best_blend_w = DRP_BLEND_GEMS
            print(f"  Blend 모드 (w={DRP_BLEND_GEMS}): R²={best_blend_r2:.3f}")

        use_residual_final = r2_residual >= best_blend_r2
        print(f"  → 선택: {'Residual' if use_residual_final else f'Blend (w={best_blend_w})'} 모드")

    progress("DRP 최종 모델 학습")
    if USE_TA_EC_FOR_DRP:
        # PRIOR_DECAY_KM 튜닝 (TUNE_PRIOR_DECAY 시)
        _best_decay_km = PRIOR_DECAY_KM
        if TUNE_PRIOR_DECAY and DRP_USE_RESIDUAL and _distance_km is not None:
            progress("DRP prior decay (PRIOR_DECAY_KM) 튜닝")
            best_decay_r2 = -np.inf
            for d in PRIOR_DECAY_CANDIDATES:
                _, _, res_d, _, _, _, _, _, _ = run_pipeline_drp(
                    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                    model_TA, model_EC, scaler_TA, scaler_EC,
                    log_target=DRP_LOG_TARGET,
                    blend_gems=0.0,
                    residual_mode=True,
                    log_residual=DRP_USE_LOG_RESIDUAL,
                    use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
                    drp_config_override=config_DRP,
                    config_TA=config_TA,
                    config_EC=config_EC,
                    distance_km=_distance_km,
                    decay_km=d,
                    aggregation_cols=_agg,
                    verbose=False,
                )
                r2_d = res_d['R2_CV_mean'].values[0]
                print(f"  decay_km={d}: R²={r2_d:.3f}")
                if r2_d > best_decay_r2:
                    best_decay_r2 = r2_d
                    _best_decay_km = d
            print(f"  → 선택: decay_km={_best_decay_km}")

        _final_blend = best_blend_w if not use_residual_final else 0.0
        model_DRP, scaler_DRP, results_DRP, drp_log_target, drp_residual_mode, drp_log_residual, drp_decay_km, drp_pred_TA_std_mean, drp_pred_EC_std_mean = run_pipeline_drp(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            log_target=DRP_LOG_TARGET,
            blend_gems=_final_blend,
            residual_mode=use_residual_final,
            log_residual=DRP_USE_LOG_RESIDUAL if use_residual_final else False,
            use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
            drp_config_override=config_DRP,
            config_TA=config_TA,
            config_EC=config_EC,
            distance_km=_distance_km,
            decay_km=_best_decay_km,
            use_ta_ec_uncertainty=DRP_USE_TA_EC_UNCERTAINTY,
            use_fold_safe_top_k=USE_FOLD_SAFE_DRP_TOP_K,
            drp_top_k=DRP_TOP_K,
            aggregation_cols=_agg,
        )
    else:
        model_DRP, scaler_DRP, results_DRP, _ = run_pipeline(X_DRP, y_DRP, groups, "Dissolved Reactive Phosphorus", model_config=config_DRP, aggregation_cols=_agg)
        drp_log_target = False
        drp_residual_mode = False
        drp_log_residual = False
        drp_decay_km = PRIOR_DECAY_KM
        drp_pred_TA_std_mean = drp_pred_EC_std_mean = None

    progress("Validation 예측 및 submission 생성")
    # --- Results summary ---
    results_summary = pd.concat(
        [results_TA, results_EC, results_DRP], ignore_index=True
    )
    print("\nResults Summary:")
    print(results_summary)

    # --- CV 누수 제거 전후 비교 (leaky vs safe) ---
    print("\n" + "=" * 60)
    print("CV 누수 제거 전후 비교 (기존 방식 vs 누수 제거 방식)")
    print("=" * 60)
    _n = 5  # 메인과 동일 fold 수
    _, _, res_TA_leaky, _ = run_pipeline(
        X_TA, y_TA, groups_TA, "Total Alkalinity (leaky)", n_folds=_n,
        model_config=config_TA, aggregation_cols=_agg, leaky=True
    )
    _, _, res_EC_leaky, _ = run_pipeline(
        X_EC, y_EC, groups_EC, "Electrical Conductance (leaky)", n_folds=_n,
        model_config=config_EC, aggregation_cols=_agg, leaky=True
    )
    if USE_TA_EC_FOR_DRP:
        _decay_cmp = _best_decay_km
    else:
        _decay_cmp = PRIOR_DECAY_KM
    if USE_TA_EC_FOR_DRP and model_TA is not None and model_EC is not None:
        _, _, res_DRP_leaky, _, _, _, _, _, _ = run_pipeline_drp(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            n_folds=_n, log_target=DRP_LOG_TARGET, blend_gems=0.0,
            residual_mode=use_residual_final, log_residual=DRP_USE_LOG_RESIDUAL if use_residual_final else False,
            use_ta_ec_true_in_cv=DRP_USE_TA_EC_TRUE_IN_CV,
            drp_config_override=config_DRP, config_TA=config_TA, config_EC=config_EC,
            distance_km=_distance_km, decay_km=_decay_cmp,
            aggregation_cols=_agg, leaky=True, verbose=False,
        )
    else:
        _, _, res_DRP_leaky, _ = run_pipeline(
            X_DRP, y_DRP, groups, "Dissolved Reactive Phosphorus (leaky)", n_folds=_n,
            model_config=config_DRP, aggregation_cols=_agg, leaky=True
        )
    for name, res_leaky, res_safe in [
        ("TA (Total Alkalinity)", res_TA_leaky, results_TA),
        ("EC (Electrical Conductance)", res_EC_leaky, results_EC),
        ("DRP (Dissolved Reactive Phosphorus)", res_DRP_leaky, results_DRP),
    ]:
        r2_l = res_leaky["R2_CV_mean"].values[0]
        std_l = res_leaky["R2_CV_std"].values[0]
        rmse_l = res_leaky["RMSE_CV_mean"].values[0]
        rmse_std_l = res_leaky["RMSE_CV_std"].values[0]
        r2_s = res_safe["R2_CV_mean"].values[0]
        std_s = res_safe["R2_CV_std"].values[0]
        rmse_s = res_safe["RMSE_CV_mean"].values[0]
        rmse_std_s = res_safe["RMSE_CV_std"].values[0]
        print(f"\n  {name}:")
        print(f"    기존(leaky): R2_CV_mean={r2_l:.4f} (std={std_l:.4f}), RMSE_CV_mean={rmse_l:.4f} (std={rmse_std_l:.4f})")
        print(f"    누수제거(safe): R2_CV_mean={r2_s:.4f} (std={std_s:.4f}), RMSE_CV_mean={rmse_s:.4f} (std={rmse_std_s:.4f})")
        print(f"    코멘트: 누수 제거 후 R² 하락 → 이전 CV가 test 정보 유입으로 과대평가되었을 가능성." if r2_s < r2_l else "    코멘트: safe CV가 leaky와 비슷하거나 더 높음.")
    print("=" * 60)

    # --- Load validation data (template 기준, key로 merge → pred와 행 정확히 매칭) ---
    key = ["Latitude", "Longitude", "Sample Date"]
    test_file = pd.read_csv(DATA_DIR / "submission_template.csv")
    landsat_val = pd.read_csv(DATA_DIR / "landsat_features_validation.csv")
    terra_val = pd.read_csv(DATA_DIR / "terraclimate_features_validation.csv")

    val_data = test_file[key].copy()
    val_data = val_data.merge(
        landsat_val[key + ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]],
        on=key, how="left"
    )
    val_data = val_data.merge(
        terra_val[key + ["pet"]],
        on=key, how="left"
    )

    # GEMS 피처 추가 (가장 가까운 지점의 지역 전형값, validation에 사용 가능)
    if use_gems:
        gems_val_path = DATA_DIR / "gems_features_validation.csv"
        if gems_val_path.exists():
            gems_val = pd.read_csv(gems_val_path)
            for col in GEMS_FEATURE_COLS:
                if col in gems_val.columns:
                    val_data[col] = gems_val[col].values
        # gems_partial_P: enriched에 없으면 계산
        if "gems_partial_P" not in val_data.columns and "gems_TP" in val_data.columns and "gems_DRP" in val_data.columns:
            val_data["gems_partial_P"] = np.maximum(
                val_data["gems_TP"].fillna(0).values - val_data["gems_DRP"].fillna(0).values, 0
            )

    # Landsat 결측: 공간 기반 보간 (동일좌표 → 근처 validation → 근처 training → median)
    landsat_cols_used = [c for c in LANDSAT_COLS if c in val_data.columns]
    val_data = impute_landsat_spatial(
        val_data, landsat_train_features, landsat_cols=landsat_cols_used
    )
    # Add seasonality and wetness features (same as training)
    val_data = add_seasonality_features(val_data)
    val_data = add_wetness_features(val_data)

    # 파생 피처. Fold-safe면 집계는 나중에 fp_TA.transform으로 적용
    if USE_DERIVED_FEATURES and not use_pure_features:
        if USE_FOLD_SAFE_AGGREGATIONS:
            val_data = add_derived_features(val_data, skip_aggregations=True)
        else:
            loc_stats = wq_data.groupby(["Latitude", "Longitude"])[["NDMI", "MNDWI"]].mean().reset_index()
            loc_stats.columns = ["Latitude", "Longitude", "NDMI_mean", "MNDWI_mean"]
            val_data = add_derived_features(val_data, loc_stats=loc_stats)

    # 검증용 확장 피처 (val_with_hyriv_era5_events.csv) — 키 기반 merge만 사용 (행 순서 가정 금지)
    val_with_path = DATA_DIR / "val_with_hyriv_era5_events.csv"
    key_val = ["Latitude", "Longitude", "Sample Date"]
    if (USE_EXTERNAL_FEATURES or USE_HYRIV_ERA5_EVENTS) and val_with_path.exists():
        vw = pd.read_csv(val_with_path)
        need_cols = set(FEATURES_TA_f + FEATURES_EC_f + FEATURES_DRP_f) - set(val_data.columns)
        to_add_val = [c for c in vw.columns if c in need_cols]
        merge_on_val = [k for k in key_val if k in val_data.columns and k in vw.columns]
        if to_add_val and merge_on_val:
            vw_sub = vw[merge_on_val + to_add_val].drop_duplicates(subset=merge_on_val, keep="first")
            val_data = val_data.merge(vw_sub, on=merge_on_val, how="left")

    # Fold-safe 집계 피처: train에서 fit한 FoldPreprocessor로 val에 적용
    if fp_TA is not None:
        val_data = fp_TA.transform(val_data)

    val_data = val_data.fillna(val_data.median(numeric_only=True))

    # Ensure validation has all required columns (fill missing with 0)
    for c in FEATURES_TA_f:
        if c not in val_data.columns:
            val_data[c] = 0
    for c in FEATURES_EC_f:
        if c not in val_data.columns:
            val_data[c] = 0
    for c in FEATURES_DRP_f:
        if c not in val_data.columns:
            val_data[c] = 0

    # Scaler가 fit 시 본 컬럼 순서와 동일하게 맞춤 (sklearn feature_names_in_ 검사 통과)
    def _align_to_scaler(val_df, feat_list, scaler):
        if hasattr(scaler, "feature_names_in_"):
            order = list(scaler.feature_names_in_)
            return val_df.reindex(columns=order).fillna(0)
        return val_df[feat_list]

    submission_val_data_TA = _align_to_scaler(val_data, FEATURES_TA_f, scaler_TA)
    submission_val_data_EC = _align_to_scaler(val_data, FEATURES_EC_f, scaler_EC)
    submission_val_data_DRP = _align_to_scaler(val_data, FEATURES_DRP_f, scaler_DRP)

    # --- Predict ---
    X_sub_scaled_TA = scaler_TA.transform(submission_val_data_TA)
    pred_TA_submission = model_TA.predict(X_sub_scaled_TA)

    X_sub_scaled_EC = scaler_EC.transform(submission_val_data_EC)
    pred_EC_submission = model_EC.predict(X_sub_scaled_EC)

    # Step 2: weak gems blend (final = 0.85*model + 0.15*gems)
    if use_weak_gems_blend:
        w_model, w_gems = 0.85, 0.15
        if "gems_Alk_Tot" in val_data.columns:
            g_ta = np.maximum(np.nan_to_num(val_data["gems_Alk_Tot"].values, nan=0.0), 0)
            pred_TA_submission = w_model * pred_TA_submission + w_gems * g_ta
        if "gems_EC" in val_data.columns:
            g_ec = np.maximum(np.nan_to_num(val_data["gems_EC"].values, nan=0.0), 0)
            pred_EC_submission = w_model * pred_EC_submission + w_gems * g_ec

    if USE_TA_EC_FOR_DRP:
        submission_val_data_drp = submission_val_data_DRP.copy()
        submission_val_data_drp['pred_TA'] = pred_TA_submission
        submission_val_data_drp['pred_EC'] = pred_EC_submission
        if drp_pred_TA_std_mean is not None and drp_pred_EC_std_mean is not None:
            n_val = len(submission_val_data_drp)
            submission_val_data_drp['pred_TA_std'] = np.full(n_val, drp_pred_TA_std_mean)
            submission_val_data_drp['pred_EC_std'] = np.full(n_val, drp_pred_EC_std_mean)
        submission_val_data_drp = _align_to_scaler(submission_val_data_drp, submission_val_data_drp.columns.tolist(), scaler_DRP)
        X_sub_scaled_DRP = scaler_DRP.transform(submission_val_data_drp)
    else:
        X_sub_scaled_DRP = scaler_DRP.transform(submission_val_data_DRP)

    # DRP prediction: residual 모드면 final = prior_weight*gems_DRP + residual_pred
    pred_residual = model_DRP.predict(X_sub_scaled_DRP)
    if drp_residual_mode:
        gems_val = np.maximum(np.nan_to_num(val_data["gems_DRP"].values, nan=0.0), 0)
        # 거리 가중 prior: 멀수록 prior 신뢰도 하락
        if "gems_distance_km" in val_data.columns:
            dist = np.nan_to_num(val_data["gems_distance_km"].values, nan=999.0)
            prior_w = np.exp(-dist / drp_decay_km)
            gems_val = gems_val * prior_w
        if drp_log_residual:
            g_val = np.asarray(gems_val, dtype=np.float64)
            pred_DRP_submission = np.maximum(np.expm1(np.log1p(g_val) + pred_residual), 0)
        else:
            pred_DRP_submission = np.maximum(gems_val + pred_residual, 0)
    else:
        pred_DRP_submission = np.expm1(pred_residual) if drp_log_target else pred_residual
        pred_DRP_submission = np.maximum(pred_DRP_submission, 0)

    # --- Create submission ---
    submission_df = pd.DataFrame({
        'Latitude': test_file['Latitude'].values,
        'Longitude': test_file['Longitude'].values,
        'Sample Date': test_file['Sample Date'].values,
        'Total Alkalinity': pred_TA_submission,
        'Electrical Conductance': pred_EC_submission,
        'Dissolved Reactive Phosphorus': pred_DRP_submission
    })
    output_path = DATA_DIR / "submission_base.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")


if __name__ == "__main__":
    main()
