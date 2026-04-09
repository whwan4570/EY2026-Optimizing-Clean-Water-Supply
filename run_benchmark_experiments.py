"""
A. 기준점: GEMS merged enriched (고정)
B. 파생 실험: full_gems, weak_gems, no_gems, no_latlon_no_gems, compact_no_gems
C. 비교: Average R² (LB proxy), harsh spatial split = region holdout (GroupKFold by block)
D. 각 실험별 TA/EC/DRP 피처 개수 + 한 줄 판단

실행: python run_benchmark_experiments.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False

try:
    from benchmark_model import add_derived_features as _add_derived_features
    HAS_BM_DERIVED = True
except Exception:
    HAS_BM_DERIVED = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]
TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
RANDOM_STATE = 42
N_FOLDS = 5
SPATIAL_BLOCK_DEG = 0.5

# 공통 정의 (run_benchmark_notebook와 동일)
BASE_FEATURES = ["Latitude", "Longitude", "swir22", "NDMI", "MNDWI", "pet", "month", "dayofyear", "sin_doy", "cos_doy"]
BASE_NO_LATLON = ["swir22", "NDMI", "MNDWI", "pet", "month", "dayofyear", "sin_doy", "cos_doy"]
COMPACT_BASE = ["swir22", "NDMI", "MNDWI", "pet", "month", "sin_doy", "cos_doy"]
GEMS_TA = ["gems_Alk_Tot", "gems_Ca_Dis", "gems_Mg_Dis", "gems_Si_Dis", "gems_pH"]
GEMS_EC = ["gems_EC", "gems_Cl_Dis", "gems_SO4_Dis", "gems_Na_Dis", "gems_Ca_Dis", "gems_Mg_Dis", "gems_Sal", "gems_pH"]
GEMS_DRP_FULL = ["wet_index", "water_stress", "gems_TP", "gems_TP_log", "gems_DRP", "gems_DRP_log", "gems_partial_P", "gems_NOxN", "gems_NH4N", "gems_pH"]
WETNESS = ["wet_index", "water_stress"]

# benchmark_model.py와 동일: 파생·확장 피처 (90개 이상 나오는 이유)
DERIVED_COMMON = [
    "WRI", "green_nir_ratio", "green_swir22_ratio", "nir_swir16_ratio", "nir_swir22_ratio",
    "NDMI_sq", "MNDWI_sq", "NDMI_MNDWI", "NDMI_anom", "MNDWI_anom",
    "log1p_pet", "pet_norm", "pet_seasonal_anom", "pet_sin_doy", "pet_cos_doy", "pet_lat",
    "season", "growing_season",
]
DERIVED_TA = ["hardness_like", "cation_balance", "Ca_Mg_ratio"]
DERIVED_EC = ["major_ions", "na_cl_ratio", "sulphate_chloride_ratio", "sal_over_EC", "dry_index"]
DERIVED_DRP_COMMON = ["NDMI_sq", "MNDWI_sq", "NDMI_MNDWI", "NDMI_anom", "MNDWI_anom", "log1p_pet", "pet_seasonal_anom", "season"]
DERIVED_DRP = ["N_total_like", "N_to_P_ratio", "DRP_to_TP_ratio", "wet_TP", "wet_DRP", "NDMI_TP", "MNDWI_TP"]
EXTENDED_BASE = [
    "pr", "era5_rh", "era5_sm", "precip_anom", "cumulative_anom_3m", "precip_anom_lag1", "wetness_index",
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


def add_seasonality_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:
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
    result = df.copy()
    if "NDMI" in result.columns and "pet" in result.columns:
        result["wet_index"] = result["NDMI"] * result["pet"]
        result["water_stress"] = result["NDMI"] / (result["pet"] + 1e-6)
    return result


def get_spatial_block_groups(df: pd.DataFrame, block_deg: float) -> np.ndarray:
    lat = df["Latitude"].values
    lon = df["Longitude"].values
    lat_min, lon_min = lat.min(), lon.min()
    block = (
        np.floor((lat - lat_min) / block_deg).astype(int) * 10_000
        + np.floor((lon - lon_min) / block_deg).astype(int)
    )
    groups, _ = pd.factorize(block)
    return groups


def get_gems_quality_split(df: pd.DataFrame, use_distance_fallback: bool = True, distance_median: bool = True):
    """
    GEMS 품질 기반 train/val split.
    train: gems_within_limit == True (또는 distance 짧은 쪽)
    val:   gems_within_limit == False (또는 distance 긴 쪽)
    Returns (train_idx, test_idx) or None if columns missing or one class empty.
    """
    if "gems_within_limit" in df.columns:
        within = df["gems_within_limit"].fillna(False)
        train_mask = within == True
        test_mask = ~train_mask
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            return None
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx
    if use_distance_fallback and "gems_distance_km" in df.columns:
        dist = df["gems_distance_km"].fillna(np.nanmedian(df["gems_distance_km"]))
        if distance_median:
            thresh = np.nanmedian(dist)
        else:
            thresh = np.nanpercentile(dist, 50)
        train_mask = (dist <= thresh).values
        test_mask = ~train_mask
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            return None
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx
    return None


def combine_two_datasets(d1, d2, d3):
    data = pd.concat([d1, d2, d3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def _train(X_s, y):
    if USE_XGB:
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0)
    else:
        m = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    m.fit(X_s, y)
    return m


def _filter_exist(cols, df):
    return [c for c in cols if c in df.columns]


def load_baseline_data():
    """GEMS merged enriched 기준 데이터 1회 로드 (고정 기준점)."""
    wq_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_path.exists():
        wq_path = DATA_DIR / "water_quality_training_dataset.csv"
    water_quality = pd.read_csv(wq_path)
    landsat_train = pd.read_csv(DATA_DIR / "landsat_features_training.csv")
    terra_train = pd.read_csv(DATA_DIR / "terraclimate_features_training.csv")
    wq_data = combine_two_datasets(water_quality, landsat_train, terra_train)

    if (DATA_DIR / "precipitation_training.csv").exists():
        pr_df = pd.read_csv(DATA_DIR / "precipitation_training.csv")
        if all(k in wq_data.columns and k in pr_df.columns for k in KEY_COLS):
            add_pr = [c for c in pr_df.columns if c not in KEY_COLS and c not in wq_data.columns]
            if add_pr:
                wq_data = wq_data.merge(pr_df[KEY_COLS + add_pr], on=KEY_COLS, how="left")
    if (DATA_DIR / "water_quality_with_precip_anomaly.csv").exists():
        pa_df = pd.read_csv(DATA_DIR / "water_quality_with_precip_anomaly.csv")
        extra = [c for c in pa_df.columns if c not in wq_data.columns and c not in KEY_COLS]
        if extra and all(k in wq_data.columns and k in pa_df.columns for k in KEY_COLS):
            wq_data = wq_data.merge(pa_df[KEY_COLS + extra], on=KEY_COLS, how="left")
    if (DATA_DIR / "train_with_hyriv_era5_events.csv").exists():
        tw = pd.read_csv(DATA_DIR / "train_with_hyriv_era5_events.csv")
        skip = set(KEY_COLS) | set(TARGET_COLS)
        to_add = [c for c in tw.columns if c not in wq_data.columns and c not in skip]
        if to_add and all(k in wq_data.columns and k in tw.columns for k in KEY_COLS):
            tw_sub = tw[KEY_COLS + to_add].drop_duplicates(subset=KEY_COLS, keep="first")
            wq_data = wq_data.merge(tw_sub, on=KEY_COLS, how="left")

    wq_data = add_seasonality_features(wq_data)
    wq_data = add_wetness_features(wq_data)
    if "gems_TP" in wq_data.columns and "gems_DRP" in wq_data.columns and "gems_partial_P" not in wq_data.columns:
        wq_data["gems_partial_P"] = np.maximum(wq_data["gems_TP"].fillna(0).values - wq_data["gems_DRP"].fillna(0).values, 0)

    # benchmark_model과 동일: 파생 피처 추가 (WRI, NDMI_sq, hardness_like 등 -> 90개 이상 피처)
    if HAS_BM_DERIVED:
        loc_stats = None
        if "Latitude" in wq_data.columns and "Longitude" in wq_data.columns:
            loc_stats = wq_data.groupby(["Latitude", "Longitude"])[["NDMI", "MNDWI"]].mean().reset_index()
            loc_stats.columns = ["Latitude", "Longitude", "NDMI_mean", "MNDWI_mean"]
        wq_data = _add_derived_features(wq_data, loc_stats=loc_stats, skip_aggregations=(loc_stats is None))

    wq_data = wq_data.fillna(wq_data.median(numeric_only=True))
    return wq_data


def get_baseline_turn_off_order(wq_data: pd.DataFrame) -> list:
    """
    GEMS merged baseline(full_gems)에서 전체 데이터로 TA/EC/DRP 학습 후
    피처 중요도 평균 기준 '낮은 순' 정렬 -> 이 순서대로 끄면 됨.
    반환: [feat_lowest_imp, ..., feat_highest_imp]
    """
    feat_TA, _, _ = get_experiment_features(wq_data, "full_gems")
    if not feat_TA:
        return []
    X = wq_data[feat_TA].fillna(wq_data[feat_TA].median())
    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    imp_ta = _train(X_s, y_TA).feature_importances_
    imp_ec = _train(X_s, y_EC).feature_importances_
    imp_drp = _train(X_s, y_DRP).feature_importances_
    mean_imp = (np.asarray(imp_ta) + np.asarray(imp_ec) + np.asarray(imp_drp)) / 3.0
    order_idx = np.argsort(mean_imp)
    return [feat_TA[i] for i in order_idx]


def get_experiment_features(wq_data: pd.DataFrame, experiment: str, turn_off_order: list = None):
    """
    실험별 TA/EC/DRP 피처 리스트 반환. (feat_TA, feat_EC, feat_DRP) 각각 존재하는 컬럼만.
    turn_off_order: full_gems 기준 끄는 순서 (낮은 중요도 먼저). pruned_gems에서 사용.
    """
    base = _filter_exist(BASE_FEATURES, wq_data)
    base_no_ll = _filter_exist(BASE_NO_LATLON, wq_data)
    compact = _filter_exist(COMPACT_BASE, wq_data)
    wet = _filter_exist(WETNESS, wq_data)
    g_ta = _filter_exist(GEMS_TA, wq_data)
    g_ec = _filter_exist(GEMS_EC, wq_data)
    g_drp_full = _filter_exist(GEMS_DRP_FULL, wq_data)
    gems_drp_only = _filter_exist(["gems_DRP"], wq_data)

    if experiment == "full_gems":
        # benchmark_model과 동일: BASE + GEMS + 파생(DERIVED) + 확장(EXTENDED) -> 90개 이상
        derived = list(dict.fromkeys(
            DERIVED_COMMON + DERIVED_TA + DERIVED_EC + DERIVED_DRP_COMMON + DERIVED_DRP
        ))
        extended = list(dict.fromkeys(
            EXTENDED_BASE + EXTENDED_EXTERNAL + EXTENDED_HYRIV_ERA5 + EXTENDED_TA + EXTENDED_EC + EXTENDED_DRP
        ))
        common = list(dict.fromkeys(base + wet + g_ta + g_ec + g_drp_full + derived + extended))
        common = _filter_exist(common, wq_data)
        return common, common, common

    if experiment == "pruned_gems" and turn_off_order:
        # full_gems에서 하위 50% 중요도 피처 제거 (끄는 순서 앞쪽 절반 제거)
        full, _, _ = get_experiment_features(wq_data, "full_gems")
        n_off = max(0, len(turn_off_order) // 2)
        to_drop = set(turn_off_order[:n_off])
        common = [c for c in full if c not in to_drop]
        return common, common, common

    if experiment == "weak_gems":
        # TA/EC: no GEMS. DRP: BASE + wet + gems_DRP only
        ta_ec = list(dict.fromkeys(base + wet))
        ta_ec = _filter_exist(ta_ec, wq_data)
        drp = list(dict.fromkeys(base + wet + gems_drp_only))
        drp = _filter_exist(drp, wq_data)
        return ta_ec, ta_ec, drp

    if experiment == "no_gems":
        common = list(dict.fromkeys(base + wet))
        common = _filter_exist(common, wq_data)
        return common, common, common

    if experiment == "no_latlon_no_gems":
        common = list(dict.fromkeys(base_no_ll + wet))
        common = _filter_exist(common, wq_data)
        return common, common, common

    if experiment == "compact_no_gems":
        common = _filter_exist(compact, wq_data)
        return common, common, common

    raise ValueError(f"Unknown experiment: {experiment}")


def run_one_experiment(wq_data: pd.DataFrame, experiment: str, groups: np.ndarray, turn_off_order: list = None):
    """한 실험에 대해 GroupKFold (region holdout) CV, Average R² LB proxy + 피처 수 반환."""
    feat_TA, feat_EC, feat_DRP = get_experiment_features(wq_data, experiment, turn_off_order)
    n_ta, n_ec, n_drp = len(feat_TA), len(feat_EC), len(feat_DRP)

    X_TA = wq_data[feat_TA].fillna(wq_data[feat_TA].median())
    X_EC = wq_data[feat_EC].fillna(wq_data[feat_EC].median())
    X_DRP = wq_data[feat_DRP].fillna(wq_data[feat_DRP].median())
    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    r2_ta, r2_ec, r2_drp = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_TA, y_TA, groups)):
        for X_df, y_ser, feat_list, r2_list in [
            (X_TA, y_TA, feat_TA, r2_ta),
            (X_EC, y_EC, feat_EC, r2_ec),
            (X_DRP, y_DRP, feat_DRP, r2_drp),
        ]:
            X_tr = X_df.iloc[train_idx]
            X_te = X_df.iloc[test_idx]
            y_tr = y_ser.iloc[train_idx]
            y_te = y_ser.iloc[test_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model = _train(X_tr_s, y_tr)
            pred = model.predict(X_te_s)
            r2_list.append(r2_score(y_te, pred))

    avg_r2_ta = np.mean(r2_ta)
    avg_r2_ec = np.mean(r2_ec)
    avg_r2_drp = np.mean(r2_drp)
    lb_proxy = (avg_r2_ta + avg_r2_ec + avg_r2_drp) / 3.0
    return {
        "n_TA": n_ta,
        "n_EC": n_ec,
        "n_DRP": n_drp,
        "R2_TA": avg_r2_ta,
        "R2_EC": avg_r2_ec,
        "R2_DRP": avg_r2_drp,
        "Avg_R2_LB_proxy": lb_proxy,
    }


def run_one_experiment_with_fixed_split(
    wq_data: pd.DataFrame, experiment: str, train_idx: np.ndarray, test_idx: np.ndarray, turn_off_order: list = None
):
    """고정 train/val split으로 한 번만 학습·평가 (GEMS 품질 holdout 등)."""
    feat_TA, feat_EC, feat_DRP = get_experiment_features(wq_data, experiment, turn_off_order)
    n_ta, n_ec, n_drp = len(feat_TA), len(feat_EC), len(feat_DRP)

    X_TA = wq_data[feat_TA].fillna(wq_data[feat_TA].median())
    X_EC = wq_data[feat_EC].fillna(wq_data[feat_EC].median())
    X_DRP = wq_data[feat_DRP].fillna(wq_data[feat_DRP].median())
    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]

    r2_ta, r2_ec, r2_drp = [], [], []
    for X_df, y_ser, feat_list, r2_list in [
        (X_TA, y_TA, feat_TA, r2_ta),
        (X_EC, y_EC, feat_EC, r2_ec),
        (X_DRP, y_DRP, feat_DRP, r2_drp),
    ]:
        X_tr = X_df.iloc[train_idx]
        X_te = X_df.iloc[test_idx]
        y_tr = y_ser.iloc[train_idx]
        y_te = y_ser.iloc[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = _train(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        r2_list.append(r2_score(y_te, pred))

    avg_r2_ta = float(r2_ta[0])
    avg_r2_ec = float(r2_ec[0])
    avg_r2_drp = float(r2_drp[0])
    lb_proxy = (avg_r2_ta + avg_r2_ec + avg_r2_drp) / 3.0
    return {
        "n_TA": n_ta,
        "n_EC": n_ec,
        "n_DRP": n_drp,
        "R2_TA": avg_r2_ta,
        "R2_EC": avg_r2_ec,
        "R2_DRP": avg_r2_drp,
        "Avg_R2_LB_proxy": lb_proxy,
    }


def verdict(row, baseline_proxy: float):
    """한 줄 판단."""
    p = row["Avg_R2_LB_proxy"]
    if p >= baseline_proxy * 0.98:
        return "LB proxy baseline 수준 유지"
    if p >= 0:
        return "GEMS 의존 낮춤, LB proxy 일부 하락"
    return "region holdout에서 약함, LB용 별도 튜닝 필요"


def main():
    print("=" * 72)
    print("A. 기준점: GEMS merged enriched (고정)")
    print("B. 파생 실험: full_gems, weak_gems, no_gems, no_latlon_no_gems, compact_no_gems")
    print("C. 비교: Average R² (LB proxy), harsh spatial split (region holdout)")
    print("=" * 72)

    wq_data = load_baseline_data()
    print("\n[데이터] GEMS merged enriched 로드 완료")
    print(f"  rows={len(wq_data)}, cols={len(wq_data.columns)}")
    if HAS_BM_DERIVED:
        print("  파생 피처: benchmark_model.add_derived_features 적용 (full_gems 시 90개 이상)")
    else:
        print("  파생 피처: benchmark_model 미로드 -> BASE+GEMS만 사용 (약 29개)")

    # GEMS baseline에서 끄는 순서 (낮은 중요도 먼저)
    print("\n[Baseline] 피처 중요도 기준 끄는 순서 계산 중 ...")
    turn_off_order = get_baseline_turn_off_order(wq_data)
    print("  끄는 순서 (먼저 끌수록 LB 영향 적음):")
    for i, f in enumerate(turn_off_order):
        print(f"    {i+1}. {f}")
    print("  -> pruned_gems 실험: 위 순서 앞쪽 50% 제거 후 학습")

    if "Latitude" not in wq_data.columns or "Longitude" not in wq_data.columns:
        print("\n  Lat/Lon 없음 -> no_latlon 실험만 유효, 공간 그룹 스킵")
        groups = np.zeros(len(wq_data), dtype=int)
    else:
        groups = get_spatial_block_groups(wq_data, SPATIAL_BLOCK_DEG)
        print(f"\n  공간 블록 수: {len(np.unique(groups))} (block_deg={SPATIAL_BLOCK_DEG})")

    experiments = ["full_gems", "pruned_gems", "weak_gems", "no_gems", "no_latlon_no_gems", "compact_no_gems"]
    results = []
    for exp in experiments:
        print(f"\n  실험: {exp} ...")
        res = run_one_experiment(wq_data, exp, groups, turn_off_order=turn_off_order)
        res["experiment"] = exp
        results.append(res)

    # target-wise hybrid: TA=pruned_gems, EC=full_gems, DRP=full_gems
    full_res = next(r for r in results if r["experiment"] == "full_gems")
    pruned_res = next(r for r in results if r["experiment"] == "pruned_gems")
    hybrid_proxy = (pruned_res["R2_TA"] + full_res["R2_EC"] + full_res["R2_DRP"]) / 3.0
    hybrid_row = {
        "experiment": "hybrid (TA=pruned, EC=full, DRP=full)",
        "n_TA": pruned_res["n_TA"],
        "n_EC": full_res["n_EC"],
        "n_DRP": full_res["n_DRP"],
        "R2_TA": pruned_res["R2_TA"],
        "R2_EC": full_res["R2_EC"],
        "R2_DRP": full_res["R2_DRP"],
        "Avg_R2_LB_proxy": hybrid_proxy,
    }
    baseline_proxy = full_res["Avg_R2_LB_proxy"]
    hybrid_row["verdict"] = verdict(hybrid_row, baseline_proxy)
    results.append(hybrid_row)

    for r in results:
        if "verdict" not in r:
            r["verdict"] = verdict(r, baseline_proxy)

    # 테이블 (spatial holdout)
    df = pd.DataFrame(results)
    df = df[["experiment", "n_TA", "n_EC", "n_DRP", "R2_TA", "R2_EC", "R2_DRP", "Avg_R2_LB_proxy", "verdict"]]
    df.columns = ["실험", "N_TA", "N_EC", "N_DRP", "R2_TA", "R2_EC", "R2_DRP", "Avg_R2(LB proxy)", "한 줄 판단"]
    print("\n" + "=" * 72)
    print("결과 (harsh spatial split = region holdout)")
    print("=" * 72)
    print(df.to_string(index=False))
    print("\n  [실험별 피처 수] TA/EC/DRP 각 모델에 사용한 피처 개수:")
    for r in results:
        print(f"    {r['experiment']}: N_TA={r['n_TA']}, N_EC={r['n_EC']}, N_DRP={r['n_DRP']}")
    print()
    print("  기준: full_gems = 강한 local baseline. hybrid = TA만 pruned, EC/DRP는 full.")
    print("=" * 72)

    # GEMS 품질 기반 holdout (train=within_limit True, val=within_limit False 또는 distance 긴 쪽)
    gems_split = get_gems_quality_split(wq_data)
    if gems_split is not None:
        train_idx, test_idx = gems_split
        n_train, n_test = len(train_idx), len(test_idx)
        print("\n" + "=" * 72)
        print("GEMS 품질 기반 holdout (train=GEMS 좋은 지역, val=GEMS 약한 지역)")
        print("=" * 72)
        if "gems_within_limit" in wq_data.columns:
            print(f"  split: gems_within_limit True -> train ({n_train}), False -> val ({n_test})")
        else:
            print(f"  split: gems_distance_km 기준 train ({n_train}) / val ({n_test})")
        gems_results = []
        for exp in experiments:
            print(f"  실험: {exp} ...")
            res = run_one_experiment_with_fixed_split(wq_data, exp, train_idx, test_idx, turn_off_order=turn_off_order)
            res["experiment"] = exp
            gems_results.append(res)
        full_g = next(r for r in gems_results if r["experiment"] == "full_gems")
        pruned_g = next(r for r in gems_results if r["experiment"] == "pruned_gems")
        hybrid_g_proxy = (pruned_g["R2_TA"] + full_g["R2_EC"] + full_g["R2_DRP"]) / 3.0
        hybrid_g = {
            "experiment": "hybrid (TA=pruned, EC=full, DRP=full)",
            "n_TA": pruned_g["n_TA"],
            "n_EC": full_g["n_EC"],
            "n_DRP": full_g["n_DRP"],
            "R2_TA": pruned_g["R2_TA"],
            "R2_EC": full_g["R2_EC"],
            "R2_DRP": full_g["R2_DRP"],
            "Avg_R2_LB_proxy": hybrid_g_proxy,
        }
        hybrid_g["verdict"] = verdict(hybrid_g, full_g["Avg_R2_LB_proxy"])
        gems_results.append(hybrid_g)
        for r in gems_results:
            if "verdict" not in r:
                r["verdict"] = verdict(r, full_g["Avg_R2_LB_proxy"])
        df_gems = pd.DataFrame(gems_results)
        df_gems = df_gems[["experiment", "n_TA", "n_EC", "n_DRP", "R2_TA", "R2_EC", "R2_DRP", "Avg_R2_LB_proxy", "verdict"]]
        df_gems.columns = ["실험", "N_TA", "N_EC", "N_DRP", "R2_TA", "R2_EC", "R2_DRP", "Avg_R2(LB proxy)", "한 줄 판단"]
        print(df_gems.to_string(index=False))
        print("=" * 72)
    else:
        if "gems_within_limit" in wq_data.columns:
            n_true = (wq_data["gems_within_limit"].fillna(False) == True).sum()
            n_false = (wq_data["gems_within_limit"].fillna(False) == False).sum()
            print(f"\n  [GEMS holdout 스킵] gems_within_limit 있으나 샘플 부족: True={n_true}, False={n_false} (각 10 이상 필요)")
        elif "gems_distance_km" in wq_data.columns:
            print("\n  [GEMS holdout 스킵] gems_distance_km만 있음; 샘플 부족이면 distance 기준 split 실패")
        else:
            print("\n  [GEMS holdout 스킵] gems_within_limit, gems_distance_km 없음 (enriched CSV에 포함되어 있어야 함)")


if __name__ == "__main__":
    main()
