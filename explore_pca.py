"""
주성분 분석(PCA) 탐색 스크립트 — 수질·날씨·유량 등 변수가 많을 때 사용.

효과:
- 서로 연관성이 높은 변수들을 합쳐 핵심 정보(주성분)만 남김
- 모델 복잡도 감소 → 과적합(Overfitting) 방지

사용법: 프로젝트 루트에서
  python explore_pca.py

옵션:
  --target TA|EC|DRP|ALL  : PCA할 타깃 (기본: ALL = TA, EC, DRP 각각 따로)
  --variance 0.95     : 유지할 누적 설명 분산 비율 (기본: 0.95)
  --no-compare        : PCA vs 원본 회귀 비교 생략
"""

import argparse
import sys
from pathlib import Path

# Windows 콘솔 한글 출력 (UTF-8)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# 스크립트 위치 = 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent

# benchmark와 동일한 전체 피처 목록 (파생·확장 포함). 존재하는 컬럼만 실제 사용.
BASE_FEATURES = [
    "Latitude", "Longitude", "spatial_block",
    "swir22", "NDMI", "MNDWI", "pet", "month", "dayofyear", "sin_doy", "cos_doy",
]
# Landsat 원본 밴드 (merge 시 포함되는 경우)
BASE_EXTRA = ["nir", "green", "swir16"]
DERIVED_FEATURES_COMMON = [
    "WRI", "green_nir_ratio", "green_swir22_ratio", "nir_swir16_ratio", "nir_swir22_ratio",
    "NDMI_sq", "MNDWI_sq", "NDMI_MNDWI", "NDMI_anom", "MNDWI_anom",
    "log1p_pet", "pet_norm", "pet_seasonal_anom", "pet_sin_doy", "pet_cos_doy", "pet_lat",
    "NDMI_pet", "MNDWI_pet", "lat_cos_doy", "lat_sin_doy", "season", "growing_season",
]
DERIVED_FEATURES_TA = ["hardness_like", "cation_balance", "Ca_Mg_ratio"]
DERIVED_FEATURES_EC = ["major_ions", "na_cl_ratio", "sulphate_chloride_ratio", "sal_over_EC", "dry_index"]
DERIVED_FEATURES_DRP_COMMON = [
    "NDMI_sq", "MNDWI_sq", "NDMI_MNDWI", "NDMI_anom", "MNDWI_anom",
    "log1p_pet", "pet_seasonal_anom", "NDMI_pet", "MNDWI_pet", "lat_cos_doy", "lat_sin_doy", "season",
]
DERIVED_FEATURES_DRP = [
    "N_total_like", "N_to_P_ratio", "DRP_to_TP_ratio",
    "wet_TP", "wet_DRP", "NDMI_TP", "MNDWI_TP",
]
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
EXTENDED_TA = ["te_tot_spatial", "lag_tot_prev", "lag_tot_days", "clay_pet", "elev_pet", "soc_NDMI", "hardness_season", "gems_Alk_Tot_NDMI", "block_rain_6m"]
EXTENDED_EC = ["te_ele_spatial", "lag_ele_prev", "lag_ele_days", "clay_pet", "elev_pet", "log_pop_density", "gems_EC_NDMI", "major_ions_season", "block_rain_6m"]
EXTENDED_DRP = [
    "te_dis_spatial", "lag_dis_prev", "lag_dis_days", "lag_tot_prev", "lag_ele_prev",
    "gems_TP_season", "precip_anom_mm", "precip_anom_TP", "precip_anom_DRP", "precip_anom_N",
    "cum_anom_3m_TP", "cum_anom_3m_DRP", "wetness_TP", "wetness_DRP", "anom_pos_TP", "anom_neg_DRP",
    "storm_pet_TP", "pr_TP", "pr_DRP", "drp_extreme", "precip_anom_extreme", "wetness_extreme", "storm_spike",
    "partialP_ratio", "MNDWI_N", "cum_anom_3m_N", "cropland_TP", "cropland_precip", "urban_pop",
    "clay_pet", "elev_pet", "soc_NDMI", "log_pop_density", "block_rain_6m",
]
AGGREGATION_FEATURES = ["NDMI_anom", "MNDWI_anom", "pet_norm", "pet_seasonal_anom"]

# 타깃별 기본 피처 (benchmark와 동일)
FEATURES_TA_BASE = BASE_FEATURES + BASE_EXTRA + [
    "gems_Alk_Tot", "gems_Ca_Dis", "gems_Mg_Dis", "gems_Si_Dis", "gems_pH", "gems_H_T",
]
FEATURES_EC_BASE = BASE_FEATURES + BASE_EXTRA + [
    "gems_EC", "gems_Cl_Dis", "gems_SO4_Dis", "gems_Na_Dis",
    "gems_Ca_Dis", "gems_Mg_Dis", "gems_Sal", "gems_pH",
]
FEATURES_DRP_BASE = BASE_FEATURES + BASE_EXTRA + [
    "wet_index", "water_stress", "gems_TP", "gems_TP_log", "gems_NOxN", "gems_NH4N", "gems_pH",
    "gems_DRP", "gems_DRP_log", "gems_partial_P", "prior_weight",
]

def build_full_feature_list_for_target(df_columns, target):
    """benchmark와 동일한 전체 피처 목록 중 데이터에 존재하는 것만 반환 (62개 수준)."""
    if target == "TA":
        base = FEATURES_TA_BASE + DERIVED_FEATURES_COMMON + DERIVED_FEATURES_TA
    elif target == "EC":
        base = FEATURES_EC_BASE + DERIVED_FEATURES_COMMON + DERIVED_FEATURES_EC
    else:
        base = FEATURES_DRP_BASE + DERIVED_FEATURES_DRP_COMMON + DERIVED_FEATURES_DRP
    ext_common = [c for c in (EXTENDED_BASE + EXTENDED_EXTERNAL + EXTENDED_HYRIV_ERA5) if c in df_columns]
    ext_ta = [c for c in EXTENDED_TA if c in df_columns]
    ext_ec = [c for c in EXTENDED_EC if c in df_columns]
    ext_drp = [c for c in EXTENDED_DRP if c in df_columns]
    agg = [c for c in AGGREGATION_FEATURES if c in df_columns]
    if target == "TA":
        full = list(dict.fromkeys(base + ext_common + ext_ta + agg))
    elif target == "EC":
        full = list(dict.fromkeys(base + ext_common + ext_ec + agg))
    else:
        full = list(dict.fromkeys(base + ext_common + ext_drp + agg))
    return [c for c in full if c in df_columns]

TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]


def load_and_merge_data():
    """Water Quality + Landsat + Terraclimate 병합. 없으면 있는 것만 사용."""
    wq_path = BASE_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_path.exists():
        wq_path = BASE_DIR / "water_quality_training_dataset.csv"
    if not wq_path.exists():
        print("water_quality_training_dataset(.csv / _enriched.csv) 없음. 샘플 데이터로 PCA만 시연합니다.")
        return make_demo_data()

    wq = pd.read_csv(wq_path)
    wq["Sample Date"] = pd.to_datetime(wq["Sample Date"], dayfirst=True, errors="coerce")
    wq["month"] = wq["Sample Date"].dt.month
    wq["dayofyear"] = wq["Sample Date"].dt.dayofyear
    wq["sin_doy"] = np.sin(2 * np.pi * wq["dayofyear"] / 365.25)
    wq["cos_doy"] = np.cos(2 * np.pi * wq["dayofyear"] / 365.25)

    def _parse_date(ser):
        return pd.to_datetime(ser, dayfirst=True, errors="coerce")

    landsat_path = BASE_DIR / "landsat_features_training.csv"
    if landsat_path.exists():
        ls = pd.read_csv(landsat_path)
        if "Sample Date" in ls.columns:
            ls["Sample Date"] = _parse_date(ls["Sample Date"])
        key = [c for c in KEY_COLS if c in wq.columns and c in ls.columns]
        if key:
            wq = wq.merge(ls.drop_duplicates(key), on=key, how="left")
    terra_path = BASE_DIR / "terraclimate_features_training.csv"
    if terra_path.exists():
        tc = pd.read_csv(terra_path)
        if "Sample Date" in tc.columns:
            tc["Sample Date"] = _parse_date(tc["Sample Date"])
        key = [c for c in KEY_COLS if c in wq.columns and c in tc.columns]
        if key:
            wq = wq.merge(tc.drop_duplicates(key), on=key, how="left")

    # 외부/HydroRIVERS/ERA5 + 공개 피처 병합 (benchmark와 동일하게 전체 피처 수 확대)
    train_with_path = BASE_DIR / "train_with_hyriv_era5_events.csv"
    key_ext = ["Latitude", "Longitude", "Sample Date"]
    if train_with_path.exists():
        tw = pd.read_csv(train_with_path)
        if "Sample Date" in tw.columns:
            tw["Sample Date"] = _parse_date(tw["Sample Date"])
        skip = set(TARGET_COLS) | set(key_ext)
        to_add = [c for c in tw.columns if c not in wq.columns and c not in skip]
        merge_on = [k for k in key_ext if k in wq.columns and k in tw.columns]
        if to_add and len(merge_on) == len(key_ext):
            tw_sub = tw[merge_on + to_add].drop_duplicates(subset=merge_on, keep="first")
            wq = wq.merge(tw_sub, on=merge_on, how="left")
    public_path = BASE_DIR / "public_features.csv"
    if public_path.exists():
        pub = pd.read_csv(public_path)
        if "Sample Date" in pub.columns:
            pub["Sample Date"] = _parse_date(pub["Sample Date"])
        key_pub = [c for c in KEY_COLS if c in pub.columns]
        skip_pub = set(key_pub) | set(TARGET_COLS)
        to_add_pub = [c for c in pub.columns if c not in wq.columns and c not in skip_pub]
        merge_on_pub = [k for k in key_pub if k in wq.columns and k in pub.columns]
        if to_add_pub and len(merge_on_pub) == len(key_pub):
            pub_sub = pub[merge_on_pub + to_add_pub].drop_duplicates(subset=merge_on_pub, keep="first")
            wq = wq.merge(pub_sub, on=merge_on_pub, how="left")

    # 파생 피처 일부 추가 (PCA에서 사용할 수 있도록, 존재하는 컬럼만)
    eps = 1e-6
    if "NDMI" in wq.columns:
        wq["NDMI_sq"] = wq["NDMI"] ** 2
    if "MNDWI" in wq.columns:
        wq["MNDWI_sq"] = wq["MNDWI"] ** 2
    if "NDMI" in wq.columns and "MNDWI" in wq.columns:
        wq["NDMI_MNDWI"] = wq["NDMI"] * wq["MNDWI"]
    if "pet" in wq.columns:
        wq["log1p_pet"] = np.log1p(np.maximum(wq["pet"], 0))
    if "Latitude" in wq.columns and "cos_doy" in wq.columns:
        wq["lat_cos_doy"] = wq["Latitude"] * wq["cos_doy"]
    if "Latitude" in wq.columns and "sin_doy" in wq.columns:
        wq["lat_sin_doy"] = wq["Latitude"] * wq["sin_doy"]
    if "month" in wq.columns:
        m = wq["month"]
        wq["season"] = np.where(m.isin([12, 1, 2]), 1, np.where(m.isin([3, 4, 5]), 2, np.where(m.isin([6, 7, 8]), 3, 4)))
        wq["growing_season"] = (m >= 4) & (m <= 9)
    if "green" in wq.columns and "nir" in wq.columns:
        wq["green_nir_ratio"] = wq["green"] / (wq["nir"] + eps)
    if "nir" in wq.columns and "swir22" in wq.columns:
        wq["nir_swir22_ratio"] = wq["nir"] / (wq["swir22"] + eps)
    if "NDMI" in wq.columns and "pet" in wq.columns:
        wq["NDMI_pet"] = wq["NDMI"] * wq["pet"]
    if "MNDWI" in wq.columns and "pet" in wq.columns:
        wq["MNDWI_pet"] = wq["MNDWI"] * wq["pet"]
    if "gems_Ca_Dis" in wq.columns and "gems_Mg_Dis" in wq.columns:
        wq["hardness_like"] = wq["gems_Ca_Dis"].fillna(0) + wq["gems_Mg_Dis"].fillna(0)
    if "gems_EC" in wq.columns and "gems_Sal" in wq.columns:
        wq["sal_over_EC"] = wq["gems_Sal"] / (wq["gems_EC"] + eps)
    if "NDMI" in wq.columns and "MNDWI" in wq.columns:
        wq["dry_index"] = -wq["NDMI"] + (1 - wq["MNDWI"])
    if "gems_TP" in wq.columns and "gems_partial_P" in wq.columns:
        wq["partialP_ratio"] = wq["gems_partial_P"] / (wq["gems_TP"] + eps)
    if "Latitude" in wq.columns and "Longitude" in wq.columns:
        wq["spatial_block"] = (np.round(wq["Latitude"].values, 2) * 100 + np.round(wq["Longitude"].values, 2)).astype(int)

    return wq


def make_demo_data():
    """데이터 파일이 없을 때: 연관성 높은 변수 시뮬레이션 (날씨/유량/탁도 유사)."""
    np.random.seed(42)
    n = 2000
    # 공통 잠재 요인 3개 → 변수 12개 (상관 높음)
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.randn(n)
    cols = {}
    for i in range(4):
        cols[f"weather_{i}"] = f1 + 0.3 * np.random.randn(n)
    for i in range(4):
        cols[f"flow_{i}"] = f2 + 0.3 * np.random.randn(n)
    for i in range(4):
        cols[f"turbidity_{i}"] = f3 + 0.3 * np.random.randn(n)
    cols["Latitude"] = np.random.uniform(-35, -25, n)
    cols["Longitude"] = np.random.uniform(15, 32, n)
    return pd.DataFrame(cols)


def get_feature_matrix(df, feature_list=None, for_demo=False):
    """숫자형 피처만 추출, NaN은 열 중앙값으로 채움. feature_list=None이면 가능한 전체."""
    if for_demo:
        use = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
    elif feature_list is not None:
        use = [c for c in feature_list if c in df.columns]
    else:
        use = list(BASE_FEATURES) + [c for c in df.columns if c.startswith("gems_") and c in df.columns]
        use = [c for c in use if c in df.columns]
    if len(use) < 2:
        return None, []
    X = df[use].copy()
    for c in use:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    X = np.asarray(X.values, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # 상수 열 제거 (분산 0 → PCA/상관계수 오류 방지)
    var = np.var(X, axis=0)
    nonconst = np.isfinite(var) & (var > 1e-12)
    if nonconst.sum() < 2:
        return None, []
    X = X[:, nonconst]
    use = [use[i] for i in range(len(use)) if nonconst[i]]
    return X, use


def run_pca_analysis(X, feature_names, variance_keep=0.95, target_label=""):
    """PCA 적합 후 설명 분산, 누적 분산, 권장 성분 수 출력."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_features = X_scaled.shape[1]

    pca = PCA(n_components=min(n_features, min(X_scaled.shape[0], n_features) - 1))
    pca.fit(X_scaled)

    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)

    title = f"주성분 분석 (PCA) — {target_label}" if target_label else "주성분 분석 (PCA) 결과"
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"원본 변수 수: {n_features}")
    print(f"샘플 수: {X.shape[0]}")
    print()
    print("설명 분산 비율 (앞 10개 주성분):")
    for i in range(min(10, len(evr))):
        print(f"  PC{i+1}: {evr[i]:.4f}  (누적: {cumsum[i]:.4f})")
    if len(evr) > 10:
        print(f"  ... (총 {len(evr)}개)")
    print()

    for thresh in [0.80, 0.90, 0.95, 0.99]:
        n_need = int(np.searchsorted(cumsum, thresh)) + 1
        n_need = min(n_need, len(cumsum))
        actual = cumsum[n_need - 1]
        print(f"  설명 분산 {thresh:.0%} 유지 → 주성분 {n_need}개 (실제 누적: {actual:.4f})")
    print()

    n_components = int(np.searchsorted(cumsum, variance_keep)) + 1
    n_components = min(max(1, n_components), len(cumsum))
    print(f"권장: variance_keep={variance_keep} → n_components={n_components}")
    print("=" * 60)

    return pca, scaler, n_components


def show_correlation_summary(X, feature_names, top_n=15):
    """원본 변수 간 상관 요약 — 연관성 높은 변수 확인."""
    if len(feature_names) < 2 or X.shape[1] != len(feature_names):
        return
    try:
        C = np.corrcoef(X.T)
    except Exception:
        C = pd.DataFrame(X, columns=feature_names).corr().values
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0)
    high_corr = []
    for i in range(min(top_n, len(feature_names))):
        for j in range(i + 1, len(feature_names)):
            if abs(C[i, j]) > 0.7:
                high_corr.append((feature_names[i], feature_names[j], C[i, j]))
    if high_corr:
        print("\n[연관성 높은 변수 쌍 (|상관|>0.7)] - PCA로 요약하면 과적합 완화에 도움:")
        for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2]))[:15]:
            print(f"  {a} ↔ {b}: {r:.3f}")
    else:
        print("\n[상관 > 0.7인 쌍 없음] (변수 수가 적거나 상관이 낮음)")


def compare_regression(X, y, feature_names, target_name, n_components_suggested=None, n_splits=5):
    """원본 피처 vs PCA 축소 피처로 Ridge 회귀 비교. n_components_suggested=95% 분산 유지 개수 권장."""
    if y is None or np.any(np.isnan(y)) or len(y) < 50:
        print("\n[회귀 비교 생략] 타깃 또는 샘플 부족")
        return
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(y)
    if valid.sum() < 50:
        return
    X, y = X[valid], y[valid]
    # 95% 분산 유지 개수를 쓰면 비교가 공정함. 없으면 원본의 약 절반
    n_comp = n_components_suggested
    if n_comp is None or n_comp < 2:
        n_comp = max(2, min(20, X.shape[1] // 2, X.shape[0] // 10))
    n_comp = min(n_comp, X.shape[1] - 1, X.shape[0] // 5)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ridge = Ridge(alpha=1.0)

    r2_raw, r2_pca = [], []
    for tr, va in kf.split(X_scaled):
        ridge.fit(X_scaled[tr], y[tr])
        r2_raw.append(r2_score(y[va], ridge.predict(X_scaled[va])))
        ridge.fit(X_pca[tr], y[tr])
        r2_pca.append(r2_score(y[va], ridge.predict(X_pca[va])))

    mean_raw = np.mean(r2_raw)
    mean_pca = np.mean(r2_pca)
    drop = mean_raw - mean_pca

    print("\n[과적합 방지 확인] Ridge 회귀 (5-fold CV R²)")
    print(f"  타깃: {target_name}")
    print(f"  원본 피처 ({X.shape[1]}개): R² = {mean_raw:.4f} ± {np.std(r2_raw):.4f}")
    print(f"  PCA 피처 ({n_comp}개, 약 95% 분산): R² = {mean_pca:.4f} ± {np.std(r2_pca):.4f}  (차이: {drop:+.4f})")
    if drop <= 0.02:
        print("  -> PCA로 줄여도 성능이 비슷함. 과적합 완화 효과 기대.")
    elif drop > 0.05:
        print("  -> PCA로 줄이면 성능 하락이 큼. 이 데이터에서는 원본 피처 유지 권장 (PCA는 해석/시각화용으로만 고려).")
    print()


def main():
    parser = argparse.ArgumentParser(description="PCA 탐색 (수질·날씨 변수 축소) — TA/EC/DRP 각각")
    parser.add_argument("--target", default="ALL", choices=["TA", "EC", "DRP", "ALL"],
                        help="PCA할 타깃 (ALL=3개 모델 각각)")
    parser.add_argument("--variance", type=float, default=0.95, help="유지할 누적 설명 분산")
    parser.add_argument("--no-compare", action="store_true", help="회귀 비교 생략")
    args = parser.parse_args()

    targets = ["TA", "EC", "DRP"] if args.target == "ALL" else [args.target]
    target_to_col = {
        "TA": "Total Alkalinity",
        "EC": "Electrical Conductance",
        "DRP": "Dissolved Reactive Phosphorus",
    }

    df = load_and_merge_data()
    demo = "weather_" in str(df.columns)

    for t in targets:
        target_col = target_to_col[t]
        # benchmark와 동일한 전체 피처 사용 (62개 수준, 데이터에 존재하는 것만)
        feat_list = None if demo else build_full_feature_list_for_target(df.columns, t)
        X, feature_names = get_feature_matrix(df, feature_list=feat_list, for_demo=demo)
        if X is None or X.shape[1] < 2:
            print(f"\n[{t}] 숫자형 피처 2개 미만. 스킵.")
            continue

        print("\n" + "#" * 60)
        print(f"# 모델: {t} ({target_col})")
        print(f"# 사용 피처: {len(feature_names)}개 (benchmark 전체 목록 중 데이터 존재분, 상수열 제외)")
        print("#" * 60)
        show_correlation_summary(X, feature_names)
        pca, scaler, n_comp_rec = run_pca_analysis(X, feature_names, variance_keep=args.variance, target_label=t)
        if not args.no_compare and not demo and target_col in df.columns:
            y = df[target_col].values
            compare_regression(X, y, feature_names, target_col, n_components_suggested=n_comp_rec)

    if len(targets) > 1 and not args.no_compare and not demo:
        print("=" * 60)
        print("요약: PCA로 줄이면 R² 하락이 크면 원본 피처 유지 권장. PCA는 해석/시각화용으로만 고려.")
        print("=" * 60)


if __name__ == "__main__":
    main()
