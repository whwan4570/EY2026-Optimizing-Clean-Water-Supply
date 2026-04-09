"""
모델 비교: XGBoost vs LightGBM vs CatBoost vs HistGradientBoosting
TA, EC, DRP 각각 GroupKFold(5) CV R²/RMSE 비교

실행: python compare_models.py
필요 시: pip install lightgbm catboost
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# === 모델 등록 (설치된 것만 사용) ===
MODELS = {}

try:
    import xgboost as xgb
    MODELS["XGBoost"] = lambda cfg: xgb.XGBRegressor(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        min_child_weight=cfg.get("min_child_weight", 2),
        reg_alpha=cfg.get("reg_alpha", 0.05),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        random_state=42,
        verbosity=0,
    )
except ImportError:
    pass

try:
    import lightgbm as lgb
    MODELS["LightGBM"] = lambda cfg: lgb.LGBMRegressor(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        min_child_samples=cfg.get("min_child_weight", 2) * 10,  # LGB uses samples
        reg_alpha=cfg.get("reg_alpha", 0.05),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        random_state=42,
        verbosity=-1,
    )
except ImportError:
    pass

try:
    import catboost as cb  # type: ignore[import-untyped]
    MODELS["CatBoost"] = lambda cfg: cb.CatBoostRegressor(
        iterations=cfg.get("n_estimators", 500),
        depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        l2_leaf_reg=cfg.get("reg_lambda", 1.0),
        random_seed=42,
        verbose=False,
    )
except ImportError:
    pass

MODELS["HistGradientBoosting"] = lambda cfg: HistGradientBoostingRegressor(
    max_iter=cfg.get("n_estimators", 500),
    max_depth=cfg.get("max_depth", 6),
    learning_rate=cfg.get("learning_rate", 0.05),
    min_samples_leaf=cfg.get("min_child_weight", 2),
    l2_regularization=cfg.get("reg_lambda", 1.0),
    random_state=42,
)

# === Config (benchmark_model과 동일) ===
CONFIG_TA = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 2, "reg_alpha": 0.05, "reg_lambda": 1.0}
CONFIG_EC = {"n_estimators": 600, "max_depth": 7, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0}
CONFIG_DRP = {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.15, "reg_lambda": 1.5}


def add_seasonality_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result[date_col], format="mixed", dayfirst=True)
    result["month"] = dates.dt.month
    result["dayofyear"] = dates.dt.dayofyear
    result["sin_doy"] = np.sin(2 * np.pi * result["dayofyear"] / 365)
    result["cos_doy"] = np.cos(2 * np.pi * result["dayofyear"] / 365)
    return result


def add_wetness_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "NDMI" in result.columns and "pet" in result.columns:
        result["wet_index"] = result["NDMI"] * result["pet"]
        result["water_stress"] = result["NDMI"] / (result["pet"] + 1e-6)
    return result


def combine_two_datasets(d1, d2, d3):
    data = pd.concat([d1, d2, d3], axis=1)
    return data.loc[:, ~data.columns.duplicated()]


def get_location_groups(df: pd.DataFrame) -> np.ndarray:
    loc_key = df['Latitude'].round(6).astype(str) + '_' + df['Longitude'].round(6).astype(str)
    groups, _ = pd.factorize(loc_key)
    return groups


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def run_cv_one_model(model_name: str, model_factory, config: dict, X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_folds: int = 5) -> dict:
    """한 모델에 대해 GroupKFold CV 실행, R²/RMSE 평균 반환"""
    gkf = GroupKFold(n_splits=n_folds)
    r2_list, rmse_list = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_s, X_test_s, _ = scale_data(X_train, X_test)
        model = model_factory(config)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        r2_list.append(r2_score(y_test, pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))

    return {
        "R2_mean": np.mean(r2_list),
        "R2_std": np.std(r2_list),
        "RMSE_mean": np.mean(rmse_list),
        "RMSE_std": np.std(rmse_list),
    }


def load_and_prepare_data():
    """benchmark_model과 동일한 데이터 준비"""
    wq_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_path.exists():
        wq_path = DATA_DIR / "water_quality_training_dataset.csv"
    wq = pd.read_csv(wq_path)
    landsat = pd.read_csv(DATA_DIR / "landsat_features_training.csv")
    terra = pd.read_csv(DATA_DIR / "terraclimate_features_training.csv")

    data = combine_two_datasets(wq, landsat, terra)
    data = add_seasonality_features(data)
    data = add_wetness_features(data)

    if "gems_TP" in data.columns and "gems_DRP" in data.columns and "gems_partial_P" not in data.columns:
        data["gems_partial_P"] = np.maximum(data["gems_TP"].fillna(0).values - data["gems_DRP"].fillna(0).values, 0)

    data = data.fillna(data.median(numeric_only=True))

    BASE = ['Latitude', 'Longitude', 'swir22', 'NDMI', 'MNDWI', 'pet', 'month', 'dayofyear', 'sin_doy', 'cos_doy']
    FEAT_TA = BASE + [c for c in ['gems_Alk_Tot', 'gems_Ca_Dis', 'gems_Mg_Dis', 'gems_Si_Dis', 'gems_pH', 'gems_H_T'] if c in data.columns]
    FEAT_EC = BASE + [c for c in ['gems_EC', 'gems_Cl_Dis', 'gems_SO4_Dis', 'gems_Na_Dis', 'gems_Ca_Dis', 'gems_Mg_Dis', 'gems_Sal', 'gems_pH'] if c in data.columns]
    FEAT_DRP = BASE + ['wet_index', 'water_stress'] + [c for c in ['gems_TP', 'gems_TP_log', 'gems_NOxN', 'gems_NH4N', 'gems_pH', 'gems_DRP', 'gems_DRP_log', 'gems_partial_P'] if c in data.columns]

    return data, FEAT_TA, FEAT_EC, FEAT_DRP


def main():
    print("=" * 70)
    print("모델 비교: XGBoost vs LightGBM vs CatBoost vs HistGradientBoosting")
    print("TA, EC, DRP 각각 GroupKFold(5) CV")
    print("=" * 70)

    data, FEAT_TA, FEAT_EC, FEAT_DRP = load_and_prepare_data()
    groups = get_location_groups(data)

    X_TA = data[FEAT_TA].copy()
    X_EC = data[FEAT_EC].copy()
    X_DRP = data[FEAT_DRP].copy()
    y_TA = data['Total Alkalinity']
    y_EC = data['Electrical Conductance']
    y_DRP = data['Dissolved Reactive Phosphorus']

    targets = [
        ("TA", X_TA, y_TA, CONFIG_TA),
        ("EC", X_EC, y_EC, CONFIG_EC),
        ("DRP", X_DRP, y_DRP, CONFIG_DRP),
    ]

    results = []
    for model_name, model_factory in MODELS.items():
        print(f"\n--- {model_name} ---")
        for tname, X, y, cfg in targets:
            out = run_cv_one_model(model_name, model_factory, cfg, X, y, groups)
            results.append({
                "Model": model_name,
                "Target": tname,
                "R2_mean": out["R2_mean"],
                "R2_std": out["R2_std"],
                "RMSE_mean": out["RMSE_mean"],
            })
            print(f"  {tname}: R²={out['R2_mean']:.3f} ± {out['R2_std']:.3f}, RMSE={out['RMSE_mean']:.3f}")

    df = pd.DataFrame(results)

    # Pivot 표
    print("\n" + "=" * 70)
    print("결과 요약 (R² CV mean)")
    print("=" * 70)
    pivot_r2 = df.pivot(index="Model", columns="Target", values="R2_mean")
    pivot_r2["평균"] = pivot_r2.mean(axis=1)
    pivot_r2 = pivot_r2.round(3)
    print(pivot_r2.to_string())

    print("\n" + "-" * 70)
    print("RMSE (낮을수록 좋음)")
    pivot_rmse = df.pivot(index="Model", columns="Target", values="RMSE_mean")
    pivot_rmse = pivot_rmse.round(3)
    print(pivot_rmse.to_string())

    # Best per target
    print("\n" + "=" * 70)
    print("타겟별 최고 모델")
    for t in ["TA", "EC", "DRP"]:
        subset = df[df["Target"] == t]
        best_row = subset.loc[subset["R2_mean"].idxmax()]
        print(f"  {t}: {best_row['Model']} (R²={best_row['R2_mean']:.3f})")

    out_path = DATA_DIR / "model_comparison_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
