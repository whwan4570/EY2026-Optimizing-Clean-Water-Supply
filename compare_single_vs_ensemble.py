"""
단일 모델 vs 앙상블 여러 조합 비교 + 공간별 train/test 분리 (LB 스타일 평가).
- TA/EC: XGBoost 우선. DRP: XGBoost 우선, 필요 시 CatBoost 보조 앙상블 포함 비교.
- 공간 분할: 동일 블록은 train 또는 test에만 속함 (GroupShuffleSplit).
실행: python compare_single_vs_ensemble.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SPATIAL_BLOCK_DEG = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42

N_EST = 200
MAX_DEPTH = 10
LR = 0.1


def get_location_groups(df: pd.DataFrame) -> np.ndarray:
    key = df["Latitude"].round(6).astype(str) + "_" + df["Longitude"].round(6).astype(str)
    groups, _ = pd.factorize(key)
    return groups


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


def combine_two_datasets(d1, d2, d3):
    data = pd.concat([d1, d2, d3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---- 단일 모델 빌더 (X, y) -> model ----
def _train_rf(X_s, y):
    m = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    m.fit(X_s, y)
    return m


def _train_et(X_s, y):
    m = ExtraTreesRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    m.fit(X_s, y)
    return m


def _train_hgb(X_s, y):
    m = HistGradientBoostingRegressor(
        max_iter=N_EST, max_depth=MAX_DEPTH, learning_rate=LR, random_state=RANDOM_STATE,
        early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
    )
    m.fit(X_s, y)
    return m


def _train_xgb(X_s, y):
    if not HAS_XGBOOST:
        return None
    m = xgb.XGBRegressor(
        n_estimators=N_EST, max_depth=MAX_DEPTH, learning_rate=LR,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0,
    )
    m.fit(X_s, y)
    return m


def _train_lgb(X_s, y):
    if not HAS_LIGHTGBM:
        return None
    m = lgb.LGBMRegressor(
        n_estimators=N_EST, max_depth=MAX_DEPTH, learning_rate=LR,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1,
    )
    m.fit(X_s, y)
    return m


def _train_cb(X_s, y):
    if not HAS_CATBOOST:
        return None
    m = cb.CatBoostRegressor(
        iterations=N_EST, depth=min(MAX_DEPTH, 10), learning_rate=LR,
        subsample=0.8, random_seed=RANDOM_STATE, verbose=0,
    )
    m.fit(X_s, y)
    return m


# ---- 앙상블: 이름 -> (train_fn, predict_fn). train_fn returns list of models ----
def _predict_multi(models, X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


def _build_ensemble(keys):
    """keys: list of 'rf','et','hgb','xgb','lgb','cb'. Returns (train_fn, predict_fn)."""
    builders = {"rf": _train_rf, "et": _train_et, "hgb": _train_hgb, "xgb": _train_xgb, "lgb": _train_lgb, "cb": _train_cb}

    def train_fn(X_s, y):
        models = []
        for k in keys:
            b = builders.get(k)
            if b is None:
                continue
            m = b(X_s, y)
            if m is not None:
                models.append(m)
        return models

    def predict_fn(models, X):
        if not models:
            return np.zeros(X.shape[0])
        return _predict_multi(models, X)

    return train_fn, predict_fn


# TA/EC: XGB 우선. 단일은 XGB 먼저, 그 다음 RF, LGB, CB.
SINGLE_CONFIGS = [
    ("XGB", _train_xgb),
    ("RF", _train_rf),
    ("LGB", _train_lgb),
    ("CB", _train_cb),
]
# 앙상블: XGB 중심, DRP 보조용 XGB+CB 조합 포함
ENSEMBLE_CONFIGS = [
    "XGB+CB",
    "XGB+LGB",
    "RF+XGB",
    "RF+XGB+CB",
    "RF+ET+HGB",
    "RF+ET+HGB+XGB",
    "RF+ET+HGB+XGB+CB",
    "RF+ET+HGB+LGB",
    "RF+ET+HGB+XGB+LGB",
    "RF+ET+HGB+XGB+LGB+CB",
]


def _parse_ensemble_keys(label):
    """e.g. 'RF+ET+HGB' -> ['rf','et','hgb']."""
    parts = label.replace("+", " ").split()
    key_map = {"RF": "rf", "ET": "et", "HGB": "hgb", "XGB": "xgb", "LGB": "lgb", "CB": "cb"}
    return [key_map[p] for p in parts if p in key_map]


def run_one_split(X, y_TA, y_EC, y_DRP, groups, use_spatial=True):
    """한 번의 train/test 분할로 모든 설정 학습·평가. 반환: dict[config_name] = {TA, EC, DRP, Average}."""
    if use_spatial and groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y_TA, groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    X_train = X.iloc[train_idx].copy().fillna(X.iloc[train_idx].median())
    X_test = X.iloc[test_idx].copy().fillna(X_train.median())
    y_TA_t, y_TA_v = y_TA.iloc[train_idx], y_TA.iloc[test_idx]
    y_EC_t, y_EC_v = y_EC.iloc[train_idx], y_EC.iloc[test_idx]
    y_DRP_t, y_DRP_v = y_DRP.iloc[train_idx], y_DRP.iloc[test_idx]

    all_results = {}

    for name, train_fn in SINGLE_CONFIGS:
        if name == "XGB" and not HAS_XGBOOST:
            continue
        if name == "LGB" and not HAS_LIGHTGBM:
            continue
        if name == "CB" and not HAS_CATBOOST:
            continue
        out = {}
        for y_train, y_test, key in [(y_TA_t, y_TA_v, "TA"), (y_EC_t, y_EC_v, "EC"), (y_DRP_t, y_DRP_v, "DRP")]:
            X_tr_s, X_te_s, _ = scale_data(X_train, X_test)
            m = train_fn(X_tr_s, y_train)
            if m is None:
                out[key] = np.nan
                continue
            pred = m.predict(X_te_s)
            out[key] = r2_score(y_test, pred)
        out["Average"] = np.nanmean(list(out.values()))
        all_results[name] = out

    for label in ENSEMBLE_CONFIGS:
        keys = _parse_ensemble_keys(label)
        if not keys:
            continue
        if "xgb" in keys and not HAS_XGBOOST:
            continue
        if "lgb" in keys and not HAS_LIGHTGBM:
            continue
        if "cb" in keys and not HAS_CATBOOST:
            continue
        train_fn, predict_fn = _build_ensemble(keys)
        out = {}
        for y_train, y_test, key in [(y_TA_t, y_TA_v, "TA"), (y_EC_t, y_EC_v, "EC"), (y_DRP_t, y_DRP_v, "DRP")]:
            X_tr_s, X_te_s, _ = scale_data(X_train, X_test)
            models = train_fn(X_tr_s, y_train)
            if not models:
                out[key] = np.nan
                continue
            pred = predict_fn(models, X_te_s)
            out[key] = r2_score(y_test, pred)
        out["Average"] = np.nanmean(list(out.values()))
        all_results[label] = out

    return all_results


def main():
    print("=" * 70)
    print("단일 모델 vs 앙상블 여러 조합 비교 (공간 분할 = LB 스타일 평가)")
    print("=" * 70)
    print(f"  XGBoost: {'O' if HAS_XGBOOST else 'X'},  LightGBM: {'O' if HAS_LIGHTGBM else 'X'},  CatBoost: {'O' if HAS_CATBOOST else 'X'}")

    water_quality = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    landsat_train = pd.read_csv(DATA_DIR / "landsat_features_training.csv")
    terra_train = pd.read_csv(DATA_DIR / "terraclimate_features_training.csv")
    wq_data = combine_two_datasets(water_quality, landsat_train, terra_train)
    wq_data = wq_data.fillna(wq_data.median(numeric_only=True))

    feature_cols = ["swir22", "NDMI", "MNDWI", "pet"]
    keep = feature_cols + ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    if "Latitude" in wq_data.columns and "Longitude" in wq_data.columns:
        keep = ["Latitude", "Longitude"] + keep
    wq_data = wq_data[[c for c in keep if c in wq_data.columns]]

    X = wq_data[feature_cols].copy()
    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]

    if "Latitude" in wq_data.columns:
        groups_block = get_spatial_block_groups(wq_data, SPATIAL_BLOCK_DEG)
        n_blocks = len(np.unique(groups_block))
        print(f"\n공간 그룹: 블록(block_deg={SPATIAL_BLOCK_DEG}°) 수={n_blocks}")
    else:
        groups_block = None

    # 공간 분할
    print("\n--- 공간 분할 (블록 홀드아웃, test_size=20%) ---")
    results_spatial = run_one_split(X, y_TA, y_EC, y_DRP, groups_block, use_spatial=True)

    # 랜덤 분할 (참고)
    print("--- 랜덤 분할 (참고) ---")
    results_random = run_one_split(X, y_TA, y_EC, y_DRP, None, use_spatial=False)

    # 테이블: 공간
    print("\n" + "=" * 70)
    print("결과 요약 - 공간 분할 (R2)")
    print("=" * 70)
    rows_spatial = []
    for name in sorted(results_spatial.keys(), key=lambda x: (-results_spatial[x]["Average"], x)):
        r = results_spatial[name]
        rows_spatial.append({
            "모델": name,
            "R²_TA": r["TA"],
            "R²_EC": r["EC"],
            "R²_DRP": r["DRP"],
            "Average_R²": r["Average"],
        })
    df_spatial = pd.DataFrame(rows_spatial)
    print(df_spatial.to_string(index=False))

    best_name = rows_spatial[0]["모델"]
    best_avg = rows_spatial[0]["Average_R²"]
    print(f"\n  [공간 분할] 최고: {best_name} (Average R² = {best_avg:.4f})")

    # 테이블: 랜덤 (요약만)
    print("\n" + "-" * 70)
    print("참고 - 랜덤 분할 (상위 5)")
    rows_rand = []
    for name in sorted(results_random.keys(), key=lambda x: -results_random[x]["Average"])[:5]:
        r = results_random[name]
        rows_rand.append({"모델": name, "Average_R²": r["Average"]})
    print(pd.DataFrame(rows_rand).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
