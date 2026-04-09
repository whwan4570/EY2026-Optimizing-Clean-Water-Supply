"""
DRP Grid Search: Systematically test DRP hyperparameter/feature combinations.
Reuses the existing pipeline infrastructure from run_benchmark_notebook.py.
Outputs a ranked CSV of all experiments.

Usage: python drp_grid_search.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import itertools
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

try:
    import xgboost as xgb
except ImportError:
    raise RuntimeError("xgboost required")

from run_benchmark_experiments import load_baseline_data, get_experiment_features
from run_benchmark_notebook import (
    is_unsafe_feature, get_drp_feature_variants, get_drp_expanded_features,
    add_drp_derived_features, get_cluster_groups, get_has_prior,
    _oof_predictions_same_splits, train_model,
    USE_DRP_HYDRO_FEATURES, USE_DRP_CLUSTER_FEATURE, USE_DRP_DISTANCE_PRIOR,
    USE_FOLD_MEDIAN_IMPUTE, USE_EARLY_STOPPING, EARLY_STOPPING_ROUNDS,
    N_CLUSTERS, RANDOM_STATE, LANDSAT_COLS,
    DRP_BASELINE_PATTERNS, DRP_EVENT_PATTERNS,
)

BASE_DIR = Path(__file__).resolve().parent


def run_drp_cv_experiment(
    X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
    model_TA, model_EC, scaler_TA, scaler_EC,
    n_splits=5,
    xgb_params=None,
    residual_mode=True, log_residual=True,
    use_ta_ec=True, ta_ec_blend=1.0,
    distance_km=None, decay_km=None,
    sample_weight=None,
):
    """Lightweight DRP CV: returns (r2_test_avg, rmse_test_avg, r2_train_avg, fold_r2s)."""
    n = len(y_DRP)
    gkf = GroupKFold(n_splits=n_splits)

    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(n)
    gems_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    if residual_mode:
        if log_residual:
            y_res = np.log1p(np.maximum(y_DRP.values, 0)) - np.log1p(np.maximum(gems_safe, 0))
        else:
            y_res = y_DRP.values - gems_safe
    else:
        y_res = np.log1p(np.maximum(y_DRP.values, 0)) if log_residual else y_DRP.values
    q01, q99 = np.percentile(y_res, 1), np.percentile(y_res, 99)
    y_res = np.clip(y_res, q01, q99)
    y_res = pd.Series(y_res, index=y_DRP.index)

    if use_ta_ec:
        oof_TA = _oof_predictions_same_splits(X_TA, y_TA, groups, n_splits)
        oof_EC = _oof_predictions_same_splits(X_EC, y_EC, groups, n_splits)
        if ta_ec_blend != 1.0:
            oof_TA = oof_TA * ta_ec_blend
            oof_EC = oof_EC * ta_ec_blend

    params = xgb_params or {}
    r2_tests, rmse_tests, r2_trains = [], [], []

    for fold, (tr_ix, te_ix) in enumerate(gkf.split(X_DRP, y_DRP, groups)):
        X_tr = X_DRP.iloc[tr_ix].copy().fillna(X_DRP.iloc[tr_ix].median())
        X_te = X_DRP.iloc[te_ix].copy().fillna(X_DRP.iloc[tr_ix].median())
        y_tr = y_res.iloc[tr_ix]

        if use_ta_ec:
            X_ta_tr = X_TA.iloc[tr_ix].fillna(X_TA.iloc[tr_ix].median())
            X_ta_te = X_TA.iloc[te_ix].fillna(X_TA.iloc[tr_ix].median())
            X_ec_tr = X_EC.iloc[tr_ix].fillna(X_EC.iloc[tr_ix].median())
            X_ec_te = X_EC.iloc[te_ix].fillna(X_EC.iloc[tr_ix].median())
            sc_ta, sc_ec = StandardScaler(), StandardScaler()
            m_ta = train_model(sc_ta.fit_transform(X_ta_tr), y_TA.iloc[tr_ix])
            m_ec = train_model(sc_ec.fit_transform(X_ec_tr), y_EC.iloc[tr_ix])
            X_tr = X_tr.copy()
            X_te = X_te.copy()
            X_tr["pred_TA"] = oof_TA[tr_ix]
            X_tr["pred_EC"] = oof_EC[tr_ix]
            X_te["pred_TA"] = m_ta.predict(sc_ta.transform(X_ta_te)) * ta_ec_blend
            X_te["pred_EC"] = m_ec.predict(sc_ec.transform(X_ec_te)) * ta_ec_blend

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        sw = sample_weight[tr_ix] if sample_weight is not None else None
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr_s, y_tr, sample_weight=sw)

        raw_te = m.predict(X_te_s)
        raw_tr = m.predict(X_tr_s)

        def _to_drp(raw, ix):
            if residual_mode:
                g = gems_safe[ix]
                if log_residual:
                    return np.maximum(np.expm1(np.log1p(np.maximum(g, 0)) + raw), 0)
                return np.maximum(g + raw, 0)
            return np.maximum(np.expm1(raw) if log_residual else raw, 0)

        pred_te = np.clip(_to_drp(raw_te, te_ix), 0, 1e4)
        pred_tr = np.clip(_to_drp(raw_tr, tr_ix), 0, 1e4)

        if USE_DRP_DISTANCE_PRIOR and distance_km is not None and decay_km is not None:
            dist_te = np.nan_to_num(distance_km.iloc[te_ix].values if hasattr(distance_km, "iloc") else distance_km[te_ix], nan=999.0)
            pw = np.exp(-dist_te / decay_km)
            pred_te = np.maximum(pred_te + (pw - 1.0) * gems_safe[te_ix], 0)
            pred_te = np.clip(pred_te, 0, 1e4)

        r2_tests.append(r2_score(y_DRP.iloc[te_ix], pred_te))
        rmse_tests.append(np.sqrt(mean_squared_error(y_DRP.iloc[te_ix], pred_te)))
        r2_trains.append(r2_score(y_DRP.iloc[tr_ix], pred_tr))

    return np.mean(r2_tests), np.mean(rmse_tests), np.mean(r2_trains), r2_tests


def compute_dro_weights(X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups, n_splits, worst_weight, n_worst):
    """Pre-compute DRO sample weights from cluster-level RMSE."""
    n = len(y_DRP)
    gems_DRP = X_DRP["gems_DRP"].values if "gems_DRP" in X_DRP.columns else np.zeros(n)
    gems_safe = np.maximum(np.nan_to_num(gems_DRP, nan=0.0), 0)
    y_res = np.log1p(np.maximum(y_DRP.values, 0)) - np.log1p(np.maximum(gems_safe, 0))
    gkf = GroupKFold(n_splits=n_splits)
    cluster_sq = {}
    for _, (tr_ix, te_ix) in enumerate(gkf.split(X_DRP, y_DRP, groups)):
        X_tr = X_DRP.iloc[tr_ix].fillna(X_DRP.median())
        X_te = X_DRP.iloc[te_ix].fillna(X_DRP.median())
        sc = StandardScaler()
        m = train_model(sc.fit_transform(X_tr), pd.Series(y_res).iloc[tr_ix])
        raw = m.predict(sc.transform(X_te))
        pred = np.maximum(np.expm1(np.log1p(np.maximum(gems_safe[te_ix], 0)) + raw), 0)
        for i, idx in enumerate(te_ix):
            g = groups[idx]
            cluster_sq.setdefault(g, []).append((y_DRP.values[idx] - pred[i]) ** 2)
    rmse_per = {g: np.sqrt(np.mean(sq)) for g, sq in cluster_sq.items()}
    worst = sorted(rmse_per, key=lambda g: rmse_per[g], reverse=True)[:n_worst]
    sw = np.ones(n, dtype=np.float64)
    for i in range(n):
        if groups[i] in worst:
            sw[i] = worst_weight
    return sw


def main():
    print("=" * 70)
    print("DRP Grid Search")
    print("=" * 70)

    target_cols = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    wq_data = load_baseline_data()
    feat_TA, feat_EC, _ = get_experiment_features(wq_data, "full_gems")
    feat_TA = [c for c in feat_TA if c in wq_data.columns and c not in target_cols and not is_unsafe_feature(c)]
    feat_EC = [c for c in feat_EC if c in wq_data.columns and c not in target_cols and not is_unsafe_feature(c)]
    if USE_DRP_HYDRO_FEATURES:
        wq_data = add_drp_derived_features(wq_data)
    drp_variants = get_drp_feature_variants(wq_data, target_cols)

    groups, km_model = get_cluster_groups(wq_data, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, return_model=True)
    wq_data = wq_data.copy()
    wq_data["cluster_id"] = groups.astype(np.float32)
    n_splits = min(5, len(np.unique(groups)))

    y_TA = wq_data["Total Alkalinity"]
    y_EC = wq_data["Electrical Conductance"]
    y_DRP = wq_data["Dissolved Reactive Phosphorus"]
    X_TA = wq_data[feat_TA].copy()
    X_EC = wq_data[feat_EC].copy()

    print("  Training TA/EC models...")
    from run_benchmark_notebook import run_pipeline_cv
    _out_TA = run_pipeline_cv(X_TA, y_TA, "Total Alkalinity", groups=groups, n_splits=n_splits)
    _out_EC = run_pipeline_cv(X_EC, y_EC, "Electrical Conductance", groups=groups, n_splits=n_splits)
    model_TA, scaler_TA = _out_TA[0], _out_TA[1]
    model_EC, scaler_EC = _out_EC[0], _out_EC[1]

    distance_km = wq_data["gems_distance_km"] if "gems_distance_km" in wq_data.columns else None

    # --- Define search space ---
    VARIANT_NAMES = [
        "drp_compact_plus_20a_bare",
        "drp_compact_plus_20a_bare_partial_p",
        "drp_compact_plus_20a_bare_month_add",
        "drp_compact_plus_20a_bare_water",
    ]

    XGB_CONFIGS = [
        {"label": "orig_500_d5", "n_estimators": 500, "max_depth": 5, "learning_rate": 0.04,
         "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.5},
        {"label": "reg_450_d5", "n_estimators": 450, "max_depth": 5, "learning_rate": 0.04,
         "subsample": 0.72, "colsample_bytree": 0.68, "reg_alpha": 0.3, "reg_lambda": 1.5, "gamma": 0.8},
        {"label": "reg_400_d4", "n_estimators": 400, "max_depth": 4, "learning_rate": 0.04,
         "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 1.0},
        {"label": "deep_300_d6", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.3},
        {"label": "shallow_600_d3", "n_estimators": 600, "max_depth": 3, "learning_rate": 0.03,
         "subsample": 0.75, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0, "gamma": 1.5},
    ]

    TA_EC_BLENDS = [1.0, 0.9, 0.8]
    DECAY_KMS = [10.0, 15.0, 25.0]
    DRO_CONFIGS = [None, {"weight": 1.5, "n_worst": 2}]
    HYDRO_OPTIONS = [True, False]

    # Pre-compute DRO weights (once)
    print("  Pre-computing DRO weights...")
    base_feat = drp_variants.get("drp_compact_plus_20a_bare_partial_p") or []
    base_feat_exp = get_drp_expanded_features(base_feat, wq_data) + ["cluster_id"]
    base_feat_exp = [c for c in base_feat_exp if c in wq_data.columns]
    X_DRP_base = wq_data[base_feat_exp].copy()
    dro_sw = compute_dro_weights(X_TA, X_EC, X_DRP_base, y_TA, y_EC, y_DRP, groups, n_splits, 1.5, 2)

    results = []
    exp_id = 0

    # Phase 1: Feature variant + XGB config (most impactful)
    print("\n  Phase 1: Feature variants x XGB configs")
    for vname in VARIANT_NAMES:
        feat_base = drp_variants.get(vname) or []
        if not feat_base:
            continue
        feat_exp = get_drp_expanded_features(feat_base, wq_data) + ["cluster_id"]
        feat_exp = [c for c in feat_exp if c in wq_data.columns]
        X_DRP = wq_data[feat_exp].copy()

        for cfg in XGB_CONFIGS:
            label = cfg["label"]
            params = {k: cfg[k] for k in cfg if k != "label"}
            params["random_state"] = 42
            params["verbosity"] = 0

            t0 = time.time()
            r2_te, rmse_te, r2_tr, fold_r2s = run_drp_cv_experiment(
                X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
                model_TA, model_EC, scaler_TA, scaler_EC,
                n_splits=n_splits, xgb_params=params,
                distance_km=distance_km, decay_km=15.0,
            )
            elapsed = time.time() - t0
            exp_id += 1
            row = {"exp": exp_id, "variant": vname, "xgb": label, "blend": 1.0,
                   "decay_km": 15.0, "dro": False, "hydro": True,
                   "R2_Test": r2_te, "RMSE_Test": rmse_te, "R2_Train": r2_tr,
                   "folds": [round(v, 4) for v in fold_r2s], "time_s": round(elapsed, 1)}
            results.append(row)
            fstr = " ".join(f"f{i}={v:.4f}" for i, v in enumerate(fold_r2s))
            print(f"    [{exp_id:2d}] {vname[-20:]:>20s} | {label:>15s} | R2={r2_te:.4f} Tr={r2_tr:.3f} ({fstr}) {elapsed:.0f}s")

    # Phase 2: Best variant + TA_EC blend + decay_km
    results_df = pd.DataFrame(results)
    best_variant = results_df.loc[results_df["R2_Test"].idxmax(), "variant"]
    best_xgb_label = results_df.loc[results_df["R2_Test"].idxmax(), "xgb"]
    best_cfg = next(c for c in XGB_CONFIGS if c["label"] == best_xgb_label)
    best_params = {k: best_cfg[k] for k in best_cfg if k != "label"}
    best_params["random_state"] = 42
    best_params["verbosity"] = 0
    print(f"\n  Phase 1 best: {best_variant} + {best_xgb_label} (R2={results_df['R2_Test'].max():.4f})")

    print(f"\n  Phase 2: TA_EC blend x decay_km x DRO (using {best_variant})")
    feat_base = drp_variants.get(best_variant) or []
    feat_exp = get_drp_expanded_features(feat_base, wq_data) + ["cluster_id"]
    feat_exp = [c for c in feat_exp if c in wq_data.columns]
    X_DRP = wq_data[feat_exp].copy()

    for blend, decay, dro_cfg in itertools.product(TA_EC_BLENDS, DECAY_KMS, DRO_CONFIGS):
        if blend == 1.0 and decay == 15.0 and dro_cfg is None:
            continue  # already tested in Phase 1
        sw = dro_sw if dro_cfg is not None else None
        t0 = time.time()
        r2_te, rmse_te, r2_tr, fold_r2s = run_drp_cv_experiment(
            X_TA, X_EC, X_DRP, y_TA, y_EC, y_DRP, groups,
            model_TA, model_EC, scaler_TA, scaler_EC,
            n_splits=n_splits, xgb_params=best_params,
            ta_ec_blend=blend, distance_km=distance_km, decay_km=decay,
            sample_weight=sw,
        )
        elapsed = time.time() - t0
        exp_id += 1
        row = {"exp": exp_id, "variant": best_variant, "xgb": best_xgb_label, "blend": blend,
               "decay_km": decay, "dro": dro_cfg is not None, "hydro": True,
               "R2_Test": r2_te, "RMSE_Test": rmse_te, "R2_Train": r2_tr,
               "folds": [round(v, 4) for v in fold_r2s], "time_s": round(elapsed, 1)}
        results.append(row)
        dro_str = "DRO" if dro_cfg else "---"
        fstr = " ".join(f"f{i}={v:.4f}" for i, v in enumerate(fold_r2s))
        print(f"    [{exp_id:2d}] blend={blend} decay={decay:4.0f} {dro_str:>3s} | R2={r2_te:.4f} Tr={r2_tr:.3f} ({fstr}) {elapsed:.0f}s")

    # Phase 3: Hydro on/off with best
    print(f"\n  Phase 3: Hydro features on/off")
    feat_base_noh = drp_variants.get(best_variant) or []
    feat_noh = [c for c in feat_base_noh if c in wq_data.columns] + ["cluster_id"]
    feat_noh = [c for c in feat_noh if c in wq_data.columns]
    X_DRP_noh = wq_data[feat_noh].copy()
    t0 = time.time()
    r2_te, rmse_te, r2_tr, fold_r2s = run_drp_cv_experiment(
        X_TA, X_EC, X_DRP_noh, y_TA, y_EC, y_DRP, groups,
        model_TA, model_EC, scaler_TA, scaler_EC,
        n_splits=n_splits, xgb_params=best_params,
        distance_km=distance_km, decay_km=15.0,
    )
    elapsed = time.time() - t0
    exp_id += 1
    row = {"exp": exp_id, "variant": best_variant, "xgb": best_xgb_label, "blend": 1.0,
           "decay_km": 15.0, "dro": False, "hydro": False,
           "R2_Test": r2_te, "RMSE_Test": rmse_te, "R2_Train": r2_tr,
           "folds": [round(v, 4) for v in fold_r2s], "time_s": round(elapsed, 1)}
    results.append(row)
    fstr = " ".join(f"f{i}={v:.4f}" for i, v in enumerate(fold_r2s))
    print(f"    [{exp_id:2d}] hydro=OFF | R2={r2_te:.4f} Tr={r2_tr:.3f} ({fstr}) {elapsed:.0f}s")

    # Save all results
    all_df = pd.DataFrame(results)
    all_df = all_df.sort_values("R2_Test", ascending=False).reset_index(drop=True)
    out_path = BASE_DIR / "drp_grid_search_results.csv"
    all_df.drop(columns=["folds"], errors="ignore").to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by R2_Test)")
    print("=" * 70)
    for i, row in all_df.head(10).iterrows():
        print(f"  #{i+1:2d} R2={row['R2_Test']:.4f} Tr={row['R2_Train']:.3f} | "
              f"{row['variant'][-25:]:>25s} {row['xgb']:>15s} blend={row['blend']} "
              f"decay={row['decay_km']:4.0f} dro={row['dro']} hydro={row['hydro']}")

    best = all_df.iloc[0]
    print(f"\n  BEST: R2_Test={best['R2_Test']:.4f}")
    print(f"    variant:  {best['variant']}")
    print(f"    xgb:      {best['xgb']}")
    print(f"    blend:    {best['blend']}")
    print(f"    decay_km: {best['decay_km']}")
    print(f"    dro:      {best['dro']}")
    print(f"    hydro:    {best['hydro']}")
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
