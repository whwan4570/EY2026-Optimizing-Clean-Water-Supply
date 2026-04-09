"""
train_with_hyriv_era5_events.csv의 비어 있는 rain_sum_*, rain_max_*, storm_cnt_* 등을
precipitation_training.csv의 pr(월별 강수)로 대체 채웁니다.

- ERA5 일별 시계열이 없을 때 사용하는 proxy: rain_sum_1m=pr, rain_sum_3m=3*pr, rain_sum_6m=6*pr, rain_sum_12m=12*pr
- benchmark merge가 동작하고 rain_sum_6m 등이 NaN이 아니게 됨.

실행: python fill_rain_sum_from_pr.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KEY = ["Latitude", "Longitude", "Sample Date"]

RAIN_COLS = [
    "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
    "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
    "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
]
SM_WETNESS = [
    "sm_mean_1m", "sm_mean_3m", "sm_lag_1m", "sm_lag_3m",
    "wetness_rain_sm_1m", "wetness_rain_sm_3m",
    "dilution_proxy_1m", "ionic_flush_proxy_3m",
]


def main():
    tw_path = DATA_DIR / "train_with_hyriv_era5_events.csv"
    pr_path = DATA_DIR / "precipitation_training.csv"
    if not tw_path.exists():
        print("train_with_hyriv_era5_events.csv not found in data/")
        return
    if not pr_path.exists():
        print("precipitation_training.csv not found in data/. Cannot fill rain_sum from pr.")
        return

    tw = pd.read_csv(tw_path)
    if "Sample Date" not in tw.columns:
        print("train_with_hyriv_era5_events.csv has no 'Sample Date'. Run fix_train_hyriv_add_sample_date.py first.")
        return

    pr_df = pd.read_csv(pr_path)
    if "pr" not in pr_df.columns or not all(k in pr_df.columns for k in KEY):
        print("precipitation_training.csv must have Latitude, Longitude, Sample Date, pr")
        return

    pr_merge = pr_df[KEY + ["pr"]].drop_duplicates(subset=KEY, keep="first").copy()
    pr_merge = pr_merge.rename(columns={"pr": "_pr"})
    merged = tw.merge(pr_merge, on=KEY, how="left")
    pr_vals = np.asarray(merged["_pr"], dtype=float)
    pr_vals = np.nan_to_num(pr_vals, nan=0.0, posinf=0.0, neginf=0.0)

    filled = 0
    for c in RAIN_COLS:
        if c not in tw.columns:
            continue
        if "rain_sum" in c:
            n = int(c.replace("rain_sum_", "").replace("m", ""))
            merged[c] = np.where(merged[c].isna(), n * pr_vals, merged[c])
        elif "rain_max" in c:
            merged[c] = np.where(merged[c].isna(), pr_vals, merged[c])
        elif "storm_cnt" in c:
            merged[c] = np.where(merged[c].isna(), 0.0, merged[c])
        if merged[c].notna().any():
            filled += 1

    for c in SM_WETNESS:
        if c in merged.columns and merged[c].isna().all():
            merged[c] = 0.0
            filled += 1

    merged = merged.drop(columns=["_pr"], errors="ignore")
    # Keep original column order
    cols = [c for c in tw.columns if c in merged.columns]
    extra = [c for c in merged.columns if c not in cols]
    merged = merged[cols + extra]

    merged.to_csv(tw_path, index=False)
    print(f"Updated {tw_path}")
    nn_6m = merged["rain_sum_6m"].notna().sum() if "rain_sum_6m" in merged.columns else 0
    print(f"  rain_sum_6m non-null: {nn_6m}/{len(merged)}")


if __name__ == "__main__":
    main()
