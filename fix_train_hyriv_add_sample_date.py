"""
train_with_hyriv_era5_events.csv에 'Sample Date' 컬럼을 추가하고 DD-MM-YYYY 형식으로 맞춥니다.
수질 학습 데이터와 행 순서가 동일하다고 가정하고, 해당 파일에서 Sample Date를 가져와 삽입합니다.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

MERGE_KEY = ["Latitude", "Longitude", "Sample Date"]


def main():
    wq_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_path.exists():
        wq_path = DATA_DIR / "water_quality_training_dataset.csv"
    if not wq_path.exists():
        print("water_quality_training_dataset(.enriched).csv not found in data/")
        return

    tw_path = DATA_DIR / "train_with_hyriv_era5_events.csv"
    if not tw_path.exists():
        print("train_with_hyriv_era5_events.csv not found in data/")
        return

    wq = pd.read_csv(wq_path, usecols=["Latitude", "Longitude", "Sample Date"])
    tw = pd.read_csv(tw_path)

    if "Sample Date" in tw.columns:
        print("Sample Date already present in train_with_hyriv_era5_events.csv. Exiting.")
        return

    n = len(wq)
    if len(tw) != n:
        print(f"Row count mismatch: water_quality={n}, train_with_hyriv={len(tw)}. Cannot align by index.")
        return

    # Check row alignment by (Lat, Lon)
    match = (wq["Latitude"].values == tw["Latitude"].values) & (wq["Longitude"].values == tw["Longitude"].values)
    if not match.all():
        mismatched = (~match).sum()
        print(f"Lat/Lon mismatch in {mismatched} rows. First mismatch at index: {(~match).argmax()}")
        return

    # Ensure DD-MM-YYYY format (keep as string if already in that form)
    sample_dates = wq["Sample Date"].astype(str)
    # Normalise: if parsed as datetime would give DD-MM-YYYY, leave as is
    tw.insert(2, "Sample Date", sample_dates.values)

    # Reorder: Latitude, Longitude, Sample Date first, then rest
    cols = ["Latitude", "Longitude", "Sample Date"] + [c for c in tw.columns if c not in MERGE_KEY]
    tw = tw[cols]

    tw.to_csv(tw_path, index=False)
    print(f"Updated {tw_path}")
    print(f"  Added 'Sample Date' (DD-MM-YYYY), rows={len(tw)}, sample: {tw['Sample Date'].iloc[0]}")


if __name__ == "__main__":
    main()
