"""
EY 2026 Optimizing Clean Water Supply — Precipitation anomaly from GRIB.

- Discovers monthly GRIB files in PptData/ (pattern *_YYYYMM_*).
- Extracts nearest-grid precipitation anomaly per training row.
- Adds lag and hydrology features; saves water_quality_with_precip_anomaly.csv.

Requires: pandas, numpy, xarray, cfgrib (and eccodes for GRIB support).
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Register cfgrib engine so xarray can open GRIB files (optional: pip install cfgrib)
try:
    import cfgrib  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    print(
        "cfgrib is not installed. xarray cannot open GRIB files without it.\n"
        "Install with:\n  pip install cfgrib\n"
        "On Windows, if that fails, try:\n  conda install -c conda-forge eccodes cfgrib"
    )
    sys.exit(1)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PPTDATA_DIR = BASE_DIR / "PptData"
TRAINING_CSV = DATA_DIR / "water_quality_training_dataset_enriched.csv"
OUTPUT_CSV = DATA_DIR / "water_quality_with_precip_anomaly.csv"

# South Africa bounding box for coverage check
SA_LAT_MIN, SA_LAT_MAX = -35, -22
SA_LON_MIN, SA_LON_MAX = 16, 33


def find_grib_files(pptdata_dir: Path) -> dict:
    """
    Scan pptdata_dir for .grib files containing YYYYMM in the name.
    Returns mapping "YYYYMM" (str) -> filepath for reliable lookup across environments.
    """
    if not pptdata_dir.exists():
        return {}
    pattern1 = re.compile(r"_(\d{6})_")
    pattern2 = re.compile(r"(?<!\d)(201[1-5])(0[1-9]|1[0-2])(?!\d)")
    month_to_file = {}
    for path in pptdata_dir.glob("*.grib"):
        name = path.name.lower()
        if "anomaly" not in name:
            continue
        match = pattern1.search(path.name)
        if not match:
            match = pattern2.search(path.name)
            if match:
                y, m = int(match.group(1)), int(match.group(2))
                month_to_file[f"{y}{m:02d}"] = path
        else:
            yyyymm = match.group(1)
            y, m = int(yyyymm[:4]), int(yyyymm[4:6])
            if 1 <= m <= 12:
                month_to_file[f"{y}{m:02d}"] = path
    return month_to_file


def inspect_grib(path: Path) -> dict:
    """
    Load one GRIB with xarray/cfgrib and return summary info.
    Handles lat/lon vs latitude/longitude; converts lon 0-360 to -180-180 if needed.
    """
    try:
        import xarray as xr
    except ImportError:
        return {"error": "xarray not installed. pip install xarray"}
    try:
        ds = xr.open_dataset(path, engine="cfgrib")
    except Exception as e:
        msg = str(e).lower()
        if "eccodes" in msg or "cfgrib" in msg or "grib" in msg:
            return {
                "error": f"GRIB open failed: {e}",
                "hint": "Install eccodes (e.g. conda install -c conda-forge eccodes) and cfgrib (pip install cfgrib).",
            }
        return {"error": f"GRIB open failed: {e}"}

    info = {"path": str(path), "summary": str(ds)}
    # Coordinate names: GRIB often uses latitude/longitude
    coord_candidates = (
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("y", "x"),
    )
    lat_dim, lon_dim = None, None
    for lat_name, lon_name in coord_candidates:
        if lat_name in ds.coords and lon_name in ds.coords:
            lat_dim, lon_dim = lat_name, lon_name
            break
    if lat_dim is None:
        # Try from first data variable dims
        for v in ds.data_vars:
            dims = list(ds[v].dims)
            for la, lo in coord_candidates:
                if la in dims and lo in dims:
                    lat_dim, lon_dim = la, lo
                    break
            if lat_dim:
                break
    info["lat_dim"] = lat_dim
    info["lon_dim"] = lon_dim

    if lat_dim and lon_dim:
        lats = ds.coords[lat_dim].values
        lons = ds.coords[lon_dim].values
        info["lat_min"] = float(np.min(lats))
        info["lat_max"] = float(np.max(lats))
        info["lon_min"] = float(np.min(lons))
        info["lon_max"] = float(np.max(lons))
        # Convert longitude 0-360 to -180-180 for display/use
        if np.any(lons >= 180):
            lons_180 = np.where(lons > 180, lons - 360, lons)
            info["lon_min_180"] = float(np.min(lons_180))
            info["lon_max_180"] = float(np.max(lons_180))
            info["lon_was_0_360"] = True
        else:
            info["lon_min_180"] = info["lon_min"]
            info["lon_max_180"] = info["lon_max"]
            info["lon_was_0_360"] = False
        # South Africa coverage
        info["covers_sa_lat"] = info["lat_min"] <= SA_LAT_MAX and info["lat_max"] >= SA_LAT_MIN
        info["covers_sa_lon"] = info["lon_min_180"] <= SA_LON_MAX and info["lon_max_180"] >= SA_LON_MIN
    # Time dimension if present
    for tdim in ("time", "valid_time", "date"):
        if tdim in ds.coords:
            info["time_dim"] = tdim
            tvals = ds.coords[tdim].values
            info["time_values"] = str(tvals[:3]) if np.ndim(tvals) >= 1 else str(tvals)
            break
    ds.close()
    return info


def _parse_sample_date(s: str) -> tuple[int, int] | None:
    """Parse DD-MM-YYYY to (year, month)."""
    s = str(s).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.year, dt.month
        except ValueError:
            continue
    return None


def _normalize_lon(da, lon_dim):
    """Convert longitude 0-360 to -180-180 and sort so coordinate is monotonic (nearest selection works)."""
    lon = np.asarray(da.coords[lon_dim].values)
    if not np.any(lon > 180):
        return da
    lon180 = np.where(lon > 180, lon - 360, lon)
    perm = np.argsort(lon180)
    da = da.assign_coords({lon_dim: lon180}).isel({lon_dim: perm})
    return da


def _get_anomaly_var(ds):
    """Return the main anomaly data array (single data var or first one)."""
    if len(ds.data_vars) == 0:
        return None
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    # Prefer names suggesting anomaly/precip
    for name in ("tp", "precipitation", "precip", "anomaly", "unknown"):
        if name in ds.data_vars:
            return name
    return list(ds.data_vars)[0]


def extract_monthly_anomaly(df: pd.DataFrame, month_to_file: dict) -> pd.DataFrame:
    """
    For each row, set precip_anom from the GRIB file for that row's (year, month).
    Uses vectorized nearest sampling per unique YYYYMM where possible.
    """
    df = df.copy()
    # Parse Sample Date to (year, month) and YYYYMM (use object dtype for tuple column)
    parsed = df["Sample Date"].map(_parse_sample_date)
    df["_year"] = [p[0] if p else np.nan for p in parsed]
    df["_month"] = [p[1] if p else np.nan for p in parsed]
    df["_yyyymm"] = pd.Series(
        [(int(y), int(m)) if pd.notna(y) else None for y, m in zip(df["_year"], df["_month"])],
        index=df.index,
        dtype=object,
    )
    df["precip_anom"] = np.nan

    try:
        import xarray as xr
    except ImportError:
        print("extract_monthly_anomaly: xarray not installed.")
        return df

    missing_yyyymm = []
    unique_months = df["_yyyymm"].dropna().drop_duplicates().tolist()

    for yyyymm in unique_months:
        if not isinstance(yyyymm, tuple) or len(yyyymm) != 2:
            continue
        y, m = int(yyyymm[0]), int(yyyymm[1])
        key = f"{y}{m:02d}"
        path = month_to_file.get(key)
        if path is None:
            missing_yyyymm.append(key)
            continue
        try:
            ds = xr.open_dataset(path, engine="cfgrib")
        except Exception as e:
            print(f"  Warning: could not open {path.name}: {e}")
            missing_yyyymm.append(f"{y}{m:02d}")
            continue
        var_name = _get_anomaly_var(ds)
        if var_name is None:
            ds.close()
            continue
        da = ds[var_name].squeeze()
        # Coordinate names
        dims = list(da.dims)
        lat_dim = next((d for d in ("latitude", "lat", "y") if d in dims), None)
        lon_dim = next((d for d in ("longitude", "lon", "x") if d in dims), None)
        if not lat_dim or not lon_dim:
            ds.close()
            continue
        # Longitude 0-360 -> -180-180 for correct nearest in SA
        da = _normalize_lon(da, lon_dim)
        # Time: select single slice if present
        time_dim = next((d for d in ("time", "valid_time", "date") if d in dims), None)
        if time_dim is not None:
            da = da.isel({time_dim: 0}).squeeze()
        # Rows for this month (compare as scalars to avoid numpy vs Python int mismatch)
        mask = df["_yyyymm"].apply(
            lambda x: (int(x[0]) == y and int(x[1]) == m) if isinstance(x, tuple) and len(x) == 2 else False
        )
        n_pts = mask.sum()
        if n_pts == 0:
            ds.close()
            continue
        lats = df.loc[mask, "Latitude"].values.astype(float)
        lons = df.loc[mask, "Longitude"].values.astype(float)
        try:
            # Vectorized nearest selection
            pts_lat = xr.DataArray(lats, dims="points")
            pts_lon = xr.DataArray(lons, dims="points")
            sampled = da.sel(
                **{lat_dim: pts_lat, lon_dim: pts_lon},
                method="nearest",
            )
            vals = np.asarray(sampled.values).ravel()
        except Exception:
            # Fallback: loop over points (still only once per month)
            vals = np.full(n_pts, np.nan)
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                try:
                    v = float(da.sel(**{lat_dim: lat, lon_dim: lon}, method="nearest").values)
                    vals[i] = v
                except Exception:
                    pass
        df.loc[mask, "precip_anom"] = vals[:n_pts]
        ds.close()

    if missing_yyyymm:
        print(f"  Missing GRIB for YYYYMM: {sorted(set(missing_yyyymm))[:20]}{'...' if len(missing_yyyymm) > 20 else ''}")
    # Drop helpers
    df = df.drop(columns=["_year", "_month", "_yyyymm"], errors="ignore")
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add precip_anom_lag1, lag2, lag3 by (Latitude, Longitude) in chronological order.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["Sample Date"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(["Latitude", "Longitude", "_date"]).reset_index(drop=True)
    grouped = df.groupby(["Latitude", "Longitude"], sort=False)
    df["precip_anom_lag1"] = grouped["precip_anom"].shift(1)
    df["precip_anom_lag2"] = grouped["precip_anom"].shift(2)
    df["precip_anom_lag3"] = grouped["precip_anom"].shift(3)
    df = df.drop(columns=["_date"], errors="ignore")
    return df


def add_hydrology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wetness_index, storm_pet_ratio, cumulative_anom_3m, cumulative_wetness_3m,
    dry_spike_index, anom_pos, anom_neg. Requires pet and precip_anom (and lags).
    """
    df = df.copy()
    eps = 1e-6
    pet = df["pet"].fillna(0) + eps

    df["wetness_index"] = df["precip_anom"] / pet
    df["storm_pet_ratio"] = (df["precip_anom"] - df["precip_anom_lag1"]) / pet
    df["dry_spike_index"] = df["precip_anom"] * (1.0 / pet)
    df["anom_pos"] = np.maximum(df["precip_anom"], 0)
    df["anom_neg"] = np.maximum(-df["precip_anom"], 0)

    # Rolling 3-month per (Lat, Lon) — need order preserved
    if "_date" not in df.columns:
        df["_date"] = pd.to_datetime(df["Sample Date"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(["Latitude", "Longitude", "_date"]).reset_index(drop=True)
    grouped = df.groupby(["Latitude", "Longitude"], sort=False)
    df["cumulative_anom_3m"] = grouped["precip_anom"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["cumulative_wetness_3m"] = grouped["wetness_index"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df = df.drop(columns=["_date"], errors="ignore")
    return df


def main():
    print("1) Discover GRIB files and parse YYYYMM")
    month_to_file = find_grib_files(PPTDATA_DIR)
    n_files = len(month_to_file)
    print(f"   Found {n_files} GRIB files in {PPTDATA_DIR}")
    if not month_to_file:
        print("   PptData/ missing or no *.grib with _YYYYMM_. Exiting.")
        return
    keys_sorted = sorted(month_to_file.keys())
    print(f"   YYYYMM range: {keys_sorted[0]} to {keys_sorted[-1]}")
    print(f"   Sample keys: {keys_sorted[:3]} ... {keys_sorted[-2:]}")

    print("\n2) Load one representative GRIB and inspect")
    first_path = month_to_file[keys_sorted[0]]
    info = inspect_grib(first_path)
    if "error" in info:
        print(f"   Error: {info['error']}")
        if "hint" in info:
            print(f"   Hint: {info['hint']}")
        return
    print(f"   Path: {info['path']}")
    print(f"   lat_dim={info.get('lat_dim')}, lon_dim={info.get('lon_dim')}")
    print(f"   latitude: [{info.get('lat_min')}, {info.get('lat_max')}]")
    print(f"   longitude (original): [{info.get('lon_min')}, {info.get('lon_max')}]")
    print(f"   longitude (-180..180): [{info.get('lon_min_180')}, {info.get('lon_max_180')}]")
    print(f"   Covers SA bbox (lat {SA_LAT_MIN}..{SA_LAT_MAX}, lon {SA_LON_MIN}..{SA_LON_MAX}): "
          f"lat={info.get('covers_sa_lat')}, lon={info.get('covers_sa_lon')}")
    if "time_dim" in info:
        print(f"   time_dim: {info['time_dim']}, sample values: {info.get('time_values', '')}")

    print("\n3) Load training CSV and extract anomaly per row")
    if not TRAINING_CSV.exists():
        print(f"   Training CSV not found: {TRAINING_CSV}")
        return
    df = pd.read_csv(TRAINING_CSV)
    print("   Columns count:", len(df.columns))
    print("   Has pet?", "pet" in df.columns)
    print("   Sample columns:", df.columns[:25].tolist())
    print(f"   Rows: {len(df)}")
    df = extract_monthly_anomaly(df, month_to_file)
    missing_rate = df["precip_anom"].isna().mean()
    print(f"   precip_anom missing rate: {missing_rate:.2%}")
    _missing = df.loc[df["precip_anom"].isna()].copy()
    _missing["y"] = pd.to_datetime(_missing["Sample Date"], format="%d-%m-%Y", errors="coerce").dt.year
    _missing["m"] = pd.to_datetime(_missing["Sample Date"], format="%d-%m-%Y", errors="coerce").dt.month
    _missing = _missing.dropna(subset=["y", "m"]).drop_duplicates(subset=["y", "m"])
    missing_yyyymm = [f"{int(r['y'])}{int(r['m']):02d}" for _, r in _missing[["y", "m"]].iterrows()]
    if missing_yyyymm:
        print(f"   Missing YYYYMM (sample): {sorted(set(missing_yyyymm))[:15]}")

    print("\n4) Add lag features")
    df = add_lags(df)

    print("5) Hydrology feature engineering")
    if "pet" not in df.columns:
        terra_path = DATA_DIR / "terraclimate_features_training.csv"
        if terra_path.exists():
            terra = pd.read_csv(terra_path)
            if "pet" in terra.columns and all(k in terra.columns for k in ["Latitude", "Longitude", "Sample Date"]):
                df = df.merge(
                    terra[["Latitude", "Longitude", "Sample Date", "pet"]],
                    on=["Latitude", "Longitude", "Sample Date"],
                    how="left",
                )
                print("   Merged 'pet' from terraclimate_features_training.csv")
        if "pet" not in df.columns:
            print("   Warning: 'pet' not in dataframe; using 0 for pet in hydrology features.")
            df["pet"] = 0.0
    df = add_hydrology_features(df)

    print("6) Save output")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved: {OUTPUT_CSV}")

    print("\n7) Validation prints")
    print(df.head().to_string())
    cols = ["precip_anom", "precip_anom_lag1", "wetness_index"]
    print("\nDescribe (precip_anom, precip_anom_lag1, wetness_index):")
    print(df[cols].describe().to_string())
    new_cols = [
        "precip_anom", "precip_anom_lag1", "precip_anom_lag2", "precip_anom_lag3",
        "wetness_index", "storm_pet_ratio", "cumulative_anom_3m", "cumulative_wetness_3m",
        "dry_spike_index", "anom_pos", "anom_neg",
    ]
    print("\nMissing value counts for new columns:")
    for c in new_cols:
        if c in df.columns:
            print(f"   {c}: {df[c].isna().sum()}")


if __name__ == "__main__":
    main()
