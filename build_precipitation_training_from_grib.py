"""
GRIB 월별 파일에서 (Latitude, Longitude, Sample Date) 기준으로 강수값을 추출해
precipitation_training.csv를 생성합니다. benchmark_model.py에서 pr merge 시 사용.

- PptData/ 내 *.grib 중 파일명에 _YYYYMM_ 이 있는 것을 사용.
- anomaly 전용이 아닌 모든 YYYYMM GRIB을 사용 (절대 강수 파일이 있으면 pr=절대값,
  anomaly만 있으면 pr=anomaly 값으로 채움 → 주석 참고).
- 요구: pandas, numpy, xarray, cfgrib (eccodes).

실행: 이 스크립트가 있는 폴더에서
  python build_precipitation_training_from_grib.py
  python build_precipitation_training_from_grib.py PptData
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import cfgrib  # noqa: F401  # type: ignore[import-untyped]
except ImportError:
    print("cfgrib 필요. pip install cfgrib (및 conda install -c conda-forge eccodes)")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PPTDATA_DIR = BASE_DIR / "PptData"
TRAINING_CSV = DATA_DIR / "water_quality_training_dataset_enriched.csv"
OUTPUT_CSV = DATA_DIR / "precipitation_training.csv"


def find_grib_files(pptdata_dir: Path, anomaly_only: bool = False) -> dict:
    """
    Scan for .grib with YYYYMM in name.
    anomaly_only=False: 모든 YYYYMM GRIB 사용 (절대 강수용).
    Returns "YYYYMM" -> path.
    """
    if not pptdata_dir.exists():
        return {}
    pattern1 = re.compile(r"_(\d{6})_")
    pattern2 = re.compile(r"(?<!\d)(201[1-5])(0[1-9]|1[0-2])(?!\d)")
    month_to_file = {}
    for path in pptdata_dir.glob("*.grib"):
        name = path.name.lower()
        if anomaly_only and "anomaly" not in name:
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


def _parse_sample_date(s: str) -> tuple | None:
    """Sample Date -> (year, month)."""
    s = str(s).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return (dt.year, dt.month)
        except ValueError:
            continue
    return None


def _normalize_lon(da, lon_dim):
    """Longitude 0-360 -> -180-180, monotonic."""
    lon = np.asarray(da.coords[lon_dim].values)
    if not np.any(lon > 180):
        return da
    lon180 = np.where(lon > 180, lon - 360, lon)
    perm = np.argsort(lon180)
    da = da.assign_coords({lon_dim: lon180}).isel({lon_dim: perm})
    return da


def _get_precip_var(ds):
    """First data var, preferring tp / precip / precipitation."""
    if len(ds.data_vars) == 0:
        return None
    for name in ("tp", "precip", "precipitation", "unknown"):
        if name in ds.data_vars:
            return name
    return list(ds.data_vars)[0]


def extract_monthly_pr(df: pd.DataFrame, month_to_file: dict) -> pd.DataFrame:
    """각 행의 (Latitude, Longitude, Sample Date)에 대해 해당 월 GRIB에서 값을 샘플링해 pr 컬럼 채움."""
    df = df.copy()
    parsed = df["Sample Date"].map(_parse_sample_date)
    df["_year"] = [p[0] if p else np.nan for p in parsed]
    df["_month"] = [p[1] if p else np.nan for p in parsed]
    df["_yyyymm"] = pd.Series(
        [(int(y), int(m)) if pd.notna(y) else None for y, m in zip(df["_year"], df["_month"])],
        index=df.index,
        dtype=object,
    )
    df["pr"] = np.nan

    import xarray as xr

    missing_yyyymm = []
    unique_months = df["_yyyymm"].dropna().drop_duplicates().tolist()

    for yyyyymm in unique_months:
        if not isinstance(yyyyymm, tuple) or len(yyyyymm) != 2:
            continue
        y, m = int(yyyyymm[0]), int(yyyyymm[1])
        key = f"{y}{m:02d}"
        path = month_to_file.get(key)
        if path is None:
            missing_yyyymm.append(key)
            continue
        try:
            ds = xr.open_dataset(path, engine="cfgrib")
        except Exception as e:
            print(f"  Warning: open failed {path.name}: {e}")
            missing_yyyymm.append(key)
            continue
        var_name = _get_precip_var(ds)
        if var_name is None:
            ds.close()
            continue
        da = ds[var_name].squeeze()
        dims = list(da.dims)
        lat_dim = next((d for d in ("latitude", "lat", "y") if d in dims), None)
        lon_dim = next((d for d in ("longitude", "lon", "x") if d in dims), None)
        if not lat_dim or not lon_dim:
            ds.close()
            continue
        da = _normalize_lon(da, lon_dim)
        time_dim = next((d for d in ("time", "valid_time", "date") if d in dims), None)
        if time_dim is not None:
            da = da.isel({time_dim: 0}).squeeze()
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
            pts_lat = xr.DataArray(lats, dims="points")
            pts_lon = xr.DataArray(lons, dims="points")
            sampled = da.sel(**{lat_dim: pts_lat, lon_dim: pts_lon}, method="nearest")
            vals = np.asarray(sampled.values).ravel()
        except Exception:
            vals = np.full(n_pts, np.nan)
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                try:
                    v = float(da.sel(**{lat_dim: lat, lon_dim: lon}, method="nearest").values)
                    vals[i] = v
                except Exception:
                    pass
        df.loc[mask, "pr"] = vals[:n_pts]
        ds.close()

    if missing_yyyymm:
        print(f"  Missing GRIB for YYYYMM: {sorted(set(missing_yyyymm))[:20]}{'...' if len(missing_yyyymm) > 20 else ''}")
    df = df.drop(columns=["_year", "_month", "_yyyymm"], errors="ignore")
    return df


def main():
    pptdata_dir = PPTDATA_DIR
    if len(sys.argv) >= 2:
        pptdata_dir = Path(sys.argv[1])

    print("1) Discover GRIB files (all YYYYMM, not only anomaly)")
    month_to_file = find_grib_files(pptdata_dir, anomaly_only=False)
    n_files = len(month_to_file)
    print(f"   Found {n_files} GRIB files in {pptdata_dir}")
    if not month_to_file:
        print("   No *.grib with _YYYYMM_ in name. Exiting.")
        sys.exit(1)
    keys_sorted = sorted(month_to_file.keys())
    print(f"   YYYYMM range: {keys_sorted[0]} to {keys_sorted[-1]}")
    # 경고: 전부 anomaly 파일이면 pr은 anomaly 값
    any_non_anom = any("anomaly" not in month_to_file[k].name.lower() for k in month_to_file)
    if not any_non_anom:
        print("   Note: All files contain 'anomaly' in name → pr column will be anomaly (not absolute precip).")

    print("\n2) Load training CSV")
    wq_path = TRAINING_CSV if TRAINING_CSV.exists() else DATA_DIR / "water_quality_training_dataset.csv"
    if not wq_path.exists():
        print(f"   Not found: {wq_path}")
        sys.exit(1)
    df = pd.read_csv(wq_path, usecols=["Latitude", "Longitude", "Sample Date"])
    df = df.drop_duplicates(subset=["Latitude", "Longitude", "Sample Date"]).reset_index(drop=True)
    print(f"   Rows: {len(df)}")

    print("\n3) Extract pr per (Lat, Lon, Sample Date)")
    df = extract_monthly_pr(df, month_to_file)
    missing = df["pr"].isna().mean()
    print(f"   pr missing rate: {missing:.2%}")

    out_df = df[["Latitude", "Longitude", "Sample Date", "pr"]].copy()
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n4) Saved: {OUTPUT_CSV} (cols: {list(out_df.columns)})")
    print("   Use in benchmark_model.py: merge on (Latitude, Longitude, Sample Date).")


if __name__ == "__main__":
    main()
