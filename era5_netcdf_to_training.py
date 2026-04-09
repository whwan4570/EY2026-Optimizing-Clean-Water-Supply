"""
ERA5 0.25° NetCDF (PptData 등) → (Latitude, Longitude, Sample Date) 기준으로
최근접(nearest) 또는 선형 보간 후 precipitation_training.csv / precipitation_validation.csv 생성.

지원 변수 (CDS ecv-for-climate-change 등):
  - Precipitation         → pr
  - Surface air relative humidity → era5_rh
  - 0-7cm volumetric soil moisture → era5_sm

NetCDF 차원: time(또는 month), lat/latitude, lon/longitude.
Sample Date는 월 단위로 매칭 (year-month).
실행: 이 스크립트가 있는 폴더에서 python era5_netcdf_to_training.py [netcdf_path]
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
DATE_START = "2011-01-01"
DATE_END = "2015-12-31"

# NetCDF 변수명 후보 (실제 파일에 있는 이름으로 매핑)
VAR_PRECIP = ("precipitation", "pr", "precip", "tp", "total_precipitation")
VAR_RH = ("surface_air_relative_humidity", "relative_humidity", "rh", "hurs")
VAR_SM = (
    "0_7cm_volumetric_soil_moisture",
    "volumetric_soil_water_layer_1",
    "swvl1",
    "soil_moisture",
    "sm",
)


def parse_sample_date(s: str) -> str | None:
    """Sample Date → 'YYYY-MM'."""
    s = str(s).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    return None


def find_var(ds, candidates, data_vars=None):
    """Dataset에서 후보 이름 중 존재하는 변수명 반환."""
    names = list(ds.data_vars) if data_vars is None else data_vars
    for c in candidates:
        if c in names:
            return c
    # case-insensitive
    lower_map = {k.lower(): k for k in names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def get_lat_lon_time_dims(da):
    """DataArray 차원에서 lat, lon, time 차원 이름 반환."""
    dims = list(da.dims)
    lat = next((d for d in ("lat", "latitude", "y") if d in dims), None)
    lon = next((d for d in ("lon", "longitude", "x") if d in dims), None)
    time = next((d for d in ("time", "month", "date") if d in dims), None)
    return lat, lon, time


def extract_at_points(ds, var_name, lat_dim, lon_dim, time_dim, points_df, date_start, date_end, use_interp=False):
    """
    points_df: columns Latitude, Longitude, year_month (str 'YYYY-MM').
    use_interp: True면 lat/lon 선형 보간, False면 최근접(nearest).
    반환: 길이 len(points_df) 배열.
    """
    import xarray as xr

    da = ds[var_name].squeeze()
    if time_dim and time_dim in ds.dims:
        ds_slice = ds.sel({time_dim: slice(date_start, date_end)})
        da = ds_slice[var_name].squeeze()
    out = []
    for _, row in points_df.iterrows():
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        ym = row["year_month"]
        t = pd.Timestamp(ym + "-01")
        try:
            if use_interp and lat_dim in da.dims and lon_dim in da.dims:
                # 시간은 nearest, lat/lon만 선형 보간
                if time_dim and time_dim in da.dims:
                    da_t = da.sel(**{time_dim: t}, method="nearest").squeeze()
                else:
                    da_t = da
                val = float(da_t.interp(**{lat_dim: lat, lon_dim: lon}, method="linear").values)
            else:
                sel = {lat_dim: lat, lon_dim: lon}
                if time_dim and time_dim in da.dims:
                    sel[time_dim] = t
                val = float(da.sel(**sel, method="nearest").values)
        except Exception:
            val = np.nan
        out.append(val)
    return np.array(out)


def build_merge_df(ds, wq_path, date_start, date_end, use_interp=False):
    """
    훈련/검증 CSV에서 (Latitude, Longitude, Sample Date) 로드 후
    NetCDF에서 pr, era5_rh, era5_sm 추출해 DataFrame 반환.
    use_interp: True면 lat/lon 선형 보간, False면 최근접.
    """
    import xarray as xr

    if not wq_path.exists():
        return None
    df = pd.read_csv(wq_path, usecols=["Latitude", "Longitude", "Sample Date"])
    df = df.drop_duplicates(subset=["Latitude", "Longitude", "Sample Date"]).reset_index(drop=True)
    df["year_month"] = df["Sample Date"].map(parse_sample_date)
    df = df.dropna(subset=["year_month"])

    var_pr = find_var(ds, VAR_PRECIP)
    var_rh = find_var(ds, VAR_RH)
    var_sm = find_var(ds, VAR_SM)

    first_var = var_pr or var_rh or var_sm
    if not first_var:
        print("NetCDF에 precipitation/rh/soil_moisture 변수 없음. 변수:", list(ds.data_vars))
        return None
    lat_dim, lon_dim, time_dim = get_lat_lon_time_dims(ds[first_var])
    if not lat_dim or not lon_dim:
        print("차원 확인 실패:", list(ds[first_var].dims))
        return None

    if var_pr:
        df["pr"] = extract_at_points(ds, var_pr, lat_dim, lon_dim, time_dim, df, date_start, date_end, use_interp)
    if var_rh:
        df["era5_rh"] = extract_at_points(ds, var_rh, lat_dim, lon_dim, time_dim, df, date_start, date_end, use_interp)
    if var_sm:
        df["era5_sm"] = extract_at_points(ds, var_sm, lat_dim, lon_dim, time_dim, df, date_start, date_end, use_interp)

    df = df.drop(columns=["year_month"])
    return df


def main():
    try:
        import xarray as xr
    except ImportError:
        print("pip install xarray")
        sys.exit(1)

    # NetCDF 경로: 인자 > PptData*.nc > era5_ecv*.nc > 기타 *.nc
    if len(sys.argv) >= 2:
        nc_path = Path(sys.argv[1])
    else:
        candidates = list(BASE_DIR.glob("PptData*.nc")) + list(BASE_DIR.glob("era5_ecv*.nc")) + list(BASE_DIR.glob("*era5*.nc"))
        nc_path = candidates[0] if candidates else None
    if not nc_path or not nc_path.exists():
        print("Usage: python era5_netcdf_to_training.py <path_to_netcdf> [0|1]")
        print("  0 = nearest (default), 1 = linear interp for lat/lon")
        print("  Or place PptData*.nc / era5_ecv*.nc in this folder.")
        sys.exit(1)

    # 최근접(0) vs 선형보간(1): 인자 두 번째가 "1"이면 보간
    use_interp = len(sys.argv) >= 3 and sys.argv[2] == "1"

    print(f"Opening NetCDF: {nc_path}")
    ds = xr.open_dataset(nc_path)
    print("  dims:", dict(ds.dims))
    print("  data_vars:", list(ds.data_vars))
    print("  method:", "linear interp (lat/lon)" if use_interp else "nearest")

    # 훈련
    wq_train = BASE_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_train.exists():
        wq_train = BASE_DIR / "water_quality_training_dataset.csv"
    train_df = build_merge_df(ds, wq_train, DATE_START, DATE_END, use_interp=use_interp)
    if train_df is not None:
        out_train = BASE_DIR / "precipitation_training.csv"
        train_df.to_csv(out_train, index=False)
        print(f"Saved: {out_train} (rows={len(train_df)}, cols={list(train_df.columns)})")
    else:
        print("Training merge failed (missing CSV or variables).")

    # 검증 (파일 있으면)
    wq_val = BASE_DIR / "water_quality_validation_dataset.csv"
    val_df = build_merge_df(ds, wq_val, DATE_START, DATE_END, use_interp=use_interp)
    if val_df is not None:
        out_val = BASE_DIR / "precipitation_validation.csv"
        val_df.to_csv(out_val, index=False)
        print(f"Saved: {out_val} (rows={len(val_df)})")

    ds.close()
    print("Done. Merge in benchmark_model on (Latitude, Longitude, Sample Date).")


if __name__ == "__main__":
    main()
