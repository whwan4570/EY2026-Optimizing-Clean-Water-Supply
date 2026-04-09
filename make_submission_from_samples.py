"""
submission_template 각 행에 대해 samples(GEMS 실측)의 TA/EC/DRP를 매칭해 제출 파일 생성.
- (Lat, Lon) → 가장 가까운 GEMS 지점 선택 (템플릿 좌표 = GEMS 지점이므로 동일 지점)
- (지점, Sample Date)로 samples에서 Alk-Tot, EC, DRP 실측 조회 (같은 날 없으면 해당 지점 중앙값 사용)
실행: python make_submission_from_samples.py
      python make_submission_from_samples.py --mm-dd-yyyy  # 템플릿 날짜를 MM-DD-YYYY로 해석
검증: python verify_submission_from_samples.py  (행 순서·키 일치 여부)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_PATH = DATA_DIR / "samples_with_coordinates.csv"
if not SAMPLES_PATH.exists():
    SAMPLES_PATH = BASE_DIR / "samples_with_coordinates.csv"
if not SAMPLES_PATH.exists():
    SAMPLES_PATH = BASE_DIR / "samples.csv"

KEY_COLS = ["Latitude", "Longitude", "Sample Date"]
TARGET_MAP = {"Alk-Tot": "Total Alkalinity", "EC": "Electrical Conductance", "DRP": "Dissolved Reactive Phosphorus"}
PARAMS = list(TARGET_MAP.keys())
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


# 날짜 형식: Day-Month-Year (DD-MM-YYYY). 예: 01-09-2014 = 1일 9월 2014년
DATE_FMT_DMY = "%d-%m-%Y"   # day-month-year
DATE_FMT_MDY = "%m-%d-%Y"   # (테스트용) month-day-year


def parse_date(ser, use_mm_dd_yyyy=False):
    """Day-Month-Year (DD-MM-YYYY) 또는 테스트용 MM-DD-YYYY -> datetime."""
    fmt = DATE_FMT_MDY if use_mm_dd_yyyy else DATE_FMT_DMY
    out = pd.to_datetime(ser, format=fmt, errors="coerce")
    if out.isna().all() and not use_mm_dd_yyyy:
        out = pd.to_datetime(ser, format="%Y-%m-%d", errors="coerce")
    return out


def main():
    ap = argparse.ArgumentParser(description="GEMS samples로 제출 파일 생성")
    ap.add_argument("--mm-dd-yyyy", dest="mm_dd_yyyy", action="store_true", help="템플릿만 MM-DD-YYYY로 해석 (기본: Day-Month-Year, DD-MM-YYYY)")
    args = ap.parse_args()
    use_mm_dd_yyyy = getattr(args, "mm_dd_yyyy", False)

    template_path = DATA_DIR / "submission_template.csv"
    if not template_path.exists():
        print(f"없음: {template_path}")
        return
    if not SAMPLES_PATH.exists():
        print(f"없음: {SAMPLES_PATH}. samples_with_coordinates.csv 또는 samples.csv 필요.")
        return

    sub = pd.read_csv(template_path)
    sep = ";" if ";" in open(SAMPLES_PATH, encoding="utf-8").readline() else ","
    samples = pd.read_csv(SAMPLES_PATH, sep=sep, encoding="utf-8", low_memory=False)
    samples["Value"] = pd.to_numeric(samples["Value"], errors="coerce")
    samples = samples[samples["Parameter Code"].isin(PARAMS)].copy()

    # Day-Month-Year (DD-MM-YYYY)로 통일, 매칭은 하루(day) 단위로
    samples["Sample Date"] = parse_date(samples["Sample Date"], use_mm_dd_yyyy=False)
    sub["Sample Date_parsed"] = parse_date(sub["Sample Date"], use_mm_dd_yyyy=use_mm_dd_yyyy)
    # 같은 날짜 비교를 위해 시간 제거 → 하루(day) 단위로 매칭
    samples["Sample Date"] = pd.to_datetime(samples["Sample Date"]).dt.normalize()
    sub["Sample Date_parsed"] = pd.to_datetime(sub["Sample Date_parsed"]).dt.normalize()

    station_coords = (
        samples.groupby("GEMS Station Number")
        .agg({"Latitude": "first", "Longitude": "first"})
        .reset_index()
    )
    stations = station_coords["GEMS Station Number"].values
    st_lat = station_coords["Latitude"].astype(float).values
    st_lon = station_coords["Longitude"].astype(float).values

    # (Station, Date) 별 Alk-Tot, EC, DRP 값 (같은 날 여러 개면 중앙값)
    by_station_date = (
        samples.groupby(["GEMS Station Number", "Sample Date", "Parameter Code"])["Value"]
        .median()
        .reset_index()
    )
    # 지점별 중앙값 (날짜 무관, fallback용)
    by_station = (
        samples.groupby(["GEMS Station Number", "Parameter Code"])["Value"]
        .median()
        .reset_index()
    )

    n = len(sub)
    out_ta = np.full(n, np.nan)
    out_ec = np.full(n, np.nan)
    out_drp = np.full(n, np.nan)

    sub_lat = sub["Latitude"].astype(float).values
    sub_lon = sub["Longitude"].astype(float).values
    sub_dates = sub["Sample Date_parsed"].values

    for i in range(n):
        lat, lon = sub_lat[i], sub_lon[i]
        dt = sub_dates[i]
        if pd.isna(lat) or pd.isna(lon):
            continue
        dists = haversine_km(lat, lon, st_lat, st_lon)
        nearest_ix = np.argmin(dists)
        station = stations[nearest_ix]
        for param, col in TARGET_MAP.items():
            # 같은 (지점, 날짜) 실측
            row = by_station_date[
                (by_station_date["GEMS Station Number"] == station)
                & (by_station_date["Sample Date"] == dt)
                & (by_station_date["Parameter Code"] == param)
            ]
            if len(row) > 0:
                val = row["Value"].iloc[0]
            else:
                # fallback: 지점 중앙값
                row = by_station[
                    (by_station["GEMS Station Number"] == station)
                    & (by_station["Parameter Code"] == param)
                ]
                val = row["Value"].iloc[0] if len(row) > 0 else np.nan
            if col == "Total Alkalinity":
                out_ta[i] = val
            elif col == "Electrical Conductance":
                out_ec[i] = val
            else:
                out_drp[i] = val

    sub["Total Alkalinity"] = out_ta
    sub["Electrical Conductance"] = out_ec
    sub["Dissolved Reactive Phosphorus"] = out_drp

    # 결측: 지점 중앙값으로 한 번 더 (날짜 매칭 실패 시)
    for i in range(n):
        lat, lon = sub_lat[i], sub_lon[i]
        if pd.notna(sub["Total Alkalinity"].iloc[i]):
            continue
        dists = haversine_km(lat, lon, st_lat, st_lon)
        nearest_ix = np.argmin(dists)
        station = stations[nearest_ix]
        for param, col in TARGET_MAP.items():
            row = by_station[
                (by_station["GEMS Station Number"] == station)
                & (by_station["Parameter Code"] == param)
            ]
            if len(row) == 0:
                continue
            val = row["Value"].iloc[0]
            if col == "Total Alkalinity":
                sub.iloc[i, sub.columns.get_loc(col)] = val
            elif col == "Electrical Conductance":
                sub.iloc[i, sub.columns.get_loc(col)] = val
            else:
                sub.iloc[i, sub.columns.get_loc(col)] = val

    sub = sub.drop(columns=["Sample Date_parsed"], errors="ignore")
    out_path = DATA_DIR / ("submission_from_samples_mmdd.csv" if use_mm_dd_yyyy else "submission_from_samples.csv")
    sub.to_csv(out_path, index=False)
    print(f"날짜 해석: {'MM-DD-YYYY' if use_mm_dd_yyyy else 'Day-Month-Year (DD-MM-YYYY)'}")
    print(f"저장: {out_path} ({len(sub)}행)")
    for col in TARGET_MAP.values():
        v = sub[col]
        filled = v.notna().sum()
        print(f"  {col}: 채워진 행 {filled}/{n}, mean={v.mean():.4f}, min={v.min():.4f}, max={v.max():.4f}")


if __name__ == "__main__":
    main()
