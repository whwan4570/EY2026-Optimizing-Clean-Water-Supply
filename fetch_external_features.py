"""
외부 데이터 fetch + 피처 생성 스크립트.
각 측정 지점 (Latitude, Longitude) 기준으로:
  1. ESA WorldCover 2021 → 토지이용 비율 (cropland, urban, forest, water, grassland)
  2. SoilGrids → 토양 특성 (점토 비율, 유기탄소, pH)
  3. OpenTopography/SRTM → 고도 (elevation)
  4. WorldPop → 인구밀도

결과: external_features_training.csv, external_features_validation.csv
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

BUFFER_DEG = 0.01  # ~1km radius for land cover sampling


# ─────────────────────────────────────────────
# 1. SoilGrids (ISRIC REST API)
# ─────────────────────────────────────────────

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["clay", "soc", "phh2o"]  # clay%, organic carbon, pH
SOIL_DEPTHS = ["0-5cm"]
SOIL_VALUES = ["mean"]

def fetch_soilgrids_point(lat: float, lon: float, retries=3) -> dict:
    """SoilGrids REST API에서 한 지점의 토양 특성 조회."""
    params = {
        "lon": round(lon, 5),
        "lat": round(lat, 5),
        "property": SOIL_PROPERTIES,
        "depth": SOIL_DEPTHS,
        "value": SOIL_VALUES,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(SOILGRIDS_URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                result = {}
                for layer in data.get("properties", {}).get("layers", []):
                    prop_name = layer["name"]
                    depths = layer.get("depths", [])
                    if depths:
                        val = depths[0].get("values", {}).get("mean")
                        if val is not None:
                            if prop_name == "clay":
                                result["soil_clay_pct"] = val / 10.0  # g/kg → %
                            elif prop_name == "soc":
                                result["soil_organic_carbon"] = val / 10.0  # dg/kg → g/kg
                            elif prop_name == "phh2o":
                                result["soil_ph"] = val / 10.0  # pH*10 → pH
                return result
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return {}
        except Exception:
            time.sleep(1)
    return {}


# ─────────────────────────────────────────────
# 2. Open-Elevation (free DEM API)
# ─────────────────────────────────────────────

ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

def fetch_elevation_batch(lats: list, lons: list, batch_size=100) -> list:
    """Open-Elevation API에서 고도(m) 일괄 조회."""
    elevations = [np.nan] * len(lats)
    for start in range(0, len(lats), batch_size):
        end = min(start + batch_size, len(lats))
        locations = [
            {"latitude": float(lats[i]), "longitude": float(lons[i])}
            for i in range(start, end)
        ]
        for attempt in range(3):
            try:
                resp = requests.post(
                    ELEVATION_URL,
                    json={"locations": locations},
                    timeout=60,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for j, r in enumerate(results):
                        elevations[start + j] = r.get("elevation", np.nan)
                    break
                else:
                    time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2)
        time.sleep(0.5)
    return elevations


# ─────────────────────────────────────────────
# 3. WorldPop (인구밀도) — REST API
# ─────────────────────────────────────────────

WORLDPOP_URL = "https://api.worldpop.org/v1/services/stats"

def fetch_population_point(lat: float, lon: float, year: int = 2015) -> Optional[float]:
    """WorldPop API에서 한 지점 인구밀도 조회 (people/km²)."""
    params = {
        "dataset": "wpgppop",
        "year": year,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
    }
    for attempt in range(3):
        try:
            resp = requests.get(WORLDPOP_URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and "total_population" in data["data"]:
                    pop = data["data"]["total_population"]
                    return float(pop) if pop is not None else np.nan
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return np.nan


# ─────────────────────────────────────────────
# 4. ESA WorldCover 2021 (Planetary Computer STAC)
# ─────────────────────────────────────────────

def fetch_landcover_ratios(lats: list, lons: list, buffer_deg: float = 0.005) -> pd.DataFrame:
    """
    Planetary Computer의 ESA WorldCover 2021에서
    각 지점 주변 buffer_deg 범위의 토지이용 비율 산출.
    """
    try:
        import planetary_computer
        import pystac_client
        import rioxarray  # noqa
        import xarray as xr
    except ImportError:
        print("  [WorldCover] planetary_computer / pystac_client / rioxarray 미설치 → skip")
        return pd.DataFrame()

    lc_classes = {
        10: "lc_tree_pct",
        20: "lc_shrub_pct",
        30: "lc_grassland_pct",
        40: "lc_cropland_pct",
        50: "lc_urban_pct",
        60: "lc_bare_pct",
        80: "lc_water_pct",
    }

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    unique_pts = pd.DataFrame({"lat": lats, "lon": lons}).drop_duplicates().reset_index(drop=True)
    results = {col: np.full(len(unique_pts), np.nan) for col in lc_classes.values()}

    for i, row in unique_pts.iterrows():
        lat, lon = row["lat"], row["lon"]
        bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        try:
            search = catalog.search(
                collections=["esa-worldcover"],
                bbox=bbox,
                query={"esa_worldcover:version": {"eq": "v200"}},
            )
            items = list(search.items())
            if not items:
                search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
                items = list(search.items())
            if not items:
                continue

            item = items[0]
            signed_href = item.assets["map"].href
            ds = xr.open_dataarray(signed_href, engine="rasterio")
            clipped = ds.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
            )
            vals = clipped.values.flatten()
            total = len(vals)
            if total == 0:
                continue
            for code, col_name in lc_classes.items():
                results[col_name][i] = np.sum(vals == code) / total * 100.0
        except Exception as e:
            if i < 3:
                print(f"  [WorldCover] point {i} ({lat:.4f},{lon:.4f}) error: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  [WorldCover] {i+1}/{len(unique_pts)} points processed")

    unique_pts = unique_pts.assign(**results)

    out = pd.DataFrame({"lat": lats, "lon": lons})
    out = out.merge(unique_pts, on=["lat", "lon"], how="left")
    return out.drop(columns=["lat", "lon"])


# ─────────────────────────────────────────────
# Main: 모든 외부 데이터 fetch → CSV 저장
# ─────────────────────────────────────────────

def get_unique_locations(df: pd.DataFrame):
    """고유 (Latitude, Longitude) 추출."""
    return df[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)


def process_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """한 데이터셋의 모든 지점에 대해 외부 피처 수집."""
    locs = get_unique_locations(df)
    n = len(locs)
    print(f"\n{'='*60}")
    print(f"[{name}] {n}개 고유 지점에서 외부 피처 수집 시작")
    print(f"{'='*60}")

    lats = locs["Latitude"].tolist()
    lons = locs["Longitude"].tolist()

    # 1. SoilGrids
    print(f"\n[1/4] SoilGrids 토양 데이터 ({n} points)...")
    soil_data = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        soil = fetch_soilgrids_point(lat, lon)
        soil_data.append(soil)
        if (i + 1) % 20 == 0:
            print(f"  SoilGrids: {i+1}/{n}")
        time.sleep(0.3)  # rate limiting

    soil_df = pd.DataFrame(soil_data, index=locs.index)
    for col in ["soil_clay_pct", "soil_organic_carbon", "soil_ph"]:
        if col not in soil_df.columns:
            soil_df[col] = np.nan
    locs = pd.concat([locs, soil_df], axis=1)
    print(f"  SoilGrids 완료: {soil_df.notna().sum().to_dict()}")

    # 2. Elevation
    print(f"\n[2/4] Elevation 고도 ({n} points)...")
    elevations = fetch_elevation_batch(lats, lons)
    locs["elevation_m"] = elevations
    valid_elev = sum(1 for e in elevations if not np.isnan(e))
    print(f"  Elevation 완료: {valid_elev}/{n} valid")

    # 3. WorldPop 인구밀도
    print(f"\n[3/4] WorldPop 인구밀도 ({n} points)...")
    pop_data = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        pop = fetch_population_point(lat, lon)
        pop_data.append(pop)
        if (i + 1) % 20 == 0:
            print(f"  WorldPop: {i+1}/{n}")
        time.sleep(0.3)
    locs["population_density"] = pop_data
    valid_pop = sum(1 for p in pop_data if not np.isnan(p))
    print(f"  WorldPop 완료: {valid_pop}/{n} valid")

    # 4. ESA WorldCover
    print(f"\n[4/4] ESA WorldCover 토지이용 ({n} points)...")
    lc_df = fetch_landcover_ratios(lats, lons)
    if not lc_df.empty:
        for col in lc_df.columns:
            locs[col] = lc_df[col].values
        print(f"  WorldCover 완료: {lc_df.columns.tolist()}")
    else:
        print("  WorldCover: skipped (dependencies missing or API error)")
        for col in ["lc_tree_pct", "lc_shrub_pct", "lc_grassland_pct",
                     "lc_cropland_pct", "lc_urban_pct", "lc_bare_pct", "lc_water_pct"]:
            locs[col] = np.nan

    # Merge back to original df rows
    result = df[["Latitude", "Longitude"]].merge(locs, on=["Latitude", "Longitude"], how="left")
    result = result.drop(columns=["Latitude", "Longitude"])
    return result


def main():
    # Training data
    train_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    if not train_path.exists():
        train_path = DATA_DIR / "water_quality_training_dataset.csv"
    if not train_path.exists():
        print("Training data not found!")
        return

    train_df = pd.read_csv(train_path)
    train_ext = process_dataset(train_df, "Training")
    out_train = pd.concat([train_df[["Latitude", "Longitude", "Sample Date"]], train_ext], axis=1)
    out_path = DATA_DIR / "external_features_training.csv"
    out_train.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")

    # Validation data
    val_path = DATA_DIR / "landsat_features_validation.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        val_ext = process_dataset(val_df, "Validation")
        out_val = pd.concat([val_df[["Latitude", "Longitude", "Sample Date"]], val_ext], axis=1)
        out_val_path = DATA_DIR / "external_features_validation.csv"
        out_val.to_csv(out_val_path, index=False)
        print(f"\n저장: {out_val_path}")

    print("\n" + "="*60)
    print("외부 피처 수집 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
