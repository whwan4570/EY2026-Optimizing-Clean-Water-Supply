"""
공간 merge를 활용해 samples_with_coordinates의 GEMS 피처를
water_quality_training_dataset과 validation 데이터에 추가하는 스크립트

- Training: 가장 가까운 GEMS 지점의 중앙값(median) 사용
- Validation: 동일 로직 (지역별 전형적 수질 = 공간 맥락, 데이터 누수 없음)
- 거리 제한: 50km 이내 (configurable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

try:
    import openpyxl  # noqa: F401
except ImportError:
    pass  # only needed if reading metadata

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# GEMS 피처 매핑 (Parameter Code → 피처명)
PARAM_MAP = {
    "pH": "gems_pH",
    "Ca-Dis": "gems_Ca_Dis",
    "Mg-Dis": "gems_Mg_Dis",
    "Cl-Dis": "gems_Cl_Dis",
    "SO4-Dis": "gems_SO4_Dis",
    "Na-Dis": "gems_Na_Dis",
    "Si-Dis": "gems_Si_Dis",
    "NOxN": "gems_NOxN",
    # 타겟과 직접 대응 (지역 전형값 = 강한 prior)
    "Alk-Tot": "gems_Alk_Tot",   # Total Alkalinity
    "EC": "gems_EC",             # Electrical Conductance
    "DRP": "gems_DRP",           # Dissolved Reactive Phosphorus
    "TP": "gems_TP",             # Total Phosphorus (DRP와 강한 상관, 피처로 유용)
    # 추가 화학 지표
    "H-T": "gems_H_T",           # Hardness Total (TA 관련)
    "Sal": "gems_Sal",           # Salinity (EC 관련)
    "NH4N": "gems_NH4N",         # Ammonia (영양염)
}

# 사용할 피처 (결측 적고 TA/EC/DRP와 상관 있을 것)
SELECTED_PARAMS = [
    "pH", "Ca-Dis", "Mg-Dis", "Cl-Dis", "SO4-Dis", "Na-Dis", "Si-Dis",
    "Alk-Tot", "EC", "DRP", "TP", "H-T", "Sal", "NH4N", "NOxN",
]

# 거리 제한 (km)
MAX_DISTANCE_KM = 50.0

# 지구 반경 (km)
EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """두 위경도 점 사이의 거리(km)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def load_gems_station_stats(samples_path: Path) -> pd.DataFrame:
    """
    samples_with_coordinates를 pivot하여 GEMS 지점별 중앙값 계산
    """
    samples = pd.read_csv(samples_path, sep=";", encoding="utf-8")

    # Value를 숫자로 변환 (Value Flags '<' 등 처리)
    samples["Value"] = pd.to_numeric(samples["Value"], errors="coerce")

    # 필요한 파라미터만 필터
    samples = samples[samples["Parameter Code"].isin(SELECTED_PARAMS)].copy()

    # 지점별 좌표 (unique)
    station_coords = (
        samples.groupby("GEMS Station Number")
        .agg({"Latitude": "first", "Longitude": "first"})
        .reset_index()
    )

    # 지점별 파라미터 중앙값
    station_medians = (
        samples.groupby(["GEMS Station Number", "Parameter Code"])["Value"]
        .median()
        .reset_index()
    )

    # Wide format으로 pivot
    pivot = station_medians.pivot(
        index="GEMS Station Number",
        columns="Parameter Code",
        values="Value",
    ).reset_index()

    # 컬럼명 매핑
    rename_cols = {p: PARAM_MAP.get(p, f"gems_{p}") for p in SELECTED_PARAMS}
    pivot = pivot.rename(columns=rename_cols)

    # 좌표와 merge
    gems_stats = station_coords.merge(pivot, on="GEMS Station Number")

    # gems_DRP, gems_TP가 0 근처 많음 → log(1+x) 변환 추가 (피처로 유용)
    if "gems_DRP" in gems_stats.columns:
        gems_stats["gems_DRP_log"] = np.log1p(gems_stats["gems_DRP"])
    if "gems_TP" in gems_stats.columns:
        gems_stats["gems_TP_log"] = np.log1p(gems_stats["gems_TP"].fillna(0))
    # gems_partial_P: 입자성 인(particulate P) 프록시 = max(TP - DRP, 0) (DRP 전용 파생 피처)
    if "gems_TP" in gems_stats.columns and "gems_DRP" in gems_stats.columns:
        gems_stats["gems_partial_P"] = np.maximum(
            gems_stats["gems_TP"].fillna(0) - gems_stats["gems_DRP"].fillna(0), 0
        )

    return gems_stats


def add_gems_features(
    target_df: pd.DataFrame,
    gems_stats: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    max_dist_km: float = MAX_DISTANCE_KM,
) -> pd.DataFrame:
    """
    target_df의 각 행에 대해 가장 가까운 GEMS 지점의 피처를 merge
    """
    target_lat = target_df[lat_col].values.astype(float)
    target_lon = target_df[lon_col].values.astype(float)

    # 위경도 → 구면 좌표 (cKDTree용)
    def latlon_to_cartesian(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
        y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
        z = EARTH_RADIUS_KM * np.sin(lat_rad)
        return np.column_stack([x, y, z])

    gems_xyz = latlon_to_cartesian(
        gems_stats["Latitude"].values,
        gems_stats["Longitude"].values,
    )
    target_xyz = latlon_to_cartesian(target_lat, target_lon)

    tree = cKDTree(gems_xyz)
    _, indices = tree.query(target_xyz, k=1)

    # 정확한 거리(km) 계산 (haversine)
    matched_lat = gems_stats["Latitude"].values[indices]
    matched_lon = gems_stats["Longitude"].values[indices]
    dist_km = haversine_distance_km(target_lat, target_lon, matched_lat, matched_lon)

    # GEMS 피처 컬럼만 선택
    feature_cols = [c for c in gems_stats.columns if c.startswith("gems_")]

    result = target_df.copy()
    for i, col in enumerate(feature_cols):
        values = gems_stats[col].values[indices]
        # 거리 제한 초과 시 NaN
        values[dist_km > max_dist_km] = np.nan
        result[col] = values

    result["gems_distance_km"] = dist_km
    result["gems_within_limit"] = dist_km <= max_dist_km

    return result


def main():
    samples_path = DATA_DIR / "samples_with_coordinates.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"없음: {samples_path}")

    TOTAL_STEPS = 4
    def progress(step, msg):
        pct = int(100 * step / TOTAL_STEPS)
        print(f"\n[{step}/{TOTAL_STEPS} ({pct}%)] {msg}")
        print("-" * 50)

    progress(1, "GEMS 지점별 통계 계산 중...")
    gems_stats = load_gems_station_stats(samples_path)
    print(f"  GEMS 지점 수: {len(gems_stats)}")
    print(f"  추가 피처: {[c for c in gems_stats.columns if c.startswith('gems_')]}")

    progress(2, "Training 데이터 GEMS merge 중...")
    train_path = DATA_DIR / "water_quality_training_dataset.csv"
    train_df = pd.read_csv(train_path)
    print(f"  Training 데이터: {len(train_df)}행")

    train_enriched = add_gems_features(train_df, gems_stats, max_dist_km=MAX_DISTANCE_KM)
    within = train_enriched["gems_within_limit"].sum()
    print(f"  {MAX_DISTANCE_KM}km 이내 매칭: {within}행 ({100*within/len(train_df):.1f}%)")

    out_train = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    train_enriched.to_csv(out_train, index=False)
    print(f"  저장: {out_train}")

    progress(3, "Validation 데이터 GEMS merge 중...")
    # submission_template과 동일한 (Lat, Long, Sample Date) 순서 사용
    val_path = DATA_DIR / "submission_template.csv"
    val_df = pd.read_csv(val_path)
    print(f"\nValidation 데이터: {len(val_df)}행")

    val_enriched = add_gems_features(
        val_df[["Latitude", "Longitude", "Sample Date"]],
        gems_stats,
        max_dist_km=MAX_DISTANCE_KM,
    )
    within_val = val_enriched["gems_within_limit"].sum()
    print(f"  {MAX_DISTANCE_KM}km 이내 매칭: {within_val}행 ({100*within_val/len(val_df):.1f}%)")

    # validation용 gems 피처 저장 (거리 가중 prior용 gems_distance_km 포함)
    gems_cols = ["Latitude", "Longitude", "Sample Date"] + [
        c for c in val_enriched.columns if c.startswith("gems_")
    ]
    gems_val = val_enriched[gems_cols].copy()

    progress(4, "저장 및 완료")
    out_val = DATA_DIR / "gems_features_validation.csv"
    gems_val.to_csv(out_val, index=False)
    print(f"  저장: {out_val}")

    print("\n처음 3행 (training enriched):")
    print(train_enriched.head(3).to_string())


if __name__ == "__main__":
    main()
