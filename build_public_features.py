"""
public_features.csv 생성.
1) external_features_training.csv + external_features_validation.csv 가 있으면 합쳐서 저장.
2) 없으면 submission_template의 (Lat, Lon)에 대해 SoilGrids + 고도 API 호출 후 저장.

실행: python build_public_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "public_features.csv"


def main():
    # 1) 기존 외부 피처 CSV가 있으면 합쳐서 public_features.csv 로 저장
    ext_train = DATA_DIR / "external_features_training.csv"
    ext_val = DATA_DIR / "external_features_validation.csv"
    if ext_train.exists() or ext_val.exists():
        dfs = []
        if ext_train.exists():
            dfs.append(pd.read_csv(ext_train))
        if ext_val.exists():
            dfs.append(pd.read_csv(ext_val))
        out = pd.concat(dfs, ignore_index=True)
        key = ["Latitude", "Longitude", "Sample Date"]
        if all(c in out.columns for c in key):
            out = out.drop_duplicates(subset=key, keep="first")
        out.to_csv(OUT_PATH, index=False)
        print(f"저장: {OUT_PATH} (행={len(out)}, 컬럼={list(out.columns)})")
        return

    # 2) 없으면 submission_template 기준으로 API에서 조회 (SoilGrids + 고도)
    sub_path = DATA_DIR / "submission_template.csv"
    if not sub_path.exists():
        print("external_features_*.csv 도 없고 submission_template.csv 도 없습니다.")
        print("  먼저 fetch_external_features.py 를 실행한 뒤 이 스크립트를 다시 실행하세요.")
        return

    try:
        from fetch_external_features import (
            fetch_soilgrids_point,
            fetch_elevation_batch,
        )
    except ImportError:
        print("fetch_external_features 모듈을 불러올 수 없습니다.")
        print("  submission_template 기준 키만 있는 public_features.csv 를 생성합니다.")
        sub = pd.read_csv(sub_path)
        key = ["Latitude", "Longitude", "Sample Date"]
        sub[key].drop_duplicates(subset=key, keep="first").to_csv(OUT_PATH, index=False)
        print(f"저장: {OUT_PATH} (키만 있음. fetch_external_features.py 실행 후 build_public_features.py 재실행 권장)")
        return

    sub = pd.read_csv(sub_path)
    key = ["Latitude", "Longitude", "Sample Date"]
    if not all(c in sub.columns for c in key):
        print("submission_template에 Latitude, Longitude, Sample Date 가 없습니다.")
        return

    # 고유 (Lat, Lon) 에 대해 API 호출
    locs = sub[key].drop_duplicates(subset=key, keep="first")
    lats = locs["Latitude"].tolist()
    lons = locs["Longitude"].tolist()
    n = len(locs)
    print(f"submission_template 기준 {n}개 고유 (Lat,Lon) 에서 SoilGrids + 고도 조회 중...")

    soil_rows = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        row = fetch_soilgrids_point(lat, lon)
        soil_rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  SoilGrids: {i+1}/{n}")
        import time
        time.sleep(0.25)

    soil_df = pd.DataFrame(soil_rows)
    elevations = fetch_elevation_batch(lats, lons)
    soil_df["elevation_m"] = elevations

    # (Lat, Lon) 별 피처를 원본 행 수만큼 확장
    locs = locs.reset_index(drop=True)
    feats = pd.concat([locs[["Latitude", "Longitude"]], soil_df], axis=1)
    out = sub[key].merge(
        feats.drop_duplicates(subset=["Latitude", "Longitude"], keep="first"),
        on=["Latitude", "Longitude"],
        how="left",
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH} (행={len(out)}, 컬럼={list(out.columns)})")


if __name__ == "__main__":
    main()
