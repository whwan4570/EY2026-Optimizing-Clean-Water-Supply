"""
이스턴케이프 지역에서 submission_template vs GEMS 데이터 비교.
실행: python compare_ec_gems_submission.py
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EASTERN_CAPE_LAT = (-34.2, -30.5)
EASTERN_CAPE_LON = (22.5, 30.0)
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]


def in_eastern_cape(df: pd.DataFrame) -> pd.Series:
    lat = df["Latitude"].astype(float)
    lon = df["Longitude"].astype(float)
    return (
        (lat >= EASTERN_CAPE_LAT[0]) & (lat <= EASTERN_CAPE_LAT[1])
        & (lon >= EASTERN_CAPE_LON[0]) & (lon <= EASTERN_CAPE_LON[1])
    )


def main():
    sub = pd.read_csv(DATA_DIR / "submission_template.csv")
    print("=" * 60)
    print("1) submission_template")
    print("=" * 60)
    print(f"   총 행 수: {len(sub)}")
    ec_sub = in_eastern_cape(sub)
    print(f"   이스턴케이프 내 행 수: {ec_sub.sum()} ({100*ec_sub.mean():.1f}%)")
    if ec_sub.sum() < len(sub):
        out_ec = sub.loc[~ec_sub, KEY_COLS].drop_duplicates()
        print(f"   EC 밖 좌표 수: {len(out_ec)} (고유)")
    lat_range = sub["Latitude"].astype(float).agg(["min", "max"])
    lon_range = sub["Longitude"].astype(float).agg(["min", "max"])
    print(f"   위도 범위: [{lat_range['min']:.4f}, {lat_range['max']:.4f}]")
    print(f"   경도 범위: [{lon_range['min']:.4f}, {lon_range['max']:.4f}]")
    sub_keys = sub[KEY_COLS].astype(str).agg("_".join, axis=1)
    print(f"   고유 (Lat,Lon,Date): {sub_keys.nunique()}")

    if not (DATA_DIR / "gems_features_validation.csv").exists():
        print("\n   gems_features_validation.csv 없음. create_enriched_dataset.py 실행 후 다시 비교.")
        return

    gems = pd.read_csv(DATA_DIR / "gems_features_validation.csv")
    print("\n" + "=" * 60)
    print("2) gems_features_validation (내 GEMS 데이터)")
    print("=" * 60)
    print(f"   총 행 수: {len(gems)}")
    ec_gems = in_eastern_cape(gems)
    print(f"   이스턴케이프 내 행 수: {ec_gems.sum()} ({100*ec_gems.mean():.1f}%)")

    # 키 일치 여부
    if list(sub[KEY_COLS].columns) == list(gems[KEY_COLS].columns):
        sub_join = sub[KEY_COLS].astype(str).agg("_".join, axis=1)
        gems_join = gems[KEY_COLS].astype(str).agg("_".join, axis=1)
        in_both = set(sub_join) & set(gems_join)
        only_sub = set(sub_join) - set(gems_join)
        only_gems = set(gems_join) - set(sub_join)
        print(f"   submission_template과 동일 키 행: {len(in_both)} / {len(sub)}")
        if only_sub:
            print(f"   template에만 있는 키: {len(only_sub)}개")
        if only_gems:
            print(f"   GEMS에만 있는 키: {len(only_gems)}개")

    # GEMS 핵심 컬럼 결측/통계 (이스턴케이프 행만)
    gem_cols = [c for c in gems.columns if c.startswith("gems_") and c not in KEY_COLS]
    if gem_cols:
        gems_ec = gems.loc[ec_gems] if ec_gems.any() else gems
        print("\n   [이스턴케이프 행 기준] GEMS 컬럼 요약:")
        for col in ["gems_distance_km", "gems_within_limit", "gems_Alk_Tot", "gems_EC", "gems_DRP", "gems_pH"]:
            if col not in gems_ec.columns:
                continue
            s = gems_ec[col]
            if s.dtype == bool or s.dtype == object:
                print(f"     {col}: {s.value_counts().to_dict()}")
            else:
                valid = s.dropna()
                print(f"     {col}: non-null={len(valid)}/{len(gems_ec)} ({100*valid.count()/len(gems_ec):.1f}%), mean={valid.mean():.4f}, min={valid.min():.4f}, max={valid.max():.4f}")

    # template 행 순서와 GEMS 행 순서 일치 여부 (Lat, Lon만)
    if len(sub) == len(gems) and "Latitude" in sub.columns and "Latitude" in gems.columns:
        match = ((sub["Latitude"].astype(float).values == gems["Latitude"].astype(float).values)
                 & (sub["Longitude"].astype(float).values == gems["Longitude"].astype(float).values))
        date_ok = (sub["Sample Date"].astype(str).values == gems["Sample Date"].astype(str).values).sum()
        print(f"\n   submission_template과 GEMS 행 순서: Lat/Lon 일치 {match.sum()}/{len(sub)}, Sample Date 일치 {date_ok}/{len(sub)}")

    # train 쪽 EC와의 겹침 (선택)
    train_path = DATA_DIR / "train_with_hyriv_era5_events.csv"
    if train_path.exists():
        train = pd.read_csv(train_path, nrows=0)
        if "Latitude" in train.columns:
            train = pd.read_csv(train_path, usecols=KEY_COLS)
            train_ec = in_eastern_cape(train)
            train_ec_loc = train.loc[train_ec, ["Latitude", "Longitude"]].drop_duplicates()
            sub_loc = sub[["Latitude", "Longitude"]].drop_duplicates()
            overlap = train_ec_loc.merge(sub_loc, on=["Latitude", "Longitude"], how="inner")
            print("\n" + "=" * 60)
            print("3) Train(EC) vs submission_template 좌표 겹침")
            print("=" * 60)
            print(f"   Train 이스턴케이프 고유 (Lat,Lon): {len(train_ec_loc)}")
            print(f"   submission_template 고유 (Lat,Lon): {len(sub_loc)}")
            print(f"   겹치는 (Lat,Lon): {len(overlap)} (제출 위치 중 train에 있는 EC 좌표 수)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
