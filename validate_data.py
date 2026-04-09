"""
데이터 검증 스크립트: benchmark_model.py에서 사용하는 모든 데이터/피처를 점검합니다.

- 파일 존재 여부, 행/열 수, merge 키 컬럼
- rain_sum_6m, rain_sum_3m, pr 등 강수 피처의 실제 채움 비율
- HydroRIVERS+ERA5, 공개 피처 등 merge 후 결측 현황
- 권장 조치 (merge 키 정합성, 누락 파일 생성 방법)

실행: python validate_data.py
(한글 출력이 깨지면: set PYTHONIOENCODING=utf-8 후 실행 또는 IDE 터미널에서 실행)
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# benchmark에서 기대하는 merge 키
MERGE_KEY = ["Latitude", "Longitude", "Sample Date"]

# 점검할 데이터 파일 (경로, 설명, merge 키 필요 여부)
DATA_FILES = [
    ("water_quality_training_dataset.csv", "수질 학습(기본)", True),
    ("water_quality_training_dataset_enriched.csv", "수질 학습(GEMS enriched)", True),
    ("landsat_features_training.csv", "Landsat 학습", True),
    ("terraclimate_features_training.csv", "TerraClimate 학습", True),
    ("precipitation_training.csv", "ERA5 강수(pr)", True),
    ("water_quality_with_precip_anomaly.csv", "강수 이상치 피처", True),
    ("train_with_hyriv_era5_events.csv", "HydroRIVERS+ERA5(rain_sum_*, hyriv_*)", True),
    ("val_with_hyriv_era5_events.csv", "HydroRIVERS+ERA5 검증용", True),
    ("public_features.csv", "공개 피처(토양/토지이용 등)", True),
    ("submission_template.csv", "제출 템플릿", True),
    ("landsat_features_validation.csv", "Landsat 검증", True),
    ("terraclimate_features_validation.csv", "TerraClimate 검증", True),
    ("gems_features_validation.csv", "GEMS 검증", True),
    ("external_features_training.csv", "외부 피처 학습", True),
    ("external_features_validation.csv", "외부 피처 검증", True),
]

# benchmark에서 중요한 강수·HydroRIVERS 관련 컬럼
PRECIP_AND_HYRIV_COLS = [
    "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
    "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
    "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
    "hyriv_flow_order", "hyriv_log_upcells", "hyriv_log_q", "river_dist_m",
    "pr", "precip_anom", "precip_anom_mm", "precip_anom_TP", "precip_anom_DRP",
    "block_rain_6m",
]

# TA/EC/DRP 확장 피처 목록 (benchmark 상수와 동일)
EXTENDED_HYRIV_ERA5 = [
    "river_dist_m", "hyriv_dist_bin", "hyriv_log_upcells", "hyriv_flow_order",
    "hyriv_log_q", "hyriv_q_over_up", "hyriv_order_x_up",
    "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
    "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
    "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
    "sm_mean_1m", "sm_mean_3m", "sm_lag_1m", "sm_lag_3m",
    "wetness_rain_sm_1m", "wetness_rain_sm_3m", "dilution_proxy_1m", "ionic_flush_proxy_3m",
]
EXTENDED_TA_EC_RAIN = ["block_rain_6m", "rain_sum_3m", "rain_sum_6m"]


def check_file(path: Path, label: str, need_key: bool) -> dict:
    """단일 파일 존재·형태·키 컬럼 점검."""
    out = {"path": path, "label": label, "exists": path.exists()}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, nrows=0)
        out["columns"] = list(df.columns)
        out["n_columns"] = len(df.columns)
        full = pd.read_csv(path)
        out["n_rows"] = len(full)
        out["dtypes"] = {c: str(full[c].dtype) for c in full.columns[:20]}
        if need_key:
            for k in MERGE_KEY:
                if k in full.columns:
                    out[f"key_{k}"] = {
                        "dtype": str(full[k].dtype),
                        "non_null": int(full[k].notna().sum()),
                        "sample": full[k].dropna().iloc[0] if full[k].notna().any() else None,
                    }
                else:
                    out[f"key_{k}"] = "MISSING"
    except Exception as e:
        out["error"] = str(e)
    return out


def normalize_sample_date(ser: pd.Series) -> pd.Series:
    """Sample Date를 datetime으로 통일 (merge 키 비교용)."""
    if ser is None or len(ser) == 0:
        return ser
    return pd.to_datetime(ser, dayfirst=True, errors="coerce")


def run_checks() -> None:
    print("=" * 70)
    print("EY 2026 데이터 검증 (validate_data.py)")
    print(f"DATA_DIR = {DATA_DIR}")
    print("=" * 70)

    if not DATA_DIR.exists():
        print("\n[오류] data 폴더가 없습니다. benchmark에서 사용 중인 CSV는 data/ 아래에 있어야 합니다.")
        return

    # ----- 1) 파일별 점검 -----
    print("\n--- 1) 파일 존재·행/열·merge 키 ---\n")
    file_results = []
    for fname, label, need_key in DATA_FILES:
        path = DATA_DIR / fname
        r = check_file(path, label, need_key)
        file_results.append(r)
        status = "OK" if r["exists"] else "없음"
        n_rows = r.get("n_rows", "-")
        n_cols = r.get("n_columns", "-")
        print(f"  [{status:>4}] {fname}")
        print(f"         설명: {label} | rows={n_rows}, cols={n_cols}")
        if r.get("exists") and need_key:
            for k in MERGE_KEY:
                info = r.get(f"key_{k}", "?")
                if isinstance(info, dict):
                    print(f"         키 '{k}': dtype={info['dtype']}, non_null={info['non_null']}, sample={info.get('sample')}")
                else:
                    print(f"         키 '{k}': {info}")
        if r.get("error"):
            print(f"         오류: {r['error']}")
        print()

    # ----- 2) 학습용 통합 데이터 재현 (merge 시뮬레이션) -----
    print("\n--- 2) 학습 데이터 통합 시뮬레이션 (benchmark와 동일 merge 순서) ---\n")

    wq_path = DATA_DIR / "water_quality_training_dataset_enriched.csv"
    if not wq_path.exists():
        wq_path = DATA_DIR / "water_quality_training_dataset.csv"
    if not wq_path.exists():
        print("  수질 학습 CSV가 없어 통합 시뮬레이션을 건너뜁니다.")
        return

    wq = pd.read_csv(wq_path)
    landsat = pd.read_csv(DATA_DIR / "landsat_features_training.csv") if (DATA_DIR / "landsat_features_training.csv").exists() else None
    terra = pd.read_csv(DATA_DIR / "terraclimate_features_training.csv") if (DATA_DIR / "terraclimate_features_training.csv").exists() else None

    if not all(k in wq.columns for k in MERGE_KEY):
        print("  수질 데이터에 merge 키(Latitude, Longitude, Sample Date)가 없습니다.")
        return
    wq["Sample Date"] = normalize_sample_date(wq["Sample Date"])

    merged = wq.copy()
    merge_steps = ["water_quality"]

    if landsat is not None and all(k in landsat.columns for k in MERGE_KEY):
        landsat["Sample Date"] = normalize_sample_date(landsat["Sample Date"])
        merged = merged.merge(landsat, on=MERGE_KEY, how="left", suffixes=("", "_ls"))
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merge_steps.append("landsat")
    if terra is not None and all(k in terra.columns for k in MERGE_KEY):
        terra["Sample Date"] = normalize_sample_date(terra["Sample Date"])
        merged = merged.merge(terra, on=MERGE_KEY, how="left", suffixes=("", "_tc"))
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merge_steps.append("terraclimate")

    # ERA5 pr
    pr_path = DATA_DIR / "precipitation_training.csv"
    if pr_path.exists():
        pr_df = pd.read_csv(pr_path)
        if all(k in pr_df.columns and k in merged.columns for k in MERGE_KEY):
            pr_df = pr_df.copy()
            pr_df["Sample Date"] = normalize_sample_date(pr_df["Sample Date"])
            add_pr = [c for c in pr_df.columns if c not in MERGE_KEY and c not in merged.columns]
            if add_pr or "pr" in pr_df.columns:
                cols = MERGE_KEY + (add_pr if add_pr else ["pr"])
                merged = merged.merge(pr_df[[c for c in cols if c in pr_df.columns]].drop_duplicates(subset=MERGE_KEY, keep="first"), on=MERGE_KEY, how="left")
                merge_steps.append("precipitation(pr)")
    # Precip anomaly
    pa_path = DATA_DIR / "water_quality_with_precip_anomaly.csv"
    if pa_path.exists():
        pa_df = pd.read_csv(pa_path)
        if all(k in pa_df.columns and k in merged.columns for k in MERGE_KEY):
            pa_df = pa_df.copy()
            pa_df["Sample Date"] = normalize_sample_date(pa_df["Sample Date"])
            extra = [c for c in pa_df.columns if c not in merged.columns and c not in MERGE_KEY]
            if extra:
                merged = merged.merge(pa_df[MERGE_KEY + extra].drop_duplicates(subset=MERGE_KEY, keep="first"), on=MERGE_KEY, how="left")
                merge_steps.append("precip_anomaly")
    # HydroRIVERS + ERA5 events (rain_sum_6m 등)
    tw_path = DATA_DIR / "train_with_hyriv_era5_events.csv"
    if tw_path.exists():
        tw = pd.read_csv(tw_path)
        if all(k in tw.columns and k in merged.columns for k in MERGE_KEY):
            tw = tw.copy()
            tw["Sample Date"] = normalize_sample_date(tw["Sample Date"])
            to_add = [c for c in tw.columns if c not in merged.columns and c not in MERGE_KEY
                      and c not in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]]
            if to_add:
                tw_sub = tw[MERGE_KEY + to_add].drop_duplicates(subset=MERGE_KEY, keep="first")
                merged = merged.merge(tw_sub, on=MERGE_KEY, how="left")
                merge_steps.append("train_with_hyriv_era5_events")
    # Public
    pub_path = DATA_DIR / "public_features.csv"
    if pub_path.exists():
        pub = pd.read_csv(pub_path)
        if "Sample Date" in pub.columns:
            pub["Sample Date"] = normalize_sample_date(pub["Sample Date"])
        if all(k in pub.columns and k in merged.columns for k in MERGE_KEY):
            to_add_pub = [c for c in pub.columns if c not in merged.columns and c not in MERGE_KEY
                          and c not in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]]
            if to_add_pub:
                pub_sub = pub[MERGE_KEY + to_add_pub].drop_duplicates(subset=MERGE_KEY, keep="first")
                merged = merged.merge(pub_sub, on=MERGE_KEY, how="left")
                merge_steps.append("public_features")

    n_final = len(merged)
    print(f"  Merge 단계: {' → '.join(merge_steps)}")
    print(f"  최종 행 수: {n_final}, 컬럼 수: {len(merged.columns)}")

    # ----- 3) 강수·HydroRIVERS 피처 채움 현황 -----
    print("\n--- 3) 강수·HydroRIVERS 피처 채움 현황 (실제 사용 가능 여부) ---\n")

    for col in PRECIP_AND_HYRIV_COLS:
        if col not in merged.columns:
            print(f"  [없음] {col}  (통합 데이터에 컬럼 없음)")
            continue
        non_null = merged[col].notna().sum()
        pct = 100.0 * non_null / n_final if n_final else 0
        status = "OK" if pct >= 80 else ("주의" if pct >= 10 else "미사용(대부분 NaN)")
        print(f"  [{status:>12}] {col}: non_null={non_null}/{n_final} ({pct:.1f}%)")

    # ----- 4) rain_sum_6m / rain_sum_3m 상세 (근본 개선용) -----
    print("\n--- 4) rain_sum_6m / rain_sum_3m 상세 (benchmark TA/EC/DRP 확장 피처) ---\n")

    for col in ["rain_sum_6m", "rain_sum_3m"]:
        if col not in merged.columns:
            print(f"  {col}: 컬럼 없음 → train_with_hyriv_era5_events.csv에 해당 컬럼이 있어야 하며, merge 키(Lat,Lon,Sample Date)가 학습 데이터와 일치해야 합니다.")
            continue
        s = merged[col]
        nn = s.notna().sum()
        pct = 100.0 * nn / n_final if n_final else 0
        if nn == 0:
            print(f"  {col}: 전부 NaN (0/{n_final})")
            print("    → 원인: train_with_hyriv_era5_events.csv가 없거나, merge 키 불일치(날짜 형식/위도·경도 소수 자리).")
            print("    → 조치: era5_event_features.py 등으로 (Latitude, Longitude, Sample Date) 기준으로 rain_sum_* 를 계산한 CSV를 생성 후 병합.")
        else:
            print(f"  {col}: non_null={nn}/{n_final} ({pct:.1f}%), min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")

    # ----- 5) EXTENDED 피처 존재·채움 요약 -----
    print("\n--- 5) 확장 피처(EXTENDED_TA/EC/DRP·HydroRIVERS) 존재·채움 요약 ---\n")

    extended_all = list(dict.fromkeys(EXTENDED_HYRIV_ERA5 + EXTENDED_TA_EC_RAIN))
    missing = []
    low_fill = []
    for c in extended_all:
        if c not in merged.columns:
            missing.append(c)
        else:
            pct = 100.0 * merged[c].notna().sum() / n_final if n_final else 0
            if pct < 50:
                low_fill.append((c, pct))
    if missing:
        print("  통합 데이터에 없는 컬럼:")
        for m in missing:
            print(f"    - {m}")
    if low_fill:
        print("  채움 50% 미만 컬럼:")
        for c, p in low_fill:
            print(f"    - {c}: {p:.1f}%")

    if not missing and not low_fill:
        print("  확장 피처(강수·HydroRIVERS) 모두 존재하며 채움 양호.")

    # ----- 6) merge 키 정합성 (날짜 형식) -----
    print("\n--- 6) Merge 키 정합성 (Sample Date 형식) ---\n")

    for fname, label, _ in DATA_FILES[:9]:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, nrows=5)
        if "Sample Date" not in df.columns:
            continue
        sample = df["Sample Date"].iloc[0]
        print(f"  {fname}: Sample Date 예시 = {repr(sample)}")

    # ----- 7) 권장 조치 요약 -----
    print("\n--- 7) 권장 조치 요약 ---\n")

    rain_6m_ok = "rain_sum_6m" in merged.columns and merged["rain_sum_6m"].notna().sum() > 0
    if not rain_6m_ok:
        print("  [*] rain_sum_6m이 비어 있음:")
        print("    1) train_with_hyriv_era5_events.csv 존재 여부 확인.")
        print("    2) 해당 파일의 Latitude, Longitude, Sample Date 형식이 water_quality·landsat·terra와 동일한지 확인.")
        print("    3) era5_event_features.py(또는 HydroRIVERS+ERA5 파이프라인)로 학습 좌표·날짜에 대해 rain_sum_1m/3m/6m/12m 등을 계산해 CSV 생성.")
    else:
        print("  [*] rain_sum_6m 등 강수 피처가 채워져 있음. benchmark에서 정상 사용 가능.")

    if not (DATA_DIR / "train_with_hyriv_era5_events.csv").exists():
        print("  [*] train_with_hyriv_era5_events.csv 없음 -> HydroRIVERS+ERA5 이벤트 스크립트 실행 후 생성 필요.")
    else:
        hyriv_cols = pd.read_csv(DATA_DIR / "train_with_hyriv_era5_events.csv", nrows=0).columns.tolist()
        if "Sample Date" not in hyriv_cols:
            print("  [*] train_with_hyriv_era5_events.csv에 'Sample Date' 컬럼 없음 -> merge 불가, rain_sum_* 가 학습에 붙지 않음. 파이프라인에서 Sample Date 포함해 재생성 필요.")

    print("\n" + "=" * 70)
    print("검증 완료.")
    print("=" * 70)


if __name__ == "__main__":
    run_checks()
