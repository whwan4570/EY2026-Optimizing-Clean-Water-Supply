"""
외부 데이터 수집: SoilGrids(토양), NPKGRIDS(비료) 등.
데이터 경로: 스크립트와 같은 data/ 폴더. train/val의 (Latitude, Longitude) 기준으로 조회/병합.

실행 예:
  python fetch_external_data.py                    # SoilGrids만 (train+val 고유 좌표)
  python fetch_external_data.py --soilgrids-only   # 동일
  python fetch_external_data.py --npkgrids-download  # NPKGRIDS Figshare에서 1개 샘플 다운로드
  python fetch_external_data.py --max-points 100   # 테스트: 최대 100개 좌표만
"""
from pathlib import Path
import argparse
import time
import json

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# SoilGrids REST API (5 calls/min 제한)
ISRIC_BASE = "https://rest.isric.org/soilgrids/v2.0"
ISRIC_RATE_DELAY = 12  # 초 (5회/분 → 12초 간격 권장)

# Figshare NPKGRIDS (DOI 10.6084/m9.figshare.24616050 → article 24616050)
FIGSHARE_ARTICLE_ID = 24616050
FIGSHARE_API = "https://api.figshare.com/v2/articles"


def get_unique_coords(data_dir: Path, max_points: int | None = None) -> pd.DataFrame:
    """train/val/submission CSV에서 고유 (Latitude, Longitude) 추출."""
    out = []
    for name, f in [
        ("train", "train_with_hyriv_era5_events.csv"),
        ("val", "val_with_hyriv_era5_events.csv"),
        ("submission", "submission_template.csv"),
    ]:
        path = data_dir / f
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["Latitude", "Longitude"])
        df = df.drop_duplicates()
        df["_source"] = name
        out.append(df)
    if not out:
        raise FileNotFoundError(f"No train/val/submission CSV in {data_dir}")
    coords = pd.concat(out, ignore_index=True).drop_duplicates(subset=["Latitude", "Longitude"])
    coords = coords.sort_values(["Latitude", "Longitude"]).reset_index(drop=True)
    if max_points is not None and len(coords) > max_points:
        coords = coords.head(max_points)
    return coords


def fetch_soilgrids_point(lat: float, lon: float, properties: list[str] | None = None, verbose: bool = False) -> dict | None:
    """SoilGrids v2 properties/query: 단일 (lat, lon). 5회/분 제한."""
    if requests is None:
        print("pip install requests 필요")
        return None
    properties = properties or ["phh2o", "clay", "soc", "sand", "silt"]
    url = f"{ISRIC_BASE}/properties/query"
    params = {"lat": lat, "lon": lon, "depth": "0-5cm", "value": "mean", "property": ",".join(properties)}
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            if verbose:
                print(f"    API 오류: HTTP {resp.status_code} ({lat:.4f}, {lon:.4f}) - {resp.text[:100]}")
            return None
        return resp.json()
    except (requests.exceptions.Timeout, requests.exceptions.RequestException, Exception) as e:
        if verbose:
            print(f"    요청 실패: {type(e).__name__} ({lat:.4f}, {lon:.4f})")
        return None


def parse_soilgrids_response(data: dict) -> dict:
    """properties/query 응답에서 숫자 값만 추출. clay/sand/silt: g/kg→%, soc: dg/kg→g/kg, phh2o: *10→pH."""
    out = {}
    if not data:
        return out
    props_block = data.get("properties", {})
    if not props_block:
        return out
    # API v2: properties.layers[] with name, depths[].values.mean
    layers_list = props_block.get("layers", [])
    if layers_list:
        for ly in layers_list:
            prop_name = ly.get("name", "")
            for dep_block in ly.get("depths", []):
                dep = dep_block.get("depth", "0-5cm")
                vals = dep_block.get("values", {})
                raw = vals.get("mean") if isinstance(vals, dict) else None
                if raw is None:
                    continue
                key = f"soilgrids_{prop_name}_{dep.replace('-', '_')}"
                if prop_name in ("clay", "sand", "silt"):
                    out[key] = raw / 10.0  # g/kg → %
                elif prop_name == "soc":
                    out[key] = raw / 10.0  # dg/kg → g/kg
                elif prop_name == "phh2o":
                    out[key] = raw / 10.0  # pH*10 → pH
                else:
                    out[key] = raw
        return out
    # fallback: properties.<prop>.layers[]
    for prop_name, block in props_block.items():
        if not isinstance(block, dict):
            continue
        for ly in block.get("layers", []):
            dep = ly.get("depth", "0-5cm")
            vals = ly.get("properties") or ly.get("values", ly)
            if not isinstance(vals, dict) or "mean" not in vals:
                continue
            raw = vals["mean"]
            key = f"soilgrids_{prop_name}_{dep.replace('-', '_')}"
            if prop_name in ("clay", "sand", "silt"):
                out[key] = raw / 10.0
            elif prop_name == "soc":
                out[key] = raw / 10.0
            elif prop_name == "phh2o":
                out[key] = raw / 10.0
            else:
                out[key] = raw
    return out


def run_soilgrids(data_dir: Path, out_path: Path, max_points: int | None, properties: list[str], verbose: bool = False) -> pd.DataFrame:
    """고유 좌표에 대해 SoilGrids 조회 후 CSV 저장."""
    coords = get_unique_coords(data_dir, max_points)
    print(f"SoilGrids: {len(coords)}개 좌표 조회 (5회/분 제한으로 {len(coords) * ISRIC_RATE_DELAY / 60:.1f}분 예상)")
    print("  참고: ISRIC SoilGrids /properties/query API가 500 오류를 반환할 수 있음 (서버 측 이슈)")
    rows = []
    fail_count = 0
    for i, row in coords.iterrows():
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{len(coords)} ({lat:.4f}, {lon:.4f})")
        try:
            data = fetch_soilgrids_point(lat, lon, properties, verbose=verbose)
        except Exception:
            data = None
        time.sleep(ISRIC_RATE_DELAY)
        if data is None:
            fail_count += 1
            if verbose and fail_count <= 3:
                print(f"    실패 #{fail_count}: ({lat:.4f}, {lon:.4f})")
            rows.append({"Latitude": lat, "Longitude": lon})
            continue
        parsed = parse_soilgrids_response(data)
        parsed["Latitude"] = lat
        parsed["Longitude"] = lon
        rows.append(parsed)
    result = pd.DataFrame(rows)
    result = result.reindex(columns=["Latitude", "Longitude"] + [c for c in result.columns if c not in ("Latitude", "Longitude")], fill_value=float("nan"))
    result.to_csv(out_path, index=False)
    soil_cols = [c for c in result.columns if str(c).startswith("soilgrids_")]
    n_with_soil = result[soil_cols].notna().any(axis=1).sum() if soil_cols else 0
    if fail_count > 0 or n_with_soil == 0:
        print(f"저장: {out_path} ({len(result)}행) - 토양 데이터 성공: {n_with_soil}행, 실패: {fail_count}행")
        if n_with_soil == 0:
            print("  *** API가 500 오류를 반환함. ISRIC SoilGrids REST API가 일시 중단되었을 수 있음.")
            print("  *** 대안: train_with_hyriv_era5_events.csv의 soil_clay_pct, soil_organic_carbon, soil_ph 사용")
    else:
        print(f"저장: {out_path} ({len(result)}행)")
    return result


def npkgrids_figshare_list() -> list[dict]:
    """Figshare API로 NPKGRIDS(article 24616050) 파일 목록 조회."""
    if requests is None:
        return []
    r = requests.get(f"{FIGSHARE_API}/{FIGSHARE_ARTICLE_ID}", timeout=15)
    if r.status_code != 200:
        return []
    try:
        art = r.json()
        return art.get("files", [])
    except Exception:
        return []


def npkgrids_download_one(data_dir: Path, choose_index: int = 0) -> Path | None:
    """Figshare에서 NPKGRIDS 파일 1개 다운로드 (용량 큰 데이터셋이므로 샘플만)."""
    if requests is None:
        print("pip install requests 필요")
        return None
    files = npkgrids_figshare_list()
    if not files:
        print("NPKGRIDS 파일 목록 조회 실패")
        return None
    # 작은 파일 우선 (일부 메타데이터/README 등)
    files = sorted(files, key=lambda x: x.get("size", 0))
    idx = min(choose_index, len(files) - 1)
    info = files[idx]
    url = info.get("download_url")
    name = info.get("name", "npkgrids_sample")
    if not url:
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / name
    print(f"다운로드: {name} ({info.get('size', 0) / 1024 / 1024:.1f} MB)")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"저장: {out}")
        return out
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="외부 데이터 수집 (SoilGrids, NPKGRIDS)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="data 디렉터리")
    parser.add_argument("--soilgrids-only", action="store_true", help="SoilGrids만 실행")
    parser.add_argument("--npkgrids-download", action="store_true", help="Figshare에서 NPKGRIDS 샘플 1개 다운로드")
    parser.add_argument("--max-points", type=int, default=None, help="SoilGrids 조회할 최대 좌표 수 (테스트용)")
    parser.add_argument("--soilgrids-properties", type=str, default="phh2o,clay,soc,sand,silt", help="SoilGrids 속성 (쉼표)")
    parser.add_argument("--verbose", "-v", action="store_true", help="API 오류 상세 출력")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.npkgrids_download:
        npkgrids_download_one(data_dir)
        if args.soilgrids_only:
            return

    # SoilGrids: train/val 고유 좌표에 대해 API 호출 후 CSV 저장
    out_soil = data_dir / "external_soilgrids.csv"
    props = [p.strip() for p in args.soilgrids_properties.split(",") if p.strip()]
    run_soilgrids(data_dir, out_soil, args.max_points, props, verbose=args.verbose)

    print("\n병합 방법: train/val에 external_soilgrids.csv를 Latitude, Longitude 기준으로 merge 후 사용.")
    print("  예: pd.merge(train, soil, on=['Latitude','Longitude'], how='left')")


if __name__ == "__main__":
    main()
