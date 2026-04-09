"""
HydroRIVERS (Africa) GDB 로딩 및 샘플 지점–하천 매칭.
가장 가까운 river segment 매칭 후 hyriv_* 피처 추출.
"""
from pathlib import Path
from typing import Optional, Tuple, List, Any
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

# GDB 내부 경로 (폴더 안에 실제 .gdb 디렉터리가 있는 구조)
def _gdb_path(base_dir: Path) -> Path:
    outer = base_dir / "HydroRIVERS_v10_af.gdb"
    inner = outer / "HydroRIVERS_v10_af.gdb"
    if inner.exists():
        return inner
    return outer


def _rivers_data_path(base_dir: Path) -> Tuple[Path, str]:
    """
    강 레이어 로드에 쓸 경로 반환. (path, "gpkg"|"gdb")
    GDB가 Windows에서 안 열리면 같은 폴더의 HydroRIVERS_v10_af.gpkg 있으면 우선 사용.
    """
    gdb = _gdb_path(base_dir)
    gpkg = base_dir / "HydroRIVERS_v10_af.gpkg"
    if gpkg.exists():
        return gpkg, "gpkg"
    return gdb, "gdb"


def _list_layers_for_path(path: Path) -> List[str]:
    """단일 경로에 대해 레이어 목록 시도 (fiona → pyogrio → geopandas 순)."""
    s = str(path)
    try:
        import fiona
        layers = fiona.listlayers(s)
        if layers:
            return layers
    except Exception:
        pass
    try:
        import pyogrio
        if hasattr(pyogrio, "list_layers"):
            out = pyogrio.list_layers(s)
            # ndarray shape (2, n): row0=names, row1=geometry_types
            if out is not None and getattr(out, "size", 0) > 0:
                names = out[0] if hasattr(out, "__getitem__") else out
                if hasattr(names, "tolist"):
                    names = names.tolist()
                if names:
                    return list(names)
    except Exception:
        pass
    try:
        import geopandas as _gpd
        if hasattr(_gpd, "list_layers"):
            df = _gpd.list_layers(s)
            if df is not None and not df.empty and "name" in df.columns:
                return df["name"].astype(str).tolist()
    except Exception:
        pass
    try:
        import geopandas as _gpd
        d = _gpd.read_file(s, layer=None)
        if isinstance(d, dict):
            return list(d.keys())
    except Exception:
        pass
    return []


def list_gdb_layers(gdb_path: Path) -> List[str]:
    """GDB 레이어 목록. 내부 경로 → (실패 시) 상위 .gdb 폴더 순으로 시도."""
    layers = _list_layers_for_path(gdb_path)
    if layers:
        return layers
    # 상위가 *인.gdb 폴더이고, 현재가 그 안의 동일명 폴더인 경우 상위 시도 (일부 환경)
    parent = gdb_path.parent
    if parent.name.endswith(".gdb") and parent.name == gdb_path.name:
        layers = _list_layers_for_path(parent)
    return layers if layers else []


def _find_river_layer(gdb_path: Path) -> Optional[str]:
    """River 라인 레이어 이름 자동 탐색."""
    layers = list_gdb_layers(gdb_path)
    for name in layers:
        low = name.lower()
        if "river" in low or "hydro" in low or "stream" in low or "riv" in low:
            return name
    return layers[0] if layers else None


# HydroRIVERS v10 Africa 실제 스키마 (TechDoc v10): UPLAND_SKM, ORD_STRA, DIS_AVG 등
ATTR_ALIASES = {
    "up_cells": ["UPLAND_SKM", "UP_CELLS", "UP_CELL", "Up_cells", "up_cells"],
    "ord_flow": ["ORD_STRA", "ORD_FLOW", "ORD_CLAS", "STRAHLER", "Order"],
    "dis_av_cms": ["DIS_AVG", "DIS_AV_CMS", "DIS_AV", "AvgDis"],
    "slope": ["SLOPE", "slope", "SLOPE_AVG"],
    "length": ["LENGTH_KM", "LENGTH", "LEN_KM", "Length_km"],
    "id": ["HYRIV_ID", "ID", "OBJECTID", "FID", "LINK_NO"],
}


def _map_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    """실제 컬럼명 후보 중 존재하는 것 반환."""
    cols_lower = {c.upper(): c for c in columns}
    for a in aliases:
        if a.upper() in cols_lower:
            return cols_lower[a.upper()]
    return None


def load_hydrorivers_rivers(gdb_path: Path, layer_name: Optional[str] = None) -> Optional["gpd.GeoDataFrame"]:
    """HydroRIVERS 라인 레이어 로드. WGS84로 통일. 레이어명 없으면 layer=0으로 시도."""
    if not HAS_GEOPANDAS:
        return None
    layer = layer_name or _find_river_layer(gdb_path)
    gdf = None
    try:
        if layer is not None:
            gdf = gpd.read_file(str(gdb_path), layer=layer)
    except Exception:
        pass
    if gdf is None:
        try:
            gdf = gpd.read_file(str(gdb_path))
        except Exception:
            pass
    if gdf is None:
        try:
            gdf = gpd.read_file(str(gdb_path), layer=0)
        except Exception:
            return None
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def extract_river_attributes(row: pd.Series, cols: dict) -> dict:
    """한 행(매칭된 river)에서 hyriv_* 속성 추출."""
    eps = 1e-6
    out = {}
    up = row.get(cols.get("up_cells")) if cols.get("up_cells") else None
    ord_f = row.get(cols.get("ord_flow")) if cols.get("ord_flow") else None
    dis = row.get(cols.get("dis_av_cms")) if cols.get("dis_av_cms") else None
    if up is not None and pd.notna(up):
        out["hyriv_log_upcells"] = np.log1p(float(up))
        if dis is not None and pd.notna(dis) and float(up) + eps != 0:
            out["hyriv_q_over_up"] = float(dis) / (float(up) + eps)
        if ord_f is not None and pd.notna(ord_f):
            out["hyriv_order_x_up"] = float(ord_f) * np.log1p(float(up))
    if ord_f is not None and pd.notna(ord_f):
        out["hyriv_flow_order"] = float(ord_f)
    if dis is not None and pd.notna(dis):
        out["hyriv_log_q"] = np.log1p(float(dis))
    return out


def river_dist_bin(dist_m: float, edges: Optional[List[float]] = None) -> int:
    """river_dist_m을 0-250, 250-1000, 1000-5000, 5000+ bin. edges는 fold train에서 계산해 전달."""
    if edges is not None:
        for i, e in enumerate(edges):
            if dist_m < e:
                return i
        return len(edges)
    if dist_m < 250:
        return 0
    if dist_m < 1000:
        return 1
    if dist_m < 5000:
        return 2
    return 3


def _deg_to_meter_approx(deg: float, lat: float = 0) -> float:
    """WGS84 경도 1도 ≈ 111320*cos(lat) m, 위도 1도 ≈ 110540 m."""
    import math
    return float(deg) * 111_320 * max(0.01, math.cos(math.radians(lat)))


def build_hyriv_features(
    points_gdf: "gpd.GeoDataFrame",
    rivers_gdf: "gpd.GeoDataFrame",
    river_col_map: dict,
    dist_bin_edges: Optional[List[float]] = None,
) -> pd.DataFrame:
    """points_gdf에 대해 nearest join 후 hyriv_* + river_dist_m + hyriv_dist_bin 반환."""
    from geopandas.tools import sjoin_nearest
    pts = points_gdf.copy()
    pts["_idx"] = np.arange(len(pts))
    joined = sjoin_nearest(pts[["geometry", "_idx"]], rivers_gdf, distance_col="river_dist_m", how="left")
    if "river_dist_m" not in joined.columns:
        joined["river_dist_m"] = 0.0
    joined["river_dist_m"] = joined["river_dist_m"].fillna(0).astype(float)
    if joined["river_dist_m"].max() < 1:
        lat = pts.geometry.y.mean() if hasattr(pts.geometry, "y") else 0
        joined["river_dist_m"] = joined["river_dist_m"].apply(lambda d: _deg_to_meter_approx(d, lat))
    if "index_right" in joined.columns:
        river_attrs = rivers_gdf.drop(columns=["geometry"], errors="ignore")
        joined = joined.merge(river_attrs, left_on="index_right", right_index=True, how="left", suffixes=("", "_r"))
    out_list = []
    for _, row in joined.iterrows():
        d = {"_idx": row["_idx"], "river_dist_m": row["river_dist_m"]}
        d["hyriv_dist_bin"] = river_dist_bin(row["river_dist_m"], dist_bin_edges)
        for k, v in extract_river_attributes(row, river_col_map).items():
            d[k] = v
        out_list.append(d)
    out_df = pd.DataFrame(out_list)
    for c in ["hyriv_log_upcells", "hyriv_flow_order", "hyriv_log_q", "hyriv_q_over_up", "hyriv_order_x_up"]:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df.sort_values("_idx").drop(columns=["_idx"]).reset_index(drop=True)
    return out_df


def compute_dist_bin_edges_from_train(dist_m_series: pd.Series, n_bins: int = 4) -> List[float]:
    """train에서만 호출: river_dist_m 분포 기반 경계 (quantile)."""
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = dist_m_series.quantile(q).tolist()
    return sorted(set(edges))


def get_hyriv_features_for_dataframe(
    df: pd.DataFrame,
    gdb_path: Path,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    river_layer: Optional[str] = None,
    dist_bin_edges: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    df에 Latitude, Longitude가 있을 때 가장 가까운 river 매칭 후 hyriv_* 피처 DataFrame 반환.
    반환: (feature_df, stats_dict)
    stats_dict: river_col_map, match_rate, dist_bin_edges_used 등 (로깅용).
    """
    if not HAS_GEOPANDAS or gdb_path is None or not gdb_path.exists():
        empty = pd.DataFrame()
        for c in ["river_dist_m", "hyriv_dist_bin", "hyriv_log_upcells", "hyriv_flow_order",
                  "hyriv_log_q", "hyriv_q_over_up", "hyriv_order_x_up"]:
            empty[c] = np.nan
        return empty.iloc[:len(df)].assign(**{c: np.nan for c in empty.columns}), {}
    rivers = load_hydrorivers_rivers(gdb_path, layer_name=river_layer)
    if rivers is None or len(rivers) == 0:
        empty = pd.DataFrame(index=range(len(df)))
        for c in ["river_dist_m", "hyriv_dist_bin", "hyriv_log_upcells", "hyriv_flow_order",
                  "hyriv_log_q", "hyriv_q_over_up", "hyriv_order_x_up"]:
            empty[c] = np.nan
        return empty, {}
    cols = list(rivers.columns)
    river_col_map = {}
    for key, aliases in ATTR_ALIASES.items():
        river_col_map[key] = _map_column(cols, aliases)
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326",
    )
    if dist_bin_edges is None and "river_dist_m" in (rivers.columns if hasattr(rivers, "columns") else []):
        dist_bin_edges = None
    feat_df = build_hyriv_features(points, rivers, river_col_map, dist_bin_edges=dist_bin_edges)
    if dist_bin_edges is None and "river_dist_m" in feat_df.columns and feat_df["river_dist_m"].notna().any():
        dist_bin_edges = compute_dist_bin_edges_from_train(feat_df["river_dist_m"].dropna())
        feat_df["hyriv_dist_bin"] = feat_df["river_dist_m"].apply(lambda x: river_dist_bin(x, dist_bin_edges))
    stats = {
        "river_col_map": river_col_map,
        "match_rate": feat_df["river_dist_m"].notna().mean() if "river_dist_m" in feat_df.columns else 0,
        "dist_bin_edges": dist_bin_edges,
    }
    return feat_df, stats


def verify_hydrorivers_gdb(gdb_path: Path) -> dict:
    """
    GDB 경로가 HydroRIVERS_v10_af.gdb 실제 파일과 맞는지 열어보고,
    레이어명·속성 목록·매핑 결과를 반환. 로그 출력용.
    """
    out = {"path": str(gdb_path), "exists": gdb_path.exists(), "layers": [], "river_layer": None, "columns": [], "mapped": {}}
    if not gdb_path.exists():
        return out
    out["layers"] = list_gdb_layers(gdb_path)
    out["river_layer"] = _find_river_layer(gdb_path)
    rivers = load_hydrorivers_rivers(gdb_path, layer_name=out["river_layer"])
    if rivers is not None and len(rivers) > 0:
        out["columns"] = list(rivers.columns)
        for key, aliases in ATTR_ALIASES.items():
            out["mapped"][key] = _map_column(out["columns"], aliases)
    return out


def _diagnose_gdb_open(gdb_path: Path) -> None:
    """GDB 열기 실패 시 원인 확인용: fiona/pyogrio/ogr 예외 메시지 출력."""
    s = str(gdb_path)
    print("  [진단] GDB 열기 시도 및 오류 메시지:")
    for name, func in [
        ("fiona.listlayers", lambda: __import__("fiona").listlayers(s)),
        ("pyogrio.list_layers", lambda: __import__("pyogrio").list_layers(s)),
        ("geopandas.read_file(layer=0)", lambda: __import__("geopandas").read_file(s, layer=0)),
    ]:
        try:
            out = func()
            print(f"    {name}: OK -> {type(out).__name__}", end="")
            if hasattr(out, "__len__") and not isinstance(out, (str, dict)):
                print(f" len={len(out)}", end="")
            print()
        except Exception as e:
            print(f"    {name}: {type(e).__name__}: {e}")
    try:
        from osgeo import ogr
        ds = ogr.Open(s)
        if ds is None:
            print("    ogr.Open: ds is None (GDAL could not open)")
        else:
            n = ds.GetLayerCount()
            print(f"    ogr.Open: OK, LayerCount={n}")
            for i in range(n):
                ly = ds.GetLayer(i)
                print(f"      layer[{i}] = {ly.GetName()}")
    except ImportError:
        print("    ogr.Open: osgeo (GDAL) not installed")
    except Exception as e:
        print(f"    ogr.Open: {type(e).__name__}: {e}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    gdb = _gdb_path(base)
    print("GDB path:", gdb)
    print("Exists:", gdb.exists())
    info = verify_hydrorivers_gdb(gdb)
    print("Layers:", info["layers"])
    print("River layer:", info["river_layer"])
    print("Columns:", info["columns"][:20] if info["columns"] else [])
    print("Mapped:", info["mapped"])
    if not info["layers"] and gdb.exists():
        _diagnose_gdb_open(gdb)
