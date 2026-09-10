"""
ERA5 daily timeseries → 이벤트/누적 강수, 토양수분 lag, storm count 등.
fold 내 누수 방지: storm threshold는 fit(train)에서만 계산, transform(val)에 적용.
"""
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# ERA5 daily 컬럼 후보 (자동 매핑)
DATE_COLS = ["date", "Date", "time", "Time", "datetime"]
LAT_COLS = ["Latitude", "lat", "latitude", "y"]
LON_COLS = ["Longitude", "lon", "longitude", "x"]
PRECIP_COLS = ["tp", "pr", "precip", "precipitation", "tp_mm", "pr_mm"]
SM_COLS = ["swvl1", "sm", "soil_moisture", "swvl1_0_7cm", "sm_0_7"]
RUNOFF_COLS = ["ro", "runoff", "ro_mm"]


def _infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if col.lower() in [x.lower() for x in candidates]:
            return col
    return None


def load_era5_daily(path: Path) -> pd.DataFrame:
    """parquet 또는 CSV 로드. 날짜 파싱."""
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, nrows=100000)
    date_col = _infer_column(df, DATE_COLS)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def map_era5_columns(df: pd.DataFrame) -> Dict[str, str]:
    """실제 컬럼명 → 표준명 매핑 (date, lat, lon, precip, sm)."""
    m = {}
    c = _infer_column(df, DATE_COLS)
    if c:
        m["date"] = c
    c = _infer_column(df, LAT_COLS)
    if c:
        m["lat"] = c
    c = _infer_column(df, LON_COLS)
    if c:
        m["lon"] = c
    c = _infer_column(df, PRECIP_COLS)
    if c:
        m["precip"] = c
    c = _infer_column(df, SM_COLS)
    if c:
        m["sm"] = c
    c = _infer_column(df, RUNOFF_COLS)
    if c:
        m["runoff"] = c
    return m


def _round_coords(lat: float, lon: float, grid_deg: float = 0.25) -> Tuple[float, float]:
    """가장 가까운 grid point (round to grid_deg)."""
    lat_r = round(lat / grid_deg) * grid_deg
    lon_r = round(lon / grid_deg) * grid_deg
    return lat_r, lon_r


def _get_series_at_point(
    era5: pd.DataFrame,
    lat: float,
    lon: float,
    date_col: str,
    lat_col: str,
    lon_col: str,
    precip_col: Optional[str],
    sm_col: Optional[str],
    grid_deg: float = 0.25,
) -> Tuple[pd.Series, pd.Series]:
    """한 (lat, lon)에 대한 daily precip, sm 시계열 (날짜 인덱스)."""
    lat_r, lon_r = _round_coords(lat, lon, grid_deg)
    sub = era5[(era5[lat_col] == lat_r) & (era5[lon_col] == lon_r)].copy()
    if sub.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    sub = sub.sort_values(date_col)
    sub = sub.set_index(date_col)
    pr = sub[precip_col].astype(float) if precip_col and precip_col in sub.columns else pd.Series(dtype=float)
    sm = sub[sm_col].astype(float) if sm_col and sm_col in sub.columns else pd.Series(dtype=float)
    return pr, sm


def compute_rain_sums(pr: pd.Series, sample_date: pd.Timestamp, windows: List[int]) -> Dict[str, float]:
    """Sample Date 이전 N일 누적/최대. pr 인덱스는 date."""
    out = {}
    pr = pr.reindex(pd.date_range(pr.index.min() if len(pr.index) else sample_date, sample_date, freq="D")).fillna(0)
    pr = pr.sort_index()
    for n in windows:
        window = pr.loc[:sample_date].iloc[-n:] if len(pr) >= n else pr.loc[:sample_date]
        window = window.tail(n)
        out[f"rain_sum_{n}d"] = window.sum()
        out[f"rain_max_{n}d"] = window.max()
    return out


def compute_storm_counts(
    pr: pd.Series,
    sample_date: pd.Timestamp,
    threshold: float,
    windows: List[int],
) -> Dict[str, float]:
    """임계값 이상 강수 일수 (storm_cnt_Nd). threshold는 fit에서 train 기반 계산."""
    out = {}
    pr = pr.reindex(pd.date_range(pr.index.min() if len(pr.index) else sample_date, sample_date, freq="D")).fillna(0)
    pr = pr.sort_index()
    for n in windows:
        window = pr.loc[:sample_date].iloc[-n:] if len(pr) >= n else pr.loc[:sample_date]
        window = window.tail(n)
        out[f"storm_cnt_{n}d"] = (window >= threshold).sum()
    return out


def compute_sm_features(
    sm: pd.Series,
    sample_date: pd.Timestamp,
    pr: pd.Series,
    windows: List[int],
) -> Dict[str, float]:
    """sm_mean_Nd, sm_lag_1d/7d/30d, wetness_rain_sm_Nd."""
    out = {}
    sm = sm.reindex(pd.date_range(sm.index.min() if len(sm.index) else sample_date, sample_date, freq="D"))
    sm = sm.sort_index()
    for n in windows:
        window = sm.loc[:sample_date].iloc[-n:] if len(sm) >= n else sm.loc[:sample_date]
        window = window.tail(n)
        out[f"sm_mean_{n}d"] = window.mean()
    for lag in [1, 7, 30]:
        before = sample_date - pd.Timedelta(days=lag)
        if len(sm) and sm.index.min() <= before:
            vals = sm.loc[:before]
            if len(vals):
                out[f"sm_lag_{lag}d"] = vals.iloc[-1]
        if f"sm_lag_{lag}d" not in out:
            out[f"sm_lag_{lag}d"] = np.nan
    pr = pr.reindex(pd.date_range(pr.index.min() if len(pr.index) else sample_date, sample_date, freq="D")).fillna(0)
    pr = pr.sort_index()
    for n in windows:
        wpr = pr.loc[:sample_date].tail(n)
        wsm = sm.loc[:sample_date].tail(n)
        if len(wpr) and len(wsm):
            out[f"wetness_rain_sm_{n}d"] = wpr.sum() * wsm.mean()
        else:
            out[f"wetness_rain_sm_{n}d"] = np.nan
    return out


# --- 월별(monthly) 버전: 시계열 인덱스는 월 첫날 (YYYY-MM-01) 가정 ---

def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize().replace(day=1)


def _ensure_month_index(s: pd.Series, aggregate: str = "mean") -> pd.Series:
    """인덱스를 월 첫날로 통일. 동일 월이 여러 개면 aggregate(sum 또는 mean) 적용."""
    if s.empty:
        return s
    idx = pd.to_datetime(s.index, errors="coerce")
    month_start = idx.map(lambda t: t.normalize().replace(day=1) if pd.notna(t) else pd.NaT)
    s = s.copy()
    s.index = month_start
    s = s[~s.index.isna()]
    if not s.empty and s.index.duplicated().any():
        s = s.groupby(level=0).agg(aggregate)
    return s.sort_index()


def compute_rain_sums_monthly(
    pr: pd.Series, sample_date: pd.Timestamp, windows_months: List[int]
) -> Dict[str, float]:
    """Sample Date 포함 월 기준 이전 N개월 누적/최대. pr 인덱스는 월 첫날."""
    out = {}
    pr = _ensure_month_index(pr, aggregate="sum")
    if pr.empty:
        return {f"rain_sum_{n}m": np.nan for n in windows_months} | {f"rain_max_{n}m": np.nan for n in windows_months}
    end_month = _month_start(sample_date)
    pr = pr.reindex(pd.date_range(pr.index.min(), end_month, freq="MS")).fillna(0).sort_index()
    for n in windows_months:
        window = pr.loc[:end_month].tail(n)
        out[f"rain_sum_{n}m"] = window.sum()
        out[f"rain_max_{n}m"] = window.max()
    return out


def compute_storm_counts_monthly(
    pr: pd.Series,
    sample_date: pd.Timestamp,
    threshold: float,
    windows_months: List[int],
) -> Dict[str, float]:
    """임계값 이상 강수인 월 수 (storm_cnt_Nm)."""
    out = {}
    pr = _ensure_month_index(pr, aggregate="sum")
    if pr.empty:
        return {f"storm_cnt_{n}m": np.nan for n in windows_months}
    end_month = _month_start(sample_date)
    pr = pr.reindex(pd.date_range(pr.index.min(), end_month, freq="MS")).fillna(0).sort_index()
    for n in windows_months:
        window = pr.loc[:end_month].tail(n)
        out[f"storm_cnt_{n}m"] = (window >= threshold).sum()
    return out


def compute_sm_features_monthly(
    sm: pd.Series,
    sample_date: pd.Timestamp,
    pr: pd.Series,
    windows_months: List[int],
    lags_months: List[int],
) -> Dict[str, float]:
    """sm_mean_Nm, sm_lag_1m/3m, wetness_rain_sm_Nm."""
    out = {}
    sm = _ensure_month_index(sm, aggregate="mean")
    pr = _ensure_month_index(pr, aggregate="sum")
    if sm.empty:
        for n in windows_months:
            out[f"sm_mean_{n}m"] = np.nan
        for lag in lags_months:
            out[f"sm_lag_{lag}m"] = np.nan
        for n in windows_months:
            out[f"wetness_rain_sm_{n}m"] = np.nan
        return out
    end_month = _month_start(sample_date)
    sm = sm.reindex(pd.date_range(sm.index.min(), end_month, freq="MS")).sort_index()
    pr = pr.reindex(pd.date_range(pr.index.min(), end_month, freq="MS")).fillna(0).sort_index()
    for n in windows_months:
        window = sm.loc[:end_month].tail(n)
        out[f"sm_mean_{n}m"] = window.mean()
    for lag in lags_months:
        before = end_month - pd.DateOffset(months=lag)
        vals = sm.loc[sm.index <= before]
        out[f"sm_lag_{lag}m"] = vals.iloc[-1] if len(vals) else np.nan
    for n in windows_months:
        wpr = pr.loc[:end_month].tail(n)
        wsm = sm.loc[:end_month].tail(n)
        if len(wpr) and len(wsm):
            out[f"wetness_rain_sm_{n}m"] = wpr.sum() * wsm.mean()
        else:
            out[f"wetness_rain_sm_{n}m"] = np.nan
    return out


# 월별 피처 이름 (resolution="monthly"일 때 사용)
ERA5_EVENT_FEATURE_NAMES_MONTHLY = [
    "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
    "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
    "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
    "sm_mean_1m", "sm_mean_3m",
    "sm_lag_1m", "sm_lag_3m",
    "wetness_rain_sm_1m", "wetness_rain_sm_3m",
    "dilution_proxy_1m", "ionic_flush_proxy_3m",
]


class ERA5EventFeatureBuilder:
    """
    fit(train_df): train의 (lat, lon, date)에 대해 ERA5 시계열로 storm threshold(90~95 percentile) 계산.
    transform(df): 각 행에 대해 rain_sum/max, storm_cnt, sm_mean/lag, wetness 생성.
    resolution: "daily" → 일별 시계열(7d/30d/90d), "monthly" → 월별 시계열(1m/3m/6m/12m).
    """

    def __init__(
        self,
        era5_train_path: Optional[Path] = None,
        era5_val_path: Optional[Path] = None,
        storm_percentile: float = 92,
        windows: List[int] = None,
        windows_months: List[int] = None,
        grid_deg: float = 0.25,
        resolution: str = "daily",
    ):
        self.era5_train_path = Path(era5_train_path) if era5_train_path else None
        self.era5_val_path = Path(era5_val_path) if era5_val_path else None
        self.storm_percentile = storm_percentile
        self.windows = windows or [7, 30, 90]
        self.windows_months = windows_months or [1, 3, 6, 12]
        self.grid_deg = grid_deg
        self.resolution = resolution if resolution in ("daily", "monthly") else "daily"
        self.storm_threshold_ = None
        self.era5_train_ = None
        self.era5_val_ = None
        self.col_map_ = {}
        self.precip_col_ = None
        self.sm_col_ = None
        self.date_col_ = None
        self.lat_col_ = None
        self.lon_col_ = None

    def fit(self, train_df: pd.DataFrame) -> "ERA5EventFeatureBuilder":
        """train_df에 Latitude, Longitude, Sample Date 있음. ERA5 시계열에서 storm threshold 계산."""
        if self.era5_train_path and self.era5_train_path.exists():
            self.era5_train_ = load_era5_daily(self.era5_train_path)
        else:
            self.era5_train_ = pd.DataFrame()
        if self.era5_val_path and self.era5_val_path.exists():
            self.era5_val_ = load_era5_daily(self.era5_val_path)
        else:
            self.era5_val_ = self.era5_train_
        if self.era5_train_.empty:
            return self
        self.col_map_ = map_era5_columns(self.era5_train_)
        self.date_col_ = self.col_map_.get("date")
        self.lat_col_ = self.col_map_.get("lat")
        self.lon_col_ = self.col_map_.get("lon")
        self.precip_col_ = self.col_map_.get("precip")
        self.sm_col_ = self.col_map_.get("sm")
        if not self.date_col_ or not self.lat_col_ or not self.lon_col_ or not self.precip_col_:
            return self
        precip_values = []
        for _, row in train_df.iterrows():
            lat, lon = row["Latitude"], row["Longitude"]
            lat_r, lon_r = _round_coords(lat, lon, self.grid_deg)
            sub = self.era5_train_[
                (self.era5_train_[self.lat_col_] == lat_r) & (self.era5_train_[self.lon_col_] == lon_r)
            ]
            if self.precip_col_ in sub.columns:
                pr = sub[self.precip_col_].dropna().astype(float)
                if self.resolution == "monthly":
                    pr = _ensure_month_index(pr, aggregate="sum")
                precip_values.append(pr)
        if precip_values:
            all_pr = pd.concat(precip_values, ignore_index=False)
            all_pr = all_pr.astype(float)
            self.storm_threshold_ = float(np.nanpercentile(all_pr.values, self.storm_percentile))
        else:
            self.storm_threshold_ = 0.0
        return self

    def transform(self, df: pd.DataFrame, era5_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """df 각 행에 대해 ERA5 event 피처 추가. era5_df 없으면 self.era5_train_/era5_val_ 사용."""
        if self.resolution == "monthly":
            out_cols = [
                "rain_sum_1m", "rain_sum_3m", "rain_sum_6m", "rain_sum_12m",
                "rain_max_1m", "rain_max_3m", "rain_max_6m", "rain_max_12m",
                "storm_cnt_1m", "storm_cnt_3m", "storm_cnt_6m", "storm_cnt_12m",
                "sm_mean_1m", "sm_mean_3m",
                "sm_lag_1m", "sm_lag_3m",
                "wetness_rain_sm_1m", "wetness_rain_sm_3m",
            ]
        else:
            out_cols = [
                "rain_sum_7d", "rain_sum_30d", "rain_sum_90d",
                "rain_max_7d", "rain_max_30d", "rain_max_90d",
                "storm_cnt_7d", "storm_cnt_30d", "storm_cnt_90d",
                "sm_mean_7d", "sm_mean_30d",
                "sm_lag_1d", "sm_lag_7d", "sm_lag_30d",
                "wetness_rain_sm_7d", "wetness_rain_sm_30d",
            ]
        result = pd.DataFrame(index=df.index)
        for c in out_cols:
            result[c] = np.nan
        if self.era5_train_ is None or self.era5_train_.empty or not self.precip_col_:
            return result
        era5 = era5_df if era5_df is not None else self.era5_train_
        date_col = self.col_map_.get("date") or self.date_col_
        lat_col = self.col_map_.get("lat") or self.lat_col_
        lon_col = self.col_map_.get("lon") or self.lon_col_
        if not date_col or not lat_col or not lon_col:
            return result
        sample_date_col = "Sample Date" if "Sample Date" in df.columns else "Sample_Date"
        if sample_date_col not in df.columns:
            return result
        threshold = self.storm_threshold_ if self.storm_threshold_ is not None else 0.0
        for i, row in df.iterrows():
            lat, lon = row["Latitude"], row["Longitude"]
            try:
                sd = pd.Timestamp(row[sample_date_col])
            except Exception:
                continue
            pr, sm = _get_series_at_point(
                era5, lat, lon, date_col, lat_col, lon_col,
                self.precip_col_, self.sm_col_, self.grid_deg,
            )
            if pr.empty:
                continue
            if self.resolution == "monthly":
                rain = compute_rain_sums_monthly(pr, sd, self.windows_months)
                storm = compute_storm_counts_monthly(pr, sd, threshold, self.windows_months)
                sm_f = compute_sm_features_monthly(sm, sd, pr, [1, 3], lags_months=[1, 3])
            else:
                rain = compute_rain_sums(pr, sd, self.windows)
                storm = compute_storm_counts(pr, sd, threshold, self.windows)
                sm_f = compute_sm_features(sm, sd, pr, [7, 30])
            for k, v in rain.items():
                result.loc[i, k] = v
            for k, v in storm.items():
                result.loc[i, k] = v
            for k, v in sm_f.items():
                result.loc[i, k] = v
        return result


def add_dilution_ionic_proxies(
    df: pd.DataFrame,
    era5_features: pd.DataFrame,
    pet_col: str = "pet",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """TA/EC용: dilution_proxy (7d 또는 1m), ionic_flush_proxy (30d 또는 3m). df에 pet 필요."""
    extra = pd.DataFrame(index=df.index)
    if "rain_sum_7d" in era5_features.columns and pet_col in df.columns:
        extra["dilution_proxy_7d"] = era5_features["rain_sum_7d"] / (df[pet_col].values + eps)
    elif "rain_sum_1m" in era5_features.columns and pet_col in df.columns:
        extra["dilution_proxy_1m"] = era5_features["rain_sum_1m"] / (df[pet_col].values + eps)
    else:
        extra["dilution_proxy_7d"] = np.nan
    if "rain_max_30d" in era5_features.columns and "sm_mean_30d" in era5_features.columns:
        extra["ionic_flush_proxy_30d"] = era5_features["rain_max_30d"] * (1.0 / (era5_features["sm_mean_30d"] + eps))
    elif "rain_max_3m" in era5_features.columns and "sm_mean_3m" in era5_features.columns:
        extra["ionic_flush_proxy_3m"] = era5_features["rain_max_3m"] * (1.0 / (era5_features["sm_mean_3m"] + eps))
    elif "ionic_flush_proxy_30d" not in extra.columns:
        extra["ionic_flush_proxy_30d"] = np.nan
    return extra
