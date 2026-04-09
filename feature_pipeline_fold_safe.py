"""
Fold-safe feature builder: HydroRIVERS + ERA5 event features.
fit(train_df): train만으로 storm threshold, river_dist_bin 경계, ERA5 builder fit.
transform(df): 저장된 파라미터로 feature 생성 (누수 없음).
"""
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from hydrorivers_features import (
    get_hyriv_features_for_dataframe,
    compute_dist_bin_edges_from_train,
    _gdb_path,
    HAS_GEOPANDAS,
)
from era5_event_features import (
    ERA5EventFeatureBuilder,
    add_dilution_ionic_proxies,
    ERA5_EVENT_FEATURE_NAMES_MONTHLY,
)


# 새로 추가되는 피처 이름 (TA/EC/DRP feature list에 추가할 목록)
HYRIV_FEATURE_NAMES = [
    "river_dist_m", "hyriv_dist_bin",
    "hyriv_log_upcells", "hyriv_flow_order", "hyriv_log_q",
    "hyriv_q_over_up", "hyriv_order_x_up",
]
ERA5_EVENT_FEATURE_NAMES = [
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d",
    "rain_max_7d", "rain_max_30d", "rain_max_90d",
    "storm_cnt_7d", "storm_cnt_30d", "storm_cnt_90d",
    "sm_mean_7d", "sm_mean_30d",
    "sm_lag_1d", "sm_lag_7d", "sm_lag_30d",
    "wetness_rain_sm_7d", "wetness_rain_sm_30d",
    "dilution_proxy_7d", "ionic_flush_proxy_30d",
]


class FoldSafeFeatureBuilder:
    """
    fit(train_df): train만 사용해
      - HydroRIVERS 매칭 후 river_dist_m 분포로 dist_bin_edges 계산
      - ERA5 storm threshold 계산
    transform(df, is_validation=False): 동일 파라미터로 df에 피처 추가.
    """

    def __init__(
        self,
        base_dir: Path,
        gdb_path: Optional[Path] = None,
        era5_train_path: Optional[Path] = None,
        era5_val_path: Optional[Path] = None,
        storm_percentile: float = 92,
        use_hyriv: bool = True,
        use_era5_events: bool = True,
        era5_resolution: str = "daily",
    ):
        self.base_dir = Path(base_dir)
        self.gdb_path = Path(gdb_path) if gdb_path else _gdb_path(self.base_dir)
        self.era5_train_path = Path(era5_train_path) if era5_train_path else self.base_dir / "era5_daily_timeseries_train.parquet"
        self.era5_val_path = Path(era5_val_path) if era5_val_path else self.base_dir / "era5_daily_timeseries_val.parquet"
        if not self.era5_val_path.exists():
            self.era5_val_path = self.base_dir / "era5_daily_timeseries_val.csv"
        if not self.era5_train_path.exists():
            self.era5_train_path = self.base_dir / "era5_daily_timeseries_train.csv"
        self.storm_percentile = storm_percentile
        self.use_hyriv = use_hyriv and HAS_GEOPANDAS
        self.use_era5_events = use_era5_events
        self.era5_resolution = era5_resolution if era5_resolution in ("daily", "monthly") else "daily"
        # fit 시 저장
        self.dist_bin_edges_ = None
        self.hyriv_stats_ = {}
        self.era5_builder_ = ERA5EventFeatureBuilder(
            era5_train_path=self.era5_train_path,
            era5_val_path=self.era5_val_path,
            storm_percentile=storm_percentile,
            resolution=self.era5_resolution,
        )

    def fit(self, train_df: pd.DataFrame) -> "FoldSafeFeatureBuilder":
        """train_df만 사용해 통계/임계값 계산."""
        if self.use_hyriv and self.gdb_path.exists():
            hyriv_train, stats = get_hyriv_features_for_dataframe(
                train_df, self.gdb_path, dist_bin_edges=None
            )
            self.hyriv_stats_ = stats
            if "river_dist_m" in hyriv_train.columns and hyriv_train["river_dist_m"].notna().any():
                self.dist_bin_edges_ = compute_dist_bin_edges_from_train(hyriv_train["river_dist_m"].dropna())
                mr = stats.get("match_rate", 0)
                print(f"  [HydroRIVERS] match_rate={100*mr:.1f}%, river_dist_m 분포: "
                      f"min={hyriv_train['river_dist_m'].min():.0f}, max={hyriv_train['river_dist_m'].max():.0f} m")
            else:
                self.dist_bin_edges_ = None
        if self.use_era5_events:
            self.era5_builder_.fit(train_df)
            if self.era5_builder_.storm_threshold_ is not None:
                print(f"  [ERA5 event] storm_threshold (train {self.era5_builder_.storm_percentile}%ile) = {self.era5_builder_.storm_threshold_:.6f}")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        is_validation: bool = False,
    ) -> pd.DataFrame:
        """
        df에 hyriv_* + ERA5 event 피처 추가한 DataFrame 반환.
        기존 df 컬럼 + 새 컬럼만 추가 (merge).
        """
        out = df.copy()
        # 데이터 없어도 컬럼은 생성(NaN) → 진단/다운스트림에서 "(컬럼 없음)" 대신 결측률 확인 가능
        for c in HYRIV_FEATURE_NAMES:
            if c not in out.columns:
                out[c] = np.nan
        if self.use_hyriv and self.gdb_path.exists():
            hyriv_df, _ = get_hyriv_features_for_dataframe(
                df, self.gdb_path,
                dist_bin_edges=self.dist_bin_edges_,
            )
            for c in hyriv_df.columns:
                out[c] = hyriv_df[c].values
        era5_names = ERA5_EVENT_FEATURE_NAMES_MONTHLY if self.era5_resolution == "monthly" else ERA5_EVENT_FEATURE_NAMES
        for c in era5_names:
            if c not in out.columns:
                out[c] = np.nan
        if self.use_era5_events and self.era5_builder_.era5_train_ is not None and not self.era5_builder_.era5_train_.empty:
            era5_df = self.era5_builder_.era5_val_ if is_validation else self.era5_builder_.era5_train_
            era5_feat = self.era5_builder_.transform(df, era5_df=era5_df)
            for c in era5_feat.columns:
                out[c] = era5_feat[c].values
            extra = add_dilution_ionic_proxies(df, era5_feat, pet_col="pet")
            for c in extra.columns:
                out[c] = extra[c].values
        return out

    def get_feature_names(self) -> List[str]:
        """추가되는 피처 이름 목록 (존재하는 것만)."""
        names = []
        if self.use_hyriv:
            names.extend(HYRIV_FEATURE_NAMES)
        if self.use_era5_events:
            names.extend(
                ERA5_EVENT_FEATURE_NAMES_MONTHLY if self.era5_resolution == "monthly" else ERA5_EVENT_FEATURE_NAMES
            )
        return names


def build_fold_safe_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    base_dir: Path,
    era5_train_path: Optional[Path] = None,
    era5_val_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], FoldSafeFeatureBuilder]:
    """
    fit(train) → transform(train), transform(val) 후 반환.
    반환: train_with_features, val_with_features (val_df 없으면 None), builder.
    """
    builder = FoldSafeFeatureBuilder(
        base_dir=base_dir,
        era5_train_path=era5_train_path,
        era5_val_path=era5_val_path,
    )
    builder.fit(train_df)
    train_out = builder.transform(train_df, is_validation=False)
    val_out = builder.transform(val_df, is_validation=True) if val_df is not None and len(val_df) else None
    return train_out, val_out, builder
