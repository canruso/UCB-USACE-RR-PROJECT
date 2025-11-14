from typing import List, Dict, Union, Tuple
from pathlib import Path
import pandas as pd
import xarray
from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class SyntheticRussianRiver(BaseDataset):
    """Synthetic Russian River dataset supporting multi-date ranges for training, validation, and testing.

    Handles semicolon-delimited or YAML list-style date ranges and subsets data accordingly.
    """

    def __init__(self, cfg: Config, is_train: bool, period: str, basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):

        # expand date ranges before initializing BaseDataset
        self._init_date_ranges(cfg)

        super(SyntheticRussianRiver, self).__init__(
            cfg=cfg, is_train=is_train, period=period, basin=basin,
            additional_features=additional_features, id_to_int=id_to_int, scaler=scaler
        )

    # ────────────────────────────────────────────────────────────────
    # CONFIG RANGE HANDLING
    def _init_date_ranges(self, cfg: Config):
        """Parse and expand semicolon-separated or YAML-list style date ranges."""
        for phase in ["train", "validation", "test"]:
            start_key, end_key = f"{phase}_start_date", f"{phase}_end_date"
            ranges_key = f"{phase}_ranges"

            # Skip if ranges already defined
            if ranges_key in cfg._cfg:
                continue

            start_val = cfg._cfg.get(start_key)
            end_val = cfg._cfg.get(end_key)

            # Handle semicolon-separated string inputs
            if isinstance(start_val, str) and ";" in start_val:
                starts = [s.strip() for s in start_val.split(";")]
                ends = [e.strip() for e in cfg._cfg[end_key].split(";")]
                cfg.update_config({ranges_key: list(zip(starts, ends))}, dev_mode=True)

            # Handle list inputs directly
            elif isinstance(start_val, list) and isinstance(end_val, list):
                cfg.update_config({ranges_key: list(zip(start_val, end_val))}, dev_mode=True)

            # Single fallback range
            elif start_val and end_val:
                cfg.update_config({ranges_key: [(start_val, end_val)]}, dev_mode=True)

            else:
                print(f"[WARN] No date range found for {phase}, skipping split.")

        # add dummy single fields to satisfy NH
        for phase in ["train", "validation", "test"]:
            if f"{phase}_ranges" in cfg._cfg:
                first = cfg._cfg[f"{phase}_ranges"][0]
                cfg.update_config({
                    f"{phase}_start_date": first[0],
                    f"{phase}_end_date": first[1]
                }, dev_mode=True)

    # ────────────────────────────────────────────────────────────────
    # BASIN DATA LOADING
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        is_mts = self.cfg.as_dict().get("is_mts_data", False)
        df = self._load_mts_data(basin) if is_mts else self._load_single_freq(basin)
        return self._subset_by_ranges(df)

    def _load_mts_data(self, basin: str) -> pd.DataFrame:
        """Merge daily/hourly data for MTS mode."""
        daily_path = self.cfg.data_dir / "daily_mts_shift.csv"
        hourly_path = self.cfg.data_dir / "hourly_shared.csv"

        daily_df = clean_df(pd.read_csv(daily_path, low_memory=False))
        daily_df = daily_df.resample("1H").ffill()
        hourly_df = clean_df(pd.read_csv(hourly_path, low_memory=False))

        df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)

        # optional physics merge
        if self.cfg.physics_informed and self.cfg.physics_data_file:
            phys_df = clean_df(pd.read_csv(self.cfg.physics_data_file, low_memory=False))
            df = pd.merge(df, phys_df, how="outer", left_index=True, right_index=True)

        return df

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """Load hourly or daily data."""
        path = self.cfg.data_dir / ("hourly.csv" if self.cfg.hourly else "daily_shift.csv")
        df = clean_df(pd.read_csv(path, low_memory=False))

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            phys_df = clean_df(pd.read_csv(self.cfg.physics_data_file, low_memory=False))
            df = pd.merge(df, phys_df, how="outer", left_index=True, right_index=True)
        elif self.cfg.physics_informed:
            print("[WARNING] physics_informed=True but no physics_data_file found; skipping merge.")

        return df

    # ────────────────────────────────────────────────────────────────
    # SUBSETTING LOGIC
    def _subset_by_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subset dataset by date ranges based on current period."""
        phase = self.period.lower()
        ranges_key = f"{phase}_ranges"

        if ranges_key not in self.cfg._cfg:
            print(f"[WARN] No {ranges_key} found, returning full dataset.")
            return df

        df.index = pd.to_datetime(df.index, errors="coerce")
        valid_segments = []

        for (start, end) in self.cfg._cfg[ranges_key]:
            try:
                start_dt = pd.to_datetime(start, dayfirst=True)
                end_dt = pd.to_datetime(end, dayfirst=True)
                seg = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
                valid_segments.append(seg)
            except Exception as e:
                print(f"[WARN] Invalid date range {start}–{end}: {e}")

        if not valid_segments:
            print(f"[WARN] No valid data segments found for {phase}.")
            return df

        subset_df = pd.concat(valid_segments).sort_index()
        print(f"[INFO] {phase.title()} subset shape: {subset_df.shape}")
        return subset_df

    # ────────────────────────────────────────────────────────────────
    def _load_attributes(self) -> pd.DataFrame:
        return load_russian_river_attributes(self.cfg.data_dir)


def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    """Return empty attributes (no static basin features)."""
    return pd.DataFrame()