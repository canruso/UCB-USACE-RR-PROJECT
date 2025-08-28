from typing import List, Dict, Union
from pathlib import Path
import pandas as pd
import xarray
from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class RussianRiver(BaseDataset):
    """Multi-basin, multi-timescale dataset loader for the Russian River region.

    If cfg.is_mts=True, merges daily.csv + hourly.csv + daily physics + hourly physics
    for the given basin, upsampling daily data to hourly. Otherwise, single-frequency logic.
    """

    def __init__(self, cfg: Config, is_train: bool, period: str, basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [], id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):

        super(RussianRiver, self).__init__(cfg=cfg, is_train=is_train, period=period, basin=basin,
                                           additional_features=additional_features, id_to_int=id_to_int, scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        cfg_dict = self.cfg.as_dict()
        is_mts_data_flag = cfg_dict.get("is_mts_data", False)

        if is_mts_data_flag:
            return self._load_mts_data(basin)
        else:
            return self._load_single_freq(basin)

    def _load_mts_data(self, basin: str) -> pd.DataFrame:
        daily_path = self.cfg.data_dir / "daily_mts_shift.csv"
        hourly_path = self.cfg.data_dir / "hourly_shared.csv"
        daily_df = pd.read_csv(daily_path, low_memory=False)
        daily_df = clean_df(daily_df)
        daily_df = daily_df.resample("1H").ffill()
        hourly_df = pd.read_csv(hourly_path, low_memory=False)
        hourly_df = clean_df(hourly_df)
        df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)

        if self.cfg.physics_informed:
            pass

        return df

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """Load single-frequency data (daily or hourly)."""
        if self.cfg.hourly:
            path = self.cfg.data_dir / "hourly.csv"
        else:
            path = self.cfg.data_dir / "daily_shift.csv"

        raw_df = pd.read_csv(path, low_memory=False)
        df = clean_df(raw_df)

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            physics_path = self.cfg.physics_data_file
            phys_df = pd.read_csv(physics_path, low_memory=False)
            phys_df = clean_df(phys_df)
            df = pd.merge(df, phys_df, how='outer', left_index=True, right_index=True)

        else:
            if self.cfg.physics_informed:
                print(f"[WARNING:_load_single_freq] => No physics_data_file found, skipping merges.")

        return df

    def _load_attributes(self) -> pd.DataFrame:
        # If you have static basin attributes, load them here
        return load_russian_river_attributes(self.cfg.data_dir)

def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    # if no static attributes are needed, just return None or empty DataFrame
    return pd.DataFrame()