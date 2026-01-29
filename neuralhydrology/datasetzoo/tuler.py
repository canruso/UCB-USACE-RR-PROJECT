# neuralhydrology/datasetzoo/tuler.py
from typing import List, Dict, Union
from pathlib import Path
import pandas as pd
import xarray
from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class Tuler(BaseDataset):
    """
    Tule River dataset loader.

    Expects under cfg.data_dir:
      - daily_shift.csv                  (required for daily runs)
      - Tule_daily_shift.csv             (optional; used if physics_informed=True)
      # If/when you add hourly:
      # - hourly.csv
      # - Tule_hourly_shift.csv

    Returns a datetime-indexed DataFrame; BaseDataset will subset to the
    features specified in the YAML (dynamic_inputs, target_variables, etc.).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(Tuler, self).__init__(cfg=cfg,
                                    is_train=is_train,
                                    period=period,
                                    basin=basin,
                                    additional_features=additional_features,
                                    id_to_int=id_to_int,
                                    scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        # MAIN file (daily by default)
        hourly = bool(getattr(self.cfg, "hourly", False))
        main_name = "hourly.csv" if hourly else "daily_shift.csv"
        main_path = Path(self.cfg.data_dir) / main_name
        if not main_path.exists():
            raise FileNotFoundError(f"[Tuler] Missing required file: {main_path}")

        df = clean_df(pd.read_csv(main_path, low_memory=False))

        # Optionally merge physics/HMS split if requested
        if getattr(self.cfg, "physics_informed", False):
            if getattr(self.cfg, "physics_data_file", None):
                phys_path = Path(self.cfg.physics_data_file)
            else:
                phys_name = "Tule_hourly_shift.csv" if hourly else "Tule_daily_shift.csv"
                phys_path = Path(self.cfg.data_dir) / phys_name

            if phys_path.exists():
                phys_df = clean_df(pd.read_csv(phys_path, low_memory=False))
                # Avoid duplicate target columns if they appear in physics file
                for t in getattr(self.cfg, "target_variables", []):
                    if t in phys_df.columns:
                        phys_df = phys_df.drop(columns=[t])
                df = df.join(phys_df, how="outer")
            else:
                print(f"[Tuler] physics_informed=True but physics file not found → {phys_path}. Continuing without it.")

        return df

    def _load_attributes(self) -> pd.DataFrame:
        # No static attributes; return empty DataFrame
        return pd.DataFrame()