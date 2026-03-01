from typing import List, Dict, Union, Tuple
from pathlib import Path
import pandas as pd
import xarray
from datetime import datetime
from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class SyntheticRussianRiver(BaseDataset):
    """
    Multi-basin, multi-timescale dataset loader with custom synthetic ranges.
    Relabels disjoint training years into a consecutive synthetic timeline 
    starting at year 2200 (e.g., 2200, 2201, 2202, etc). 
    Years labeled back in UCB_train.py.
    """

    def __init__(
        self,
        cfg: Config,
        is_train: bool,
        period: str,
        basin: str = None,
        train_ranges: List[Tuple[str, str]] = None,
        validation_ranges: List[Tuple[str, str]] = None,
        test_ranges: List[Tuple[str, str]] = None,
        additional_features: List[Dict[str, pd.DataFrame]] = [],
        id_to_int: Dict[str, int] = {},
        scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {},
    ):
        # Relabel dates into 22XX format, where XX is the year index in the range.
        dummy_start = pd.Timestamp("2200-01-01")
        dummy_end = pd.Timestamp("2201-01-01")

        required_keys = [
            "train_start_date", "train_end_date",
            "validation_start_date", "validation_end_date",
            "test_start_date", "test_end_date",
        ]
        for key in required_keys:
            if key not in cfg._cfg or cfg._cfg[key] is None:
                cfg._cfg[key] = dummy_start if "start" in key else dummy_end

        for key in ["train_ranges", "validation_ranges", "test_ranges"]:
            if key not in cfg._cfg or cfg._cfg[key] is None:
                cfg._cfg[key] = []
        
        def _parse_range_list(range_list):
            parsed = []
            if range_list:
                for item in range_list:
                    if isinstance(item, str) and "-" in item:
                        start, end = item.split("-", 1)
                        parsed.append((start.strip(), end.strip()))
            return parsed

        cfg_dict = cfg.as_dict()
        self.custom_ranges = {
            "train": train_ranges or _parse_range_list(cfg_dict.get("train_ranges")),
            "validation": validation_ranges or _parse_range_list(cfg_dict.get("validation_ranges")),
            "test": test_ranges or _parse_range_list(cfg_dict.get("test_ranges")),
        }
        
        def _get_synthetic_bounds(ranges, start_year_idx=0):
            if not ranges:
                return None, None
            
            # start is always Jan 1st of the base year (2200)
            syn_start = pd.Timestamp(f"{2200 + start_year_idx}-01-01")
            
            last_year_idx = start_year_idx + len(ranges) - 1
            
            # sets valid window to end at Dec 31st of the last year.
            syn_end = pd.Timestamp(f"{2200 + last_year_idx}-12-31")
            
            return syn_start, syn_end

        # set training bounds
        tr_s, tr_e = _get_synthetic_bounds(self.custom_ranges["train"], 0)
        if tr_s:
            cfg._cfg["train_start_date"] = tr_s
            cfg._cfg["train_end_date"]   = tr_e

        # set validation bounds
        val_s, val_e = _get_synthetic_bounds(self.custom_ranges["validation"], 0)
        if val_s:
            cfg._cfg["validation_start_date"] = val_s
            cfg._cfg["validation_end_date"]   = val_e

        # set test bounds
        te_s, te_e = _get_synthetic_bounds(self.custom_ranges["test"], 0)
        if te_s:
            cfg._cfg["test_start_date"] = te_s
            cfg._cfg["test_end_date"]   = te_e

        # print(f"[SyntheticRussianRiver] Synthetic 22XX bounds applied for {period}: {tr_s} to {tr_e}")

        super(SyntheticRussianRiver, self).__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
            additional_features=additional_features,
            id_to_int=id_to_int,
            scaler=scaler,
        )

        self._log_ranges_to_csv() 

    def _log_ranges_to_csv(self):
        """write all custom ranges into a csv for verification"""
        try:
            proof_dir = getattr(self.cfg, "output_dir", Path.cwd())
            csv_path = Path(proof_dir) / "custom_ranges_log.csv"

            range_data = []
            for period, ranges in self.custom_ranges.items():
                for i, (start, end) in enumerate(ranges):
                    range_data.append({
                        'period': period,
                        'original_index': i,
                        'synthetic_year': 2200 + i,
                        'orig_start': start,
                        'orig_end': end
                    })

            if range_data:
                pd.DataFrame(range_data).to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[WARN] Could not write custom ranges CSV: {e}")

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """load basin dataset and shift/stitch it to 22XX"""
        cfg_dict = self.cfg.as_dict()
        is_mts_data_flag = cfg_dict.get("is_mts_data", False)

        if is_mts_data_flag:
            df = self._load_mts_data(basin)
        else:
            df = self._load_single_freq(basin)

        df = self._clip_to_date_range(df)
        return df

    def _load_mts_data(self, basin: str) -> pd.DataFrame:
        """load multi-timescale mts dataset"""
        daily_path = self.cfg.data_dir / "daily_mts_shift.csv"
        hourly_path = self.cfg.data_dir / "hourly_shared.csv"

        daily_df = clean_df(pd.read_csv(daily_path, low_memory=False))
        daily_df = daily_df.resample("1H").ffill()

        hourly_df = clean_df(pd.read_csv(hourly_path, low_memory=False))
        df = pd.merge(hourly_df, daily_df, how="outer", left_index=True, right_index=True)
        return df

    def _load_single_freq(self, basin: str) -> pd.DataFrame:
        """load either hourly or daily data"""
        if self.cfg.hourly:
            path = self.cfg.data_dir / "hourly.csv"
        else:
            path = self.cfg.data_dir / "daily_shift.csv"

        df = clean_df(pd.read_csv(path, low_memory=False))

        if self.cfg.physics_informed and self.cfg.physics_data_file:
            phys_df = clean_df(pd.read_csv(self.cfg.physics_data_file, low_memory=False))
            df = pd.merge(df, phys_df, how="outer", left_index=True, right_index=True)

        return df

    def _remove_leap_days(self, df: Union[pd.DataFrame, pd.DatetimeIndex]) -> Union[pd.DataFrame, pd.DatetimeIndex]:
        """Removes Feb 29th rows from a DataFrame or DatetimeIndex"""
        if isinstance(df, pd.DatetimeIndex):
            mask = ~((df.month == 2) & (df.day == 29))
            return df[mask]
        
        if isinstance(df.index, pd.DatetimeIndex):
            mask = ~((df.index.month == 2) & (df.index.day == 29))
            return df.loc[mask]
        return df

    def _clip_to_date_range(self, df: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        ranges = self.custom_ranges.get(self.period, [])
        if not ranges:
            print(f"[WARN] No ranges for {self.period}, returning full df (no relabeling).")
            return df

        relabelled_chunks = []
        target_freq = "1H" if getattr(self.cfg, "hourly", False) else "1D"

        for i, (start, end) in enumerate(ranges):
            # select original slice
            start_dt = pd.to_datetime(start, dayfirst=True)
            end_dt = pd.to_datetime(end, dayfirst=True)
            
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            chunk = df.loc[mask].copy()

            if chunk.empty:
                print(f"[WARN] Chunk {i} ({start}-{end}) is empty. Skipping.")
                continue

            chunk = self._remove_leap_days(chunk)

            syn_year = 2200 + i
            syn_start = pd.Timestamp(f"{syn_year}-01-01")
            
            new_index = pd.date_range(
                start=syn_start, 
                periods=len(chunk), 
                freq=target_freq
            )
            
            
            if new_index.is_leap_year.any():
                padded_index = pd.date_range(
                    start=syn_start,
                    periods=len(chunk) + 48, # ample buffer
                    freq=target_freq
                )
                padded_index = self._remove_leap_days(padded_index)
                new_index = padded_index[:len(chunk)]

            if len(new_index) != len(chunk):
                # should not happen with padded logic, but safe fallback
                print(f"[ERR] Index mismatch in chunk {i}. Chunk: {len(chunk)}, New: {len(new_index)}")
                min_len = min(len(new_index), len(chunk))
                chunk = chunk.iloc[:min_len]
                new_index = new_index[:min_len]

            chunk.index = new_index
            relabelled_chunks.append(chunk)
            
            # Debug log
            # print(f"[DEBUG] Chunk {i}: {start_dt.date()}->{end_dt.date()} mapped to {new_index[0]}->{new_index[-1]} (Len: {len(chunk)})")

        if not relabelled_chunks:
            return pd.DataFrame()

        out_df = pd.concat(relabelled_chunks).sort_index()

        # final frequency check
        try:
            out_df = out_df.asfreq(target_freq)
        except Exception:
            pass 

        out_df.index.name = "date" 

        self._write_proof_file(out_df)
        return out_df

    def _write_proof_file(self, df):
        proof_dir = getattr(self.cfg, "output_dir", Path.cwd())
        proof_path = Path(proof_dir) / f"range_proof_{self.period}.txt"
        try:
            with open(proof_path, "w") as f:
                f.write(f"Period: {self.period}\n")
                f.write(f"Start: {df.index.min()}\n")
                f.write(f"End: {df.index.max()}\n")
                f.write(f"Rows: {len(df)}\n")
        except:
            pass

    def _load_attributes(self) -> pd.DataFrame:
        """return empty static attribute frame"""
        return pd.DataFrame()
    