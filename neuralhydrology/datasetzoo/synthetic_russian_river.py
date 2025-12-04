from typing import List, Dict, Union, Tuple
from pathlib import Path
import pandas as pd
import xarray
from datetime import datetime
from UCB_training.UCB_utils import clean_df
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class SyntheticRussianRiver(BaseDataset):
    """multi-basin, multi-timescale dataset loader with custom synthetic ranges"""

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
        # --- EARLY PADDING: ensure NH validation succeeds before dataset override ---
        dummy_start = pd.Timestamp("1900-01-01")
        dummy_end = pd.Timestamp("1901-01-01")

        required_keys = [
            "train_start_date", "train_end_date",
            "validation_start_date", "validation_end_date",
            "test_start_date", "test_end_date",
        ]
        for key in required_keys:
            if key not in cfg._cfg or cfg._cfg[key] is None:
                cfg._cfg[key] = dummy_start if "start" in key else dummy_end

        # Pad missing range lists so NH doesn't error on missing keys
        for key in ["train_ranges", "validation_ranges", "test_ranges"]:
            if key not in cfg._cfg or cfg._cfg[key] is None:
                cfg._cfg[key] = []
        print("[SyntheticRussianRiver] Injected dummy date placeholders into config.")

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

        print("[SyntheticRussianRiver] Loaded custom ranges:")
        for k, v in self.custom_ranges.items():
            print(f"  {k}: {v}")

        def _safe_parse(date_str):
            try:
                return pd.to_datetime(date_str, dayfirst=True)
            except Exception:
                return dummy_start

        def _assign_if_exists(key, value):
            if key in cfg._cfg:
                cfg._cfg[key] = value

        # Build synthetic unified NH period bounds from *all* custom ranges
        all_train_starts = []
        all_train_ends   = []

        for start, end in self.custom_ranges["train"]:
            all_train_starts.append(pd.to_datetime(start, dayfirst=True))
            all_train_ends.append(pd.to_datetime(end,   dayfirst=True))

        # Only assign if ranges exist
        if all_train_starts:
            cfg._cfg["train_start_date"] = min(all_train_starts)
            cfg._cfg["train_end_date"]   = max(all_train_ends)

        # Same for validation
        all_val_starts = []
        all_val_ends   = []
        for start, end in self.custom_ranges["validation"]:
            all_val_starts.append(pd.to_datetime(start, dayfirst=True))
            all_val_ends.append(pd.to_datetime(end,   dayfirst=True))

        if all_val_starts:
            cfg._cfg["validation_start_date"] = min(all_val_starts)
            cfg._cfg["validation_end_date"]   = max(all_val_ends)

        # Same for test
        all_test_starts = []
        all_test_ends   = []
        for start, end in self.custom_ranges["test"]:
            all_test_starts.append(pd.to_datetime(start, dayfirst=True))
            all_test_ends.append(pd.to_datetime(end,   dayfirst=True))

        if all_test_starts:
            cfg._cfg["test_start_date"] = min(all_test_starts)
            cfg._cfg["test_end_date"]   = max(all_test_ends)

        print("[SyntheticRussianRiver] Overrode cfg date fields with first synthetic ranges.")

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
                        'range_index': i,
                        'start_date': start,
                        'end_date': end
                    })

            if range_data:
                ranges_df = pd.DataFrame(range_data)
                ranges_df.to_csv(csv_path, index=False)
                print(f"[RANGE_LOG] Custom ranges logged to {csv_path}")
                print(f"[RANGE_LOG] Total ranges: {len(range_data)} across all periods")
            else:
                print("[RANGE_LOG] No custom ranges to log")

        except Exception as e:
            print(f"[WARN] Could not write custom ranges CSV: {e}")

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """load basin dataset and clip it to synthetic merged ranges"""
        cfg_dict = self.cfg.as_dict()
        is_mts_data_flag = cfg_dict.get("is_mts_data", False)

        if is_mts_data_flag:
            df = self._load_mts_data(basin)
        else:
            df = self._load_single_freq(basin)

        ranges = self.custom_ranges.get(self.period, [])
        if ranges:
            merged = pd.concat([
                pd.Series(pd.date_range(
                    pd.to_datetime(start, dayfirst=True), 
                    pd.to_datetime(end, dayfirst=True), 
                    freq="H"
                ))
                for start, end in ranges
            ]).drop_duplicates().sort_values().reset_index(drop=True)

            if len(merged) % 24 != 0:
                excess = len(merged) % 24
                trimmed_rows = merged.iloc[-excess:]

                print(f"[WARN] Trimming {excess} hours to make merged dataset length divisible by 24.")
                print("[WARN] Trimmed hours (timestamps):")
                print(trimmed_rows.to_list())

                merged = merged.iloc[:-excess]

            self.merged_dates = merged
            df = self._clip_to_date_range(df)
        else:
            print(f"[WARN] No custom ranges found for {self.period}; skipping clip_to_date_range().")

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
        elif self.cfg.physics_informed:
            print("[WARNING:_load_single_freq] => No physics_data_file found, skipping merges.")

        return df

    def _clip_to_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """clip dataframe to merged synthetic ranges"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

        if hasattr(self, "merged_dates") and isinstance(self.merged_dates, pd.Series):
            mask = df.index.isin(self.merged_dates)
            out_df = df.loc[mask]
            print(f"[DEBUG] Using merged date set: {mask.sum()} rows kept.")
        else:
            ranges = self.custom_ranges.get(self.period, [])
            if not ranges:
                print("[WARN] No custom ranges found; using full dataset (fallback).")
                return df
            subset_dfs = []
            for start, end in ranges:
                # Force dayfirst parsing here to prevent 01/10 becoming Jan 10th
                start_dt, end_dt = pd.to_datetime(start, dayfirst=True), pd.to_datetime(end, dayfirst=True)
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                subset_dfs.append(df.loc[mask])
                print(f"[DEBUG] Included {start_dt.date()} → {end_dt.date()}, {mask.sum()} rows")
            out_df = pd.concat(subset_dfs).sort_index()

        try:
            target_freq = "1H" if getattr(self.cfg, "hourly", False) else "1D"
            out_df = out_df.asfreq(target_freq)
            print(f"[DEBUG] Enforced uniform frequency: {target_freq}")
        except Exception as e:
            print(f"[WARN] Could not enforce frequency for {self.period}: {e}")
        print(f"[DEBUG] Final subset shape for {self.period}: {out_df.shape}")

        proof_dir = getattr(self.cfg, "output_dir", Path.cwd())
        proof_path = Path(proof_dir) / f"range_proof_{self.period}.txt"
        try:
            with open(proof_path, "w") as f:
                f.write(f"Period: {self.period}\n")
                f.write(f"Custom ranges: {self.custom_ranges.get(self.period)}\n")
                f.write(f"Subset index min: {out_df.index.min()}\n")
                f.write(f"Subset index max: {out_df.index.max()}\n")
                f.write(f"Rows: {out_df.shape[0]}\n")
            print(f"[PROOF] Wrote dataset verification to {proof_path}")
        except Exception as e:
            print(f"[WARN] Could not write range proof file: {e}")

        return out_df

    def _load_attributes(self) -> pd.DataFrame:
        """return empty static attribute frame"""
        return load_russian_river_attributes(self.cfg.data_dir)


def load_russian_river_attributes(data_dir: Path) -> pd.DataFrame:
    """load static basin attributes (empty)"""
    return pd.DataFrame()