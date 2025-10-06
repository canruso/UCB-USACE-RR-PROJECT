from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional
from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics
import xarray as xr


class UCB_trainer:
    """
    A wrapper to facilitate easier training/evaluation of neural hydrology models.
    """

    def __init__(self, path_to_csv_folder: Path, yaml_path: Path, hyperparams: dict, input_features: List[str] = None,
                 num_ensemble_members: int = 1, physics_informed: bool = False, physics_data_file: Path = None,
                 hourly: bool = False, extend_train_period: bool = False, gpu: int = -1, is_mts: bool = False,
                 is_mts_data: bool = False, basin: bool = None, verbose: bool = True, runs_parent: Path = None,
                 run_label: str = None, run_stamp: str = None):
        """
        Initialize the UCB_trainer class with configurations and training parameters.
        """
        self._hyperparams = hyperparams
        self._num_ensemble_members = num_ensemble_members
        self._physics_informed = physics_informed
        self._physics_data_file = physics_data_file
        self._gpu = gpu
        self._data_dir = path_to_csv_folder
        self._dynamic_inputs = input_features
        self._yaml_path = yaml_path
        self._hourly = hourly
        self._extended_train_period = extend_train_period
        self._is_mts = is_mts
        self._is_mts_data = is_mts_data
        self._basin = basin
        self._verbose = verbose

        self._config = None
        self._model = None
        self._predictions = None
        self._observed = None
        self._metrics = None
        self._basin_name = None
        self._target_variable = None
        self._runs_parent = runs_parent
        self._run_label = run_label
        self._run_stamp = run_stamp
        self._create_config()

    def train(self):
        """
        Train the model or ensemble based on the specified number of ensemble members.
        """
        if self._num_ensemble_members == 1:
            path = self._train_model()
            self._eval_model(path, period="validation")
            self._model = path
        else:
            self._model = self._train_ensemble()
            for model_path in self._model:
                self._eval_model(model_path, period="validation")

        return self._model

    def results(self, period='validation', mts_trk="1H") -> (Path, dict):
        """
        Public method to return a CSV path and a metrics dict for a given period.
        """
        if self._is_mts:
            time_resolution_key = mts_trk
        else:
            time_resolution_key = '1h' if self._hourly else '1D'

        self._get_predictions(time_resolution_key, period)

        if self._verbose:
            self._generate_obs_sim_plt(period)

        self._metrics = calculate_all_metrics(self._observed, self._predictions)
        csv_path = self._generate_csv(period, freq_key=(time_resolution_key if self._is_mts else None))
        return csv_path, self._metrics

    @classmethod
    def from_run_dir(cls, run_dirs, *, gpu: int = -1, verbose: bool = True) -> "UCB_trainer":
        """
        Create an evaluation‑only stub from one run directory or an ensemble.
        """
        paths = [Path(p) for p in ([run_dirs] if isinstance(run_dirs, (str, Path)) else run_dirs)]
        cfg = Config(paths[0] / "config.yml")
        stub = cls.__new__(cls)
        stub._model = paths if len(paths) > 1 else paths[0]
        stub._gpu = gpu
        stub._verbose = verbose
        stub._is_mts = getattr(cfg, "is_mts", False)
        stub._physics_informed = getattr(cfg, "physics_informed", False)
        stub._config = cfg
        stub._basin_name = None
        stub._target_variable = None
        return stub

    def _dbg(self, tag: str, *msg):
        """Compact internal debug printer controlled by `self._verbose`."""
        if getattr(self, "_verbose", False):
            print(f"[DEBUG:{tag}]", *msg)

    def _predict_core(self, period: str = "test", mts_trk: Optional[str] = None, epoch: Optional[int] = None,
                      gpu: Optional[int] = None):

        gpu = self._gpu if gpu is None else gpu
        parts = {"train_validation": ["train", "validation"]}.get(period, [period])
        self._dbg(self, "_predict_core", f"period={period} → parts={parts}",
                  f"gpu={gpu}", f"epoch={epoch}", f"mts_trk={mts_trk}")

        def _flatten(da: xr.DataArray) -> xr.DataArray:
            self._dbg(self, "_flatten‑in ", da.dims, da.shape)
            if "time_step" not in da.dims:
                return da
            if da.sizes["time_step"] == 1:
                da = da.isel(time_step=0)
            else:
                da = da.stack(time=("date", "time_step"))
                mi = da["time"].to_index()
                da = da.assign_coords(time=("time", pd.to_datetime(mi.get_level_values("date")) +
                                            pd.to_timedelta(mi.get_level_values("time_step"), unit="h"))).rename(
                    {"time": "date"})
            self._dbg(self, "_flatten‑out", da.dims, da.shape)

            return da

        def _pick_freq(bkt: dict) -> str:
            choice = mts_trk if mts_trk and mts_trk in bkt else ("1H" if "1H" in bkt else next(iter(bkt)))
            self._dbg(self, "_pick_freq", f"bucket keys={list(bkt)} → '{choice}'")

            return choice

        def _read_single(run_dir: Path, per: str):
            self._dbg(self, "_read_single", run_dir, per)
            eval_run(run_dir, period=per, epoch=epoch, gpu=gpu)
            cfg = Config(run_dir / "config.yml")
            ep = epoch or cfg.epochs
            pkl = run_dir / per / f"model_epoch{ep:03d}" / f"{per}_results.p"
            res = pickle.load(open(pkl, "rb"))

            basin = next(iter(res))
            freq = _pick_freq(res[basin])
            xr_d = res[basin][freq]["xr"]
            tgt = cfg.target_variables[0]
            self._dbg(self, "_read_single", f"basin={basin}", f"freq={freq}", f"tgt={tgt}", f"keys={list(xr_d)}")
            self._basin_name = basin
            self._target_variable = tgt

            return _flatten(xr_d[f"{tgt}_sim"]), _flatten(xr_d[f"{tgt}_obs"])

        def _read_ensemble(per: str):
            self._dbg(self, "_read_ensemble", per)
            rds = [Path(p) for p in self._model]
            for rd in rds:
                eval_run(rd, period=per, epoch=epoch, gpu=gpu)

            ens = create_results_ensemble(rds, period=per, epoch=epoch)
            basin = next(iter(ens))
            freq = _pick_freq(ens[basin])
            xr_d = ens[basin][freq]["xr"]
            tgt = Config(rds[0] / "config.yml").target_variables[0]
            self._dbg(self, "read_ensemble", f"basin={basin}", f"freq={freq}", f"tgt={tgt}", f"keys={list(xr_d)}")
            self._basin_name = basin
            self._target_variable = tgt

            return _flatten(xr_d[f"{tgt}_sim"]), _flatten(xr_d[f"{tgt}_obs"])

        if isinstance(self._model, (str, Path)):
            sim, obs = zip(*[_read_single(Path(self._model), p) for p in parts])
        else:
            sim, obs = zip(*[_read_ensemble(p) for p in parts])

        sim_da = xr.concat(sim, dim=sim[0].dims[0]) if len(sim) > 1 else sim[0]
        obs_da = xr.concat(obs, dim=obs[0].dims[0]) if len(obs) > 1 else obs[0]
        self._dbg(self, "_predict_core", "final shapes:", f"sim={sim_da.shape}", f"obs={obs_da.shape}")
        return sim_da, obs_da, None

    def predict(self, period: str = "test", *, partner_trainer: Optional["UCB_trainer"] = None,
                hms_csv: Optional[Path] = None, mts_trk: Optional[str] = None, epoch: Optional[int] = None,
                gpu: Optional[int] = None, metrics: Optional[List[str]] = None, start_date: Optional[str] = None,
                end_date: Optional[str] = None, plot_filename: Optional[Path] = None,
                timeseries_filename: Optional[Path] = None, metrics_filename: Optional[Path] = None,
                plot_title: Optional[str] = None):

        self._dbg(self, "predict", f"period={period}", f"partner={partner_trainer is not None}",
                  f"hms_csv={hms_csv}", f"mts_trk={mts_trk}")

        pred_lstm, obs, _ = self._predict_core(period, mts_trk, epoch, gpu)

        for cand in ("date", "time"):
            if cand in obs.coords:
                x_raw = obs[cand].values
                break
        else:
            x_raw = obs.coords[obs.dims[0]].values
        try:
            x = pd.to_datetime(x_raw)

        except Exception:
            x = x_raw

        title_txt = plot_title or f"{self._basin_name} – {period}"

        if partner_trainer is None:
            if self._verbose:
                plt.style.use("default")
                plt.figure(figsize=(14, 5))
                plt.plot(x, obs, label="Observed", linewidth=2)
                plt.plot(x, pred_lstm, label="Simulated", linewidth=2)
                plt.title(title_txt)
                plt.xlabel("Date")
                plt.ylabel(self._target_variable)
                plt.grid(alpha=.4)
                plt.legend(fontsize=18, fancybox=True, framealpha=0.9, borderpad=1.2)
                plt.tight_layout()
                if plot_filename:
                    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                plt.show()

            m = calculate_all_metrics(obs, pred_lstm)
            m["PBIAS"] = self.calculate_pbias(obs, pred_lstm)
            return pred_lstm, obs, pd.DataFrame([m])

        if hms_csv is None:
            raise ValueError("`hms_csv` must be provided when partner_trainer is used.")

        pred_pilstm, obs_p, _ = partner_trainer._predict_core(period, mts_trk, epoch, gpu)
        if obs.shape != obs_p.shape or not np.allclose(obs.values, obs_p.values, equal_nan=True):
            raise ValueError("Observed vectors differ between trainers.")

        hms_df = (self._clean_hms(hms_csv).reset_index().rename(columns={"date": "Date"}))

        df = (pd.DataFrame({"Date": x, "Observed": obs.values,
                            "LSTM_Predicted": pred_lstm.values if not self._physics_informed else pred_pilstm.values,
                            "PLSTM_Predicted": pred_pilstm.values if not self._physics_informed else pred_lstm.values})
              .merge(hms_df, on="Date", how="left"))

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
        hms_da = xr.DataArray(df["HMS_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
        lstm_da = xr.DataArray(df["LSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})
        pil_da = xr.DataArray(df["PLSTM_Predicted"].values, dims=["date"], coords={"date": df["Date"]})

        def _m(da):
            d = calculate_all_metrics(obs_da, da)
            d["PBIAS"] = self.calculate_pbias(obs_da, da)
            return d

        m_hms, m_lstm, m_pls = map(_m, (hms_da, lstm_da, pil_da))

        if metrics_filename:
            pd.DataFrame({"HMS": m_hms, "LSTM": m_lstm, "PILSTM": m_pls}).to_csv(metrics_filename)

        keys = metrics or ["NSE", "PBIAS"]
        fmt = lambda d: ", ".join(f"{k}={d[k]:.3f}" for k in keys if k in d)

        plt.style.use("default")
        plt.figure(figsize=(15, 6))
        plt.plot(df["Date"], df["Observed"], label="Observed", linewidth=2)
        plt.plot(df["Date"], df["HMS_Predicted"], label="HMS " + fmt(m_hms), linewidth=2, alpha=.8)
        plt.plot(df["Date"], df["LSTM_Predicted"], label="LSTM " + fmt(m_lstm), linewidth=2, alpha=.8)
        plt.plot(df["Date"], df["PLSTM_Predicted"], label="PILSTM " + fmt(m_pls), linewidth=2, alpha=.8)
        plt.title(title_txt)
        plt.xlabel("Date")
        plt.ylabel("Inflow (cfs)")
        plt.grid(alpha=.3)
        plt.legend(fontsize=18, fancybox=True, framealpha=0.9, borderpad=1.2)
        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.show()

        if timeseries_filename:
            df.to_csv(timeseries_filename, index=False)

        return lstm_da, obs_da, pd.DataFrame([m_lstm])

    def _eval_model(self, run_directory: Path, period="validation"):
        eval_run(run_dir=run_directory, period=period)

        results_file = run_directory / period / f"model_epoch{str(self._config.epochs).zfill(3)}" / f"{period}_results.p"
        if results_file.exists():
            with open(results_file, "rb") as fp:
                results = pickle.load(fp)
            basin_name = list(results.keys())[0]
            aggregator_keys = list(results[basin_name].keys())
            for aggregator_key in aggregator_keys:
                xr_dict = results[basin_name][aggregator_key]["xr"]
                for var_name in xr_dict:
                    da = xr_dict[var_name]
        else:
            print(f"[WARN] => results_file not found: {results_file}")

    def _get_predictions(self, time_resolution_key, period='validation'):
        """
            Load predictions from the model's result files for the given period.
            Only apply multi-timescale flattening logic if self._is_mts is True.
            Otherwise, keep old behavior.
            """
        if self._num_ensemble_members == 1:
            results_file = self._model / period / (f"model_"
                                                   f"epoch{str(self._config.epochs).zfill(3)}") / f"{period}_results.p"
            if not results_file.exists():
                self._eval_model(self._model, period)
            if not results_file.exists():
                raise FileNotFoundError(f"Failed to evaluate or locate results for {period} => {results_file}")

            with open(results_file, "rb") as fp:
                results = pickle.load(fp)

            self._basin_name = next(iter(results.keys()))
            basin_dict = results[self._basin_name]

            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if time_resolution_key not in basin_dict:
                raise KeyError(
                    f"time_resolution_key '{time_resolution_key}' not in results for basin '{self._basin_name}'. "
                    f"Found keys: {list(basin_dict.keys())}")

            xr_dict = basin_dict[time_resolution_key]["xr"]
            if observed_key not in xr_dict or simulated_key not in xr_dict:
                raise KeyError(
                    f"Missing '{observed_key}' or '{simulated_key}' in aggregator "
                    f"'{time_resolution_key}' for basin '{self._basin_name}'.")

            obs_da = xr_dict[observed_key]
            sim_da = xr_dict[simulated_key]

            if self._is_mts:
                if time_resolution_key == "1D":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)

                elif time_resolution_key == "1H":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.stack(stacked_time=("date", "time_step"))
                        sim_da = sim_da.stack(stacked_time=("date", "time_step"))
                        obs_da = obs_da.rename({"stacked_time": "time"})
                        sim_da = sim_da.rename({"stacked_time": "time"})
                    else:
                        print("[WARN] => The 1H aggregator has no 'time_step' dimension? shape=", obs_da.shape)
                else:
                    print("[DEBUG:_get_predictions] => MTS ignoring unknown aggregator key:", time_resolution_key)

            else:
                if "time_step" in obs_da.dims:
                    obs_da = obs_da.isel(time_step=0)
                    sim_da = sim_da.isel(time_step=0)

            self._observed = obs_da
            self._predictions = sim_da

        else:
            results = create_results_ensemble(run_dirs=self._model, period=period)
            self._basin_name = next(iter(results.keys()))
            basin_dict = results[self._basin_name]

            self._target_variable = self._config.target_variables[0]
            observed_key = f"{self._target_variable}_obs"
            simulated_key = f"{self._target_variable}_sim"

            if time_resolution_key not in basin_dict:
                raise KeyError(
                    f"time_resolution_key '{time_resolution_key}' not in ensemble results for "
                    f"basin '{self._basin_name}'. Found keys: {list(basin_dict.keys())}")

            xr_dict = basin_dict[time_resolution_key]["xr"]
            obs_da = xr_dict[observed_key]
            sim_da = xr_dict[simulated_key]

            if self._is_mts:
                if time_resolution_key == "1D":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)
                elif time_resolution_key == "1H":
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.stack(stacked_time=("date", "time_step"))
                        sim_da = sim_da.stack(stacked_time=("date", "time_step"))
                        obs_da = obs_da.rename({"stacked_time": "time"})
                        sim_da = sim_da.rename({"stacked_time": "time"})

                self._observed = obs_da
                self._predictions = sim_da

            else:  #  Ensemble logic
                if "time_step" in obs_da.dims:
                    obs_da = obs_da.isel(time_step=0)
                    sim_da = sim_da.isel(time_step=0)

                self._observed = obs_da
                self._predictions = sim_da

    def _generate_obs_sim_plt(self, period='validation'):
        """
        Plot observed vs. simulated values (matplotlib).
        """
        if self._observed is None or self._predictions is None:
            print("[WARN:_generate_obs_sim_plt] => cannot plot, observed or predictions = None")
            return

        fig, ax = plt.subplots(figsize=(16, 10))
        if self._physics_informed:
            simulated_label = "HybridSimulation"
        else:
            simulated_label = "Simulated"

        if self._num_ensemble_members == 1:
            if "date" in self._observed.coords:
                ax.plot(self._observed["date"], self._observed, label="Observed", linewidth=1.5)
                ax.plot(self._predictions["date"], self._predictions, label=simulated_label, linewidth=1.5)
            else:
                print("[WARN:_generate_obs_sim_plt] => 'date' not in coords, cannot plot easily.")

        else:  # ensemble
            pass

        ax.set_ylabel(f"{self._target_variable} (units)", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_title(f"{self._basin_name} - {self._target_variable} Over Time ({period})", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def _generate_csv(self, period='validation', freq_key: str = None) -> Path:
        """
        Save predictions to a CSV file in the run directory, e.g. "results_output_validation_1H.csv".
        Only do special flattening/timestamp logic if self._is_mts == True and freq_key == '1H'.
        """
        if self._observed is None or self._predictions is None:
            print("[ERROR] Observed or predicted are None => skipping CSV.")
            return Path()

        base_name = f"results_output_{period}"
        if self._is_mts and freq_key:
            base_name += f"_{freq_key}"
        out = self._config.run_dir / f"{base_name}.csv"

        try:
            # SINGLE-MODEL RUN
            if self._num_ensemble_members == 1:
                # MTS 1H
                if self._is_mts and freq_key == "1H":
                    obs_df = self._observed.reset_index(self._observed.dims).to_dataframe(name="Observed")
                    sim_df = self._predictions.reset_index(self._predictions.dims).to_dataframe(name="Predicted")
                    merged_df = obs_df.join(sim_df, how="inner", lsuffix="_obs", rsuffix="_sim")

                    if "date_obs" in merged_df.columns and "time_step_obs" in merged_df.columns:
                        merged_df["Date"] = pd.to_datetime(merged_df["date_obs"]) \
                                            + pd.to_timedelta(merged_df["time_step_obs"], unit="h")

                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

                    elif "time" in merged_df.columns and merged_df["time"].dtype == "object":
                        def combine_tuple(tup):
                            base_ts, hour_offset = tup
                            return pd.to_datetime(base_ts) + pd.to_timedelta(hour_offset, unit="h")

                        merged_df["Date"] = merged_df["time"].apply(combine_tuple)
                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

                    else:
                        possible_timecols = [c for c in merged_df.columns if "time" in c]
                        if possible_timecols:
                            time_col = possible_timecols[0]
                            merged_df.rename(columns={time_col: "Date"}, inplace=True)
                            final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                        else:
                            final_df = merged_df[["Observed", "Predicted"]]

                    df = final_df.reset_index(drop=True)

                else:
                    if "date" in self._observed.coords:
                        obs_times = self._observed["date"].values
                        sim_times = self._predictions["date"].values

                        df = pd.DataFrame({
                            "Date": obs_times,
                            "Observed": self._observed.values,
                            "Predicted": self._predictions.values
                        })
                    else:
                        df = pd.DataFrame({
                            "Date": range(len(self._observed)),
                            "Observed": self._observed.values,
                            "Predicted": self._predictions.values
                        })
            # ENSEMBLE
            else:
                df = pd.DataFrame({
                    "Date": self._observed["datetime"].values,
                    "Observed": self._observed.values,
                    "Predicted": self._predictions.values
                })

            df.to_csv(out, index=False)

        except Exception as exc:
            print(f"[ERROR:_generate_csv] => Could not save CSV: {exc}")

        return out

    def _train_model(self) -> Path:
        """Train a single model instance with start_training()."""
        start_training(self._config)
        return self._config.run_dir

    def _train_ensemble(self) -> List[Path]:
        """Train multiple models as an ensemble."""
        run_dirs = []
        for i in range(self._num_ensemble_members):
            path = self._train_model()
            run_dirs.append(path)

        for rd in run_dirs:
            self._eval_model(rd, period="validation")
            self._eval_model(rd, period="test")

        return run_dirs

    def _create_config(self) -> Config:
        """
        Create a Config in dev_mode, apply notebook-provided hyperparams/flags,
        and (optionally) route NH runs into a notebook-specified parent folder.
        """
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {self._yaml_path}")

        config = Config(self._yaml_path, dev_mode=True)

        if 'save_weights_every' not in self._hyperparams:
            self._hyperparams['save_weights_every'] = self._hyperparams['epochs']

        if self._dynamic_inputs is not None:
            config.update_config({'dynamic_inputs': self._dynamic_inputs}, dev_mode=True)

        if self._extended_train_period:
            config.update_config({'train_end_date': config.validation_end_date}, dev_mode=True)

        config.update_config(self._hyperparams, dev_mode=True)
        config.update_config({'data_dir': self._data_dir}, dev_mode=True)
        config.update_config({'physics_informed': self._physics_informed}, dev_mode=True)
        config.update_config({'hourly': self._hourly}, dev_mode=True)
        config.update_config({'is_mts': self._is_mts}, dev_mode=True)
        config.update_config({'is_mts_data': self._is_mts_data}, dev_mode=True)
        config.update_config({'verbose': self._verbose}, dev_mode=True)

        if self._physics_informed and self._physics_data_file:
            config.update_config({'physics_data_file': self._physics_data_file}, dev_mode=True)

        # Device selection (unchanged)
        if self._gpu == 0:
            selected_device = "cuda:0"
            if self._verbose:
                print("[UCB Trainer] Using CUDA device: 'cuda:0'")
        elif self._gpu == -2:
            selected_device = "cpu"
            if self._verbose:
                print("[UCB Trainer] Forcing CPU (gpu=-2).")
        else:
            selected_device = "cpu"
            if self._verbose:
                print(f"[UCB Trainer] Using CPU (unhandled gpu={self._gpu}).")
        config.update_config({'device': selected_device}, dev_mode=True)

        if self._runs_parent is not None:
            run_base = Path(self._runs_parent).resolve()
            run_base.mkdir(parents=True, exist_ok=True)
            config.update_config({'run_dir': run_base}, dev_mode=True)

            if self._run_label is not None:
                config.update_config({'run_label': self._run_label}, dev_mode=True)
            if self._run_stamp is not None:
                config.update_config({'run_stamp': self._run_stamp}, dev_mode=True)

        self._config = config
        return config

    def calculate_pbias(self, observed, simulated):
        if observed.shape != simulated.shape:
            raise ValueError("Observed and simulated DataArrays must have the same shape.")

        pbias = ((observed - simulated).sum() / observed.sum()) * 100
        return pbias.item()

    @staticmethod
    def _clean_hms(csv: Path, *, debug: bool = False) -> pd.DataFrame:
        """
        Robustly clean an HMS DSS export (daily **or** hourly).

        Returns
        -------
        pd.DataFrame  with datetime index and single column 'HMS_Predicted'
        """

        def _dbg(*msg):
            if debug:
                print("[HMS‑DEBUG]", *msg)

        df = pd.read_csv(csv, header=None, dtype=str)
        hdr = df.apply(lambda r: r.str.contains("date", case=False, na=False).any(), axis=1).idxmax()
        df.columns = df.iloc[hdr].str.strip()
        df = df.iloc[hdr + 1:].copy()
        df.rename(columns={c: c.strip().lower() for c in df.columns}, inplace=True)
        day_col = next((c for c in df.columns if c.startswith("date")), None)
        time_col = next((c for c in df.columns if c.startswith("time")), None)
        if day_col is None:
            raise RuntimeError(f"'Date' column not found in {csv}")

        if "ordinate" in df.columns:
            df.drop(columns="ordinate", inplace=True)

        hourly_mode = (time_col is not None and df[time_col].notna().any() and df[time_col].str.strip().ne("").any())
        _dbg("hourly_mode:", hourly_mode)

        if hourly_mode:
            df[time_col] = df[time_col].fillna("").str.strip()
            mask24 = df[time_col] == "24:00:00"
            if mask24.any():
                df.loc[mask24, day_col] = (pd.to_datetime(df.loc[mask24, day_col], format="%d-%b-%y"
                                                          ) + timedelta(days=1)).dt.strftime("%d-%b-%y")
                df.loc[mask24, time_col] = "00:00:00"

            bad_mask = ~df[time_col].str.match(r"^\d{1,2}:\d{2}:\d{2}$")
            df.loc[bad_mask, time_col] = "00:00:00"

            df["date"] = (
                        pd.to_datetime(df[day_col], format="%d-%b-%y", errors="coerce") + pd.to_timedelta(df[time_col]))
        else:
            df["date"] = pd.to_datetime(df[day_col], format="%d-%b-%y", errors="coerce")

        df.dropna(subset=["date"], inplace=True)
        df.set_index("date", inplace=True)
        flow_col = next((c for c in df.columns if "flow" in c.lower()), None)

        if flow_col is None:
            raise RuntimeError(f"No column containing 'FLOW' in {csv}")

        tidy = (df[[flow_col]].apply(pd.to_numeric, errors="coerce").rename(columns={flow_col: "HMS_Predicted"}))

        return tidy


    def cross_validate(self, intervalMonth='October', intervalLength=2, validationLength=1, gap=False, run_path=None) -> dict:
        """
        This method performs an i fold cross validation where i = ([number of years in dataset] // intervalLength) - validationLength.
        This method is currently configured to train from the start of the test set to the end of the validation set in the corresponding CSV. 

        arguments:
            intervalMonth: optional, str, the month interval for defining a year. i.e. a water year from September 30th to October 1st. Default is 'October'.

            intervalLength: optional, int, the length of the initial fold in years, default is 2

            validationLength: optional, int, the length of the validation period in years, default is 1

            gap: optional, bool, whether to include a one-year gap between the training and validation periods. Default is False.
        """
        #create a crossval run folder
        now = datetime.now()
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        second = f"{now.second}".zfill(2)
        if not run_path:
            run_dir = Path().cwd() / "runs" / f"cross_validation_{day}{month}_{hour}{minute}{second}"
        else:
            run_dir = run_path / f"cross_validation_{day}{month}_{hour}{minute}{second}"
        run_dir.mkdir(parents=True, exist_ok=True)


        MonthsLib = {'january': 'Jan', 'febuary': 'Feb', 'march': 'Mar', 'april' : 'Apr', 'may' : 'May', 'june' : 'Jun', 'july' : 'Jul', 'august': 'Aug', 'september': 'Sep', 'october': 'Oct', 'december': 'Dec'}
        interval = MonthsLib[intervalMonth.lower()]

        cross_val_results = {}

        gap = int(gap)

        #optionally adjust start and end dates based on the YAML configuration
        original_start = getattr(self._config, "train_start_date", None)
        original_start_year = int(original_start.year)

        original_end = getattr(self._config, "validation_end_date", None)
        original_end_year = int(original_end.year)

        original_validation_end = getattr(self._config, "validation_end_date", None)

        n_years = original_end_year - original_start_year + 1
        max_fold = (n_years - 2 - int(gap)) // 2 - validationLength

        i = 1
        while i <= max_fold:
            self._config.update_config({'train_start_date': pd.to_datetime(f"{str(original_start_year)}-{interval}-01", format="%Y-%b-%d")}, dev_mode=True)
            self._config.update_config({'train_end_date': pd.to_datetime(f"{str(original_start_year + (intervalLength * i))}-{interval}-01", format="%Y-%b-%d")}, dev_mode=True)
            self._config.update_config({'validation_start_date': pd.to_datetime(f"{str(original_start_year + (intervalLength * i) + gap)}-{interval}-02", format="%Y-%b-%d")}, dev_mode=True)
            self._config.update_config({'validation_end_date': pd.to_datetime(f"{str(original_start_year + (intervalLength * i + validationLength) + gap)}-{interval}-01", format="%Y-%b-%d")}, dev_mode=True)
            self._config.update_config({'run_dir': run_dir}, dev_mode=True)
            
            self.train()

            time_resolution_key = '1h' if self._hourly else '1D'
            self._get_predictions(time_resolution_key, 'validation')
            metrics = calculate_all_metrics(self._observed, self._predictions)
            
            cross_val_results[i] = metrics

            i += 1

        if original_start:
            self._config.update_config({'train_start_date': original_start})
        if original_end:
            self._config.update_config({'train_end_date': original_end})
        if original_validation_end:
            self._config.update_config({'validation_end_date': original_validation_end})

        for j in range(1, len(cross_val_results) + 1):
            print(f"Fold {j} results")
            print(cross_val_results[j])
            print("\n") 

        output = {}
        for j in cross_val_results:
            for metric in cross_val_results[j]:
                key = f"avg {metric}"
                if key not in output:
                    output[key] = []
                output[key].append(cross_val_results[j][metric])
        
        for key in output:
            output[key] = sum(output[key]) / len(output[key])
        
        return output