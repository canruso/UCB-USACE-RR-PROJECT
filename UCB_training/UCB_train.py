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
import yaml
from UCB_training.UCB_utils import data_dir as _ucb_data_dir, resolve_basin_file as _ucb_resolve_basin_file
from UCB_training.UCB_plotting import plot_loss_curves

def _train_single_member(args):
    trainer, idx = args

    trainer = pickle.loads(pickle.dumps(trainer))

    tag = "phys" if trainer._physics_informed else "nophys"
    member_run = Path(trainer._runs_parent) / f"{tag}_member_run_{idx+1:02d}"
    member_run.mkdir(parents=True, exist_ok=True)

    trainer._config.update_config({'run_dir': member_run}, dev_mode=True)

    path = trainer._train_model()

    return path

def _train_single_bootstrap_member(args):
    # needed to print from worker.
    import sys
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    trainer, idx = args

    # deep copy trainer for isolation
    trainer = pickle.loads(pickle.dumps(trainer))

    tag = "phys" if trainer._physics_informed else "nophys"
    member_run = Path(trainer._runs_parent) / f"{tag}_member_run_{idx+1:02d}"
    member_run.mkdir(parents=True, exist_ok=True)

    trainer._config.update_config({'run_dir': member_run}, dev_mode=True)

    # First member uses ORIGINAL ranges
    if trainer._bootstrap_model and idx > 0:
        trainer._bootstrap_train_ranges(idx + 1)

    if trainer._verbose:
        ranges = trainer._config._cfg.get("train_ranges", [])
        hdr = "[bootstrap member {:02d}]".format(idx + 1)
        print(f"\n{hdr} TRAIN RANGES ({len(ranges)} blocks):")
        for r in ranges:
            print(" ", r)

    path = trainer._train_model()

    if trainer._bootstrap_model and idx > 0:
        trainer._restore_train_ranges()

    return path

class UCB_trainer:
    """
    A wrapper to facilitate easier training/evaluation of neural hydrology models.
    """

    def __init__(self, path_to_csv_folder: Path, yaml_path: Path, hyperparams: dict, input_features: List[str] = None,
                 num_ensemble_members: int = 1, bootstrap_model: bool = False, physics_informed: bool = False, physics_data_file: Path = None,
                 hourly: bool = False, extend_train_period: bool = False, gpu: int = -1, is_mts: bool = False,
                 is_mts_data: bool = False, basin: bool = None, verbose: bool = True, runs_parent: Path = None,
                 run_label: str = None, run_stamp: str = None, experiment_tag: str = None, adaboost_ensemble: bool = False):
        """
        Initialize the UCB_trainer class with configurations and training parameters.
        """
        self._hyperparams = hyperparams
        self._num_ensemble_members = num_ensemble_members
        self._bootstrap_model = bootstrap_model

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

        self._adaboost_ensemble = adaboost_ensemble
        self._ada_alphas = []

        self._adaboost_cfg = {
            "eta": 0.5,
            "alpha_clip": 2.0,
            "eps": 1e-8,
            "use_validation": True,
        }

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
        self._experiment_tag = experiment_tag
        self._create_config()

    # Added for synthetic data functions:
    def _is_synthetic_rr(self) -> bool:
        """
        Is using synthetic russian river dates
        """
        try:
            return str(self._config.dataset).lower() == "synthetic_russian_river"
        except Exception:
            return False
        
    def _compute_nse(self, obs: xr.DataArray, sim: xr.DataArray) -> float:
        mask = np.isfinite(obs.values) & np.isfinite(sim.values)
        o = obs.values[mask]
        s = sim.values[mask]
        if len(o) == 0:
            return -np.inf
        denom = np.sum((o - o.mean()) ** 2)
        if denom == 0:
            return -np.inf
        return 1.0 - np.sum((o - s) ** 2) / denom
    
    def _adaboost_block_errors(self):
        """
        Compute NSE error per synthetic year block.
        Returns dict {block_index: error}
        """

        coord = (
            "date" if "date" in self._observed.coords
            else "time" if "time" in self._observed.coords
            else "datetime" if "datetime" in self._observed.coords
            else None
        )

        if coord is None:
            raise RuntimeError(f"[AdaBoost] Could not find time coord in observed: {list(self._observed.coords)}")

        dates = pd.to_datetime(self._observed.coords[coord].values)

        print("[AdaDbg] years seen:", np.unique(dates.year))
        years = dates.year

        cfg = self._config._cfg
        train_ranges = list(cfg["train_ranges"])
        k = len(train_ranges)

        errs = {}
        for i in range(k):
            y = 2200 + i
            mask = years == y

            if not mask.any():
                errs[i] = 1.0
                if self._verbose:
                    print(f"[AdaDbg] block {i} year={y} EMPTY → err=1.0")
                continue

            obs = self._observed.sel({coord: mask})
            sim = self._predictions.sel({coord: mask})

            nse = self._compute_nse(obs, sim)

            clipped = max(-1.0, min(1.0, nse))
            err = np.clip((1.0 - clipped) * 0.5, 0.0, 1.0)

            errs[i] = err

            if self._verbose:
                print(
                    f"[AdaDbg] block {i} year={y} "
                    f"N={obs.size} "
                    f"NSE={nse:.6f} "
                    f"clipped={clipped:.6f} "
                    f"err={err:.6f}"
                )

        return errs
        
    def _adaboost_update_weights(self, prev_weights, block_errs):
        """
        Standard AdaBoost weight update using block NSE.
        """

        eps = self._adaboost_cfg["eps"]

        k = len(prev_weights)
        e = np.array([block_errs[i] for i in range(k)])

        weighted_err = np.sum(prev_weights * e)
        weighted_err = np.clip(weighted_err, eps, 1 - eps)

        alpha = self._adaboost_cfg["eta"] * np.log((1 - weighted_err) / weighted_err)
        alpha = np.clip(alpha, -self._adaboost_cfg["alpha_clip"], self._adaboost_cfg["alpha_clip"])

        new_w = prev_weights * np.exp(alpha * e)
        new_w = new_w / new_w.sum()

        self._ada_alphas.append(alpha)

        return new_w, alpha
    
    def _adaboost_resample_train_ranges(self, weights, member_idx=None):
        """
        Increase dataset size via weighted resampling of synthetic year blocks.
        Never removes years.
        """

        cfg = self._config._cfg
        base = list(cfg["train_ranges"])
        k = len(base)

        # expand dataset to ~1.5x
        target = int(np.ceil(1.5 * k))

        sampled = [str(x) for x in np.random.choice(base, size=target, replace=True, p=weights)]

        cfg["train_ranges"] = sampled

        if self._verbose:
            print(f"\n[AdaBoost member {member_idx}] block weights:")
            for i, w in enumerate(weights):
                print(f"  block {i} (year {2200+i}): w={w:.4f}")

            print(f"[AdaBoost member {member_idx}] resampled train_ranges ({len(sampled)}):")
            for r in sampled:
                print(" ", r)

    def _adaboost_aggregate(self, sims: list[xr.DataArray]) -> xr.DataArray:
        alphas = np.array(self._ada_alphas)
        alphas = alphas / np.sum(np.abs(alphas))

        stacked = xr.concat(sims, dim="member")
        w = xr.DataArray(alphas, dims=["member"])

        return (stacked * w).sum("member")

    # Added 1/28/2026 for non-consecutive dates:
    def _restore_synthetic_dates(self, period: str, time_res_key: str):
        """
        Map 22XX synthetic dates back to real dates for disjoint ranges.
        Also inserts NaNs between disjoint blocks so plots are discontinuous.
        Applies to self._observed and self._predictions in-place.
        """

        cfg_dict = getattr(self._config, "_cfg", {}) or {}
        ranges = cfg_dict.get(f"{period}_ranges", None)

        if not ranges:
            return

        if getattr(self, "_verbose", False):
            print(f"[UCB_trainer] Restoring synthetic {period} dates for analysis...")

        is_hourly = (str(time_res_key).upper() == "1H") or (self._hourly and str(time_res_key).upper() != "1D")
        freq = "1H" if is_hourly else "1D"

        def _remove_leap_days(idx):
            return idx[~((idx.month == 2) & (idx.day == 29))]

        def _parse_range(r):
            if isinstance(r, dict):
                return r.get("start_date"), r.get("end_date")
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                return r[0], r[1]
            if isinstance(r, str) and "-" in r:
                s, e = r.split("-", 1)
                return s.strip(), e.strip()
            raise ValueError(f"Unrecognized range format: {r}")

        # build synthetic => real datetime map
        date_map = {}

        for i, r in enumerate(ranges):
            start, end = _parse_range(r)

            start_dt = pd.to_datetime(start, dayfirst=True, errors="raise")
            end_dt   = pd.to_datetime(end,   dayfirst=True, errors="raise")

            real_idx = pd.date_range(start=start_dt, end=end_dt, freq=freq)
            real_idx = _remove_leap_days(real_idx)

            syn_year = 2200 + i
            syn_start = pd.Timestamp(f"{syn_year}-01-01")

            syn_idx = pd.date_range(start=syn_start, periods=len(real_idx), freq=freq)

            # remove leap year days
            if syn_idx.is_leap_year.any():
                pad = 48 if freq == "1H" else 2
                padded = pd.date_range(start=syn_start, periods=len(real_idx) + pad, freq=freq)
                padded = _remove_leap_days(padded)
                syn_idx = padded[:len(real_idx)]

            for s, rdt in zip(syn_idx, real_idx):
                date_map[s] = rdt

        # helper: insert NaNs to force plot discontinuity
        def _insert_nan_gaps(da, max_gap):
            if "date" in da.coords:
                coord = "date"
            elif "time" in da.coords:
                coord = "time"
            elif "datetime" in da.coords:        # ← ADD THIS
                coord = "datetime"
            else:
                raise RuntimeError(
                    f"No temporal coordinate found. dims={da.dims}, coords={list(da.coords)}"
                )

            t = pd.to_datetime(da.coords[coord].values)

            dt = np.diff(t.values.astype("datetime64[s]").astype(np.int64))
            gap_sec = max_gap.total_seconds()
            gap_locs = np.where(dt > gap_sec)[0]

            if len(gap_locs) == 0:
                return da

            new_vals = []
            new_times = []

            for i in range(len(da)):
                new_vals.append(da.values[i])
                new_times.append(t[i])
                if i in gap_locs:
                    new_vals.append(np.nan)
                    new_times.append(t[i] + pd.Timedelta(seconds=1))

            return xr.DataArray(
                np.array(new_vals),
                dims=[coord],
                coords={coord: pd.DatetimeIndex(new_times)}
            )

        for attr_name in ["_observed", "_predictions"]:
            da = getattr(self, attr_name, None)
            if da is None:
                continue

            # Case A: MTS hourly stacked MultiIndex
            if "time" in da.coords and isinstance(da.coords["time"].to_index(), pd.MultiIndex):
                mi = da.coords["time"].to_index()
                base_dates = pd.to_datetime(mi.get_level_values(0))
                hours = mi.get_level_values(1).astype(int)
                abs_times = base_dates + pd.to_timedelta(hours, unit="h")

                mapped = pd.Series(abs_times).map(date_map)
                valid = mapped.notna().values

                if not valid.all():
                    da = da.isel(time=valid)
                    mapped = mapped[valid]

                da = da.assign_coords(time=mapped.values)

                # rename to 'date' for NH metrics + plotting work
                da = da.rename({"time": "date"})

                gap = pd.Timedelta(hours=2)
                da = _insert_nan_gaps(da, gap)

                setattr(self, attr_name, da)
                continue

            coord = (
                "date" if "date" in da.coords
                else "time" if "time" in da.coords
                else "datetime" if "datetime" in da.coords
                else None
            )
            if coord is None:
                print(f"[WARN] {attr_name} has no time coord; skipping date restore.")
                continue

            cur = pd.to_datetime(da.coords[coord].values, errors="coerce")
            mapped = pd.Series(cur).map(date_map)

            valid = mapped.notna().values
            if not valid.all():
                print(f"[WARN] Dropping {(~valid).sum()} timesteps not in date_map.")
                da = da.isel({coord: valid})
                mapped = mapped[valid]

            da = da.assign_coords({coord: mapped.values})

            gap = pd.Timedelta(days=2)
            da = _insert_nan_gaps(da, gap)

            setattr(self, attr_name, da)

        # sanity check
        if getattr(self, "_verbose", False):
            for name in ["_observed", "_predictions"]:
                da = getattr(self, name, None)
                if da is None:
                    continue
                cn = "date" if "date" in da.coords else ("time" if "time" in da.coords else None)
                if cn:
                    vals = pd.to_datetime(da.coords[cn].values)
                    print(f"[UCB_trainer] {name} restored coord '{cn}': {vals.min()} → {vals.max()}")

    def _prepare_balanced_bootstrap(self):
        """
        Precompute balanced block assignments for entire ensemble.
        Each original block appears exactly M times across ensemble.
        """

        cfg = self._config._cfg
        base = list(cfg["train_ranges"])
        N = len(base)
        M = self._num_ensemble_members

        slots = np.repeat(np.arange(N), M)

        np.random.shuffle(slots)

        assignments = slots.reshape(M, N)

        self._balanced_bootstrap_assignments = assignments
        self._orig_train_ranges = base

    def _bootstrap_train_ranges(self, member_idx: int = None):
        """
        Balanced bootstrap: each block appears exactly M times across ensemble.
        """

        if not hasattr(self, "_balanced_bootstrap_assignments"):
            self._prepare_balanced_bootstrap()

        cfg = self._config._cfg
        base = self._orig_train_ranges

        idx = member_idx - 1  # member_idx starts at 1
        block_indices = self._balanced_bootstrap_assignments[idx]

        boot = [base[i] for i in block_indices]
        cfg["train_ranges"] = boot

        if self._verbose:
            print(f"\n[balanced bootstrap member {member_idx}]")
            for r in boot:
                print(" ", r)

    def _restore_train_ranges(self):
        if hasattr(self, "_orig_train_ranges"):
            self._config._cfg["train_ranges"] = list(self._orig_train_ranges)

    def _save_member_prediction_csv(
        self,
        run_dir: Path,
        period: str,
        member_idx: int,
        sim: xr.DataArray,
        obs: xr.DataArray,
    ):

        root = self._runs_parent

        top = self._run_label or self._run_stamp or "default_run"
        exp = self._experiment_tag or "unnamed_experiment"
        member_dir = f"member_{member_idx:02d}"

        out_dir = root / "ensemble_predictions" / top / exp / member_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        coord = "date" if "date" in sim.coords else sim.dims[0]

        sim_df = sim.to_dataframe(name="Predicted").reset_index(drop=True)
        obs_df = obs.to_dataframe(name="Observed").reset_index(drop=True)

        df = pd.concat([sim_df, obs_df], axis=1)

        # ------------------------------------------

        fname = out_dir / f"{period}.csv"
        df.to_csv(fname, index=False)

        if self._verbose:
            print(f"[ensemble] saved → {fname}")   


    def train(self):
        """
        Train the model or ensemble based on the specified number of ensemble members.
        """
        if self._num_ensemble_members == 1:
            path = self._train_model()
            self._eval_model(path, period="validation")
            self._model = path

            # Generate loss curve visualization
            try:
                plot_loss_curves(path)
                if self._verbose:
                    print(f"[UCB Trainer] Loss curves saved to {path}/loss_curves.png")
            except Exception as e:
                if self._verbose:
                    print(f"[UCB Trainer] Warning: Could not generate loss curves: {e}")
        else:
            self._model = self._train_ensemble()
            for model_path in self._model:
                self._eval_model(model_path, period="validation")

                # Generate loss curve visualization for each ensemble member
                try:
                    plot_loss_curves(model_path)
                    if self._verbose:
                        print(f"[UCB Trainer] Loss curves saved to {model_path}/loss_curves.png")
                except Exception as e:
                    if self._verbose:
                        print(f"[UCB Trainer] Warning: Could not generate loss curves: {e}")

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
            replace_run_config_paths(run_dir, verbose=self._verbose)
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
                replace_run_config_paths(rd, verbose=self._verbose)
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

    def _get_last_epoch(self, run_directory: Path) -> int:
        """Find the actual last epoch that was saved (in case early stopping occurred)."""
        model_files = sorted(list(run_directory.glob('model_epoch*.pt')))
        if not model_files:
            # No checkpoint found, return configured epochs as fallback
            return self._config.epochs
        # Extract epoch number from last checkpoint filename (e.g., model_epoch025.pt -> 25)
        last_file = model_files[-1]
        epoch_str = last_file.stem.replace('model_epoch', '')
        return int(epoch_str)

    def _eval_model(self, run_directory: Path, period="validation"):
        eval_run(run_dir=run_directory, period=period)

        actual_epoch = self._get_last_epoch(run_directory)
        results_file = run_directory / period / f"model_epoch{str(actual_epoch).zfill(3)}" / f"{period}_results.p"
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

    def _get_predictions(self, time_resolution_key, period='validation', is_ensemble_member=False):
        """
            Load predictions from the model's result files for the given period.
            Only apply multi-timescale flattening logic if self._is_mts is True.
            Otherwise, keep old behavior.
            """
        if isinstance(self._model, (str, Path)):
            actual_epoch = self._get_last_epoch(self._model)
            results_file = self._model / period / (f"model_"
                                                   f"epoch{str(actual_epoch).zfill(3)}") / f"{period}_results.p"
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
            # --------------------------------------------------
            # AdaBoost synthetic ensemble (weighted aggregation)
            # --------------------------------------------------
            if self._adaboost_ensemble and self._is_synthetic_rr():

                sims = []
                obs_ref = None

                for rd in self._model:

                    actual_epoch = self._get_last_epoch(rd)
                    results_file = (
                        rd / period /
                        f"model_epoch{str(actual_epoch).zfill(3)}" /
                        f"{period}_results.p"
                    )

                    if not results_file.exists():
                        self._eval_model(rd, period)

                    with open(results_file, "rb") as fp:
                        results = pickle.load(fp)

                    basin_name = next(iter(results.keys()))
                    basin_dict = results[basin_name]

                    self._target_variable = self._config.target_variables[0]
                    obs_key = f"{self._target_variable}_obs"
                    sim_key = f"{self._target_variable}_sim"

                    xr_dict = basin_dict[time_resolution_key]["xr"]

                    obs_da = xr_dict[obs_key]
                    sim_da = xr_dict[sim_key]

                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)

                    sims.append(sim_da)

                    if obs_ref is None:
                        obs_ref = obs_da

                # Apply AdaBoost weighting
                boosted = self._adaboost_aggregate(sims)

                self._predictions = boosted
                self._observed = obs_ref

                # Restore synthetic dates if needed
                if self._is_synthetic_rr():
                    self._restore_synthetic_dates(period, time_resolution_key)

            # --------------------------------------------------
            # Normal mean ensemble
            # --------------------------------------------------
            else:

                results = create_results_ensemble(run_dirs=self._model, period=period)

                self._basin_name = next(iter(results.keys()))
                basin_dict = results[self._basin_name]

                self._target_variable = self._config.target_variables[0]
                obs_key = f"{self._target_variable}_obs"
                sim_key = f"{self._target_variable}_sim"

                xr_dict = basin_dict[time_resolution_key]["xr"]

                obs_da = xr_dict[obs_key]
                sim_da = xr_dict[sim_key]

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
                    if "time_step" in obs_da.dims:
                        obs_da = obs_da.isel(time_step=0)
                        sim_da = sim_da.isel(time_step=0)

                self._observed = obs_da
                self._predictions = sim_da


        if self._is_synthetic_rr() and not is_ensemble_member:
            self._restore_synthetic_dates(period, time_resolution_key)

    # Edited 1/28/2026 for non-consecutive date plotting:
    def _generate_obs_sim_plt(self, period='validation'):
        """
        Plot observed vs. simulated values (matplotlib).
        If dates are non-consecutive, collapse time axis and draw jump markers with gap size.
        """

        if self._observed is None or self._predictions is None:
            print("[WARN:_generate_obs_sim_plt] => cannot plot, observed or predictions = None")
            return

        fig, ax = plt.subplots(figsize=(16, 10))

        simulated_label = "HybridSimulation" if self._physics_informed else "Simulated"

        if self._num_ensemble_members == 1:
            if "date" in self._observed.coords:

                dates = pd.to_datetime(self._observed["date"].values)
                y_obs = self._observed.values
                y_sim = self._predictions.values

                # detect gaps
                dt_days = np.diff(dates.values.astype("datetime64[D]").astype(int))
                gap_idx = np.where(dt_days > 2)[0]

                has_gaps = len(gap_idx) > 0

                if has_gaps:
                    x = np.arange(len(dates))

                    ax.plot(x, y_obs, label="Observed", linewidth=1.5)
                    ax.plot(x, y_sim, label=simulated_label, linewidth=1.5)

                    # draw vertical lines + annotate gap
                    for i in gap_idx:
                        xpos = i + 0.5
                        gap_days = int(dt_days[i])

                        ax.axvline(x=xpos, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

                        ax.text(
                            xpos, ax.get_ylim()[1],
                            f"+{gap_days} days",
                            color="red",
                            fontsize=11,
                            rotation=90,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2)
                        )

                    # sparse real-date labels
                    step = max(len(x) // 10, 1)
                    ticks = np.arange(0, len(x), step)
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(
                        [dates[i].strftime("%Y-%m-%d") for i in ticks],
                        rotation=30, ha="right"
                    )

                    ax.set_xlabel("Date (discontinuous)")

                else:
                    ax.plot(dates, y_obs, label="Observed", linewidth=1.5)
                    ax.plot(dates, y_sim, label=simulated_label, linewidth=1.5)
                    ax.set_xlabel("Date")

            else:
                print("[WARN:_generate_obs_sim_plt] => 'date' not in coords, cannot plot easily.")

        else:
            key = "date" if "date" in self._observed.coords else "datetime"

            dates = pd.to_datetime(self._observed[key].values)
            y_obs = self._observed.values
            y_sim = self._predictions.values

            dt_days = np.diff(dates.values.astype("datetime64[D]").astype(int))
            gap_idx = np.where(dt_days > 2)[0]

            has_gaps = len(gap_idx) > 0

            if has_gaps:
                x = np.arange(len(dates))

                ax.plot(x, y_obs, label="Observed", linewidth=1.5)
                ax.plot(x, y_sim, label=simulated_label, linewidth=1.5)

                for i in gap_idx:
                    xpos = i + 0.5
                    gap_days = int(dt_days[i])

                    ax.axvline(x=xpos, color="red", linestyle="--", alpha=0.7, linewidth=1.5)

                    ax.text(
                        xpos, ax.get_ylim()[1],
                        f"+{gap_days} days",
                        color="red",
                        fontsize=11,
                        rotation=90,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2)
                    )

                step = max(len(x) // 10, 1)
                ticks = np.arange(0, len(x), step)

                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [dates[i].strftime("%Y-%m-%d") for i in ticks],
                    rotation=30,
                    ha="right"
                )

                ax.set_xlabel("Date (discontinuous)")

            else:
                ax.plot(dates, y_obs, label="Observed", linewidth=1.5)
                ax.plot(dates, y_sim, label=simulated_label, linewidth=1.5)
                ax.set_xlabel("Date")

        ax.set_ylabel(f"{self._target_variable} (units)", fontsize=14)
        ax.set_title(f"{self._basin_name} - {self._target_variable} Over Time ({period})", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
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

        def _combine_tuple(tup):
            base_ts, hour_offset = tup
            return pd.to_datetime(base_ts) + pd.to_timedelta(hour_offset, unit="h")

        def _finalize_df(df):
            return df.reset_index(drop=True)

        def _build_simple_df():
            if "date" in self._observed.coords:
                return pd.DataFrame({
                    "Date": self._observed["date"].values,
                    "Observed": self._observed.values,
                    "Predicted": self._predictions.values
                })
            else:
                return pd.DataFrame({
                    "Date": range(len(self._observed)),
                    "Observed": self._observed.values,
                    "Predicted": self._predictions.values
                })

        def _build_ensemble_df():
            return pd.DataFrame({
                "Date": self._observed["datetime"].values,
                "Observed": self._observed.values,
                "Predicted": self._predictions.values
            })

        def _process_mts_1h(merged_df):
            if "date_obs" in merged_df.columns and "time_step_obs" in merged_df.columns:
                merged_df["Date"] = (
                    pd.to_datetime(merged_df["date_obs"])
                    + pd.to_timedelta(merged_df["time_step_obs"], unit="h")
                )
                final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

            elif "time" in merged_df.columns and merged_df["time"].dtype == "object":
                merged_df["Date"] = merged_df["time"].apply(_combine_tuple)
                final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")

            else:
                possible_timecols = [c for c in merged_df.columns if "time" in c]
                if possible_timecols:
                    time_col = possible_timecols[0]
                    merged_df.rename(columns={time_col: "Date"}, inplace=True)
                    final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                else:
                    if "date_obs" in merged_df.columns:
                        merged_df["Date"] = pd.to_datetime(merged_df["date_obs"])
                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                    elif "Date" in merged_df.columns:
                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                    elif "date" in merged_df.columns:
                        merged_df["Date"] = merged_df["date"]
                        final_df = merged_df[["Date", "Observed", "Predicted"]].sort_values("Date")
                    else:
                        raise RuntimeError("Could not determine Date column when writing CSV")

            return _finalize_df(final_df)

        try:
            # SINGLE-MODEL RUN
            if self._num_ensemble_members == 1:
                # MTS 1H
                if self._is_mts and freq_key == "1H":
                    if self._is_synthetic_rr():
                        obs_df = self._observed.to_dataframe(name="Observed")
                        sim_df = self._predictions.to_dataframe(name="Predicted")
                        merged_df = obs_df.join(sim_df, how="inner").reset_index()
                        df = _process_mts_1h(merged_df)
                    else:
                        obs_df = self._observed.reset_index(self._observed.dims).to_dataframe(name="Observed")
                        sim_df = self._predictions.reset_index(self._predictions.dims).to_dataframe(name="Predicted")
                        merged_df = obs_df.join(sim_df, how="inner", lsuffix="_obs", rsuffix="_sim")
                        df = _process_mts_1h(merged_df)
                else:
                    df = _build_simple_df()
            # ENSEMBLE
            else:
                df = _build_ensemble_df()

            df.to_csv(out, index=False)

        except Exception as exc:
            print(f"[ERROR:_generate_csv] => Could not save CSV: {exc}")

        return out


    def _train_model(self) -> Path:
        """Train a single model instance with start_training()."""
        start_training(self._config)
        return self._config.run_dir

    
    def _train_ensemble(self) -> List[Path]:
        run_dirs = []

        if self._is_synthetic_rr() and self._adaboost_ensemble:
            if not self._adaboost_ensemble:
                raise RuntimeError("Synthetic ensembles now require adaboost_ensemble=True")

            cfg = self._config._cfg
            base_ranges = list(cfg["train_ranges"])
            k = len(base_ranges)

            # initial uniform weights
            block_weights = np.ones(k) / k

            sims_for_aggregation = []

            for i in range(self._num_ensemble_members):

                tag = "phys" if self._physics_informed else "nophys"
                member_run = Path(self._runs_parent) / f"{tag}_member_run_{i+1:02d}"
                member_run.mkdir(parents=True, exist_ok=True)

                self._config.update_config({'run_dir': member_run}, dev_mode=True)

                # first model: no resampling
                if i > 0:
                    self._adaboost_resample_train_ranges(block_weights, i+1)

                path = self._train_model()
                run_dirs.append(path)

                self._model = path

                # needed for AdaBoost weighting
                self._eval_model(path, period="train")

                # REQUIRED so create_results_ensemble(period="test") works later
                self._eval_model(path, period="test")

                self._get_predictions("1D", "train",True)

                # compute block errors + update weights AFTER first model
                if i >= 0:
                    errs = self._adaboost_block_errors()
                    block_weights, alpha = self._adaboost_update_weights(block_weights, errs)

                    if self._verbose:
                        print(f"\n[AdaBoost member {i+1}] alpha = {alpha:.4f}")

                sims_for_aggregation.append(self._predictions.copy())

                # restore original ranges for next round
                cfg["train_ranges"] = list(base_ranges)

            # final aggregation
            self._predictions = self._adaboost_aggregate(sims_for_aggregation)

            if self._is_synthetic_rr():
                self._restore_synthetic_dates("validation", "1D")

            # save ensemble prediction CSV (synthetic only)
            self._save_member_prediction_csv(
                run_dirs[-1],
                "adaboost_validation",
                self._num_ensemble_members,
                self._predictions,
                self._observed,
            )

        elif self._is_synthetic_rr() and self._bootstrap_model:

            import multiprocessing as mp

            n = self._num_ensemble_members
            workers = min(mp.cpu_count(), n)

            nested = mp.current_process().daemon

            if nested:
                if self._verbose:
                    print("[ensemble] nested multiprocessing detected → running ensemble serially")

                run_dirs = []
                for i in range(n):
                    run_dirs.append(_train_single_bootstrap_member((self, i)))

            else:
                if self._verbose:
                    print(f"[bootstrap ensemble] launching {n} members on {workers} cores")

                with mp.get_context("spawn").Pool(workers) as pool:
                    run_dirs = pool.map(
                        _train_single_bootstrap_member,
                        [(self, i) for i in range(n)]
                    )

            for idx, rd in enumerate(run_dirs, 1):
                self._eval_model(rd, period="validation")
                self._eval_model(rd, period="test")

                # load config ONCE per member
                cfg = Config(rd / "config.yml")
                tgt = cfg.target_variables[0]

                for per in ("validation", "test"):
                    ep = self._get_last_epoch(rd)

                    pkl = rd / per / f"model_epoch{ep:03d}" / f"{per}_results.p"
                    if not pkl.exists():
                        if self._verbose:
                            print(f"[ensemble] missing results: {pkl}")
                        continue

                    with open(pkl, "rb") as f:
                        res = pickle.load(f)

                    basin = next(iter(res))

                    for freq in res[basin]:   # <-- loop both 1H and 1D

                        xr_d = res[basin][freq]["xr"]

                        sim = xr_d[f"{tgt}_sim"]
                        obs = xr_d[f"{tgt}_obs"]

                        if freq == "1D" and "time_step" in sim.dims:
                            sim = sim.isel(time_step=0)
                            obs = obs.isel(time_step=0)

                            # CRITICAL: also drop leftover coord
                            sim = sim.drop_vars("time_step", errors="ignore")
                            obs = obs.drop_vars("time_step", errors="ignore")

                        elif freq == "1H" and "time_step" in sim.dims:
                            sim = sim.stack(datetime=("date", "time_step"))
                            obs = obs.stack(datetime=("date", "time_step"))

                            mi = sim["datetime"].to_index()
                            dt = pd.to_datetime(mi.get_level_values(0)) + pd.to_timedelta(
                                mi.get_level_values(1), unit="h"
                            )

                            sim = sim.assign_coords(datetime=dt).rename({"datetime": "date"})
                            obs = obs.assign_coords(datetime=dt).rename({"datetime": "date"})

                            sim = sim.drop_vars("time_step", errors="ignore")
                            obs = obs.drop_vars("time_step", errors="ignore")

                        self._observed = obs
                        self._predictions = sim
                        self._restore_synthetic_dates(per, freq)
                        obs = self._observed
                        sim = self._predictions


                        self._save_member_prediction_csv(
                            rd,
                            f"{per}_{freq}",
                            idx,
                            sim,
                            obs,
                        )

        else:
            import multiprocessing as mp

            n = self._num_ensemble_members
            workers = min(mp.cpu_count(), n)

            nested = mp.current_process().daemon

            if nested:
                if self._verbose:
                    print("[ensemble] nested multiprocessing detected → running ensemble serially")

                run_dirs = []
                for i in range(n):
                    run_dirs.append(_train_single_member((self, i)))

            else:
                if self._verbose:
                    print(f"[ensemble] launching {n} members on {workers} cores")

                with mp.get_context("spawn").Pool(workers) as pool:
                    run_dirs = pool.map(
                        _train_single_member,
                        [(self, i) for i in range(n)]
                    )

            for rd in run_dirs:
                self._eval_model(rd, period="validation")
                self._eval_model(rd, period="test")


        self._model = run_dirs

        tag = "phys" if self._physics_informed else "nophys"
        ens_dir = run_dirs[0].parent / f"{tag}_ensemble"
        ens_dir.mkdir(exist_ok=True)

        self._config.update_config({'run_dir': ens_dir})

        return run_dirs
        

    def _create_config(self) -> Config:
        """
        Create a Config in dev_mode, apply notebook-provided hyperparams/flags,
        and (optionally) route NH runs into a notebook-specified parent folder.
        """
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {self._yaml_path}")

        config = Config(self._yaml_path, dev_mode=True)

        # --- Pad values for synthetic START ---

        # check if standard ranges don't exist
        explicit_missing = (
            not config._cfg.get("train_start_date") and
            not config._cfg.get("train_end_date") and
            not config._cfg.get("validation_start_date") and
            not config._cfg.get("validation_end_date") and
            not config._cfg.get("test_start_date") and
            not config._cfg.get("test_end_date")
        )

        # check if synthetic ranges exist
        ranges_exist = (
            "train_ranges" in config._cfg and config._cfg["train_ranges"] and
            "validation_ranges" in config._cfg and config._cfg["validation_ranges"] and
            "test_ranges" in config._cfg and config._cfg["test_ranges"]
        )

        if explicit_missing and ranges_exist:
            dummy_start = "01/01/1900"
            dummy_end   = "01/01/1901"
            required_keys = [
                "train_start_date", "train_end_date",
                "validation_start_date", "validation_end_date",
                "test_start_date", "test_end_date",
            ]

            for key in required_keys:
                if key not in config._cfg or config._cfg[key] is None:
                    config._cfg[key] = dummy_start if "start" in key else dummy_end
                    if self._verbose:
                        print(f"[UCB_trainer:_create_config] Injected missing {key} → {config._cfg[key]}")

        # --- Pad values for synthetic END ---

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

    def cross_validate(self, intervalMonth='October', intervalLength=2, validationLength=1, no_leak=False, run_path=None, save_fold_details=False) -> dict:
        """
        Expanding-window time-series cross-validation.

        Performs i-fold CV where i = ([number of years in dataset] // intervalLength) - validationLength.
        Training window expands each fold; validation window slides forward.

        Arguments:
            intervalMonth: str, month boundary for water year (default 'October')
            intervalLength: int, years per training expansion (default 2)
            validationLength: int, validation window in years (default 1)
            no_leak: bool, if True validation excludes seq_length lookback (default False)
            run_path: Path, output directory (default: cwd/runs)

        Returns:
            dict of averaged metrics across folds (or tuple of daily/hourly dicts for MTS)
        """
        now = datetime.now()
        ts = f"{now.day:02d}{now.month:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}"

        root = (run_path if run_path else Path.cwd() / "runs")
        run_dir = root / f"cross_validation_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        MonthsLib = {'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr', 'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug', 'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec'}
        interval = MonthsLib[intervalMonth.lower()]

        is_mts = self._is_mts

        if not is_mts:
            cross_val_results = {}
        else:
            daily_results = {}
            hourly_results = {}

        original_start = getattr(self._config, "train_start_date", None)
        original_start_year = int(original_start.year)

        original_end = getattr(self._config, "validation_end_date", None)
        original_end_year = int(original_end.year)

        # Store original dates to restore after CV completes
        original_train_start = getattr(self._config, "train_start_date", None)
        original_train_end = getattr(self._config, "train_end_date", None)
        original_val_start = getattr(self._config, "validation_start_date", None)
        original_val_end = getattr(self._config, "validation_end_date", None)

        n_years = original_end_year - original_start_year + 1
        # Calculate max folds based on intervalLength: we need at least intervalLength years for initial training
        # plus validationLength years for validation, and each fold adds intervalLength years
        max_fold = (n_years - intervalLength) // intervalLength - validationLength


        seq_length = getattr(self._config, "seq_length", None)
        if not is_mts:
            lookback = int(seq_length)
        else:
            lookback_dict = seq_length

        def round_timedelta_up_to_day(delta: pd.Timedelta) -> pd.Timedelta:
            """Round a Timedelta up to the next whole day (ceiling to 24h multiples)."""
            days = delta / pd.Timedelta(days=1)
            days_ceiled = np.ceil(days)
            return pd.Timedelta(days=int(days_ceiled))

        def iso_date(dt) -> str:
            """Return DD/MM/YYYY string."""
            ts = pd.to_datetime(dt)
            return ts.strftime("%d/%m/%Y")

        i = 1
        while i <= max_fold:
            fold_train_start_date = pd.to_datetime(f"{str(original_start_year)}-{interval}-01", format="%Y-%b-%d")
            fold_train_end_date = pd.to_datetime(f"{original_start_year + (intervalLength * i)}-{interval}-01", format="%Y-%b-%d")
            val_eval_start = pd.to_datetime(fold_train_end_date) + pd.Timedelta(days=1)
            fold_val_end_date = pd.to_datetime(f"{original_start_year + (intervalLength * i + validationLength)}-{interval}-01", format="%Y-%b-%d")

            if no_leak:
                if not is_mts:
                    val_leak_start = val_eval_start
                else:
                    val_leak_start_d = val_eval_start
                    val_leak_start_h = val_eval_start
            else:
                if not is_mts:
                    val_leak_start = val_eval_start - pd.Timedelta(days=lookback - 1)
                else:
                    val_leak_start_d = val_eval_start - pd.Timedelta(days=lookback_dict['1D'] - 1)
                    val_leak_start_h = val_eval_start - round_timedelta_up_to_day(pd.Timedelta(hours=lookback_dict['1H']))

            fold_dir = run_dir / f"fold_{i:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if not is_mts:
                self._config.update_config({
                    "train_start_date": iso_date(fold_train_start_date),
                    "train_end_date": iso_date(fold_train_end_date),
                    "validation_start_date": iso_date(val_leak_start),
                    "validation_end_date": iso_date(fold_val_end_date),
                    "run_dir": fold_dir
                }, dev_mode=True)
            else:
                self._config.update_config({
                    "train_start_date": iso_date(fold_train_start_date),
                    "train_end_date": iso_date(fold_train_end_date),
                    "validation_start_per_frequency": {'1D': iso_date(val_leak_start_d), '1H': iso_date(val_leak_start_h)},
                    "validation_start_date": "01/01/1900",
                    "validation_end_date": iso_date(fold_val_end_date),
                    "run_dir": fold_dir
                }, dev_mode=True)

            self.train()

            if not is_mts:
                time_resolution_key = '1H' if self._hourly else '1D'
                self._get_predictions(time_resolution_key, 'validation')
                pred = self._predictions.loc[val_eval_start:fold_val_end_date]
                obs = self._observed.loc[val_eval_start:fold_val_end_date]
                metrics = calculate_all_metrics(obs, pred, resolution=time_resolution_key.upper())
                cross_val_results[i] = metrics
                
                # Save per-fold details if requested
                if save_fold_details:
                    # Save timeseries CSV
                    ts_df = pd.DataFrame({
                        'Date': pred.coords[list(pred.dims)[0]].values,
                        'Observed': obs.values,
                        'Predicted': pred.values
                    })
                    ts_df.to_csv(fold_dir / f'timeseries_validation.csv', index=False)

                    # Save metrics CSV
                    metrics_df = pd.DataFrame([metrics])
                    metrics_df.insert(0, 'fold', i)
                    metrics_df.insert(1, 'train_start', iso_date(fold_train_start_date))
                    metrics_df.insert(2, 'train_end', iso_date(fold_train_end_date))
                    metrics_df.insert(3, 'val_start', iso_date(val_eval_start))
                    metrics_df.insert(4, 'val_end', iso_date(fold_val_end_date))
                    metrics_df.to_csv(fold_dir / f'metrics_validation.csv', index=False)

                    # Generate loss curve for this fold
                    try:
                        plot_loss_curves(fold_dir, save_path=fold_dir / 'loss_curves.png')
                    except Exception as e:
                        if self._verbose:
                            print(f"Warning: Could not generate loss curve for fold {i}: {e}")

            else:
                self._get_predictions('1D', 'validation')
                pred = self._predictions.loc[val_eval_start:fold_val_end_date]
                obs = self._observed.loc[val_eval_start:fold_val_end_date]
                day_metrics = calculate_all_metrics(obs, pred, resolution='1D')

                self._get_predictions('1H', 'validation')
                obs_fixed = obs.assign_coords(
                    date=(list(obs.dims)[0], pd.date_range(start=val_eval_start,
                                                           periods=obs.sizes[list(obs.dims)[0]],
                                                           freq='H'))
                )
                pred_fixed = pred.assign_coords(
                    date=(list(pred.dims)[0], pd.date_range(start=val_eval_start,
                                                            periods=pred.sizes[list(pred.dims)[0]],
                                                            freq='H'))
                )
                hour_metrics = calculate_all_metrics(obs_fixed, pred_fixed, resolution='1H', datetime_coord='date')

                daily_results[i] = day_metrics
                hourly_results[i] = hour_metrics
                
                # Save per-fold details if requested (MTS version)
                if save_fold_details:
                    # Save daily timeseries
                    ts_df_d = pd.DataFrame({
                        'Date': pred.coords[list(pred.dims)[0]].values,
                        'Observed': obs.values,
                        'Predicted': pred.values
                    })
                    ts_df_d.to_csv(fold_dir / f'timeseries_validation_1D.csv', index=False)

                    # Save hourly timeseries
                    ts_df_h = pd.DataFrame({
                        'Date': pred_fixed.coords['date'].values,
                        'Observed': obs_fixed.values,
                        'Predicted': pred_fixed.values
                    })
                    ts_df_h.to_csv(fold_dir / f'timeseries_validation_1H.csv', index=False)

                    # Save metrics CSV (both daily and hourly)
                    metrics_df = pd.DataFrame([{**{'fold': i, 'freq': '1D'}, **day_metrics}])
                    metrics_df_h = pd.DataFrame([{**{'fold': i, 'freq': '1H'}, **hour_metrics}])
                    combined_metrics = pd.concat([metrics_df, metrics_df_h], ignore_index=True)
                    combined_metrics.insert(1, 'train_start', iso_date(fold_train_start_date))
                    combined_metrics.insert(2, 'train_end', iso_date(fold_train_end_date))
                    combined_metrics.insert(3, 'val_start', iso_date(val_eval_start))
                    combined_metrics.insert(4, 'val_end', iso_date(fold_val_end_date))
                    combined_metrics.to_csv(fold_dir / f'metrics_validation.csv', index=False)

                    # Generate loss curve for this fold
                    try:
                        plot_loss_curves(fold_dir, save_path=fold_dir / 'loss_curves.png')
                    except Exception as e:
                        if self._verbose:
                            print(f"Warning: Could not generate loss curve for fold {i}: {e}")

            i += 1

        # Restore original config dates
        if original_train_start:
            self._config.update_config({'train_start_date': original_train_start}, dev_mode=True)
        if original_train_end:
            self._config.update_config({'train_end_date': original_train_end}, dev_mode=True)
        if original_val_start:
            self._config.update_config({'validation_start_date': original_val_start}, dev_mode=True)
        if original_val_end:
            self._config.update_config({'validation_end_date': original_val_end}, dev_mode=True)
        if is_mts:
            self._config.update_config({'validation_start_per_frequency': None}, dev_mode=True)

        if self._verbose and not is_mts:
            for j in range(1, len(cross_val_results) + 1):
                print(f"Fold {j} results")
                print(cross_val_results[j])
                print("\n")

        # Aggregate metrics across folds
        if not is_mts:
            output = {}
            for j in cross_val_results:
                for metric in cross_val_results[j]:
                    key = f"avg {metric}"
                    if key not in output:
                        output[key] = []
                    output[key].append(cross_val_results[j][metric])
            for key in output:
                output[key] = sum(output[key]) / len(output[key])
        else:
            output_daily = {}
            for j in daily_results:
                for metric in daily_results[j]:
                    key = f"daily avg {metric}"
                    if key not in output_daily:
                        output_daily[key] = []
                    output_daily[key].append(daily_results[j][metric])
            output_hourly = {}
            for j in hourly_results:
                for metric in hourly_results[j]:
                    key = f"hourly avg {metric}"
                    if key not in output_hourly:
                        output_hourly[key] = []
                    output_hourly[key].append(hourly_results[j][metric])
            for key in output_daily:
                output_daily[key] = sum(output_daily[key]) / len(output_daily[key])
            for key in output_hourly:
                output_hourly[key] = sum(output_hourly[key]) / len(output_hourly[key])
            output = (output_daily, output_hourly)

        # Save CV summary CSV if fold details were saved
        if save_fold_details:
            if not is_mts:
                # Compile all fold metrics into summary
                summary_rows = []
                for j in cross_val_results:
                    row = {'fold': j, **cross_val_results[j]}
                    summary_rows.append(row)
                # Add average row
                avg_row = {'fold': 'average'}
                for metric in cross_val_results[1]:
                    avg_row[metric] = output[f"avg {metric}"]
                summary_rows.append(avg_row)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(run_dir / 'cv_summary.csv', index=False)
                if self._verbose:
                    print(f"\nCV Summary saved to: {run_dir / 'cv_summary.csv'}")
            else:
                # MTS: save daily and hourly summaries
                summary_rows_d = []
                for j in daily_results:
                    row = {'fold': j, 'freq': '1D', **daily_results[j]}
                    summary_rows_d.append(row)
                avg_row_d = {'fold': 'average', 'freq': '1D'}
                for metric in daily_results[1]:
                    avg_row_d[metric] = output_daily[f"daily avg {metric}"]
                summary_rows_d.append(avg_row_d)

                summary_rows_h = []
                for j in hourly_results:
                    row = {'fold': j, 'freq': '1H', **hourly_results[j]}
                    summary_rows_h.append(row)
                avg_row_h = {'fold': 'average', 'freq': '1H'}
                for metric in hourly_results[1]:
                    avg_row_h[metric] = output_hourly[f"hourly avg {metric}"]
                summary_rows_h.append(avg_row_h)

                summary_df = pd.DataFrame(summary_rows_d + summary_rows_h)
                summary_df.to_csv(run_dir / 'cv_summary.csv', index=False)
                if self._verbose:
                    print(f"\nCV Summary saved to: {run_dir / 'cv_summary.csv'}")

        return output


def replace_run_config_paths(run_dir: Path, verbose: bool = False) -> None:
    """
    Make a run's on-disk config.yml portable on the current machine by fixing data_dir if it does not exist,
    rewriting basin list files to absolute paths, aligning run_dir to the actual run_dir we are evaluating and
    if physics_informed, fixing physics_data_file if invalid, searching local data_dir for reasonable candidates.
    """
    yml = run_dir / "config.yml"
    if not yml.exists():
        if verbose:
            print(f"[portability] missing config.yml under {run_dir}")
        return

    try:
        with open(yml, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        if verbose:
            print(f"[portability] could not read {yml}: {e}")
        return

    changed = False

    dd = cfg.get("data_dir")
    dd_ok = isinstance(dd, (str, Path)) and str(dd) and Path(str(dd)).exists()
    if not dd_ok:
        new_dd = str(_ucb_data_dir().resolve())
        cfg["data_dir"] = new_dd
        changed = True
        if verbose:
            print(f"[portability] data_dir → {new_dd}")
    data_root = Path(cfg.get("data_dir")).resolve()

    for key in ("train_basin_file", "validation_basin_file", "test_basin_file"):
        v = cfg.get(key)
        if v is None:
            continue
        try:
            p = Path(v)
            need = (not p.is_absolute()) or (not p.exists())
        except Exception:
            need = True
        if need:
            basin_token = str(v)
            try:
                abs_file = _ucb_resolve_basin_file(basin_token, must_exist=True)
                cfg[key] = str(abs_file)
                changed = True
                if verbose:
                    print(f"[portability] {key} → {abs_file}")
            except Exception as e:
                if verbose:
                    print(f"[portability] WARNING: could not resolve '{key}'='{v}': {e}")

    rd = run_dir.resolve()
    if not cfg.get("run_dir") or Path(str(cfg.get("run_dir"))).resolve() != rd:
        cfg["run_dir"] = str(rd)
        changed = True
        if verbose:
            print(f"[portability] run_dir → {rd}")

    for key, sub in (("img_log_dir", "img_log"), ("train_dir", "train_data")):
        if key in cfg:
            expected = rd / sub
            cur = cfg.get(key)
            try:
                same = Path(str(cur)).resolve() == expected
            except Exception:
                same = False
            if not same:
                cfg[key] = str(expected)
                changed = True
                if verbose:
                    print(f"[portability] {key} → {expected}")

    def _guess_basin_prefix() -> str | None:
        src = cfg.get("train_basin_file") or cfg.get("validation_basin_file") or cfg.get("test_basin_file")
        if not src:
            return None
        name = Path(str(src)).name.lower().replace(".txt", "")
        name = name.replace("_", " ").strip()
        if "warm" in name and "spring" in name:
            return "WarmSprings_Inflow"
        return name.title().replace(" ", "")

    try:
        phys_on = bool(cfg.get("physics_informed", False))
    except Exception:
        phys_on = False

    phys_path = cfg.get("physics_data_file")
    phys_missing = (phys_path in (None, "None", "", "none", "null"))

    def _choose_candidates() -> list[Path]:
        prefix = _guess_basin_prefix()
        hourly = bool(cfg.get("hourly", False))
        cands: list[str] = []
        if hourly:
            if prefix:
                cands += [f"{prefix}_hourly.csv", f"{prefix.lower()}_hourly.csv"]
            cands += ["hourly.csv"]
        else:
            if prefix:
                cands += [f"{prefix}_daily_shift.csv", f"{prefix}_daily.csv",
                          f"{prefix.lower()}_daily_shift.csv", f"{prefix.lower()}_daily.csv"]
            cands += ["daily_shift.csv", "daily.csv"]
        return [data_root / c for c in cands]

    def _exists_case_insensitive(p: Path) -> Path | None:
        if p.exists():
            return p
        if p.parent.exists():
            target = p.name.lower()
            for child in p.parent.iterdir():
                if child.name.lower() == target:
                    return child
        return None

    if phys_on:
        need_fix = False
        if phys_missing:
            need_fix = True
        else:
            try:
                if not Path(str(phys_path)).exists():
                    need_fix = True
            except Exception:
                need_fix = True

        if need_fix:
            chosen: Path | None = None
            for cand in _choose_candidates():
                hit = _exists_case_insensitive(cand)
                if hit is not None:
                    chosen = hit
                    break
            if chosen is not None:
                cfg["physics_data_file"] = str(chosen.resolve())
                changed = True
                if verbose:
                    print(f"[portability] physics_data_file → {chosen.resolve()}")
            else:
                cfg["physics_informed"] = False
                changed = True
                if verbose:
                    print("[portability] WARNING: could not find physics_data_file locally; disabling physics_informed")

    if changed:
        try:
            with open(yml, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        except Exception as e:
            if verbose:
                print(f"[portability] could not write patched config to {yml}: {e}")