from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import xarray as xr

from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics

from UCB_training.UCB_utils import ensure_absolute_basin_files, prepare_out_path


def make_eval_config(run_dir: Path, *, basin: str, data_dir: Path, period: str,
                     start_date: str, end_date: str, gpu: int) -> Config:
    cfg = Config(run_dir / "config.yml", dev_mode=True)
    device = "cuda:0" if gpu == 0 else "cpu"
    cfg.update_config({"data_dir": Path(data_dir)}, dev_mode=True)
    cfg.update_config({"device": device}, dev_mode=True)
    cfg.update_config({f"{period}_start_date": start_date}, dev_mode=True)
    cfg.update_config({f"{period}_end_date": end_date}, dev_mode=True)
    ensure_absolute_basin_files(cfg, basin)
    if getattr(cfg, "physics_informed", False):
        phys_csv = Path(data_dir) / ("physics_hourly.csv" if getattr(cfg, "hourly", False) else "physics_daily.csv")
        cfg.update_config({"physics_data_file": phys_csv}, dev_mode=True)
    return cfg


def evaluate_runs_on_new_data(run_dirs: list[Path], *, cfg_builder, period: str,
                              start_date: str, end_date: str, gpu: int, epoch: int | None) -> None:
    for rd in run_dirs:
        cfg = cfg_builder(rd, period=period, start_date=start_date, end_date=end_date, gpu=gpu)
        start_evaluation(cfg=cfg, run_dir=rd, epoch=epoch, period=period)


def _drop_time_step(da: xr.DataArray) -> xr.DataArray:
    if "time_step" in da.dims:
        return da.isel(time_step=0)
    return da


def collect_series(run_dirs: list[Path], *, period: str, epoch: int, agg: str) -> pd.DataFrame:
    if len(run_dirs) > 1:
        ens = create_results_ensemble(run_dirs, period=period, epoch=epoch)
        basin = next(iter(ens))
        xr_d = ens[basin][agg]["xr"]
        tgt = Config(run_dirs[0] / "config.yml").target_variables[0]
    else:
        rd = run_dirs[0]
        pkl = rd / period / f"model_epoch{epoch:03d}" / f"{period}_results.p"
        with open(pkl, "rb") as fp:
            res = pickle.load(fp)
        basin = next(iter(res))
        xr_d = res[basin][agg]["xr"]
        tgt = Config(rd / "config.yml").target_variables[0]
    obs = _drop_time_step(xr_d[f"{tgt}_obs"])
    sim = _drop_time_step(xr_d[f"{tgt}_sim"])
    dates = pd.to_datetime(obs["date"].values)
    return pd.DataFrame({"Date": dates, "Observed": np.ravel(obs.values), "Predicted": np.ravel(sim.values)})


def save_ts_and_metrics(df: pd.DataFrame, *, basin: str, mode: str, period: str, tag: str) -> tuple[Path, Path]:
    ts_p = prepare_out_path(f"{basin}_{mode}_{period}_{tag}_ts.csv", kind="timeseries", period=period)
    met_p = prepare_out_path(f"{basin}_{mode}_{period}_{tag}_metrics.csv", kind="metrics", period=period)
    ts_p.parent.mkdir(parents=True, exist_ok=True)
    met_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ts_p, index=False)
    obs_da = xr.DataArray(df["Observed"].values, dims=["date"], coords={"date": df["Date"]})
    sim_da = xr.DataArray(df["Predicted"].values, dims=["date"], coords={"date": df["Date"]})
    m = calculate_all_metrics(obs_da, sim_da)
    m["PBIAS"] = float(((obs_da - sim_da).sum() / obs_da.sum()) * 100)
    pd.DataFrame([m]).to_csv(met_p, index=False)
    return ts_p, met_p


def collect_predictions(run_dirs: list[Path], *, period: str, epoch: int, agg: str,
                        colname: str = "Predicted") -> pd.DataFrame:
    """Return Date + Predicted only (no Observed)."""
    if len(run_dirs) > 1:
        from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
        ens = create_results_ensemble(run_dirs, period=period, epoch=epoch)
        basin = next(iter(ens))
        xr_d = ens[basin][agg]["xr"]
        tgt = Config(run_dirs[0] / "config.yml").target_variables[0]
    else:
        rd = run_dirs[0]
        pkl = rd / period / f"model_epoch{epoch:03d}" / f"{period}_results.p"
        with open(pkl, "rb") as fp:
            res = pickle.load(fp)
        basin = next(iter(res))
        xr_d = res[basin][agg]["xr"]
        tgt = Config(rd / "config.yml").target_variables[0]

    sim = _drop_time_step(xr_d[f"{tgt}_sim"])
    dates = pd.to_datetime(sim["date"].values)
    return pd.DataFrame({"Date": dates, colname: np.ravel(sim.values)})


def save_predictions(df: pd.DataFrame, *, basin: str, mode: str, period: str, tag: str,
                     filename_suffix: str = "forecast") -> Path:
    """Save Date,Predicted only."""
    ts_p = prepare_out_path(f"{basin}_{mode}_{period}_{tag}_{filename_suffix}.csv",
                            kind="timeseries", period=period)
    ts_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ts_p, index=False)
    return ts_p


# ─────────────────────────────────────────────────────────────────────────────
# Metrics comparison utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_test_metrics(run_dir: Path, basin: str = None) -> pd.DataFrame:
    """Load test metrics CSV from a run directory.

    Args:
        run_dir: Path to run directory (e.g., outputs/calpella/daily/BASELINE_20250815T000000Z)
        basin: Basin name for filename pattern. If None, globs for *_test_metrics.csv.

    Returns:
        DataFrame with metrics as index, models as columns.
    """
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics" / "test"
    if basin:
        csv_path = metrics_dir / f"{basin}_daily_test_metrics.csv"
    else:
        csvs = list(metrics_dir.glob("*_test_metrics.csv"))
        if not csvs:
            raise FileNotFoundError(f"No test metrics CSV found in {metrics_dir}")
        csv_path = csvs[0]
    return pd.read_csv(csv_path, index_col=0)


def pairwise_pct_change(baseline_df: pd.DataFrame, experimental_df: pd.DataFrame,
                        models: list[str] = None) -> pd.DataFrame:
    """Calculate percent change from baseline for each metric/model.

    Args:
        baseline_df: Metrics DataFrame (index=metrics, columns=models)
        experimental_df: Metrics DataFrame to compare against baseline
        models: List of model columns to compare. If None, uses all common columns.

    Returns:
        DataFrame with columns: Metric, Model, BASELINE, EXPERIMENTAL, pct_change
    """
    if models is None:
        models = [c for c in baseline_df.columns if c in experimental_df.columns]

    rows = []
    for metric in baseline_df.index:
        for model in models:
            baseline_val = baseline_df.loc[metric, model]
            exp_val = experimental_df.loc[metric, model]
            if baseline_val != 0:
                pct = abs((exp_val - baseline_val) / baseline_val * 100)
            else:
                pct = 0.0
            rows.append({
                "Metric": metric,
                "Model": model,
                "BASELINE": baseline_val,
                "EXPERIMENTAL": exp_val,
                "pct_change": pct
            })
    return pd.DataFrame(rows)


def threshold_filter(comparison_df: pd.DataFrame, threshold: float = 5.0,
                     always_keep: list[str] = None) -> pd.DataFrame:
    """Filter comparison DataFrame by percent change threshold.

    Args:
        comparison_df: DataFrame with 'Metric' and 'pct_change' columns
        threshold: Minimum percent change to keep (default 5%)
        always_keep: List of metric names to always include regardless of threshold

    Returns:
        Filtered DataFrame (without pct_change column)
    """
    always_keep = always_keep or []
    mask = (comparison_df["pct_change"] >= threshold) | (comparison_df["Metric"].isin(always_keep))
    return comparison_df[mask].drop(columns=["pct_change"])
