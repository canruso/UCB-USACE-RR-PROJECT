import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

import ipyparallel as ipp

sys.path.insert(0, '/global/home/users/ananyadua/USACE-UCB-LSTM/')
from UCB_training.UCB_train import UCB_trainer
from neuralhydrology.evaluation.metrics import calculate_all_metrics

yaml_path    = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/neuralhydrology/savio_training/calpella_hourly.yaml")
csv_folder   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data")
physics_file = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data/Calpella_hourly.csv")
output_dir   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/cv_parallel_output")

hyperparams = {
    "hidden_size":     64,
    "seq_length":      {"1H": 120, "1D": 30},
    "output_dropout":  0.5,
    "batch_size":      64,
    "epochs":          8,
    "predict_last_n":  {"1H": 24, "1D": 1},
    "experiment_name": "cv_parallel_run",
}

features_with_physics = [
    "EF RUSSIAN 20 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
    "EF RUSSIAN 20 PRECIP-INC SCREENED",
    "POTTER VALLEY CA FLOW USGS_ADJUSTED",
    "UKIAH CA HUMIDITY USAF-NOAA",
    "UKIAH CA SOLAR RADIATION USAF-NOAA",
    "UKIAH CA TEMPERATURE USAF-NOAA",
    "UKIAH CA WINDSPEED USAF-NOAA",
    "Capella Gage FLOW",
    "EF Russian 20 ET-POTENTIAL",
    "EF Russian 20 FLOW",
    "EF Russian 20 FLOW-BASE",
    "EF Russian 20 INFILTRATION",
    "EF Russian 20 PERC-SOIL",
    "EF Russian 20 SATURATION FRACTION",
]

INTERVAL_MONTH  = "October"
INTERVAL_LENGTH = 2
VALIDATION_LEN  = 1
NO_LEAK         = False

MonthsLib = {
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
    'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec'
}

def compute_fold_schedule(trainer, interval_month, interval_length, validation_length, no_leak):
    interval = MonthsLib[interval_month.lower()]
    config   = trainer._config

    original_start = getattr(config, "train_start_date", None)
    if isinstance(original_start, str):
        original_start = pd.to_datetime(original_start, dayfirst=True)
    original_end = getattr(config, "validation_end_date", None)
    if isinstance(original_end, str):
        original_end = pd.to_datetime(original_end, dayfirst=True)

    start_year = int(original_start.year)
    end_year   = int(original_end.year)
    n_years    = end_year - start_year + 1
    max_fold   = (n_years - interval_length) // interval_length - validation_length

    is_mts = trainer._is_mts
    seq    = config.seq_length
    lookback_d = int(seq['1D']) if is_mts else int(seq)
    lookback_h = seq['1H']     if is_mts else None

    def iso(dt):
        return pd.to_datetime(dt).strftime("%d/%m/%Y")

    def ceil_to_day(delta):
        return pd.Timedelta(days=int(np.ceil(delta / pd.Timedelta(days=1))))

    folds = []
    for i in range(1, max_fold + 1):
        train_start    = pd.to_datetime(f"{start_year}-{interval}-01", format="%Y-%b-%d")
        train_end      = pd.to_datetime(f"{start_year + interval_length * i}-{interval}-01", format="%Y-%b-%d")
        val_eval_start = train_end + pd.Timedelta(days=1)
        val_end        = pd.to_datetime(f"{start_year + interval_length * i + validation_length}-{interval}-01", format="%Y-%b-%d")

        if no_leak:
            val_start_d = val_eval_start
            val_start_h = val_eval_start
        else:
            val_start_d = val_eval_start - pd.Timedelta(days=lookback_d - 1)
            val_start_h = val_eval_start - ceil_to_day(pd.Timedelta(hours=lookback_h)) if lookback_h else val_start_d

        folds.append({
            "fold":           i,
            "train_start":    iso(train_start),
            "train_end":      iso(train_end),
            "val_start_d":    iso(val_start_d),
            "val_start_h":    iso(val_start_h),
            "val_end":        iso(val_end),
            "val_eval_start": val_eval_start.isoformat(),
        })

    return folds, max_fold

def run_fold(fold_spec):
    import sys, gc, json, numpy as np, pandas as pd, xarray as xr
    from pathlib import Path
    sys.path.insert(0, '/global/home/users/ananyadua/USACE-UCB-LSTM/')
    from UCB_training.UCB_train import UCB_trainer
    from neuralhydrology.evaluation.metrics import calculate_all_metrics

    yaml_path    = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/neuralhydrology/savio_training/calpella_hourly.yaml")
    csv_folder   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data")
    physics_file = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data/Calpella_hourly.csv")
    output_dir   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/cv_parallel_output")

    hyperparams = fold_spec["hyperparams"]
    f           = fold_spec["fold"]
    fold_idx    = f["fold"]

    features_with_physics = fold_spec["features"]

    fold_dir = output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fold {fold_idx}] starting — train {f['train_start']} → {f['train_end']}")
    sys.stdout.flush()

    trainer = UCB_trainer(
        path_to_csv_folder=csv_folder,
        yaml_path=yaml_path,
        hyperparams=hyperparams,
        input_features=features_with_physics,
        physics_informed=True,
        physics_data_file=physics_file,
        hourly=True,
        extend_train_period=False,
        gpu=-1,
        num_ensemble_members=2,
        is_mts=True,
        runs_parent=fold_dir,
    )

    trainer._config.update_config({
        "train_start_date":  f["train_start"],
        "train_end_date":    f["train_end"],
        "validation_start_per_frequency": {"1D": f["val_start_d"], "1H": f["val_start_h"]},
        "validation_start_date": "01/01/1900",
        "validation_end_date":   f["val_end"],
        "run_dir": fold_dir,
    }, dev_mode=True)

    trainer.train()

    val_eval_start = pd.to_datetime(f["val_eval_start"])
    val_end_dt     = pd.to_datetime(f["val_end"], dayfirst=True)

    # Daily metrics
    trainer._get_predictions('1D', 'validation')
    pred_d = trainer._predictions.loc[val_eval_start:val_end_dt]
    obs_d  = trainer._observed.loc[val_eval_start:val_end_dt]
    day_metrics = calculate_all_metrics(obs_d, pred_d, resolution='1D')

    # Hourly metrics
    trainer._get_predictions('1H', 'validation')
    pred_h = trainer._predictions.loc[val_eval_start:val_end_dt]
    obs_h  = trainer._observed.loc[val_eval_start:val_end_dt]
    n_hours = obs_h.sizes[list(obs_h.dims)[0]]
    hourly_index = pd.date_range(start=val_eval_start, periods=n_hours, freq='h')
    obs_fixed  = xr.DataArray(obs_h.values,  dims=['time'], coords={'date': ('time', hourly_index)})
    pred_fixed = xr.DataArray(pred_h.values, dims=['time'], coords={'date': ('time', hourly_index)})
    hour_metrics = calculate_all_metrics(obs_fixed, pred_fixed, resolution='1H', datetime_coord='date')

    # Save CSVs
    pd.DataFrame({'Date': pred_d.coords[list(pred_d.dims)[0]].values,
                  'Observed': obs_d.values, 'Predicted': pred_d.values}
                 ).to_csv(fold_dir / 'timeseries_validation_1D.csv', index=False)
    pd.DataFrame({'Date': pred_fixed.coords['date'].values,
                  'Observed': obs_fixed.values, 'Predicted': pred_fixed.values}
                 ).to_csv(fold_dir / 'timeseries_validation_1H.csv', index=False)

    pd.concat([
        pd.DataFrame([{'fold': fold_idx, 'freq': '1D', **day_metrics,
                       'train_start': f['train_start'], 'train_end': f['train_end'],
                       'val_start': str(val_eval_start.date()), 'val_end': f['val_end']}]),
        pd.DataFrame([{'fold': fold_idx, 'freq': '1H', **hour_metrics,
                       'train_start': f['train_start'], 'train_end': f['train_end'],
                       'val_start': str(val_eval_start.date()), 'val_end': f['val_end']}]),
    ], ignore_index=True).to_csv(fold_dir / 'metrics_validation.csv', index=False)

    meta = {
        "fold":        fold_idx,
        "train_start": f["train_start"],
        "train_end":   f["train_end"],
        "val_end":     f["val_end"],
        "result": {
            "daily":  {m: float(v) for m, v in day_metrics.items()},
            "hourly": {m: float(v) for m, v in hour_metrics.items()},
        }
    }
    with open(fold_dir / 'fold_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"[fold {fold_idx}] done. Daily NSE={day_metrics.get('NSE', float('nan')):.4f}  "
          f"Hourly NSE={hour_metrics.get('NSE', float('nan')):.4f}")
    sys.stdout.flush()

    del trainer
    gc.collect()

    return meta

def aggregate(results):
    daily_results  = {r["fold"]: r["result"]["daily"]  for r in results}
    hourly_results = {r["fold"]: r["result"]["hourly"] for r in results}

    output_daily, output_hourly = {}, {}
    for fold, metrics in daily_results.items():
        for m, v in metrics.items():
            output_daily.setdefault(f"daily avg {m}", []).append(v)
    for fold, metrics in hourly_results.items():
        for m, v in metrics.items():
            output_hourly.setdefault(f"hourly avg {m}", []).append(v)

    for k in output_daily:
        output_daily[k] = sum(output_daily[k]) / len(output_daily[k])
    for k in output_hourly:
        output_hourly[k] = sum(output_hourly[k]) / len(output_hourly[k])

    print("\n CV Results ")
    print("Daily metrics:")
    for k, v in output_daily.items():
        print(f"  {k}: {v:.4f}")
    print("Hourly metrics:")
    for k, v in output_hourly.items():
        print(f"  {k}: {v:.4f}")

    rows = []
    for fold in sorted(set(daily_results) | set(hourly_results)):
        rows.append({"fold": fold, "freq": "1D", **daily_results[fold]})
        rows.append({"fold": fold, "freq": "1H", **hourly_results[fold]})
    rows.append({"fold": "average", "freq": "1D",
                 **{k.replace("daily avg ", ""): v for k, v in output_daily.items()}})
    rows.append({"fold": "average", "freq": "1H",
                 **{k.replace("hourly avg ", ""): v for k, v in output_hourly.items()}})

    summary_path = output_dir / "cv_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    return output_daily, output_hourly

if __name__ == '__main__':

    probe_trainer = UCB_trainer(
        path_to_csv_folder=csv_folder,
        yaml_path=yaml_path,
        hyperparams=hyperparams,
        input_features=features_with_physics,
        physics_informed=True,
        physics_data_file=physics_file,
        hourly=True,
        extend_train_period=False,
        gpu=-1,
        num_ensemble_members=1,
        is_mts=True,
        runs_parent=output_dir,
        verbose=False,
    )

    folds, max_fold = compute_fold_schedule(
        probe_trainer, INTERVAL_MONTH, INTERVAL_LENGTH, VALIDATION_LEN, NO_LEAK
    )
    del probe_trainer

    print(f"CV: {max_fold} folds detected")

    # Package fold specs for workers (all data they need, no shared state)
    fold_specs = [{"fold": f, "hyperparams": hyperparams, "features": features_with_physics}
                  for f in folds]

    # Start ipyparallel cluster — one engine per fold (or cap at num_workers)
    num_workers = 7
    print(f"Starting ipyparallel cluster with {num_workers} workers...")
    mycluster = ipp.Cluster(n=num_workers)
    c = mycluster.start_and_connect_sync()
    print(f"ipyparallel version: {ipp.__version__}")

    lview = c.load_balanced_view()
    lview.block = True

    print("Running CV folds in parallel...")
    results = lview.map(run_fold, fold_specs)

    aggregate(results)
