"""
Command-Line Grid Search Runner
================================
Runs the MTS-LSTM2 cross-validation grid search from the command line,
bypassing Jupyter's mp.Pool deadlock on Windows.

Usage:
    cd notebooks/basins/calpella
    C:/Users/SabiCan/miniforge3/envs/neuralhydrology/python.exe ../../../run_grid_search.py --basin calpella

    cd notebooks/basins/guerneville
    C:/Users/SabiCan/miniforge3/envs/neuralhydrology/python.exe ../../../run_grid_search.py --basin guerneville
"""

# -- Thread-limiting env vars BEFORE any imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "3"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import sys
import io
import time
import argparse
import itertools
import multiprocessing as mp
from pathlib import Path

# -- Windows unicode safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -- Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_T0 = time.perf_counter()
def _ts():
    return f"[+{time.perf_counter() - _T0:7.2f}s]"
def _print(*args):
    print(_ts(), *args, flush=True)


# ── Basin-specific configurations ──────────────────────────────────────

BASIN_CONFIGS = {
    "calpella": {
        "yaml_key": "calpella_mtslstm2",
        "physics_file_1H": "Calpella_hourly.csv",
        "features_with_physics": [
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
        ],
    },
    "warm_springs": {
        "yaml_key": "warm_springs_mtslstm2",
        "physics_file_1H": "WarmSprings_Inflow_hourly.csv",
        "features_with_physics": [
            "DRY CREEK 20 PRECIP-INC SCREENED",
            "DRY CREEK 20 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "DRY CREEK 30 PRECIP-INC SCREENED",
            "DRY CREEK 30 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "UKIAH CA HUMIDITY USAF-NOAA",
            "UKIAH CA SOLAR RADIATION USAF-NOAA",
            "UKIAH CA TEMPERATURE USAF-NOAA",
            "UKIAH CA WINDSPEED USAF-NOAA",
            "SANTA ROSA CA HUMIDITY USAF-NOAA",
            "SANTA ROSA CA SOLAR RADIATION USAF-NOAA",
            "SANTA ROSA CA TEMPERATURE USAF-NOAA",
            "SANTA ROSA CA WINDSPEED USAF-NOAA",
            "Dry Creek 20 ET-POTENTIAL",
            "Dry Creek 20 FLOW",
            "Dry Creek 20 FLOW-BASE",
            "Dry Creek 20 INFILTRATION",
            "Dry Creek 20 PERC-SOIL",
            "Dry Creek 20 SATURATION FRACTION",
            "Dry Creek 30 ET-POTENTIAL",
            "Dry Creek 30 FLOW",
            "Dry Creek 30 FLOW-BASE",
            "Dry Creek 30 INFILTRATION",
            "Dry Creek 30 PERC-SOIL",
            "Dry Creek 30 SATURATION FRACTION",
            "Warm Springs Dam Inflow FLOW",
        ],
    },
    "guerneville": {
        "yaml_key": "guerneville_mtslstm2",
        "physics_file_1H": "Guerneville_hourly.csv",
        "features_with_physics": [
            "BIG SULPHUR CR ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "DRY CREEK 10 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "EF RUSSIAN 20 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "GREEN VALLEY ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "LAGUNA ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 20 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 30 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 40 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 50 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 60 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 70 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "SANTA ROSA CR 10 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "SANTA ROSA CR 20 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "WF RUSSIAN ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "BIG SULPHUR CR PRECIP-INC SCREENED",
            "DRY CREEK 10 PRECIP-INC SCREENED",
            "EF RUSSIAN 20 PRECIP-INC SCREENED",
            "GREEN VALLEY PRECIP-INC SCREENED",
            "LAGUNA PRECIP-INC SCREENED",
            "RUSSIAN 20 PRECIP-INC SCREENED",
            "RUSSIAN 30 PRECIP-INC SCREENED",
            "RUSSIAN 40 PRECIP-INC SCREENED",
            "RUSSIAN 50 PRECIP-INC SCREENED",
            "RUSSIAN 60 PRECIP-INC SCREENED",
            "RUSSIAN 70 PRECIP-INC SCREENED",
            "SANTA ROSA CR 10 PRECIP-INC SCREENED",
            "SANTA ROSA CR 20 PRECIP-INC SCREENED",
            "WF RUSSIAN PRECIP-INC SCREENED",
            "UKIAH CA HUMIDITY USAF-NOAA",
            "UKIAH CA SOLAR RADIATION USAF-NOAA",
            "UKIAH CA TEMPERATURE USAF-NOAA",
            "UKIAH CA WINDSPEED USAF-NOAA",
            "SANTA ROSA CA HUMIDITY USAF-NOAA",
            "SANTA ROSA CA SOLAR RADIATION USAF-NOAA",
            "SANTA ROSA CA TEMPERATURE USAF-NOAA",
            "SANTA ROSA CA WINDSPEED USAF-NOAA",
            "UKIAH CA FLOW USGS-MERGED",
            "GEYSERVILLE CA FLOW USGS-MERGED",
            "Guerneville Gage FLOW",
            "Big Sulphur Cr ET-POTENTIAL",
            "Big Sulphur Cr FLOW",
            "Big Sulphur Cr FLOW-BASE",
            "Big Sulphur Cr INFILTRATION",
            "Big Sulphur Cr PERC-SOIL",
            "Big Sulphur Cr SATURATION FRACTION",
            "Dry Creek 10 ET-POTENTIAL",
            "Dry Creek 10 FLOW",
            "Dry Creek 10 FLOW-BASE",
            "Dry Creek 10 INFILTRATION",
            "Dry Creek 10 PERC-SOIL",
            "Dry Creek 10 SATURATION FRACTION",
            "Green Valley ET-POTENTIAL",
            "Green Valley FLOW",
            "Green Valley FLOW-BASE",
            "Green Valley INFILTRATION",
            "Green Valley PERC-SOIL",
            "Green Valley SATURATION FRACTION",
            "Laguna ET-POTENTIAL",
            "Laguna FLOW",
            "Laguna FLOW-BASE",
            "Laguna INFILTRATION",
            "Laguna PERC-SOIL",
            "Laguna SATURATION FRACTION",
            "Russian 20 ET-POTENTIAL",
            "Russian 20 FLOW",
            "Russian 20 FLOW-BASE",
            "Russian 20 INFILTRATION",
            "Russian 20 PERC-SOIL",
            "Russian 20 SATURATION FRACTION",
            "Russian 30 ET-POTENTIAL",
            "Russian 30 FLOW",
            "Russian 30 FLOW-BASE",
            "Russian 30 INFILTRATION",
            "Russian 30 PERC-SOIL",
            "Russian 30 SATURATION FRACTION",
            "Russian 40 ET-POTENTIAL",
            "Russian 40 FLOW",
            "Russian 40 FLOW-BASE",
            "Russian 40 INFILTRATION",
            "Russian 40 PERC-SOIL",
            "Russian 40 SATURATION FRACTION",
            "Russian 50 ET-POTENTIAL",
            "Russian 50 FLOW",
            "Russian 50 FLOW-BASE",
            "Russian 50 INFILTRATION",
            "Russian 50 PERC-SOIL",
            "Russian 50 SATURATION FRACTION",
            "Russian 60 ET-POTENTIAL",
            "Russian 60 FLOW",
            "Russian 60 FLOW-BASE",
            "Russian 60 INFILTRATION",
            "Russian 60 PERC-SOIL",
            "Russian 60 SATURATION FRACTION",
            "Russian 70 ET-POTENTIAL",
            "Russian 70 FLOW",
            "Russian 70 FLOW-BASE",
            "Russian 70 INFILTRATION",
            "Russian 70 PERC-SOIL",
            "Russian 70 SATURATION FRACTION",
            "Santa Rosa Cr 10 ET-POTENTIAL",
            "Santa Rosa Cr 10 FLOW",
            "Santa Rosa Cr 10 FLOW-BASE",
            "Santa Rosa Cr 10 INFILTRATION",
            "Santa Rosa Cr 10 PERC-SOIL",
            "Santa Rosa Cr 10 SATURATION FRACTION",
            "Santa Rosa Cr 20 ET-POTENTIAL",
            "Santa Rosa Cr 20 FLOW",
            "Santa Rosa Cr 20 FLOW-BASE",
            "Santa Rosa Cr 20 INFILTRATION",
            "Santa Rosa Cr 20 PERC-SOIL",
            "Santa Rosa Cr 20 SATURATION FRACTION",
            "WF Russian ET-POTENTIAL",
            "WF Russian FLOW",
            "WF Russian FLOW-BASE",
            "WF Russian INFILTRATION",
            "WF Russian PERC-SOIL",
            "WF Russian SATURATION FRACTION",
        ],
    },
}

# ── Hyperparameter grid (same for all basins) ─────────────────────────

HYPERPARAM_SPACE = {
    "hidden_size": [64, 128, 256],
    "output_dropout": [0.1, 0.4],
    "seq_length_1D": [90],
    "seq_length_1H": [168, 336],
    "num_layers": [1],
    "epochs": [300],
    "batch_size": [64],
    # schedule_pairs removed — plateau ES handles LR scheduling via YAML config
}

# ── Cross-validation settings ─────────────────────────────────────────

CV_INTERVAL_MONTH = "October"
CV_INTERVAL_LENGTH = 2
CV_VALIDATION_LENGTH = 1

# Grid ranking
GRID_RANK_METRICS = ["NSE_1D", "NSE_1H"]
GRID_RANK_WEIGHTS = [0.3, 0.7]

# Other settings
RUN_LABEL = "CROSS_VAL_V4"
GPU_SETTING = -1            # -1 = CPU, 0 = cuda:0
NUM_ENSEMBLES = 1
BOOTSTRAP_MODELS = False
HYPERPARAM_ENSEMBLE = False
VERBOSE = False


def main():
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Run MTS-LSTM2 grid search from command line")
    parser.add_argument("--basin", required=True, choices=list(BASIN_CONFIGS.keys()),
                        help="Basin to run grid search for")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU device ID (-1 for CPU)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Max parallel workers (0 = auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose training output")
    parser.add_argument("--no-physics", action="store_true",
                        help="Skip physics grid search")
    args = parser.parse_args()

    basin = args.basin
    gpu_setting = args.gpu
    verbose = args.verbose
    bcfg = BASIN_CONFIGS[basin]

    _print(f"Basin: {basin}, GPU: {gpu_setting}, Verbose: {verbose}")

    # -- Pre-compile numba JIT (cached to disk with cache=True)
    _print("Pre-warming numba JIT ...")
    import warnings
    import numpy as np
    from neuralhydrology.datasetzoo.basedataset import validate_samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validate_samples(
            x_d=[np.zeros((10, 2))], x_s=None, y=[np.zeros((10, 1))],
            frequency_maps=[np.arange(10)], seq_length=[3], predict_last_n=[1])
    _print("Numba JIT pre-compiled.")

    # -- Imports
    _print("Importing UCB modules ...")
    from UCB_training.UCB_utils import (
        fractional_multi_lr, data_dir, get_yaml_path,
        ensure_shared_tree, make_run_stamp,
        hparams_exists, save_hparams,
    )
    from UCB_training.grid_search_workers import (
        run_single_experiment_nophysics,
        run_single_experiment_physics,
    )
    _print("Imports complete.")

    # -- Paths
    path_to_csv = data_dir()
    path_to_yaml = get_yaml_path(bcfg["yaml_key"])
    path_to_physics_1H = path_to_csv / bcfg["physics_file_1H"]

    _SHARED = ensure_shared_tree(basin, "mts")
    RUN_STAMP = make_run_stamp()
    RUNS_PARENT = str(_SHARED / "runs" / f"{RUN_LABEL}_{RUN_STAMP}")

    _print(f"Data dir: {path_to_csv}")
    _print(f"YAML: {path_to_yaml}")
    _print(f"Runs parent: {RUNS_PARENT}")

    # -- Build combinations
    hp_names = list(HYPERPARAM_SPACE.keys())
    all_combinations = list(itertools.product(*[HYPERPARAM_SPACE[n] for n in hp_names]))
    n_combos = len(all_combinations)

    if args.workers > 0:
        num_cores = min(n_combos, args.workers)
    else:
        num_cores = min(n_combos, max(1, mp.cpu_count() - 1))

    _print(f"{n_combos} combinations, {num_cores} workers")

    # ── No-Physics Grid Search ─────────────────────────────────────────
    _print("=" * 60)
    _print("NO-PHYSICS GRID SEARCH")
    _print("=" * 60)

    task_args_no = [
        (idx, comb, hp_names, path_to_csv, path_to_yaml,
         gpu_setting, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
         fractional_multi_lr,
         NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
         True, True,
         True, CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH)
        for idx, comb in enumerate(all_combinations)
    ]

    t0 = time.perf_counter()
    with mp.Pool(processes=num_cores) as pool:
        no_physics_results = list(tqdm(
            pool.imap(run_single_experiment_nophysics, task_args_no),
            total=n_combos, desc="Grid No-Physics", unit="it", ncols=80, ascii=True))
    _print(f"No-physics grid completed in {time.perf_counter() - t0:.1f}s")

    df_no_physics = pd.DataFrame(no_physics_results)
    df_no_physics["_rank_score"] = sum(
        w * df_no_physics[m] for m, w in zip(GRID_RANK_METRICS, GRID_RANK_WEIGHTS))
    df_no_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
    df_no_physics.reset_index(drop=True, inplace=True)
    _print("No-physics top 3:")
    _print(df_no_physics[["hidden_size", "output_dropout", "seq_length_1H", "NSE_1D", "NSE_1H", "_rank_score"]].head(3).to_string())

    # ── Physics Grid Search ────────────────────────────────────────────
    if not args.no_physics:
        _print("=" * 60)
        _print("PHYSICS GRID SEARCH")
        _print("=" * 60)

        task_args_phys = [
            (idx, comb, hp_names, path_to_csv, path_to_yaml,
             gpu_setting, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
             fractional_multi_lr,
             NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
             bcfg["features_with_physics"], path_to_physics_1H,
             True, True,
             True, CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH)
            for idx, comb in enumerate(all_combinations)
        ]

        t0 = time.perf_counter()
        with mp.Pool(processes=num_cores) as pool:
            physics_results = list(tqdm(
                pool.imap(run_single_experiment_physics, task_args_phys),
                total=n_combos, desc="Grid Physics", unit="it", ncols=80, ascii=True))
        _print(f"Physics grid completed in {time.perf_counter() - t0:.1f}s")

        df_physics = pd.DataFrame(physics_results)
        df_physics["_rank_score"] = sum(
            w * df_physics[m] for m, w in zip(GRID_RANK_METRICS, GRID_RANK_WEIGHTS))
        df_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
        df_physics.reset_index(drop=True, inplace=True)
        _print("Physics top 3:")
        _print(df_physics[["hidden_size", "output_dropout", "seq_length_1H", "NSE_1D", "NSE_1H", "_rank_score"]].head(3).to_string())
    else:
        _print("Skipping physics grid search (--no-physics)")
        df_physics = None

    # ── Save best hyperparameters ──────────────────────────────────────
    _print("=" * 60)
    _print("SAVING BEST HYPERPARAMETERS")
    _print("=" * 60)

    best_no_phys = df_no_physics.iloc[0].to_dict()
    best_no_phys["model_type"] = "no_physics"

    if df_physics is not None:
        best_phys = df_physics.iloc[0].to_dict()
        best_phys["model_type"] = "physics"
        best_params_df = pd.DataFrame([best_no_phys, best_phys])
    else:
        best_params_df = pd.DataFrame([best_no_phys])

    save_hparams(
        best_df=best_params_df,
        basin=basin,
        mode="mts",
        label=RUN_LABEL,
        run_stamp=RUN_STAMP,
        df_no=df_no_physics,
        df_phys=df_physics,
    )

    _print(f"Best no-physics: hidden={best_no_phys.get('hidden_size')}, "
           f"seq1H={best_no_phys.get('seq_length_1H')}, "
           f"NSE_1D={best_no_phys.get('NSE_1D', 'N/A'):.4f}, "
           f"NSE_1H={best_no_phys.get('NSE_1H', 'N/A'):.4f}")
    if df_physics is not None:
        _print(f"Best physics: hidden={best_phys.get('hidden_size')}, "
               f"seq1H={best_phys.get('seq_length_1H')}, "
               f"NSE_1D={best_phys.get('NSE_1D', 'N/A'):.4f}, "
               f"NSE_1H={best_phys.get('NSE_1H', 'N/A'):.4f}")

    _print("=" * 60)
    _print(f"GRID SEARCH COMPLETE  (total: {time.perf_counter() - _T0:.1f}s)")
    _print("=" * 60)


if __name__ == "__main__":
    main()
