HYPERPARAM_SPACE = {
    "hidden_size": [64, 128, 256], 
    "output_dropout": [0.1, 0.4],           
    "seq_length_1D": [90, 120],
    "seq_length_1H": [168, 336],
    "num_layers": [1, 2],
    "epochs": [300],
    "batch_size": [64, 128],
}
hyperparam_names = list(HYPERPARAM_SPACE.keys())

BASIN = "guerneville"  # "calpella", "warm_springs", "hopland", or "guerneville"
GPU_SETTING = -1
NUM_WORKERS = 0

VERBOSE = True
RUN_NO_PHYSICS_ONLY = False

USE_BAYES = False
N_BAYES_TRIALS = 36
BAYES_JOURNAL_DIR = ""

RUN_LABEL = "CROSS_VAL_V4"
# READ_STAMP = "20260304T002129Z"
READ_STAMP = ""

USE_CV = True
CV_INTERVAL_MONTH = "October"
CV_INTERVAL_LENGTH = 2
CV_VALIDATION_LENGTH = 1

GRID_RANK_METRICS = ["NSE_1D", "NSE_1H"]
GRID_RANK_WEIGHTS = [0.3, 0.7]

NUM_ENSEMBLES = 1
BOOTSTRAP_MODELS = False
HYPERPARAM_ENSEMBLE = False

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import logging
import sys
from pathlib import Path

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"gridsearch_{RUN_STAMP}.log"

    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,   # ~20MB ≈ ~100k log lines
        backupCount=1                # keep only latest logs
    )

    stream_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(process)d | %(levelname)s | %(message)s",
        handlers=[file_handler, stream_handler],
    )

    class LoggerWriter:
        def __init__(self, level):
            self.level = level
        def write(self, message):
            if message.strip():
                self.level(message.strip())
        def flush(self):
            pass

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    print(f"Logging to {log_file} (rotates at ~20MB ≈ 100k lines)")

import traceback
import sys
import io
import time
import itertools
import multiprocessing as mp
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np
import optuna
import math
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from UCB_training.UCB_utils import (
    fractional_multi_lr, data_dir, get_yaml_path,
    ensure_shared_tree, make_run_stamp,
    hparams_exists, save_hparams,
)
from UCB_training.grid_search_workers import (
    run_single_experiment_nophysics,
    run_single_experiment_physics,
)
from UCB_training.UCB_utils import (
    fractional_multi_lr,
    _artifact_root,
    save_hparams,
    load_hparams
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_T0 = time.perf_counter()
def _ts():
    return f"[+{time.perf_counter() - _T0:7.2f}s]"
def _print(*args):
    print(_ts(), *args, flush=True)

path_to_csv = None
path_to_yaml = None
path_to_physics_data_1H = None
features_with_physics = None

RUNS_PARENT = None
if READ_STAMP:
    RUN_STAMP = READ_STAMP
else:
    RUN_STAMP = make_run_stamp()

verbose = None
use_cv_for_selection = None
MODE = "mts"
 
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

def log_trial(trial_num, value, params):
    import sys

    print(f"\n[Trial {trial_num}] NSE = {value:.5f}")
    print("Hyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")


def append_trial_row(df_row: dict, *, basin: str, mode: str, label: str, run_stamp: str, tag: str):

    root = _artifact_root(basin, mode)
    hp_dir = root / "hyperparams"
    arch = hp_dir / "archive"

    prefix = f"{basin}_{mode}_{label}"

    path_latest = hp_dir / f"{prefix}_{tag}_gridsearch.csv"
    path_arch   = arch / f"{prefix}_{tag}_gridsearch_{run_stamp}.csv"

    row = pd.DataFrame([df_row])

    for p in (path_latest, path_arch):
        write_header = not p.exists()
        row.to_csv(p, mode="a", header=write_header, index=False)


def load_checkpoint(tag):

    root = _artifact_root(BASIN, MODE)
    arch = root / "hyperparams" / "archive"

    prefix = f"{BASIN}_{MODE}_{RUN_LABEL}"

    if not READ_STAMP:
        print("No READ_STAMP provided — starting fresh.")
        return None, 0

    path = arch / f"{prefix}_{tag}_gridsearch_{READ_STAMP}.csv"

    if path.exists():

        df = pd.read_csv(path)
        completed = len(df)

        print(f"[Checkpoint] Restored {completed} {tag} trials from {path}")

        return df, completed

    print(f"[Checkpoint] No archive checkpoint found at {path}")
    return None, 0

def suggest_from_space(trial):
    comb = []

    for hp, values in HYPERPARAM_SPACE.items():

        # Case 1: list or tuple of complex objects → categorical
        if isinstance(values, (list, tuple)) and len(values) > 2:
            val = trial.suggest_categorical(hp, values)

        # Case 2: numeric range (low, high)
        elif isinstance(values, tuple) and len(values) == 2:
            lo, hi = values

            # int range
            if isinstance(lo, int) and isinstance(hi, int):
                val = trial.suggest_int(hp, lo, hi)

            # float range
            else:
                val = trial.suggest_float(hp, lo, hi)

        # Case 3: list of scalars → categorical
        else:
            val = trial.suggest_categorical(hp, values)

        comb.append(val)

    return tuple(comb)


def objective_no_physics(trial):

    comb = suggest_from_space(trial)

    args = (
        trial.number, comb, hyperparam_names, path_to_csv, path_to_yaml,
        GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
        fractional_multi_lr,
        NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
        True, True,
        use_cv_for_selection,
        CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH
    )


    try:
        result = run_single_experiment_nophysics(args)
        nse_1d = result["NSE_1D"]
        nse_1h = result["NSE_1H"]

        value = 0.7 * nse_1h + 0.3 * nse_1d
    except Exception as e:
        import traceback
        print(f"[Trial {trial.number}] FAILED: {e}")
        traceback.print_exc()
        raise

    append_trial_row(
        {
            **trial.params,
            "value": value,
            "NSE_1H": nse_1h,
            "NSE_1D": nse_1d,
        },
        basin=BASIN,
        mode=MODE,
        label=RUN_LABEL,
        run_stamp=RUN_STAMP,
        tag="no_physics",
    )

    log_trial(trial.number, value, trial.params)

    return value


def objective_physics(trial):

    comb = suggest_from_space(trial)

    args = (
        trial.number, comb, hyperparam_names, path_to_csv, path_to_yaml,
        GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
        fractional_multi_lr,
        NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
        features_with_physics,
        path_to_physics_data_1H,
        True, True,
        use_cv_for_selection,
        CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH
    )

    try:
        result = run_single_experiment_physics(args)
        nse_1d = result["NSE_1D"]
        nse_1h = result["NSE_1H"]

        value = 0.7 * nse_1h + 0.3 * nse_1d

    except Exception as e:
        import traceback
        print(f"[Trial {trial.number}] FAILED: {e}")
        traceback.print_exc()
        raise

    append_trial_row(
        {
            **trial.params,
            "value": value,
            "NSE_1H": nse_1h,
            "NSE_1D": nse_1d,
        },
        basin=BASIN,
        mode=MODE,
        label=RUN_LABEL,
        run_stamp=RUN_STAMP,
        tag="physics",
    )

    log_trial(trial.number, value, trial.params)

    return value

def run_no_physics_worker(args):
    try:
        global path_to_csv, path_to_yaml, features_with_physics, path_to_physics_data_1H
        global RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose, use_cv_for_selection

        remaining_no, num_cores, runs_parent, run_label, run_stamp = args

        RUNS_PARENT = runs_parent
        RUN_LABEL  = run_label
        RUN_STAMP  = run_stamp
        verbose    = VERBOSE
        use_cv_for_selection = USE_CV

        bcfg = BASIN_CONFIGS[BASIN]
        path_to_csv = data_dir()
        path_to_yaml = get_yaml_path(bcfg["yaml_key"])
        path_to_physics_data_1H = path_to_csv / bcfg["physics_file_1H"]
        features_with_physics = bcfg["features_with_physics"]

        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_nophys_journal.log"

        study = optuna.create_study(
            study_name="journal_storage_multiprocess",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(objective_no_physics,
                       n_trials=math.ceil(remaining_no / num_cores),
                       show_progress_bar=False)

    except Exception:
        import traceback, sys
        traceback.print_exc()
        sys.stderr.flush()
        raise


def run_physics_worker(args):
    try:
        global path_to_csv, path_to_yaml, features_with_physics, path_to_physics_data_1H
        global RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose, use_cv_for_selection

        remaining_phys, num_cores, runs_parent, run_label, run_stamp = args

        RUNS_PARENT = runs_parent
        RUN_LABEL = run_label
        RUN_STAMP = run_stamp
        verbose = VERBOSE
        use_cv_for_selection = USE_CV

        bcfg = BASIN_CONFIGS[BASIN]
        path_to_csv = data_dir()
        path_to_yaml = get_yaml_path(bcfg["yaml_key"])
        path_to_physics_data_1H = path_to_csv / bcfg["physics_file_1H"]
        features_with_physics = bcfg["features_with_physics"]

        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_phys_journal.log"

        study = optuna.create_study(
            study_name="journal_storage_multiprocess",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(
            objective_physics,
            n_trials=math.ceil(remaining_phys / num_cores),
            show_progress_bar=False,
        )

    except Exception:
        import traceback, sys
        traceback.print_exc()
        sys.stderr.flush()
        raise


def main():
    global hyperparam_names, path_to_csv, path_to_yaml
    global features_with_physics, path_to_physics_data_1H
    global RUNS_PARENT, RUN_LABEL, RUN_STAMP
    global verbose, fractional_multi_lr
    global use_cv_for_selection, MODE

    os.chdir(PROJECT_ROOT / "notebooks" / "basins" / BASIN)

    bcfg = BASIN_CONFIGS[BASIN]

    _print(f"Basin: {BASIN}, GPU: {GPU_SETTING}, Verbose: {VERBOSE}")

    _print("Pre-warming numba JIT ...")
    from neuralhydrology.datasetzoo.basedataset import validate_samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validate_samples(
            x_d=[np.zeros((10, 2))], x_s=None, y=[np.zeros((10, 1))],
            frequency_maps=[np.arange(10)], seq_length=[3], predict_last_n=[1])
    _print("Numba JIT pre-compiled.")

    path_to_csv = data_dir()
    path_to_yaml = get_yaml_path(bcfg["yaml_key"])
    path_to_physics_1H = path_to_csv / bcfg["physics_file_1H"]

    print("YAML PATH:",path_to_yaml)

    _SHARED = ensure_shared_tree(BASIN, "mts")
    RUNS_PARENT = str(_SHARED / "runs" / f"{RUN_LABEL}_{RUN_STAMP}")

    Path(RUNS_PARENT).mkdir(parents=True, exist_ok=True)

    hp_names = list(HYPERPARAM_SPACE.keys())
    all_combinations = list(itertools.product(*[HYPERPARAM_SPACE[n] for n in hp_names]))
    n_combos = len(all_combinations)

    if NUM_WORKERS > 0:
        num_cores = min(n_combos, NUM_WORKERS)
    else:
        num_cores = min(n_combos, max(1, mp.cpu_count() - 1))

    num_cores = 1

    _print(f"{n_combos} combinations, {num_cores} workers")

    hyperparam_names = hp_names

    run_gridsearch = not USE_BAYES
    USE_BAYESIAN = USE_BAYES
    MODE = "mts"

    verbose = VERBOSE

    features_with_physics = bcfg["features_with_physics"]
    path_to_physics_data_1H = path_to_physics_1H

    use_cv_for_selection = USE_CV

    print("run_gridsearch =", run_gridsearch)
    print("hparams_exists =", hparams_exists(BASIN, MODE, RUN_LABEL))


    if USE_BAYESIAN:

        optuna.logging.set_verbosity(optuna.logging.INFO)

        print(f"\n[BAYESIAN OPTIMIZATION] Using {num_cores} parallel workers\n")

        _, done_no = load_checkpoint("no_physics")
        remaining_no = max(N_BAYES_TRIALS - done_no, 0)

        print(f"[Bayes] No-physics remaining trials: {remaining_no}")

        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_nophys_journal.log"
        study_no = optuna.create_study(
            study_name="journal_storage_multiprocess",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        if remaining_no > 0:
            with mp.Pool(processes=num_cores) as pool:
                worker_args = [
                    (remaining_no, num_cores, RUNS_PARENT, RUN_LABEL, RUN_STAMP)
                    for _ in range(num_cores)
                ]

                pool.map(run_no_physics_worker, worker_args)
        else:
            print("[Bayes] No-physics already complete — skipping.")

        _, done_phys = load_checkpoint("physics")
        remaining_phys = max(N_BAYES_TRIALS - done_phys, 0)

        print(f"[Bayes] Physics remaining trials: {remaining_phys}")

        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_phys_journal.log"
        study_phys = optuna.create_study(
            study_name="journal_storage_multiprocess",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        if remaining_phys > 0:
            with mp.Pool(processes=num_cores) as pool:
                worker_args = [
                    (remaining_phys, num_cores, RUNS_PARENT, RUN_LABEL, RUN_STAMP)
                    for _ in range(num_cores)
                ]

                pool.map(run_physics_worker, worker_args)
        else:
            print("[Bayes] Physics already complete — skipping.")

        df_no_physics = study_no.trials_dataframe()
        df_physics    = study_phys.trials_dataframe()

        df_no_physics.sort_values(by="value", ascending=False, inplace=True)
        df_physics.sort_values(by="value", ascending=False, inplace=True)

        df_no_physics.reset_index(drop=True, inplace=True)
        df_physics.reset_index(drop=True, inplace=True)

        best_no_phys = study_no.best_params
        best_phys = study_phys.best_params

        best_no_phys["model_type"] = "no_physics"
        best_phys["model_type"] = "physics"

        best_params_df = pd.DataFrame([best_no_phys, best_phys])

        save_hparams(
            best_df=best_params_df,
            basin=BASIN,
            mode=MODE,
            label=RUN_LABEL,
            run_stamp=RUN_STAMP,
            df_no=df_no_physics,
            df_phys=df_physics
        )


    elif run_gridsearch:

        all_combinations = list(itertools.product(
            *[HYPERPARAM_SPACE[hp] for hp in hyperparam_names]
        ))

        prev_no, done_no = load_checkpoint("no_physics")
        prev_phys, done_phys = load_checkpoint("physics")

        print(f"[Grid] Skipping first {done_no} no-physics trials")
        print(f"[Grid] Skipping first {done_phys} physics trials")

        print(f"\n[GRID SEARCH] Spawning {num_cores} workers\n")

        task_args_no = [
            (idx, comb, hyperparam_names, path_to_csv, path_to_yaml,
            GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
            fractional_multi_lr,
            NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
            True, True,
            use_cv_for_selection,
            CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH)
            for idx, comb in enumerate(all_combinations[done_no:], start=done_no)
        ]

        no_physics_results = [] if prev_no is None else prev_no.to_dict("records")

        with mp.Pool(processes=num_cores) as pool:
            for result in tqdm(
                pool.imap(run_single_experiment_nophysics, task_args_no),
                total=len(all_combinations) - done_no,
                initial=done_no,
                desc="Grid No-Physics",
                unit="it",
                ncols=60,
                ascii=True
            ):
                result["_rank_score"] = (
                    GRID_RANK_WEIGHTS[0] * result["NSE_1D"]
                    + GRID_RANK_WEIGHTS[1] * result["NSE_1H"]
                )

                no_physics_results.append(result)

                append_trial_row(
                    result,
                    basin=BASIN,
                    mode=MODE,
                    label=RUN_LABEL,
                    run_stamp=RUN_STAMP,
                    tag="no_physics",
                )

        df_no_physics = pd.DataFrame(no_physics_results)
        df_no_physics["_rank_score"] = sum(w * df_no_physics[m] for m, w in zip(GRID_RANK_METRICS, GRID_RANK_WEIGHTS))
        df_no_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
        df_no_physics.reset_index(drop=True, inplace=True)

        task_args_phys = [
            (idx, comb, hyperparam_names, path_to_csv, path_to_yaml,
            GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
            fractional_multi_lr,
            NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
            features_with_physics,
            path_to_physics_data_1H,
            True, True,
            use_cv_for_selection,
            CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH)
            for idx, comb in enumerate(all_combinations[done_phys:], start=done_phys)
        ]

        physics_results = [] if prev_phys is None else prev_phys.to_dict("records")

        with mp.Pool(processes=num_cores) as pool:
            for result in tqdm(
                pool.imap(run_single_experiment_physics, task_args_phys),
                total=len(all_combinations) - done_phys,
                initial=done_phys,
                desc="Grid Physics",
                unit="it",
                ncols=60,
                ascii=True
            ):
                result["_rank_score"] = (
                    GRID_RANK_WEIGHTS[0] * result["NSE_1D"]
                    + GRID_RANK_WEIGHTS[1] * result["NSE_1H"]
                )

                physics_results.append(result)

                append_trial_row(
                    result,
                    basin=BASIN,
                    mode=MODE,
                    label=RUN_LABEL,
                    run_stamp=RUN_STAMP,
                    tag="physics",
                )

        df_physics = pd.DataFrame(physics_results)
        df_physics["_rank_score"] = sum(w * df_physics[m] for m, w in zip(GRID_RANK_METRICS, GRID_RANK_WEIGHTS))
        df_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
        df_physics.reset_index(drop=True, inplace=True)

        best_no_phys = df_no_physics.iloc[0].to_dict()
        best_phys = df_physics.iloc[0].to_dict()

        best_no_phys["model_type"] = "no_physics"
        best_phys["model_type"] = "physics"

        best_params_df = pd.DataFrame([best_no_phys, best_phys])

        save_hparams(
            best_df=best_params_df,
            basin=BASIN,
            mode=MODE,
            label=RUN_LABEL,
            run_stamp=RUN_STAMP,
            df_no=df_no_physics,
            df_phys=df_physics
        )

    else:
        print("Skipping search!")

    print("run_gridsearch =", run_gridsearch)
    print("hparams_exists =", hparams_exists(BASIN, MODE, RUN_LABEL))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    setup_logging()
    main()
