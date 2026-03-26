# HYPERPARAM_SPACE = {
#     "hidden_size": [64, 128, 256], 
#     "output_dropout": [0.1, 0.4],           
#     "seq_length_1D": [90, 120],
#     "seq_length_1H": [168, 336],
#     "num_layers": [1, 2],
#     "epochs": [300],
#     "batch_size": [64],
# }

#Bayes
HYPERPARAM_SPACE = {
    "hidden_size": [64, 128, 256, 384],
    "output_dropout": (0.05, 0.6),
    "seq_length_1D": [60, 90, 120],
    "seq_length_1H": [168, 240, 336],
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

USE_BAYES = True
N_BAYES_TRIALS = 48
BAYES_JOURNAL_DIR = ""

RUN_LABEL = "EXTREME_YEARS"
READ_STAMP = "20260307T122221Z"

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
    sys.stderr = LoggerWriter(logging.info)


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
from neuralhydrology.utils.config import Config
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
    "hopland": {
        "yaml_key": "hopland_mtslstm2",
        "physics_file_1H": "Hopland_hourly.csv",
        "features_with_physics": [
            #from daily.csv
            "RUSSIAN 60 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "RUSSIAN 60 PRECIP-INC SCREENED",
            "RUSSIAN 70 PRECIP-INC SCREENED",
            "RUSSIAN 70 ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            "WF RUSSIAN PRECIP-INC SCREENED",
            "WF RUSSIAN ET-POTENTIAL RUN:BASIN AVERAGE 60 YR",
            #from hopland.csv
            'Hopland Gage FLOW',
            'Russian 60 ET-POTENTIAL',
            'Russian 60 FLOW',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'Russian 60 FLOW-BASE',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'Russian 60 INFILTRATION',
            'Russian 60 PERC-SOIL',
            'Russian 60 SATURATION FRACTION',
            'Russian 70 ET-POTENTIAL',
            'Russian 70 FLOW',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'Russian 70 FLOW-BASE',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'Russian 70 INFILTRATION',
            'Russian 70 PERC-SOIL',
            'Russian 70 SATURATION FRACTION',
            'WF Russian ET-POTENTIAL',
            'WF Russian FLOW',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'WF Russian FLOW-BASE',  # HMS FLOW (BC-dependent) — comment out for NOBC_V2
            'WF Russian INFILTRATION',
            'WF Russian PERC-SOIL',
            'WF Russian SATURATION FRACTION',
            "UKIAH CA HUMIDITY USAF-NOAA",
            "UKIAH CA SOLAR RADIATION USAF-NOAA",
            "UKIAH CA TEMPERATURE USAF-NOAA",
            "UKIAH CA WINDSPEED USAF-NOAA",
            "UKIAH CA FLOW USGS-MERGED",
            #"Lake Mendocino Storage",    
        ],
    },
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

    INT_FIELDS = {"hidden_size", "seq_length_1D", "seq_length_1H", "num_layers", "epochs", "batch_size"}

    clean_row = {}
    for k, v in df_row.items():
        if k in INT_FIELDS and pd.notna(v):
            clean_row[k] = int(v)
        else:
            clean_row[k] = v

    row = pd.DataFrame([clean_row])

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

def seed_optuna_from_csv(study, csv_path, model_type):
    """
    Load completed trials from CSV and register them in the Optuna study
    so the sampler can build its model from them.
    """

    import optuna
    import pandas as pd

    if not csv_path.exists():
        print("[OPTUNA SEED] no CSV found")
        return

    df = pd.read_csv(csv_path)

    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]

    print(f"[OPTUNA SEED] loading {len(df)} trials from CSV")

    for _, row in df.iterrows():

        params = {
            "hidden_size": int(row["hidden_size"]),
            "output_dropout": float(row["output_dropout"]),
            "seq_length_1D": int(row["seq_length_1D"]),
            "seq_length_1H": int(row["seq_length_1H"]),
            "num_layers": int(row["num_layers"]),
            "epochs": int(row["epochs"]),
            "batch_size": int(row["batch_size"]),
        }

        value = float(row["_rank_score"])

        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                "hidden_size": optuna.distributions.CategoricalDistribution([64,128,256,384]),
                "output_dropout": optuna.distributions.FloatDistribution(0.05,0.6),
                "seq_length_1D": optuna.distributions.CategoricalDistribution([60,90,120]),
                "seq_length_1H": optuna.distributions.CategoricalDistribution([168,240,336]),
                "num_layers": optuna.distributions.CategoricalDistribution([1,2]),
                "epochs": optuna.distributions.CategoricalDistribution([300]),
                "batch_size": optuna.distributions.CategoricalDistribution([64,128]),
            },
            value=value,
        )

        study.add_trial(trial)

    print(f"[OPTUNA SEED] study now has {len(study.trials)} trials")

def suggest_from_space(trial):
    comb = []

    for hp, values in HYPERPARAM_SPACE.items():

        # Case 1: list or tuple of complex objects → categorical
        if isinstance(values, list):
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
            study_name="journal_storage_nophys",
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
            study_name="journal_storage_phys",
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


def _parse_cfg_date(v):
    if v is None:
        return None
    return pd.to_datetime(v, dayfirst=True)

def _iso_date(dt) -> str:
    return pd.to_datetime(dt).strftime("%d/%m/%Y")

def _round_timedelta_up_to_day(delta: pd.Timedelta) -> pd.Timedelta:
    days = delta / pd.Timedelta(days=1)
    days_ceiled = np.ceil(days)
    return pd.Timedelta(days=int(days_ceiled))

def generate_cv_folds_external(yaml_path: Path,
                               intervalMonth: str = "October",
                               intervalLength: int = 2,
                               validationLength: int = 1):

    cfg = Config(yaml_path, dev_mode=True)

    MonthsLib = {
        'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
        'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
        'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec'
    }
    interval = MonthsLib[intervalMonth.lower()]

    is_mts = bool(getattr(cfg, "is_mts", False))

    _cfg = cfg._cfg
    dataset_name = _cfg.get("dataset")


    # Non-Consecutive Years: synthetic_russian_river
    if dataset_name == "synthetic_russian_river":

        train_ranges = list(_cfg.get("train_ranges", []))
        validation_ranges = list(_cfg.get("validation_ranges", []))

        if not train_ranges:
            raise ValueError("synthetic_russian_river requires train_ranges in YAML")

        all_ranges = train_ranges + validation_ranges

        print("\n[SYNTHETIC CV MODE DETECTED]")
        print("Dataset:", dataset_name)
        print("Train ranges:", train_ranges)
        print("Validation ranges:", validation_ranges)
        print("All ranges:", all_ranges)

        folds = []

        train_step = intervalLength
        val_step = validationLength

        i = train_step

        fold_id = 1

        while (i + val_step - 1) < len(all_ranges):

            train_slice = all_ranges[:i]
            val_slice = all_ranges[i:i + val_step]

            print(
                f"[SYNTHETIC FOLD {fold_id}] "
                f"train={train_slice} "
                f"val={val_slice}"
            )
            
            folds.append({
                "fold": fold_id,
                "dataset_name": dataset_name,
                "train_ranges": train_slice,
                "validation_ranges": val_slice,
                "train_start_date": None,
                "train_end_date": None,
                "validation_start_date": None,
                "validation_end_date": None,
                "val_eval_start": None,
                "val_eval_end": None,
                "validation_start_per_frequency": None,
            })

            fold_id += 1
            i += train_step

        return folds

    original_start = _parse_cfg_date(getattr(cfg, "train_start_date", None))
    original_end = _parse_cfg_date(getattr(cfg, "validation_end_date", None))

    if original_start is None or original_end is None:
        raise ValueError("Could not determine train/validation date span from YAML for external CV queue generation.")

    original_start_year = int(original_start.year)
    original_end_year = int(original_end.year)

    n_years = original_end_year - original_start_year + 1
    max_fold = (n_years - 1 - validationLength) // intervalLength

    seq_length = getattr(cfg, "seq_length", None)

    # robust detection
    if isinstance(seq_length, dict):
        is_mts = True
        lookback_dict = seq_length
    else:
        is_mts = False
        lookback = int(seq_length)

    folds = []

    i = 1
    while i <= max_fold:
        fold_train_start_date = pd.to_datetime(
            f"{str(original_start_year)}-{interval}-01", format="%Y-%b-%d"
        )
        fold_train_end_date = pd.to_datetime(
            f"{original_start_year + (intervalLength * i)}-{interval}-01", format="%Y-%b-%d"
        )
        val_eval_start = pd.to_datetime(fold_train_end_date) + pd.Timedelta(days=1)
        fold_val_end_date = pd.to_datetime(
            f"{original_start_year + (intervalLength * i + validationLength)}-{interval}-01",
            format="%Y-%b-%d"
        )

        if not is_mts:
            val_leak_start = val_eval_start - pd.Timedelta(days=lookback - 1)
            folds.append({
                "fold": i,
                "dataset_name": dataset_name,
                "train_start_date": _iso_date(fold_train_start_date),
                "train_end_date": _iso_date(fold_train_end_date),
                "validation_start_date": _iso_date(val_leak_start),
                "validation_end_date": _iso_date(fold_val_end_date),
                "val_eval_start": _iso_date(val_eval_start),
                "val_eval_end": _iso_date(fold_val_end_date),
                "validation_start_per_frequency": None,
            })
        else:
            val_leak_start_d = val_eval_start - pd.Timedelta(days=lookback_dict['1D'] - 1)
            val_leak_start_h = val_eval_start - _round_timedelta_up_to_day(
                pd.Timedelta(hours=lookback_dict['1H'])
            )
            folds.append({
                "fold": i,
                "dataset_name": dataset_name,
                "train_start_date": _iso_date(fold_train_start_date),
                "train_end_date": _iso_date(fold_train_end_date),
                "validation_start_date": _iso_date(val_eval_start),
                "validation_end_date": _iso_date(fold_val_end_date),
                "val_eval_start": _iso_date(val_eval_start),
                "val_eval_end": _iso_date(fold_val_end_date),
                "validation_start_per_frequency": {
                    "1D": _iso_date(val_leak_start_d),
                    "1H": _iso_date(val_leak_start_h),
                },
            })

        i += 1

    return folds

def build_external_cv_queue(all_combinations, folds, include_physics=True):

    queue = []

    # FIRST: all no-physics
    for idx, comb in enumerate(all_combinations):
        for fold in folds:
            queue.append({
                "job_type": "no_physics",
                "iter_idx": idx,
                "fold": fold["fold"],
                "comb": comb,
                "fold_cfg": fold,
            })

    # THEN: all physics
    if include_physics:
        for idx, comb in enumerate(all_combinations):
            for fold in folds:
                queue.append({
                    "job_type": "physics",
                    "iter_idx": idx,
                    "fold": fold["fold"],
                    "comb": comb,
                    "fold_cfg": fold,
                })

    return queue

def print_external_cv_queue(queue):
    print("\n[EXTERNAL CV QUEUE]")
    for q_idx, job in enumerate(queue, start=1):
        print(
            f"queue[{q_idx:04d}] "
            f"iter {job['iter_idx'] + 1} "
            f"fold {job['fold']} "
            f"{job['job_type']}"
        )
    print(f"[EXTERNAL CV QUEUE] total jobs = {len(queue)}\n")

def _run_external_cv_queue_job(job):

    global path_to_csv, path_to_yaml, features_with_physics
    global path_to_physics_data_1H
    global RUNS_PARENT, RUN_LABEL, RUN_STAMP

    # Ensure globals exist in worker
    if RUNS_PARENT is None:
        bcfg = BASIN_CONFIGS[BASIN]

        path_to_csv = data_dir()
        path_to_yaml = get_yaml_path(bcfg["yaml_key"])
        path_to_physics_data_1H = path_to_csv / bcfg["physics_file_1H"]
        features_with_physics = bcfg["features_with_physics"]

        _SHARED = ensure_shared_tree(BASIN, "mts")
        RUNS_PARENT = str(_SHARED / "runs" / f"{RUN_LABEL}_{RUN_STAMP}")

    job_type = job["job_type"]
    idx = job["iter_idx"]
    comb = job["comb"]
    fold = job["fold_cfg"]

    common_tail = (
        True, True,
        False,  # use_cv_for_selection -> False because CV is now externalized into the queue
        CV_INTERVAL_MONTH, CV_INTERVAL_LENGTH, CV_VALIDATION_LENGTH,
        fold["train_start_date"],
        fold["train_end_date"],
        fold["validation_start_date"],
        fold["validation_end_date"],
        fold["val_eval_start"],
        fold["val_eval_end"],
        fold["validation_start_per_frequency"],
        fold.get("train_ranges"),
        fold.get("validation_ranges"),
        fold.get("dataset_name"),
        fold["fold"],
        len(EXTERNAL_CV_FOLDS),
        True,  # cv_external_queue_mode
    )

    if job_type == "no_physics":
        args = (
            idx, comb, hyperparam_names, path_to_csv, path_to_yaml,
            GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
            fractional_multi_lr,
            NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
            *common_tail
        )
        result = run_single_experiment_nophysics(args)
    else:
        args = (
            idx, comb, hyperparam_names, path_to_csv, path_to_yaml,
            GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
            fractional_multi_lr,
            NUM_ENSEMBLES, BOOTSTRAP_MODELS, HYPERPARAM_ENSEMBLE,
            features_with_physics,
            path_to_physics_data_1H,
            *common_tail
        )
        result = run_single_experiment_physics(args)

    result["_queue_job_type"] = job_type
    result["_queue_iter_idx"] = idx
    result["_queue_fold"] = fold["fold"]

    print(
        f"[QUEUE COMPLETED] iter {idx + 1} fold {fold['fold']} {job_type} "
        f"NSE_1D={result.get('NSE_1D', float('nan')):.5f} "
        f"NSE_1H={result.get('NSE_1H', float('nan')):.5f}"
    )

    return result

def log_bayes_queue(pending_jobs, inflight):

    print("\n[BAYES QUEUE STATE]")
    print(f"pending_jobs = {len(pending_jobs)}")
    print(f"inflight_trials = {list(inflight.keys())}")

    for i, job in enumerate(pending_jobs):
        print(
            f"queue[{i:04d}] "
            f"trial {job['iter_idx'] + 1} "
            f"fold {job['fold']} "
            f"{job['job_type']}"
        )

    print("[END BAYES QUEUE]\n")

def run_bayes_streaming_queue(study, model_type, total_trials, num_cores):

    global EXTERNAL_CV_FOLDS

    folds_per_trial = len(EXTERNAL_CV_FOLDS)

    from collections import defaultdict

    pending_jobs = []
    inflight = {}

    import pandas as pd
    from pathlib import Path

    root = _artifact_root(BASIN, MODE)
    prefix = f"{BASIN}_{MODE}_{RUN_LABEL}"
    path_latest = root / "hyperparams" / "archive" / f"{prefix}_{model_type}_gridsearch_{RUN_STAMP}.csv"

    print(path_latest)

    completed_trials = 0

    if path_latest.exists():

        df_done = pd.read_csv(path_latest)

        done_trials = sorted(df_done["_queue_iter_idx"].unique())

        print("\n[REPLAYING COMPLETED TRIALS FROM CSV]\n")

        for t in done_trials:
            label = "No-Physics" if model_type == "no_physics" else "Physics"
            print(f"[Trial {t+1} {label}] already completed")

        completed_trials = len(done_trials)

        print(f"\n[REPLAY COMPLETE] {completed_trials} trials already finished\n")

    else:
        print("\n[REPLAY] No previous trials found.")

    launched_trials = completed_trials
    print(f"[RESUME] starting from trial {completed_trials}")

    if completed_trials >= total_trials:
        print(f"[BAYES] {model_type} already finished ({completed_trials}/{total_trials})")
        return

    target_queue = num_cores + 1

    def enqueue_trial():

        nonlocal launched_trials

        if launched_trials >= total_trials:
            return False

        trial = study.ask()
        comb = suggest_from_space(trial)

        trial_idx = launched_trials

        inflight[trial_idx] = {
            "trial": trial,
            "comb": comb,
            "fold_results": []
        }

        for fold in EXTERNAL_CV_FOLDS:
            pending_jobs.append({
                "job_type": model_type,
                "iter_idx": trial_idx,
                "fold": fold["fold"],
                "comb": comb,
                "fold_cfg": fold,
            })

        print(f"\n[ENQUEUE TRIAL] trial {trial_idx+1} {model_type}")
        log_bayes_queue(pending_jobs, inflight)

        launched_trials += 1
        return True


    while len(pending_jobs) < target_queue:
        if not enqueue_trial():
            break


    with mp.Pool(processes=num_cores) as pool:

        active = []

        while True:

            while pending_jobs and len(active) < num_cores:
                job = pending_jobs.pop(0)
                res = pool.apply_async(run_parallel_bayes_worker, (job,))
                active.append((res, job))

            new_active = []

            for res, job in active:

                if res.ready():

                    result = res.get()

                    trial_number = result["_queue_iter_idx"]

                    inflight[trial_number]["fold_results"].append(result)

                    if len(inflight[trial_number]["fold_results"]) == folds_per_trial:

                        folds = inflight[trial_number]["fold_results"]

                        mean_1d = np.mean([f["NSE_1D"] for f in folds])
                        mean_1h = np.mean([f["NSE_1H"] for f in folds])

                        value = 0.7 * mean_1h + 0.3 * mean_1d

                        trial = inflight[trial_number]["trial"]
                        comb = inflight[trial_number]["comb"]

                        hp_dict = dict(zip(hyperparam_names, comb))

                        append_trial_row(
                            {
                                "_queue_iter_idx": trial_number,
                                **hp_dict,
                                "NSE_1D": mean_1d,
                                "NSE_1H": mean_1h,
                                "_rank_score": value,
                                "model_type": model_type
                            },
                            basin=BASIN,
                            mode=MODE,
                            label=RUN_LABEL,
                            run_stamp=RUN_STAMP,
                            tag=model_type,
                        )

                        study.tell(trial, value)

                        print(
                            f"[TRIAL COMPLETE] trial {trial_number+1} {model_type} "
                            f"NSE_1D={mean_1d:.5f} NSE_1H={mean_1h:.5f}"
                        )

                        del inflight[trial_number]

                        while len(pending_jobs) < target_queue:
                            if not enqueue_trial():
                                break

                    completed_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))

                    print(f"[RESUME] starting from trial {completed_trials}")

                    if completed_trials >= total_trials:
                        print(f"[BAYES] {model_type} already finished ({completed_trials}/{total_trials})")
                        return

                else:
                    new_active.append((res, job))

            active = new_active

            time.sleep(0.1)

def run_parallel_bayes_worker(job):
    """
    Executes ONE fold job.
    This is now the worker entrypoint used by the streaming scheduler.
    """

    return _run_external_cv_queue_job(job)


EXTERNAL_CV_FOLDS = []


def main():
    global hyperparam_names, path_to_csv, path_to_yaml
    global features_with_physics, path_to_physics_data_1H
    global RUNS_PARENT, RUN_LABEL, RUN_STAMP
    global verbose, fractional_multi_lr
    global use_cv_for_selection, MODE
    global EXTERNAL_CV_FOLDS

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

    num_cores = 50

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

    if USE_CV:
        EXTERNAL_CV_FOLDS = generate_cv_folds_external(
            path_to_yaml,
            intervalMonth=CV_INTERVAL_MONTH,
            intervalLength=CV_INTERVAL_LENGTH,
            validationLength=CV_VALIDATION_LENGTH
        )
        print("\n[EXTERNAL CV FOLDS]")
        for f in EXTERNAL_CV_FOLDS:
            print(
                f"fold {f['fold']}: "
                f"train {f['train_start_date']} -> {f['train_end_date']} | "
                f"val {f['validation_start_date']} -> {f['validation_end_date']} | "
                f"eval {f['val_eval_start']} -> {f['val_eval_end']}"
            )
        print(f"[EXTERNAL CV FOLDS] total folds = {len(EXTERNAL_CV_FOLDS)}\n")

    if USE_BAYESIAN:

        optuna.logging.set_verbosity(optuna.logging.INFO)

        print(f"\n[BAYESIAN OPTIMIZATION] Using {num_cores} parallel workers\n")


        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_nophys_journal.log"

        study_no = optuna.create_study(
            study_name="journal_storage_no_physics",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        csv_path = (
            _artifact_root(BASIN, MODE)
            / "hyperparams"
            / "archive"
            / f"{BASIN}_{MODE}_{RUN_LABEL}_no_physics_gridsearch_{RUN_STAMP}.csv"
        )

        if len(study_no.trials) == 0:
            print("[OPTUNA SEED] journal empty → seeding from CSV")
            seed_optuna_from_csv(study_no, csv_path, "no_physics")
        else:
            print(f"[OPTUNA SEED] journal already has {len(study_no.trials)} trials → skipping CSV seed")

        workers_per_trial = len(EXTERNAL_CV_FOLDS)

        # correct scheduling
        parallel_trials = max(1, math.ceil(num_cores / workers_per_trial))

        print(
            f"[BAYES] no_physics: {parallel_trials} parallel trials "
            f"x {workers_per_trial} folds = up to {parallel_trials * workers_per_trial} workers"
        )

        run_bayes_streaming_queue(
            study_no,
            "no_physics",
            N_BAYES_TRIALS,
            num_cores
        )

        journal_path = Path(RUNS_PARENT) / f"{RUN_LABEL}_{RUN_STAMP}_phys_journal.log"

        study_phys = optuna.create_study(
            study_name="journal_storage_physics",
            storage=JournalStorage(JournalFileBackend(file_path=str(journal_path))),
            load_if_exists=True,
            direction="maximize",
        )

        csv_path = (
            _artifact_root(BASIN, MODE)
            / "hyperparams"
            / "archive"
            / f"{BASIN}_{MODE}_{RUN_LABEL}_physics_gridsearch_{RUN_STAMP}.csv"
        )

        if len(study_phys.trials) == 0:
            print("[OPTUNA SEED] physics journal empty → seeding from CSV")
            seed_optuna_from_csv(study_phys, csv_path, "physics")
        else:
            print(f"[OPTUNA SEED] physics journal already has {len(study_phys.trials)} trials → skipping CSV seed")

        print(
            f"[BAYES] physics: {parallel_trials} parallel trials "
            f"x {workers_per_trial} folds = up to {parallel_trials * workers_per_trial} workers"
        )

        run_bayes_streaming_queue(
            study_phys,
            "physics",
            N_BAYES_TRIALS,
            num_cores
        )

        root = _artifact_root(BASIN, MODE)
        arch = root / "hyperparams" / "archive"

        prefix = f"{BASIN}_{MODE}_{RUN_LABEL}"

        df_no_physics = pd.read_csv(
            arch / f"{prefix}_no_physics_gridsearch_{RUN_STAMP}.csv"
        )

        df_physics = pd.read_csv(
            arch / f"{prefix}_physics_gridsearch_{RUN_STAMP}.csv"
        )

        df_no_physics.reset_index(drop=True, inplace=True)
        df_physics.reset_index(drop=True, inplace=True)

        best_no_phys = df_no_physics.sort_values("_rank_score", ascending=False).iloc[0]
        best_phys = df_physics.sort_values("_rank_score", ascending=False).iloc[0]

        best_params_df = pd.DataFrame([
            best_no_phys,
            best_phys
        ]).reset_index(drop=True)

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

        if USE_CV:
            shared_queue = build_external_cv_queue(
                all_combinations=all_combinations,
                folds=EXTERNAL_CV_FOLDS,
                include_physics=(not RUN_NO_PHYSICS_ONLY)
            )

            print_external_cv_queue(shared_queue)

            from collections import defaultdict

            queue_results = []
            fold_buffer = defaultdict(list)
            logged_configs = set()

            with mp.Pool(processes=num_cores) as pool:
                for result in tqdm(
                    pool.imap(_run_external_cv_queue_job, shared_queue),
                    total=len(shared_queue),
                    desc="External CV Queue",
                    unit="job",
                    ncols=80,
                    ascii=True
                ):

                    queue_results.append(result)

                    iter_idx = result["_queue_iter_idx"]
                    job_type = result["_queue_job_type"]

                    key = (iter_idx, job_type)
                    fold_buffer[key].append(result)

                    # if all folds done → aggregate immediately
                    if len(fold_buffer[key]) == len(EXTERNAL_CV_FOLDS) and key not in logged_configs:

                        folds = fold_buffer[key]

                        mean_1d = np.mean([f["NSE_1D"] for f in folds])
                        mean_1h = np.mean([f["NSE_1H"] for f in folds])

                        comb = all_combinations[iter_idx]

                        hp_dict = dict(zip(hyperparam_names, comb))

                        row = {
                            "_queue_iter_idx": iter_idx,
                            **hp_dict,
                            "NSE_1D": mean_1d,
                            "NSE_1H": mean_1h,
                            "_rank_score": GRID_RANK_WEIGHTS[0]*mean_1d + GRID_RANK_WEIGHTS[1]*mean_1h,
                            "model_type": job_type
                        }

                        append_trial_row(
                            row,
                            basin=BASIN,
                            mode=MODE,
                            label=RUN_LABEL,
                            run_stamp=RUN_STAMP,
                            tag=job_type,
                        )

                        logged_configs.add(key)

                        print(
                            f"[CONFIG COMPLETE] iter {iter_idx+1} {job_type} "
                            f"NSE_1D={mean_1d:.5f} NSE_1H={mean_1h:.5f}"
                        )

            df_queue = pd.DataFrame(queue_results)

            df_no_physics_folds = df_queue[df_queue["_queue_job_type"] == "no_physics"].copy()
            df_no_physics = (
                df_no_physics_folds
                .groupby("_queue_iter_idx", as_index=False)[["NSE_1D", "NSE_1H"]]
                .mean()
            )

            hp_df_no = pd.DataFrame(
                [dict(zip(hyperparam_names, comb)) for comb in all_combinations]
            ).reset_index().rename(columns={"index": "_queue_iter_idx"})

            df_no_physics = hp_df_no.merge(df_no_physics, on="_queue_iter_idx", how="inner")
            df_no_physics["_rank_score"] = (
                GRID_RANK_WEIGHTS[0] * df_no_physics["NSE_1D"] +
                GRID_RANK_WEIGHTS[1] * df_no_physics["NSE_1H"]
            )
            df_no_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
            df_no_physics.reset_index(drop=True, inplace=True)

            if not RUN_NO_PHYSICS_ONLY:
                df_physics_folds = df_queue[df_queue["_queue_job_type"] == "physics"].copy()
                df_physics = (
                    df_physics_folds
                    .groupby("_queue_iter_idx", as_index=False)[["NSE_1D", "NSE_1H"]]
                    .mean()
                )

                hp_df_phys = pd.DataFrame(
                    [dict(zip(hyperparam_names, comb)) for comb in all_combinations]
                ).reset_index().rename(columns={"index": "_queue_iter_idx"})

                df_physics = hp_df_phys.merge(df_physics, on="_queue_iter_idx", how="inner")
                df_physics["_rank_score"] = (
                    GRID_RANK_WEIGHTS[0] * df_physics["NSE_1D"] +
                    GRID_RANK_WEIGHTS[1] * df_physics["NSE_1H"]
                )
                df_physics.sort_values(by="_rank_score", ascending=False, inplace=True)
                df_physics.reset_index(drop=True, inplace=True)
            else:
                df_physics = pd.DataFrame()

            best_no_phys = df_no_physics.iloc[0].to_dict()
            best_no_phys["model_type"] = "no_physics"

            if not df_physics.empty:
                best_phys = df_physics.iloc[0].to_dict()
                best_phys["model_type"] = "physics"
                best_params_df = pd.DataFrame([best_no_phys, best_phys])
            else:
                best_params_df = pd.DataFrame([best_no_phys])

            save_hparams(
                best_df=best_params_df,
                basin=BASIN,
                mode=MODE,
                label=RUN_LABEL,
                run_stamp=RUN_STAMP,
                df_no=df_no_physics,
                df_phys=df_physics if not df_physics.empty else pd.DataFrame()
            )

        else:
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