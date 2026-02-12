import itertools
import pandas as pd
from tqdm import tqdm

def run_single_experiment_nophysics(args):
    import os
    import warnings
    import logging

    # silence pandas / FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("neuralhydrology").setLevel(logging.ERROR)
    logging.getLogger("pandas").setLevel(logging.ERROR)

    import sys
    sys.stderr = open(os.devnull, "w")

    (idx, combinations, hyperparam_names, path_to_csv, path_to_yaml,
     GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
     UCB_trainer, fractional_multi_lr, NUM_ENSEMBLES, BOOTSTRAP_MODELS, ADABOOST_ENSEMBLE) = args

    pid = os.getpid()

    hp_run = {}
    j = 0
    schedule_pairs = None

    while j < len(hyperparam_names):
        name = hyperparam_names[j]
        val = combinations[j]

        if name in ("seq_length_1D", "seq_length_1H"):
            hp_run["seq_length"] = {"1D": combinations[j], "1H": combinations[j + 1]}
            j += 2
            continue

        if name == "schedule_pairs":
            schedule_pairs = val
            j += 1
            continue

        hp_run[name] = val
        j += 1

    if schedule_pairs is not None:
        fractions, rates = schedule_pairs
        hp_run["learning_rate"] = fractional_multi_lr(
            epochs=int(hp_run["epochs"]), fractions=list(fractions), lrs=list(rates))
    else:
        hp_run.setdefault("learning_rate", {0: 0.01, 30: 0.005, 40: 0.001})

    trainer = UCB_trainer(
        path_to_csv_folder=path_to_csv,
        yaml_path=path_to_yaml,
        hyperparams=hp_run,
        input_features=None,
        physics_informed=False,
        physics_data_file=None,
        hourly=True,
        extend_train_period=False,
        gpu=GPU_SETTING,
        is_mts=True,
        num_ensemble_members = NUM_ENSEMBLES,
        adaboost_ensemble = ADABOOST_ENSEMBLE,
        bootstrap_model = BOOTSTRAP_MODELS,
        verbose=verbose,
        runs_parent=f"{RUNS_PARENT}_grid_{idx:03d}",
        run_label=RUN_LABEL,
        run_stamp=RUN_STAMP
    )

    trainer.train()
    _, metrics_1d = trainer.results(period="validation", mts_trk="1D")
    _, metrics_1h = trainer.results(period="validation", mts_trk="1H")

    nse_1d = metrics_1d.get("NSE", None)
    nse_1h = metrics_1h.get("NSE", None)

    tqdm.write(
        f"[GRID][PID {pid}][{idx:03d}] "
        f"NSE_1D={nse_1d:.4f} NSE_1H={nse_1h:.4f} | "
        f"HP={hp_run}"
    )

    row_data = {}
    for i, name in enumerate(hyperparam_names):
        row_data[name] = combinations[i]

    row_data["learning_rate"] = str(hp_run["learning_rate"])
    for k, v in metrics_1d.items():
        row_data[f"{k}_1D"] = v
    for k, v in metrics_1h.items():
        row_data[f"{k}_1H"] = v

    return row_data

def run_single_experiment_physics(args):
    import os, warnings, logging, sys

    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    sys.stderr = open(os.devnull, "w")

    (idx, combinations, hyperparam_names, path_to_csv, path_to_yaml,
     GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
     UCB_trainer, fractional_multi_lr,
     features_with_physics, path_to_physics_data_1H) = args

    pid = os.getpid()

    hp_run = {}
    j = 0
    schedule_pairs = None

    while j < len(hyperparam_names):
        name = hyperparam_names[j]
        val = combinations[j]

        if name in ("seq_length_1D", "seq_length_1H"):
            hp_run["seq_length"] = {"1D": combinations[j], "1H": combinations[j + 1]}
            j += 2
            continue

        if name == "schedule_pairs":
            schedule_pairs = val
            j += 1
            continue

        hp_run[name] = val
        j += 1

    if schedule_pairs is not None:
        fractions, rates = schedule_pairs
        hp_run["learning_rate"] = fractional_multi_lr(
            epochs=int(hp_run["epochs"]),
            fractions=list(fractions),
            lrs=list(rates))
    else:
        hp_run.setdefault("learning_rate", {0: 0.01, 30: 0.005, 40: 0.001})

    trainer = UCB_trainer(
        path_to_csv_folder=path_to_csv,
        yaml_path=path_to_yaml,
        hyperparams=hp_run,
        input_features=features_with_physics,
        physics_informed=True,
        physics_data_file=path_to_physics_data_1H,
        hourly=True,
        extend_train_period=False,
        gpu=GPU_SETTING,
        is_mts=True,
        verbose=verbose,
        runs_parent=f"{RUNS_PARENT}_phys_grid_{idx:03d}",
        run_label=RUN_LABEL,
        run_stamp=RUN_STAMP
    )

    trainer.train()
    _, metrics_1d = trainer.results(period="validation", mts_trk="1D")
    _, metrics_1h = trainer.results(period="validation", mts_trk="1H")

    nse_1d = metrics_1d.get("NSE", None)
    nse_1h = metrics_1h.get("NSE", None)

    tqdm.write(
        f"[GRID][PID {pid}][{idx:03d}] "
        f"NSE_1D={nse_1d:.4f} NSE_1H={nse_1h:.4f} | "
        f"HP={hp_run}"
    )

    row = {name: combinations[i] for i, name in enumerate(hyperparam_names)}
    row["learning_rate"] = str(hp_run["learning_rate"])

    for k, v in metrics_1d.items():
        row[f"{k}_1D"] = v
    for k, v in metrics_1h.items():
        row[f"{k}_1H"] = v

    return row