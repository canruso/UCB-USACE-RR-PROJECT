import itertools
from pathlib import Path
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def _build_hp_run(combinations, hyperparam_names, fractional_multi_lr):
    """Parse the flat combination tuple into a hyperparams dict."""

def _build_hp_run(combinations, hyperparam_names, fractional_multi_lr):
    """Parse the flat combination tuple into a hyperparams dict."""
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
    return hp_run


def _run_and_collect(trainer, is_mts, use_cv, cv_month, cv_interval, cv_val_len):
    """Train or cross-validate. Returns (metrics_1d, metrics_1h) for MTS, or metrics_dict for standalone."""
    if use_cv:
        cv_run_path = Path(trainer._runs_parent) if trainer._runs_parent else None
        cv_out = trainer.cross_validate(
            intervalMonth=cv_month, intervalLength=cv_interval,
            validationLength=cv_val_len, no_leak=True, save_fold_details=True, run_path=cv_run_path)
        if is_mts:
            daily_d, hourly_d = cv_out
            return ({k.replace("daily avg ", ""): v for k, v in daily_d.items()},
                    {k.replace("hourly avg ", ""): v for k, v in hourly_d.items()})
        return {k.replace("avg ", ""): v for k, v in cv_out.items()}
    else:
        trainer.train()
        if is_mts:
            _, m1d = trainer.results(period="validation", mts_trk="1D")
            _, m1h = trainer.results(period="validation", mts_trk="1H")
            return m1d, m1h
        _, m = trainer.results(period="validation")
        return m


def _build_row(combinations, hyperparam_names, hp_run, metrics_result, is_mts, pid, idx):
    """Build result row dict and log to tqdm."""
    if is_mts:
        metrics_1d, metrics_1h = metrics_result
        tqdm.write(
            f"[GRID][PID {pid}][{idx:03d}] NSE_1D={metrics_1d.get('NSE'):.4f} NSE_1H={metrics_1h.get('NSE'):.4f} | HP={hp_run}")
    else:
        metrics_dict = metrics_result
        tqdm.write(f"[GRID][PID {pid}][{idx:03d}] NSE={metrics_dict.get('NSE'):.4f} | HP={hp_run}")
    row_data = {name: combinations[i] for i, name in enumerate(hyperparam_names)}
    row_data["learning_rate"] = str(hp_run["learning_rate"])
    if is_mts:
        for k, v in metrics_1d.items():
            row_data[f"{k}_1D"] = v
        for k, v in metrics_1h.items():
            row_data[f"{k}_1H"] = v
    else:
        row_data.update(metrics_dict)
    return row_data


def run_single_experiment_nophysics(args):
    import os, warnings, logging, sys
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("neuralhydrology").setLevel(logging.ERROR)
    logging.getLogger("pandas").setLevel(logging.ERROR)
    sys.stderr = open(os.devnull, "w")

    (idx, combinations, hyperparam_names, path_to_csv, path_to_yaml,
     GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
     UCB_trainer, fractional_multi_lr, NUM_ENSEMBLES, BOOTSTRAP_MODELS, ADABOOST_ENSEMBLE,
     is_mts, hourly, use_cv, cv_month, cv_interval, cv_val_len) = args

    hp_run = _build_hp_run(combinations, hyperparam_names, fractional_multi_lr)

    trainer = UCB_trainer(path_to_csv_folder=path_to_csv, yaml_path=path_to_yaml, hyperparams=hp_run,
                          input_features=None, physics_informed=False, physics_data_file=None,
                          hourly=hourly, extend_train_period=False, gpu=GPU_SETTING, is_mts=is_mts,
                          num_ensemble_members=NUM_ENSEMBLES, adaboost_ensemble=ADABOOST_ENSEMBLE,
                          bootstrap_model=BOOTSTRAP_MODELS, verbose=verbose,
                          runs_parent=f"{RUNS_PARENT}_grid_{idx:03d}", run_label=RUN_LABEL, run_stamp=RUN_STAMP)

    result = _run_and_collect(trainer, is_mts, use_cv, cv_month, cv_interval, cv_val_len)
    return _build_row(combinations, hyperparam_names, hp_run, result, is_mts, os.getpid(), idx)


def run_single_experiment_physics(args):
    import os, warnings, logging, sys
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    sys.stderr = open(os.devnull, "w")

    (idx, combinations, hyperparam_names, path_to_csv, path_to_yaml,
     GPU_SETTING, RUNS_PARENT, RUN_LABEL, RUN_STAMP, verbose,
     UCB_trainer, fractional_multi_lr, NUM_ENSEMBLES, BOOTSTRAP_MODELS, ADABOOST_ENSEMBLE,
     features_with_physics, physics_data_file,
     is_mts, hourly, use_cv, cv_month, cv_interval, cv_val_len) = args

    hp_run = _build_hp_run(combinations, hyperparam_names, fractional_multi_lr)

    trainer = UCB_trainer(path_to_csv_folder=path_to_csv, yaml_path=path_to_yaml, hyperparams=hp_run,
                          input_features=features_with_physics, physics_informed=True,
                          physics_data_file=physics_data_file,
                          hourly=hourly, extend_train_period=False, gpu=GPU_SETTING, is_mts=is_mts,
                          num_ensemble_members=NUM_ENSEMBLES, adaboost_ensemble=ADABOOST_ENSEMBLE,
                          bootstrap_model=BOOTSTRAP_MODELS, verbose=verbose,
                          runs_parent=f"{RUNS_PARENT}_phys_grid_{idx:03d}", run_label=RUN_LABEL, run_stamp=RUN_STAMP)

    result = _run_and_collect(trainer, is_mts, use_cv, cv_month, cv_interval, cv_val_len)
    return _build_row(combinations, hyperparam_names, hp_run, result, is_mts, os.getpid(), idx)