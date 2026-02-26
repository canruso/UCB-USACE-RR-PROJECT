import os
import sys
import gc
import itertools
from pathlib import Path

import ipyparallel as ipp
from UCB_training.UCB_train import UCB_trainer


def ESDL_ensemble(params):
    """Train ensemble for given config file and return NSE."""
    num_ensemble_members = 2
    print(f"Starting training for {params}")
    sys.stdout.flush()

    path_to_yaml = Path(
        "/global/home/users/ananyadua/USACE-UCB-LSTM/neuralhydrology/savio_training/calpella_hourly.yaml"
    )
    path_to_csv = Path(
        "/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data"
    )
    path_to_physics_data = Path(
        "/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data/Calpella_hourly.csv"
    )

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

    # Make a safe copy and ensure expected shapes
    member_params = dict(params)
    if isinstance(member_params.get("seq_length"), (int, float)):
        # wrap scalar seq_length into dict for MTS-LSTM
        member_params["seq_length"] = {"1H": int(member_params["seq_length"]), "1D": 90}

    member_params.setdefault("predict_last_n", {"1H": 24, "1D": 1})

    lstmPhysicsValBest = UCB_trainer(
        path_to_csv_folder=path_to_csv,
        yaml_path=path_to_yaml,
        hyperparams=member_params,
        input_features=features_with_physics,
        physics_informed=True,
        physics_data_file=path_to_physics_data,
        hourly=True,
        extend_train_period=False,
        gpu=-1,  # CPU only
        num_ensemble_members=num_ensemble_members,
    )

    lstmPhysicsValBest.train()
    physics_val_csv, physics_val_metrics = lstmPhysicsValBest.results()
    ensemble_nse = physics_val_metrics.get("NSE", float("nan"))

    # Write results safely
    output_dir = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_h = member_params["seq_length"].get("1H", "NA")
    seq_d = member_params["seq_length"].get("1D", "NA")

    output_file = output_dir / (
        f"{round(ensemble_nse, 3)}_"
        f"{member_params['hidden_size']}_"
        f"{seq_h}h_{seq_d}d_"
        f"{member_params['output_dropout']}_"
        f"{member_params['batch_size']}_"
        f"{member_params['epochs']}_"
        f"{member_params['experiment_name']}"
    )

    with open(output_file, "w") as f:
        f.write(str(ensemble_nse) + "\n")

    # Cleanup
    del lstmPhysicsValBest, physics_val_csv, physics_val_metrics
    gc.collect()

    return params, ensemble_nse


def get_param_combos():
    """Generate combinations of hyperparameters for MTS-LSTM (1H + 1D)."""
    hyperparam_space = {
        "batch_size": [32, 64],
        "seq_length": [
            {"1H": 120, "1D": 30},
            {"1H": 240, "1D": 60},
            {"1H": 336, "1D": 90},
        ],
        "hidden_size": [32, 64, 128],
        "output_dropout": [0.5],
        "epochs": [8, 12, 16],
    }

    keys = ["hidden_size", "seq_length", "output_dropout", "batch_size", "epochs"]
    combinations = list(itertools.product(*[hyperparam_space[k] for k in keys]))

    dict_combos = []
    for idx, combo in enumerate(combinations):
        param_dict = dict(zip(keys, combo))
        if isinstance(param_dict["seq_length"], dict):
            seq_val_1h = param_dict["seq_length"].get("1H", "NA")
            seq_val_1d = param_dict["seq_length"].get("1D", "NA")
            seq_tag = f"{seq_val_1h}h_{seq_val_1d}d"
        else:
            seq_tag = str(param_dict["seq_length"])

        param_dict["experiment_name"] = (
            f"calpella_attempt_1_index_{idx}_"
            f"{param_dict['hidden_size']}_"
            f"{seq_tag}_"
            f"{param_dict['output_dropout']}_"
            f"{param_dict['batch_size']}_"
            f"{param_dict['epochs']}"
        )
        dict_combos.append(param_dict)

    return dict_combos


if __name__ == "__main__":
    print("Getting param combos...")
    param_combos = get_param_combos()

    print("Creating workers...")
    num_workers = 7
    mycluster = ipp.Cluster(n=num_workers)
    c = mycluster.start_and_connect_sync()
    print(f"ipyparallel version: {ipp.__version__}")

    dview = c[:]
    dview.block = True

    # Environment setup on all workers
    setup_code = """
import os, sys, gc, itertools, logging, pickle
from pathlib import Path
sys.path.insert(0, '/global/home/users/ananyadua/USACE-UCB-LSTM/')
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.metrics import *
from UCB_training.UCB_train import UCB_trainer
from neuralhydrology.training.train import start_training
"""
    dview.execute(setup_code)

    print("Training models...")
    lview = c.load_balanced_view()
    lview.block = True
    _ = lview.map(ESDL_ensemble, param_combos)
