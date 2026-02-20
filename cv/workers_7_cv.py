"""
workers_7_cv.py
==============
Standalone cross-validation test for the MTS-LSTM model on Savio.

Mirrors workers_7_grid_ensembles.py UCB_trainer setup exactly,
then calls cross_validate() the same way your notebooks do.

Run this first to confirm CV works before integrating with the grid search.
"""

import sys
from pathlib import Path

sys.path.insert(0, '/global/home/users/ananyadua/USACE-UCB-LSTM/')

from UCB_training.UCB_train import UCB_trainer
if __name__ == '__main__':

    yaml_path    = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/neuralhydrology/savio_training/calpella_hourly.yaml")
    csv_folder   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data")
    physics_file = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/russian_river_data/Calpella_hourly.csv")
    output_dir   = Path("/global/home/users/ananyadua/USACE-UCB-LSTM/cv_test_output")

    hyperparams = {
        "hidden_size":      64,
        "seq_length":       {"1H": 120, "1D": 30},
        "output_dropout":   0.5,
        "batch_size":       64,
        "epochs":           8,
        "predict_last_n":   {"1H": 24, "1D": 1},
        "experiment_name":  "cv_test_run",
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

    print("Initialising UCB_trainer...")
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
        runs_parent=output_dir,  
    )

    print("Running cross_validate()...")
    cv_output = trainer.cross_validate(
        intervalMonth="October",
        intervalLength=2,
        validationLength=1,
        no_leak=False,
        run_path=output_dir,
        save_fold_details=True,
    )

    # cross_validate() returns (daily_dict, hourly_dict) for MTS
    daily_dict, hourly_dict = cv_output

    print("\n===== CV Results =====")
    print("Daily metrics:")
    for k, v in daily_dict.items():
        print(f"  {k}: {v:.4f}")
    print("Hourly metrics:")
    for k, v in hourly_dict.items():
        print(f"  {k}: {v:.4f}")