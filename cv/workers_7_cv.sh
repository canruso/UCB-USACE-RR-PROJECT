#!/bin/bash
#SBATCH --job-name=cv_test_mts
#SBATCH --account=ac_esdl
#SBATCH --partition=savio2
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --qos=savio_normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ananyadua@berkeley.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ucb-lstm

export PYTHONPATH=$PYTHONPATH:/global/home/users/ananyadua/USACE-UCB-LSTM

python workers_7_cv.py