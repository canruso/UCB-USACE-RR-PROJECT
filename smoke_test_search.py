"""
Smoke test for run_hyperparam_search.py rewrite.
Run on Windows PC after pulling to verify the 33/35 arg format works.

Usage:
    python smoke_test_search.py

Runs 1 combo x 10 epochs, no CV, no physics, grid mode.
Should complete in ~5 min on CPU.
"""
import run_hyperparam_search as rhs

# Override configs for minimal smoke test
rhs.HYPERPARAM_SPACE = {
    "hidden_size": [64],
    "output_dropout": [0.2],
    "seq_length_1D": [90],
    "seq_length_1H": [168],
    "num_layers": [1],
    "epochs": [10],
    "batch_size": [64],
}
rhs.hyperparam_names = list(rhs.HYPERPARAM_SPACE.keys())

rhs.BASIN = "calpella"
rhs.MODE = "mts"
rhs.GPU_SETTING = -1
rhs.NUM_WORKERS = 1
rhs.VERBOSE = True
rhs.RUN_NO_PHYSICS_ONLY = True
rhs.LOG_TO_FILE = False

rhs.USE_BAYES = False
rhs.USE_CV = False

rhs.RUN_LABEL = "SMOKE_TEST"
rhs.READ_STAMP = ""

rhs.NUM_ENSEMBLES = 1
rhs.BOOTSTRAP_MODELS = False
rhs.HYPERPARAM_ENSEMBLE = False

# Re-derive MODE flags
rhs.IS_MTS, rhs.HOURLY = rhs.MODE_FLAGS[rhs.MODE]
rhs.GRID_RANK_METRICS = ["NSE_1D", "NSE_1H"]
rhs.GRID_RANK_WEIGHTS = [0.3, 0.7]
rhs._NO_FOLD_TAIL = (None, None, None, None, None, None, None, None, None, None, 0, 1, False)

# Fresh stamp
from UCB_training.UCB_utils import make_run_stamp
rhs.RUN_STAMP = make_run_stamp()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    print("=" * 60)
    print("SMOKE TEST: 1 combo, 10 epochs, no CV, no physics, calpella")
    print("=" * 60)
    rhs.main()
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
