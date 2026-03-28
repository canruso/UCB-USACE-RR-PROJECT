"""Tests for the rewritten run_hyperparam_search.py"""
import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers: import the module without triggering heavy side-effects
# ---------------------------------------------------------------------------
def _import_module():
    """Import run_hyperparam_search with heavy deps mocked out."""
    # Mock neuralhydrology and UCB_training so import doesn't need real packages running
    mocks = {}
    for mod_name in [
        "neuralhydrology", "neuralhydrology.utils", "neuralhydrology.utils.config",
        "neuralhydrology.datasetzoo", "neuralhydrology.datasetzoo.basedataset",
        "optuna", "optuna.storages", "optuna.storages.journal",
        "optuna.distributions", "optuna.trial",
        "UCB_training", "UCB_training.UCB_utils", "UCB_training.grid_search_workers",
    ]:
        mocks[mod_name] = MagicMock()

    # Provide real-looking returns for specific functions
    mocks["UCB_training.UCB_utils"].make_run_stamp.return_value = "20260311T000000Z"
    mocks["UCB_training.UCB_utils"].data_dir.return_value = Path("/fake/data")
    mocks["UCB_training.UCB_utils"].get_yaml_path.return_value = Path("/fake/yaml.yml")
    mocks["UCB_training.UCB_utils"].ensure_shared_tree.return_value = Path("/fake/shared")
    mocks["UCB_training.UCB_utils"]._artifact_root.return_value = Path("/fake/artifacts")
    mocks["UCB_training.UCB_utils"].hparams_exists.return_value = False

    with patch.dict(sys.modules, mocks):
        # Force re-import
        if "run_hyperparam_search" in sys.modules:
            del sys.modules["run_hyperparam_search"]
        mod = importlib.import_module("run_hyperparam_search")

    return mod


@pytest.fixture(scope="module")
def rhs():
    return _import_module()


# ---------------------------------------------------------------------------
# Test: BASIN_CONFIGS completeness
# ---------------------------------------------------------------------------
class TestBasinConfigs:
    def test_four_basins_present(self, rhs):
        expected = {"calpella", "warm_springs", "guerneville", "hopland"}
        assert set(rhs.BASIN_CONFIGS.keys()) == expected

    @pytest.mark.parametrize("basin", ["calpella", "warm_springs", "guerneville", "hopland"])
    def test_all_modes_have_yaml_key(self, rhs, basin):
        cfg = rhs.BASIN_CONFIGS[basin]
        for mode in ("mts", "daily", "hourly"):
            assert mode in cfg["yaml_key"], f"{basin} missing yaml_key for mode={mode}"
            assert isinstance(cfg["yaml_key"][mode], str)

    @pytest.mark.parametrize("basin", ["calpella", "warm_springs", "guerneville", "hopland"])
    def test_all_modes_have_physics_file(self, rhs, basin):
        cfg = rhs.BASIN_CONFIGS[basin]
        for mode in ("mts", "daily", "hourly"):
            assert mode in cfg["physics_file"], f"{basin} missing physics_file for mode={mode}"
            assert isinstance(cfg["physics_file"][mode], str)

    @pytest.mark.parametrize("basin", ["calpella", "warm_springs", "guerneville", "hopland"])
    def test_features_with_physics_nonempty(self, rhs, basin):
        cfg = rhs.BASIN_CONFIGS[basin]
        assert isinstance(cfg["features_with_physics"], list)
        assert len(cfg["features_with_physics"]) > 0


# ---------------------------------------------------------------------------
# Test: MODE flags
# ---------------------------------------------------------------------------
class TestModeFlags:
    def test_mts_flags(self, rhs):
        is_mts, hourly = rhs.MODE_FLAGS["mts"]
        assert is_mts is True
        assert hourly is True

    def test_daily_flags(self, rhs):
        is_mts, hourly = rhs.MODE_FLAGS["daily"]
        assert is_mts is False
        assert hourly is False

    def test_hourly_flags(self, rhs):
        is_mts, hourly = rhs.MODE_FLAGS["hourly"]
        assert is_mts is False
        assert hourly is True

    def test_mode_flags_has_three_modes(self, rhs):
        assert set(rhs.MODE_FLAGS.keys()) == {"mts", "daily", "hourly"}


# ---------------------------------------------------------------------------
# Test: arg tuple lengths (33 for no-physics, 35 for physics)
# ---------------------------------------------------------------------------
class TestArgTupleLengths:
    def _build_no_physics_args(self, rhs):
        """Build a no-physics arg tuple the same way the script does."""
        return (
            0, (64, 0.2, 120, 168, 1, 300, 64), list(rhs.HYPERPARAM_SPACE.keys()),
            Path("/fake/csv"), Path("/fake/yaml"),
            -1, "/fake/runs", "TEST", "STAMP", True,
            MagicMock(),  # fractional_multi_lr
            1, False, False,
            True, True,  # IS_MTS, HOURLY
            False,  # use_cv
            "October", 2, 1,
            *rhs._NO_FOLD_TAIL
        )

    def _build_physics_args(self, rhs):
        """Build a physics arg tuple the same way the script does."""
        return (
            0, (64, 0.2, 120, 168, 1, 300, 64), list(rhs.HYPERPARAM_SPACE.keys()),
            Path("/fake/csv"), Path("/fake/yaml"),
            -1, "/fake/runs", "TEST", "STAMP", True,
            MagicMock(),  # fractional_multi_lr
            1, False, False,
            ["feat1", "feat2"],  # features_with_physics
            Path("/fake/physics.csv"),  # physics_data_file
            True, True,  # IS_MTS, HOURLY
            False,  # use_cv
            "October", 2, 1,
            *rhs._NO_FOLD_TAIL
        )

    def test_no_physics_arg_count_33(self, rhs):
        args = self._build_no_physics_args(rhs)
        assert len(args) == 33, f"Expected 33 args for no_physics, got {len(args)}"

    def test_physics_arg_count_35(self, rhs):
        args = self._build_physics_args(rhs)
        assert len(args) == 35, f"Expected 35 args for physics, got {len(args)}"

    def test_no_fold_tail_length_13(self, rhs):
        assert len(rhs._NO_FOLD_TAIL) == 13, f"Expected 13, got {len(rhs._NO_FOLD_TAIL)}"

    def test_no_fold_tail_values(self, rhs):
        tail = rhs._NO_FOLD_TAIL
        # First 10 should be None
        for i in range(10):
            assert tail[i] is None, f"_NO_FOLD_TAIL[{i}] should be None"
        # fold_id=0, total_folds=1, cv_external_queue_mode=False
        assert tail[10] == 0
        assert tail[11] == 1
        assert tail[12] is False


# ---------------------------------------------------------------------------
# Test: production defaults
# ---------------------------------------------------------------------------
class TestProductionDefaults:
    def test_epochs_not_test_value(self, rhs):
        epochs = rhs.HYPERPARAM_SPACE["epochs"]
        assert epochs != [2], "epochs should be [300], not test value [2]"
        assert epochs == [300]

    def test_run_label(self, rhs):
        assert rhs.RUN_LABEL == "CROSS_VAL_V5"

    def test_use_cv_default(self, rhs):
        assert rhs.USE_CV is True

    def test_mode_default(self, rhs):
        assert rhs.MODE == "mts"


# ---------------------------------------------------------------------------
# Test: import succeeds (module-level code doesn't crash)
# ---------------------------------------------------------------------------
class TestImport:
    def test_import_no_errors(self, rhs):
        assert hasattr(rhs, "main")
        assert hasattr(rhs, "BASIN_CONFIGS")
        assert hasattr(rhs, "_NO_FOLD_TAIL")
        assert hasattr(rhs, "generate_cv_folds_external")
        assert hasattr(rhs, "build_external_cv_queue")
        assert hasattr(rhs, "_run_external_cv_queue_job")
        assert hasattr(rhs, "run_bayes_streaming_queue")
        assert hasattr(rhs, "seed_optuna_from_csv")
        assert hasattr(rhs, "suggest_from_space")
        assert hasattr(rhs, "_build_optuna_distributions")


# ---------------------------------------------------------------------------
# Test: external CV queue build
# ---------------------------------------------------------------------------
class TestExternalCVQueue:
    def test_build_external_cv_queue_structure(self, rhs):
        combos = [(64, 0.2), (128, 0.3)]
        folds = [
            {"fold": 1, "train_start_date": "01/10/2000", "train_end_date": "01/10/2002",
             "validation_start_date": "02/10/2002", "validation_end_date": "01/10/2003",
             "val_eval_start": "02/10/2002", "val_eval_end": "01/10/2003",
             "validation_start_per_frequency": None, "dataset_name": None,
             "train_ranges": None, "validation_ranges": None},
            {"fold": 2, "train_start_date": "01/10/2000", "train_end_date": "01/10/2004",
             "validation_start_date": "02/10/2004", "validation_end_date": "01/10/2005",
             "val_eval_start": "02/10/2004", "val_eval_end": "01/10/2005",
             "validation_start_per_frequency": None, "dataset_name": None,
             "train_ranges": None, "validation_ranges": None},
        ]

        queue = rhs.build_external_cv_queue(combos, folds, include_physics=True)
        # 2 combos x 2 folds x 2 types = 8
        assert len(queue) == 8

        # First 4 should be no_physics, last 4 physics
        assert all(q["job_type"] == "no_physics" for q in queue[:4])
        assert all(q["job_type"] == "physics" for q in queue[4:])

    def test_build_external_cv_queue_no_physics_only(self, rhs):
        combos = [(64,)]
        folds = [{"fold": 1, "train_start_date": None, "train_end_date": None,
                  "validation_start_date": None, "validation_end_date": None,
                  "val_eval_start": None, "val_eval_end": None,
                  "validation_start_per_frequency": None, "dataset_name": None,
                  "train_ranges": None, "validation_ranges": None}]

        queue = rhs.build_external_cv_queue(combos, folds, include_physics=False)
        assert len(queue) == 1
        assert queue[0]["job_type"] == "no_physics"

    def test_queue_jobs_have_total_folds(self, rhs):
        combos = [(64,)]
        folds = [
            {"fold": 1, "train_start_date": None, "train_end_date": None,
             "validation_start_date": None, "validation_end_date": None,
             "val_eval_start": None, "val_eval_end": None,
             "validation_start_per_frequency": None, "dataset_name": None,
             "train_ranges": None, "validation_ranges": None},
            {"fold": 2, "train_start_date": None, "train_end_date": None,
             "validation_start_date": None, "validation_end_date": None,
             "val_eval_start": None, "val_eval_end": None,
             "validation_start_per_frequency": None, "dataset_name": None,
             "train_ranges": None, "validation_ranges": None},
        ]
        queue = rhs.build_external_cv_queue(combos, folds, include_physics=False)
        for job in queue:
            assert job["total_folds"] == 2


# ---------------------------------------------------------------------------
# Test: rank score computation
# ---------------------------------------------------------------------------
class TestRankScore:
    def test_compute_rank_score_mts(self, rhs):
        # Module should be in mts mode (default)
        result = {"NSE_1D": 0.8, "NSE_1H": 0.9}
        score = rhs._compute_rank_score(result)
        expected = 0.3 * 0.8 + 0.7 * 0.9  # 0.87
        assert abs(score - expected) < 1e-9

    def test_aggregate_fold_metrics(self, rhs):
        folds = [
            {"NSE_1D": 0.8, "NSE_1H": 0.9},
            {"NSE_1D": 0.7, "NSE_1H": 0.85},
        ]
        metrics = rhs._aggregate_fold_metrics(folds)
        assert abs(metrics["NSE_1D"] - 0.75) < 1e-9
        assert abs(metrics["NSE_1H"] - 0.875) < 1e-9
        assert "_rank_score" in metrics


# ---------------------------------------------------------------------------
# Test: Hopland basin config details
# ---------------------------------------------------------------------------
class TestHopland:
    def test_hopland_yaml_key_mts(self, rhs):
        assert rhs.BASIN_CONFIGS["hopland"]["yaml_key"]["mts"] == "hopland_mtslstm2"

    def test_hopland_yaml_key_daily(self, rhs):
        assert rhs.BASIN_CONFIGS["hopland"]["yaml_key"]["daily"] == "hopland_gage_nlayers"

    def test_hopland_physics_file_mts(self, rhs):
        assert rhs.BASIN_CONFIGS["hopland"]["physics_file"]["mts"] == "Hopland_hourly.csv"

    def test_hopland_features_count(self, rhs):
        feats = rhs.BASIN_CONFIGS["hopland"]["features_with_physics"]
        assert len(feats) == 30
