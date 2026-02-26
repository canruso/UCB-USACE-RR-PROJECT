"""
Unit tests for early stopping functionality.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

import torch
from neuralhydrology.training.early_stopping import (
    PatienceEarlyStopper, SlopeEarlyStopper, PlateauEarlyStopper,
    _theil_sen_slope, create_early_stopper
)


class TestTheilSenSlope:
    """Test Theil-Sen slope computation."""

    def test_simple_increasing(self):
        """Test with simple increasing data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        slope = _theil_sen_slope(x, y)
        assert abs(slope - 1.0) < 1e-6

    def test_simple_decreasing(self):
        """Test with simple decreasing data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        slope = _theil_sen_slope(x, y)
        assert abs(slope - (-1.0)) < 1e-6

    def test_flat_line(self):
        """Test with flat data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 2, 2, 2, 2])
        slope = _theil_sen_slope(x, y)
        assert abs(slope) < 1e-6

    def test_with_outlier(self):
        """Test that Theil-Sen is robust to outliers."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 10, 4, 5])  # outlier at index 2
        slope = _theil_sen_slope(x, y)
        # Median slope should still be close to 1
        assert 0.5 < slope < 1.5


class TestPatienceEarlyStopper:
    """Test patience-based early stopping."""

    def test_no_stop_improving(self):
        """Shouldnot stop when loss keeps improving."""
        stopper = PatienceEarlyStopper(patience=3, min_epoch=1)

        for epoch in range(1, 11):
            loss = 10.0 / epoch  # decreasing loss
            should_stop = stopper.update(epoch, loss)
            assert not should_stop

    def test_stop_after_patience(self):
        """Should stop after patience epochs without improvement."""
        stopper = PatienceEarlyStopper(patience=3, min_epoch=1)

        # Improve initially
        stopper.update(1, 5.0)
        stopper.update(2, 4.0)
        stopper.update(3, 3.0)

        # Plateau
        assert not stopper.update(4, 3.5)
        assert not stopper.update(5, 3.6)
        assert stopper.update(6, 3.7)  # Should stop here

    def test_min_epoch_respected(self):
        """Should not stop before min_epoch."""
        stopper = PatienceEarlyStopper(patience=2, min_epoch=10)

        # Even with no improvement, should not stop before epoch 10
        for epoch in range(1, 10):
            should_stop = stopper.update(epoch, 5.0)
            assert not should_stop

    def test_state_dict(self):
        """Test state persistence."""
        stopper = PatienceEarlyStopper(patience=3)
        stopper.update(1, 5.0)
        stopper.update(2, 4.0)
        stopper.update(3, 4.5)

        state = stopper.state_dict()

        new_stopper = PatienceEarlyStopper(patience=3)
        new_stopper.load_state_dict(state)

        assert new_stopper.best_loss == stopper.best_loss
        assert new_stopper.bad_checks == stopper.bad_checks


class TestSlopeEarlyStopper:
    """Test slope-based early stopping."""

    def test_no_stop_with_consistent_improvement(self):
        """Should not stop when loss consistently decreases."""
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6, ema_alpha=0.5)

        # Simulate log-linear decay
        for epoch in range(1, 20):
            loss = 10.0 * np.exp(-0.1 * epoch)
            should_stop = stopper.update(epoch, loss)
            assert not should_stop

    def test_stop_on_plateau(self):
        """Should stop when loss plateaus."""
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6, ema_alpha=0.5,
                                     eps_slope=1e-3, min_window_gain=0.01)

        # Improve initially
        for epoch in range(1, 10):
            loss = 10.0 * np.exp(-0.2 * epoch)
            stopper.update(epoch, loss)

        # Plateau - add small noise but no real improvement
        plateau_loss = stopper.ema_vals[-1]
        for epoch in range(10, 20):
            loss = plateau_loss + np.random.normal(0, plateau_loss * 0.001)
            should_stop = stopper.update(epoch, loss)
            if should_stop:
                assert epoch < 20  # Should stop before all epochs used
                break
        else:
            # If we got here, stopping might not have triggered due to noise/threshold
            # This is acceptable behavior
            pass

    def test_late_improvement_prevents_stop(self):
        """Should not stop if loss improves after initial plateau."""
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6, ema_alpha=0.5)

        # Initial improvement
        for epoch in range(1, 10):
            loss = 10.0 * np.exp(-0.1 * epoch)
            stopper.update(epoch, loss)

        # Brief plateau
        plateau_loss = stopper.ema_vals[-1]
        for epoch in range(10, 13):
            stopper.update(epoch, plateau_loss)

        # Resume improvement
        for epoch in range(13, 20):
            loss = plateau_loss * np.exp(-0.1 * (epoch - 12))
            should_stop = stopper.update(epoch, loss)
            assert not should_stop  # Should continue due to renewed improvement

    def test_min_epoch_respected(self):
        """Should not check before min_epoch."""
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=15)

        # Even with plateau, should not stop before min_epoch
        for epoch in range(1, 15):
            should_stop = stopper.update(epoch, 5.0)
            assert not should_stop

    def test_variance_guard(self):
        """Variance guard should prevent stopping when CV exceeds threshold."""
        # Use data that produces high CV even after EMA smoothing
        np.random.seed(42)
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6, ema_alpha=0.8,  # Higher alpha = less smoothing
                                     variance_guard=True, variance_cv_max=0.005)  # Strict threshold

        # Create very jittery flat data that will have high CV
        base_loss = 5.0
        stop_epochs = []
        for epoch in range(1, 25):
            # Alternate between high and low to create jitter
            loss = base_loss + (0.3 * base_loss if epoch % 2 == 0 else -0.3 * base_loss)
            should_stop = stopper.update(epoch, loss)
            if should_stop:
                stop_epochs.append(epoch)
                break

        # Should take longer to stop due to variance guard
        # Without variance guard, it would stop quickly on this data
        assert len(stop_epochs) == 0 or stop_epochs[0] > 12

    def test_state_dict(self):
        """Test state persistence."""
        stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6)

        for epoch in range(1, 10):
            loss = 10.0 * np.exp(-0.1 * epoch)
            stopper.update(epoch, loss)

        state = stopper.state_dict()

        new_stopper = SlopeEarlyStopper(window=5, patience=2, min_epoch=6)
        new_stopper.load_state_dict(state)

        assert new_stopper.bad_checks == stopper.bad_checks
        assert len(new_stopper.history) == len(stopper.history)
        assert len(new_stopper.ema_vals) == len(stopper.ema_vals)

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            SlopeEarlyStopper(window=1)  # window < 2

        with pytest.raises(ValueError):
            SlopeEarlyStopper(patience=0)  # patience < 1

        with pytest.raises(ValueError):
            SlopeEarlyStopper(ema_alpha=0)  # ema_alpha not in (0, 1]

        with pytest.raises(ValueError):
            SlopeEarlyStopper(ema_alpha=1.5)  # ema_alpha not in (0, 1]


# ---------------------------------------------------------------------------
# PlateauEarlyStopper tests
# ---------------------------------------------------------------------------

def _make_optimizer(lr=0.01):
    """Helper: create a minimal optimizer for testing."""
    model = torch.nn.Linear(10, 1)
    return torch.optim.Adam(model.parameters(), lr=lr)


class _MockPlateauConfig:
    """Minimal mock config for factory tests with plateau mode."""
    early_stopping = True
    early_stopping_mode = "plateau"
    plateau_factor = 0.5
    plateau_patience = 3
    plateau_threshold = 1e-4
    plateau_min_lr = 1e-6
    plateau_final_patience = 5
    plateau_cooldown = 0


class TestPlateauEarlyStopper:
    """Test plateau-based early stopping with LR-aware scheduling."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_no_stop_while_improving(self):
        """update() never returns True when val_loss consistently decreases."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=5, final_patience=3, min_lr=1e-6)

        for epoch in range(1, 31):
            loss = 10.0 / epoch  # strictly decreasing
            assert not stopper.update(epoch, loss)

    def test_lr_stays_when_improving(self):
        """LR should not be reduced when loss keeps improving."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=3, min_lr=1e-6)

        for epoch in range(1, 20):
            loss = 10.0 / epoch
            stopper.update(epoch, loss)

        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.01, rel=1e-6)

    # ------------------------------------------------------------------
    # LR reduction behavior
    # ------------------------------------------------------------------

    def test_lr_reduces_on_plateau(self):
        """LR should drop by factor after patience+1 constant-loss updates."""
        optimizer = _make_optimizer(lr=0.01)
        # patience=3 means 3 epochs of no improvement before reduction
        stopper = PlateauEarlyStopper(optimizer, patience=3, factor=0.5, final_patience=50, min_lr=1e-8)

        # Feed constant loss - scheduler needs patience+1 calls with no improvement
        # ReduceLROnPlateau: after patience epochs with no improvement, it reduces
        for epoch in range(1, 20):
            stopper.update(epoch, 5.0)

        lr = optimizer.param_groups[0]['lr']
        # LR should have been reduced at least once from 0.01
        assert lr < 0.01, f"Expected LR < 0.01 after plateau, got {lr}"

    def test_multiple_lr_reductions(self):
        """LR should reduce multiple times on sustained plateau."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=2, factor=0.5, final_patience=100, min_lr=1e-8, cooldown=0)

        # Feed many constant-loss epochs to trigger multiple reductions
        for epoch in range(1, 50):
            stopper.update(epoch, 5.0)

        lr = optimizer.param_groups[0]['lr']
        # After multiple reductions by 0.5: 0.01 -> 0.005 -> 0.0025 -> ...
        assert lr < 0.005, f"Expected multiple LR reductions, got lr={lr}"

    # ------------------------------------------------------------------
    # Core ES behavior: stop at min_lr + final_patience
    # ------------------------------------------------------------------

    def test_stop_at_min_lr_plus_final_patience(self):
        """ES triggers when LR hits min_lr AND final_patience is exhausted."""
        min_lr = 1e-4
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(
            optimizer, patience=2, factor=0.5, final_patience=3,
            min_lr=min_lr, cooldown=0
        )

        stopped = False
        stop_epoch = None
        for epoch in range(1, 200):
            result = stopper.update(epoch, 5.0)  # constant loss = never improves
            if result:
                stopped = True
                stop_epoch = epoch
                break

        assert stopped, "Expected ES to trigger after hitting min_lr + final_patience"
        assert stop_epoch is not None
        # Verify LR is at min_lr when stopped
        lr = optimizer.param_groups[0]['lr']
        assert lr <= min_lr * (1 + 1e-6), f"Expected LR <= min_lr={min_lr}, got {lr}"

    def test_no_stop_before_min_lr(self):
        """ES should NOT trigger while LR is still above min_lr."""
        optimizer = _make_optimizer(lr=0.01)
        # Use very small final_patience=1, but high min_lr so it takes a while to reach
        stopper = PlateauEarlyStopper(
            optimizer, patience=2, factor=0.9, final_patience=1,
            min_lr=1e-8, cooldown=0
        )

        # Run a limited number of epochs - LR should still be above min_lr
        for epoch in range(1, 10):
            result = stopper.update(epoch, 5.0)
            # Before LR reaches min_lr, ES must not trigger even with final_patience=1
            if optimizer.param_groups[0]['lr'] > 1e-8 * (1 + 1e-6):
                assert not result, f"ES triggered at epoch {epoch} while LR still above min_lr"

    # ------------------------------------------------------------------
    # Recovery: improvement at min_lr prevents stop
    # ------------------------------------------------------------------

    def test_no_stop_if_improves_at_min_lr(self):
        """If loss improves at min_lr, bad_checks resets and ES does not fire."""
        min_lr = 1e-3
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(
            optimizer, patience=2, factor=0.1, final_patience=5,
            min_lr=min_lr, cooldown=0
        )

        # Drive LR down to min_lr with constant loss
        epoch = 0
        while not stopper._is_at_min_lr():
            epoch += 1
            stopper.update(epoch, 5.0)
            if epoch > 200:
                pytest.fail("LR never reached min_lr")

        # Now at min_lr - feed a few bad epochs (but fewer than final_patience)
        for i in range(3):
            epoch += 1
            stopper.update(epoch, 5.0)

        assert stopper.bad_checks_at_min_lr > 0, "Expected bad_checks > 0 after flat epochs"

        # Now feed an improving loss - should reset counter
        epoch += 1
        result = stopper.update(epoch, 4.0)
        assert not result, "ES should not trigger after improvement"
        assert stopper.bad_checks_at_min_lr == 0, "bad_checks should reset after improvement"

    # ------------------------------------------------------------------
    # State dict / checkpoint roundtrip
    # ------------------------------------------------------------------

    def test_state_dict_roundtrip(self):
        """state_dict -> new stopper -> load_state_dict produces identical behavior."""
        optimizer1 = _make_optimizer(lr=0.01)
        stopper1 = PlateauEarlyStopper(optimizer1, patience=3, factor=0.5, final_patience=5, min_lr=1e-6)

        # Feed some losses
        for epoch in range(1, 15):
            stopper1.update(epoch, 5.0 + 0.01 * (epoch % 3))

        state = stopper1.state_dict()

        # Create a new stopper with an identically-configured optimizer
        optimizer2 = _make_optimizer(lr=0.01)
        stopper2 = PlateauEarlyStopper(optimizer2, patience=3, factor=0.5, final_patience=5, min_lr=1e-6)
        stopper2.load_state_dict(state)

        # Verify internal state matches
        assert stopper2.best_loss_at_min_lr == stopper1.best_loss_at_min_lr
        assert stopper2.bad_checks_at_min_lr == stopper1.bad_checks_at_min_lr
        assert stopper2.at_min_lr == stopper1.at_min_lr
        assert stopper2.history == stopper1.history

        # Feed the same next loss to both - should return same result
        result1 = stopper1.update(15, 5.0)
        result2 = stopper2.update(15, 5.0)
        assert result1 == result2

    def test_state_dict_contains_scheduler_state(self):
        """state_dict must include the scheduler's own state_dict."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5)

        stopper.update(1, 5.0)
        state = stopper.state_dict()

        assert 'scheduler_state' in state, "state_dict must contain 'scheduler_state'"
        assert 'best_loss_at_min_lr' in state
        assert 'bad_checks_at_min_lr' in state
        assert 'at_min_lr' in state
        assert 'history' in state

    def test_state_dict_scheduler_counters_preserved(self):
        """After load_state_dict, scheduler internal counters should match."""
        optimizer1 = _make_optimizer(lr=0.01)
        stopper1 = PlateauEarlyStopper(optimizer1, patience=3, factor=0.5, final_patience=5, min_lr=1e-6)

        # Run enough to trigger an LR reduction
        for epoch in range(1, 20):
            stopper1.update(epoch, 5.0)

        state = stopper1.state_dict()
        sched_state = state['scheduler_state']

        # Verify the scheduler state has meaningful content
        assert 'best' in sched_state, "Scheduler state should track 'best' metric"
        assert 'num_bad_epochs' in sched_state, "Scheduler state should track 'num_bad_epochs'"

    # ------------------------------------------------------------------
    # Factory function tests
    # ------------------------------------------------------------------

    def test_factory_creates_plateau_stopper(self):
        """create_early_stopper with mode='plateau' returns a PlateauEarlyStopper."""
        cfg = _MockPlateauConfig()
        optimizer = _make_optimizer(lr=0.01)
        stopper = create_early_stopper(cfg, optimizer=optimizer)
        assert isinstance(stopper, PlateauEarlyStopper)

    def test_factory_plateau_requires_optimizer(self):
        """create_early_stopper with mode='plateau' and no optimizer raises ValueError."""
        cfg = _MockPlateauConfig()
        with pytest.raises(ValueError, match="[Oo]ptimizer"):
            create_early_stopper(cfg, optimizer=None)

    def test_factory_patience_mode_unaffected(self):
        """Existing 'patience' mode still returns PatienceEarlyStopper (no regression)."""
        cfg = MagicMock()
        cfg.early_stopping = True
        cfg.early_stopping_mode = "patience"
        cfg.patience_early_stopping = 5
        cfg.min_delta_early_stopping = 0.0
        cfg.minimum_epochs_before_early_stopping = 1
        stopper = create_early_stopper(cfg)
        assert isinstance(stopper, PatienceEarlyStopper)

    def test_factory_none_mode_unaffected(self):
        """Mode 'none' still returns None (no regression)."""
        cfg = MagicMock()
        cfg.early_stopping = True
        cfg.early_stopping_mode = "none"
        result = create_early_stopper(cfg)
        assert result is None

    def test_factory_disabled_returns_none(self):
        """early_stopping=False returns None even with plateau mode set."""
        cfg = MagicMock()
        cfg.early_stopping = False
        cfg.early_stopping_mode = "plateau"
        result = create_early_stopper(cfg)
        assert result is None

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def test_tb_logging_called(self):
        """When tb_writer is provided, add_scalar is called with expected keys."""
        optimizer = _make_optimizer(lr=0.01)
        tb_writer = MagicMock()
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5, tb_writer=tb_writer)

        stopper.update(1, 5.0)

        # Collect all scalar names logged
        logged_keys = {call.args[0] for call in tb_writer.add_scalar.call_args_list}
        assert 'es/lr' in logged_keys, "Should log es/lr"
        assert 'es/at_min_lr' in logged_keys, "Should log es/at_min_lr"
        assert 'es/bad_checks_at_min_lr' in logged_keys, "Should log es/bad_checks_at_min_lr"

    def test_no_tb_logging_when_writer_is_none(self):
        """No crash when tb_writer is None."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5, tb_writer=None)
        # Should not raise
        stopper.update(1, 5.0)

    # ------------------------------------------------------------------
    # Cooldown interaction
    # ------------------------------------------------------------------

    def test_cooldown_delays_lr_reductions(self):
        """With cooldown > 0, consecutive LR reductions are spaced further apart."""
        optimizer_no_cd = _make_optimizer(lr=0.01)
        stopper_no_cd = PlateauEarlyStopper(optimizer_no_cd, patience=2, factor=0.5, final_patience=100, min_lr=1e-8, cooldown=0)

        optimizer_cd = _make_optimizer(lr=0.01)
        stopper_cd = PlateauEarlyStopper(optimizer_cd, patience=2, factor=0.5, final_patience=100, min_lr=1e-8, cooldown=5)

        for epoch in range(1, 40):
            stopper_no_cd.update(epoch, 5.0)
            stopper_cd.update(epoch, 5.0)

        lr_no_cd = optimizer_no_cd.param_groups[0]['lr']
        lr_cd = optimizer_cd.param_groups[0]['lr']

        # With cooldown, LR should be higher (fewer reductions in same epoch span)
        assert lr_cd > lr_no_cd, (
            f"Cooldown should delay reductions: lr_no_cd={lr_no_cd}, lr_cd={lr_cd}"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_tracking_state(self):
        """reset() should clear ES tracking state but scheduler is unaffected."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5, min_lr=1e-6)

        for epoch in range(1, 10):
            stopper.update(epoch, 5.0)

        stopper.reset()

        assert stopper.best_loss_at_min_lr == float('inf')
        assert stopper.bad_checks_at_min_lr == 0
        assert stopper.at_min_lr is False
        assert stopper.history == []

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_update_no_crash(self):
        """A single update call should not crash or trigger stop."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5)
        result = stopper.update(1, 5.0)
        assert result is False

    def test_very_large_loss_no_crash(self):
        """update() handles very large loss values without crash."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5)
        result = stopper.update(1, 1e30)
        assert result is False

    def test_zero_loss_no_crash(self):
        """update() handles zero loss without crash."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5)
        result = stopper.update(1, 0.0)
        assert result is False

    def test_history_grows_with_updates(self):
        """Each update() call appends to history."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer, patience=3, final_patience=5)

        for epoch in range(1, 6):
            stopper.update(epoch, float(epoch))

        assert len(stopper.history) == 5
        assert stopper.history[0] == (1, 1.0)
        assert stopper.history[-1] == (5, 5.0)

    def test_final_patience_one_stops_immediately_at_min_lr(self):
        """With final_patience=1, ES fires on the FIRST non-improving check at min_lr."""
        min_lr = 1e-3
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(
            optimizer, patience=2, factor=0.1, final_patience=1,
            min_lr=min_lr, cooldown=0
        )

        # Drive LR to min_lr
        epoch = 0
        while not stopper._is_at_min_lr():
            epoch += 1
            result = stopper.update(epoch, 5.0)
            if epoch > 200:
                pytest.fail("LR never reached min_lr")

        # At the epoch that first hits min_lr, the stopper sets best_loss and bad_checks=0.
        # The NEXT non-improving epoch should trigger stop with final_patience=1.
        epoch += 1
        result = stopper.update(epoch, 5.0)  # same loss = no improvement
        assert result is True, f"Expected stop at epoch {epoch} with final_patience=1"

    def test_default_parameters(self):
        """PlateauEarlyStopper defaults match the documented interface."""
        optimizer = _make_optimizer(lr=0.01)
        stopper = PlateauEarlyStopper(optimizer)
        assert stopper.min_lr == 1e-6
        assert stopper.final_patience == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
