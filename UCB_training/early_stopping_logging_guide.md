# Early Stopping Guide

This guide explains the early stopping functionality available in the NeuralHydrology training pipeline.

## Overview

Early stopping automatically halts training when the model stops improving on the validation set, preventing overfitting and saving computational resources. Three modes are supported:

1. **patience** - Traditional approach: stops after N validations without improvement
2. **slope** - Trend-based approach: stops when the validation loss trend shows no meaningful improvement
3. **none** - Disables early stopping

## Configuration

Add an `early_stopping` section to your YAML config:

```yaml
# Enable early stopping
early_stopping: True

# Choose mode: "patience", "slope", or "none"
early_stopping_mode: "patience"

# --- Patience Mode Settings ---
patience_early_stopping: 5
min_delta_early_stopping: 0.0
minimum_epochs_before_early_stopping: 10

# --- Slope Mode Settings ---
early_stopping_slope_window: 7
early_stopping_slope_patience: 2
early_stopping_slope_min_epoch: 8
early_stopping_slope_ema_alpha: 0.4
early_stopping_slope_use_log: True
early_stopping_slope_log_eps: 1.0e-12
early_stopping_slope_eps_slope: 1.0e-3
early_stopping_slope_min_window_gain: 0.01
early_stopping_slope_variance_guard: True
early_stopping_slope_variance_cv_max: 0.01
```

## Mode: Patience

Stops training when validation loss does not improve for `patience_early_stopping` consecutive validation checks.

**Key Parameters:**
- `patience_early_stopping` (int): Number of validations without improvement before stopping (default: 5)
- `min_delta_early_stopping` (float): Minimum change to qualify as improvement (default: 0.0)
- `minimum_epochs_before_early_stopping` (int): Minimum epochs before early stopping can trigger (default: 1)

**Example:**
```yaml
early_stopping: True
early_stopping_mode: "patience"
patience_early_stopping: 5
minimum_epochs_before_early_stopping: 10
```

With `validate_every: 5`, this configuration will stop training after 25 epochs (5 validations × 5 patience) without improvement, but not before epoch 10.

## Mode: Slope

Stops training when the *trend* of validation loss shows no meaningful improvement over a rolling window. Uses:
- **EMA smoothing** to reduce noise
- **Log-space analysis** so thresholds represent approximate percent changes
- **Theil-Sen regression** for robust slope estimation (resistant to outliers)

**Key Parameters:**
- `early_stopping_slope_window` (int): Rolling window size in epochs (default: 7)
- `early_stopping_slope_patience` (int): Consecutive non-improving trend checks before stopping (default: 2)
- `early_stopping_slope_min_epoch` (int): Minimum epoch before slope checks begin (default: 8)
- `early_stopping_slope_ema_alpha` (float): EMA smoothing factor in (0,1]; higher = more reactive (default: 0.4)
- `early_stopping_slope_use_log` (bool): Apply log transform to smoothed loss (default: True)
- `early_stopping_slope_eps_slope` (float): Per-epoch improvement threshold in log space (~% change/epoch) (default: 1e-3 ≈ 0.1%)
- `early_stopping_slope_min_window_gain` (float): Required total improvement across window in log space (default: 0.01 ≈ 1%)
- `early_stopping_slope_variance_guard` (bool): Check window stability before acting (default: True)
- `early_stopping_slope_variance_cv_max` (float): Max coefficient of variation to consider stable (default: 0.01 ≈ 1%)

**Example:**
```yaml
early_stopping: True
early_stopping_mode: "slope"
early_stopping_slope_window: 7
early_stopping_slope_patience: 2
early_stopping_slope_min_epoch: 8
```

### Understanding Slope Mode Thresholds

In log space, the slope and gain represent **approximate percent changes**:

- `eps_slope = 1e-3` means requiring ~0.1% improvement per epoch
- `min_window_gain = 0.01` means requiring ~1% total improvement over the window

This makes thresholds portable across different loss scales.

### Variance Guard

The variance guard (`variance_guard: True`) prevents premature stopping on jittery/unstable windows:
- If the coefficient of variation (CV) in the window exceeds `variance_cv_max`, the check is treated as inconclusive
- Training continues without incrementing the bad-check counter
- Useful for noisy validation curves

## Observability

### Training Logs

Both modes log decisions to the console:

**Patience mode:**
```
ES[patience]: epoch=15, new best=0.123456, bad=0/5
ES[patience]: epoch=20, loss=0.125000, best=0.123456, bad=1/5
ES[patience]: STOP at epoch=35 after 5 checks without improvement
```

**Slope mode:**
```
ES[slope]: epoch=15, slope=-2.3e-04, delta_win=-0.0123, cv=5.2e-03, bad=0/2, decision=continue
ES[slope]: epoch=20, slope=1.5e-04, delta_win=0.0045, cv=8.1e-03, bad=1/2, decision=counted
ES[slope]: STOP at epoch=25 after 2 consecutive non-improving trend checks
```

### TensorBoard Metrics (Slope Mode)

When `log_tensorboard: True`, slope mode writes:
- `es/slope` - Theil-Sen slope estimate
- `es/delta_window` - Total fractional change over window
- `es/cv` - Coefficient of variation (if variance_guard enabled)
- `es/bad_checks` - Consecutive non-improving checks
- `es/trigger` - 1 at the stopping epoch, 0 otherwise

### Loss Curve Plots

The `plot_loss_curves` function (in `UCB_plotting.py`) can annotate where slope-based stopping *would have* triggered:

```python
from UCB_training.UCB_plotting import plot_loss_curves

plot_loss_curves(run_dir, annotate_slope_stop=True)
```

This adds an orange vertical line showing the hypothetical slope-stop epoch.

## Checkpointing and Resume

Early stopper state is automatically saved with model checkpoints:
- State files: `early_stopper_state_epoch{N:03d}.pt`
- Restored automatically when resuming training with `is_continue_training: True`

## Choosing a Mode

| Scenario | Recommended Mode | Reason |
|----------|------------------|---------|
| Standard training, smooth validation curves | `patience` | Simple, well-understood |
| Noisy validation curves | `slope` with `variance_guard: True` | Trend-based, robust to jitter |
| Want to catch late improvements | `slope` | Looks at trend, not just best-so-far |
| Very long training (>100 epochs) | `slope` | Detects plateau without waiting for many checks |
| Short training (<20 epochs) | `patience` or `none` | Slope mode needs sufficient data |

## Backward Compatibility

Existing configs without `early_stopping` settings will work unchanged:
- `early_stopping: False` (default) - No early stopping
- Legacy `patience_early_stopping` keys are still supported

To enable:
```yaml
early_stopping: True
early_stopping_mode: "patience"  # default
```

## Advanced: Tuning Slope Mode

If slope mode stops **too early**:
- Decrease `eps_slope` (e.g., 1e-4) to require less improvement
- Decrease `min_window_gain` (e.g., 0.005) to require less total gain
- Increase `patience` (e.g., 3) to allow more non-improving checks
- Disable `variance_guard` if validation is very stable

If slope mode stops **too late**:
- Increase `eps_slope` (e.g., 5e-3) to require more improvement
- Increase `min_window_gain` (e.g., 0.02) to require more total gain
- Decrease `window` (e.g., 5) for faster response to plateau
- Decrease `patience` (e.g., 1) to stop sooner

## Example Configs

### Conservative Patience Mode
```yaml
early_stopping: True
early_stopping_mode: "patience"
patience_early_stopping: 10
minimum_epochs_before_early_stopping: 20
validate_every: 5
```

### Aggressive Slope Mode
```yaml
early_stopping: True
early_stopping_mode: "slope"
early_stopping_slope_window: 5
early_stopping_slope_patience: 1
early_stopping_slope_min_epoch: 10
early_stopping_slope_eps_slope: 2.0e-3
early_stopping_slope_min_window_gain: 0.02
validate_every: 5
```

### Slope Mode for Noisy Validation
```yaml
early_stopping: True
early_stopping_mode: "slope"
early_stopping_slope_window: 9
early_stopping_slope_patience: 3
early_stopping_slope_min_epoch: 12
early_stopping_slope_ema_alpha: 0.3  # More smoothing
early_stopping_slope_variance_guard: True
early_stopping_slope_variance_cv_max: 0.02  # More tolerant
validate_every: 3
```
