"""
Slope-based early stopping for training neural networks.

This module provides early stopping based on trend analysis of validation loss,
using exponential moving average (EMA) smoothing and Theil-Sen regression.
"""

import logging
from collections import deque
from typing import Dict, Optional, Any
import numpy as np

LOGGER = logging.getLogger(__name__)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Theil-Sen slope estimate (median of pairwise slopes).

    Robust to single outliers. O(k^2) complexity, acceptable for small windows.

    Parameters
    ----------
    x : np.ndarray
        X values (typically epoch numbers)
    y : np.ndarray
        Y values (typically log-transformed EMA validation loss)

    Returns
    -------
    float
        Median slope estimate in units of y-change per x-unit
    """
    if len(x) < 2:
        return 0.0

    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[j] - x[i]
            if abs(dx) > 1e-12:
                slopes.append((y[j] - y[i]) / dx)

    return float(np.median(slopes)) if slopes else 0.0


class PatienceEarlyStopper:
    """
    Traditional patience-based early stopping.

    Stops training when validation loss does not improve for `patience` consecutive checks.

    Parameters
    ----------
    patience : int
        Number of validations without improvement before stopping
    min_delta : float, optional
        Minimum change to qualify as an improvement (default: 0.0)
    min_epoch : int, optional
        Minimum epoch before early stopping can trigger (default: 1)
    logger : logging.Logger, optional
        Logger instance for reporting
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, min_epoch: int = 1,
                 logger: Optional[logging.Logger] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.logger = logger or LOGGER

        self.best_loss = float('inf')
        self.bad_checks = 0
        self.history = []

    def update(self, epoch: int, val_loss: float) -> bool:
        """
        Update with new validation loss.

        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss for this epoch

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        self.history.append((epoch, val_loss))

        if epoch < self.min_epoch:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_checks = 0
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"ES[patience]: epoch={epoch}, new best={val_loss:.6f}, bad=0/{self.patience}")
        else:
            self.bad_checks += 1
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"ES[patience]: epoch={epoch}, loss={val_loss:.6f}, best={self.best_loss:.6f}, "
                    f"bad={self.bad_checks}/{self.patience}"
                )

            if self.bad_checks >= self.patience:
                self.logger.info(
                    f"ES[patience]: STOP at epoch={epoch} after {self.bad_checks} checks without improvement"
                )
                return True

        return False

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'best_loss': self.best_loss,
            'bad_checks': self.bad_checks,
            'history': list(self.history)
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.best_loss = state['best_loss']
        self.bad_checks = state['bad_checks']
        self.history = state['history']

    def reset(self):
        """Reset stopper state."""
        self.best_loss = float('inf')
        self.bad_checks = 0
        self.history = []


class SlopeEarlyStopper:
    """
    Slope-based early stopping using EMA-smoothed log-loss and Theil-Sen trend.

    Stops training when the trend of validation loss over a rolling window shows no meaningful
    improvement for `patience` consecutive checks. Uses log-space analysis so thresholds represent
    approximate percent changes per epoch, making them portable across different loss scales.

    Parameters
    ----------
    window : int
        Number of epochs in the rolling window for trend analysis (default: 7)
    patience : int
        Consecutive non-improving trend checks before stopping (default: 2)
    min_epoch : int
        Minimum epoch before slope checks begin (default: 8)
    ema_alpha : float
        EMA smoothing factor in (0,1]; higher = more reactive (default: 0.4)
    use_log : bool
        Apply log transform to smoothed loss (default: True)
    log_eps : float
        Epsilon added before log if any loss <= 0 (default: 1e-12)
    eps_slope : float
        Per-epoch improvement threshold in log space (~% change/epoch) (default: 1e-3)
    min_window_gain : float
        Required total improvement across window in log space (~%) (default: 0.01)
    variance_guard : bool
        If True, check window stability before acting (default: True)
    variance_cv_max : float
        Max coefficient of variation in log space to consider stable (default: 0.01)
    logger : logging.Logger, optional
        Logger for reporting decisions
    tb_writer : object, optional
        TensorBoard SummaryWriter for logging scalars

    Notes
    -----
    - Slope in log space ≈ fractional (percent) change per epoch
    - eps_slope=1e-3 means requiring ~0.1% improvement/epoch
    - min_window_gain=0.01 means requiring ~1% total improvement over the window
    - Theil-Sen is robust to single outliers in the window
    - EMA reduces noise while remaining responsive to trends
    - Optional variance guard prevents acting on jittery/unstable windows
    """

    def __init__(self,
                 window: int = 7,
                 patience: int = 2,
                 min_epoch: int = 8,
                 ema_alpha: float = 0.4,
                 use_log: bool = True,
                 log_eps: float = 1e-12,
                 eps_slope: float = 1e-3,
                 min_window_gain: float = 0.01,
                 variance_guard: bool = True,
                 variance_cv_max: float = 0.01,
                 logger: Optional[logging.Logger] = None,
                 tb_writer: Optional[Any] = None):

        # Validate parameters
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0,1], got {ema_alpha}")
        if eps_slope < 0:
            raise ValueError(f"eps_slope must be >= 0, got {eps_slope}")
        if min_window_gain < 0:
            raise ValueError(f"min_window_gain must be >= 0, got {min_window_gain}")

        if min_epoch < window:
            if logger:
                logger.warning(
                    f"min_epoch ({min_epoch}) < window ({window}); slope checks may use partial windows initially"
                )

        self.window = window
        self.patience = patience
        self.min_epoch = min_epoch
        self.ema_alpha = ema_alpha
        self.use_log = use_log
        self.log_eps = log_eps
        self.eps_slope = eps_slope
        self.min_window_gain = min_window_gain
        self.variance_guard = variance_guard
        self.variance_cv_max = variance_cv_max
        self.logger = logger or LOGGER
        self.tb_writer = tb_writer

        # Internal state
        self.history = deque()  # (epoch, raw_val_loss)
        self.ema_vals = deque()  # smoothed values
        self.bad_checks = 0
        self._log_fallback_warned = False
        self._last_diagnostics = {}

    def update(self, epoch: int, val_loss: float) -> bool:
        """
        Update with new validation loss and check if early stop should trigger.

        Parameters
        ----------
        epoch : int
            Current epoch number
        val_loss : float
            Validation loss for this epoch

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        # Add to history
        self.history.append((epoch, val_loss))

        # Compute EMA
        if len(self.ema_vals) == 0:
            ema_val = val_loss
        else:
            ema_val = self.ema_alpha * val_loss + (1 - self.ema_alpha) * self.ema_vals[-1]
        self.ema_vals.append(ema_val)

        # Not enough data or too early
        if len(self.history) < max(self.min_epoch, 2):
            return False

        # Extract epochs and EMA values
        epochs_arr = np.array([e for e, _ in self.history])
        ema_arr = np.array(list(self.ema_vals))

        # Apply log transform
        if self.use_log:
            if np.all(ema_arr > 0):
                y = np.log(ema_arr)
            else:
                y = np.log(ema_arr + self.log_eps)
                if not self._log_fallback_warned:
                    self.logger.warning(
                        f"ES[slope]: validation loss <= 0 detected; using log(loss + {self.log_eps})"
                    )
                    self._log_fallback_warned = True
        else:
            y = ema_arr

        # Define rolling window
        k = min(self.window, len(epochs_arr))
        epochs_w = epochs_arr[-k:]
        y_w = y[-k:]

        # Compute Theil-Sen slope
        slope = _theil_sen_slope(epochs_w, y_w)

        # Compute window gain
        delta_window = (y_w[-1] - y_w[0]) / (abs(y_w[0]) + 1e-12)

        # Compute variance (CV) if guard enabled
        cv = 0.0
        if self.variance_guard:
            mean_y = np.mean(y_w)
            std_y = np.std(y_w, ddof=1) if len(y_w) > 1 else 0.0
            cv = std_y / (abs(mean_y) + 1e-12)

        # Store diagnostics
        self._last_diagnostics = {
            'slope': slope,
            'delta_window': delta_window,
            'cv': cv,
            'window_start_epoch': int(epochs_w[0]),
            'window_end_epoch': int(epochs_w[-1]),
            'ema_alpha': self.ema_alpha,
            'use_log': self.use_log,
            'eps_slope': self.eps_slope,
            'min_window_gain': self.min_window_gain,
            'checks_without_improve': self.bad_checks
        }

        # Check improvement conditions
        improving_slope = (slope < -self.eps_slope)
        improving_gain = (delta_window < -self.min_window_gain)

        # Variance guard: if window is too jittery, treat as inconclusive (continue)
        if self.variance_guard and cv >= self.variance_cv_max:
            decision = "continue (jitter)"
            # Do not increment bad_checks
        elif improving_slope and improving_gain:
            self.bad_checks = 0
            decision = "continue"
        else:
            self.bad_checks += 1
            decision = "counted" if self.bad_checks < self.patience else "STOP"

        # Log decision
        self.logger.info(
            f"ES[slope]: epoch={epoch}, slope={slope:.4e}, delta_win={delta_window:.4e}, "
            f"cv={cv:.4e}, bad={self.bad_checks}/{self.patience}, decision={decision}"
        )

        # TensorBoard logging
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('es/slope', slope, epoch)
            self.tb_writer.add_scalar('es/delta_window', delta_window, epoch)
            if self.variance_guard:
                self.tb_writer.add_scalar('es/cv', cv, epoch)
            self.tb_writer.add_scalar('es/bad_checks', self.bad_checks, epoch)
            if decision == "STOP":
                self.tb_writer.add_scalar('es/trigger', 1, epoch)
            else:
                self.tb_writer.add_scalar('es/trigger', 0, epoch)

        if self.bad_checks >= self.patience:
            self.logger.info(
                f"ES[slope]: STOP at epoch={epoch} after {self.bad_checks} consecutive non-improving trend checks"
            )
            return True

        return False

    @property
    def last_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information from the last update call."""
        return self._last_diagnostics.copy()

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'history': list(self.history),
            'ema_vals': list(self.ema_vals),
            'bad_checks': self.bad_checks,
            '_log_fallback_warned': self._log_fallback_warned,
            '_last_diagnostics': self._last_diagnostics.copy()
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.history = deque(state['history'])
        self.ema_vals = deque(state['ema_vals'])
        self.bad_checks = state['bad_checks']
        self._log_fallback_warned = state.get('_log_fallback_warned', False)
        self._last_diagnostics = state.get('_last_diagnostics', {})

    def reset(self):
        """Reset stopper state."""
        self.history.clear()
        self.ema_vals.clear()
        self.bad_checks = 0
        self._log_fallback_warned = False
        self._last_diagnostics = {}


def create_early_stopper(cfg, logger: Optional[logging.Logger] = None,
                         tb_writer: Optional[Any] = None):
    """
    Factory function to create an early stopper based on config mode.

    Parameters
    ----------
    cfg : Config
        Run configuration object with early_stopping attributes
    logger : logging.Logger, optional
        Logger for reporting
    tb_writer : object, optional
        TensorBoard SummaryWriter

    Returns
    -------
    stopper : PatienceEarlyStopper, SlopeEarlyStopper, or None
        Early stopper instance, or None if mode is "none" or early_stopping disabled
    """
    # Check if early stopping is enabled
    if not cfg.early_stopping:
        return None

    mode = cfg.early_stopping_mode

    if mode == "none":
        return None
    elif mode == "patience":
        return PatienceEarlyStopper(
            patience=cfg.patience_early_stopping,
            min_delta=cfg.min_delta_early_stopping,
            min_epoch=cfg.minimum_epochs_before_early_stopping,
            logger=logger
        )
    elif mode == "slope":
        return SlopeEarlyStopper(
            window=cfg.early_stopping_slope_window,
            patience=cfg.early_stopping_slope_patience,
            min_epoch=cfg.early_stopping_slope_min_epoch,
            ema_alpha=cfg.early_stopping_slope_ema_alpha,
            use_log=cfg.early_stopping_slope_use_log,
            log_eps=cfg.early_stopping_slope_log_eps,
            eps_slope=cfg.early_stopping_slope_eps_slope,
            min_window_gain=cfg.early_stopping_slope_min_window_gain,
            variance_guard=cfg.early_stopping_slope_variance_guard,
            variance_cv_max=cfg.early_stopping_slope_variance_cv_max,
            logger=logger,
            tb_writer=tb_writer
        )
    else:
        raise ValueError(f"Unknown early_stopping mode: '{mode}'. Must be 'patience', 'slope', or 'none'.")
