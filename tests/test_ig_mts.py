"""Tests for MTS-LSTM Integrated Gradients attribution analysis.

Tests the new MTS-specific functions in UCB_training/ucb_captum.py:
- load_model_for_ig (feature_names fix for MTS configs)
- load_input_data_mts
- _make_captum_forward_mts
- compute_ig_mts
- run_ig_analysis_mts

Uses mock models for unit tests and real checkpoints for integration tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Test checkpoint path (for integration tests)
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = Path(
    "/Users/canruso/Desktop/UCB-USACE-RR-PROJECT/outputs/calpella/"
    "mts_shared/runs/EXT_SEQ_A_NOCV_20260313T064550Z/testing_run_1303_010430"
)
_DATA_DIR = Path("/Users/canruso/Desktop/UCB-USACE-RR-PROJECT/russian_river_data")
_HAS_CHECKPOINT = _CHECKPOINT_DIR.is_dir() and (_CHECKPOINT_DIR / "model_best.pt").exists()
_HAS_DATA = _DATA_DIR.is_dir()


# ---------------------------------------------------------------------------
# Mock MTS model for unit tests
# ---------------------------------------------------------------------------
class MockMTSModel(nn.Module):
    """Minimal differentiable model mimicking MTS-LSTM's forward interface.

    Accepts data dict with x_d_1D and x_d_1H keys.
    Returns dict with y_hat_1D and y_hat_1H keys.
    Uses linear layers so gradients flow for IG computation.
    """

    def __init__(self, n_feat_1d=7, n_feat_1h=7, hidden=4):
        super().__init__()
        self.fc_1d = nn.Linear(n_feat_1d, 1, bias=False)
        self.fc_1h = nn.Linear(n_feat_1h, 1, bias=False)
        # Cross-frequency path: 1D features influence 1H output (simulates state transfer)
        self.fc_cross = nn.Linear(n_feat_1d, 1, bias=False)

    def forward(self, data):
        x_1d = data["x_d_1D"]  # [batch, seq_1d, feat_1d]
        x_1h = data["x_d_1H"]  # [batch, seq_1h, feat_1h]

        # 1D output: depends only on 1D inputs
        y_hat_1d = self.fc_1d(x_1d)  # [batch, seq_1d, 1]

        # 1H output: depends on 1H inputs + cross-frequency influence from 1D
        y_hat_1h_direct = self.fc_1h(x_1h)  # [batch, seq_1h, 1]
        # Cross-freq: mean of 1D features broadcast to 1H seq dim
        cross = self.fc_cross(x_1d.mean(dim=1, keepdim=True))  # [batch, 1, 1]
        y_hat_1h = y_hat_1h_direct + cross  # broadcasting adds cross-freq signal

        return {"y_hat_1D": y_hat_1d, "y_hat_1H": y_hat_1h}


class MockMTSModelAsymmetric(nn.Module):
    """Mock MTS model with different feature counts per frequency."""

    def __init__(self, n_feat_1d=5, n_feat_1h=9):
        super().__init__()
        self.fc_1d = nn.Linear(n_feat_1d, 1, bias=False)
        self.fc_1h = nn.Linear(n_feat_1h, 1, bias=False)

    def forward(self, data):
        x_1d = data["x_d_1D"]
        x_1h = data["x_d_1H"]
        y_hat_1d = self.fc_1d(x_1d)
        y_hat_1h = self.fc_1h(x_1h)
        return {"y_hat_1D": y_hat_1d, "y_hat_1H": y_hat_1h}


# ---------------------------------------------------------------------------
# 1. Tests for _make_captum_forward_mts
# ---------------------------------------------------------------------------
class TestMakeCaptumForwardMTS:
    """Tests for the Captum forward wrapper for MTS-LSTM."""

    def test_type_a_returns_correct_shape(self):
        """Type A (1D->1D): forward_fn(x_1d) returns scalar per sample [batch]."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 4, 90, 336, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        forward_fn = _make_captum_forward_mts(model, target_freq="1D", vary_freq="1D", hold_tensor=x_1h)
        out = forward_fn(x_1d)

        assert out.shape == (batch,), f"Expected shape ({batch},), got {out.shape}"

    def test_type_b_returns_correct_shape(self):
        """Type B (1H->1H): forward_fn(x_1h) returns scalar per sample [batch]."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 4, 90, 336, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        forward_fn = _make_captum_forward_mts(model, target_freq="1H", vary_freq="1H", hold_tensor=x_1d)
        out = forward_fn(x_1h)

        assert out.shape == (batch,), f"Expected shape ({batch},), got {out.shape}"

    def test_type_c_returns_correct_shape(self):
        """Type C (1D->1H cross-freq): forward_fn(x_1d) returns scalar per sample from y_hat_1H."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 4, 90, 336, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        forward_fn = _make_captum_forward_mts(model, target_freq="1H", vary_freq="1D", hold_tensor=x_1h)
        out = forward_fn(x_1d)

        assert out.shape == (batch,), f"Expected shape ({batch},), got {out.shape}"

    def test_type_a_hold_tensor_used(self):
        """Type A: changing hold_tensor (1H) should NOT affect output since target is 1D."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 2, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h_a = torch.randn(batch, seq_1h, feat)
        x_1h_b = torch.randn(batch, seq_1h, feat) * 10.0

        fn_a = _make_captum_forward_mts(model, target_freq="1D", vary_freq="1D", hold_tensor=x_1h_a)
        fn_b = _make_captum_forward_mts(model, target_freq="1D", vary_freq="1D", hold_tensor=x_1h_b)

        out_a = fn_a(x_1d)
        out_b = fn_b(x_1d)

        # For MockMTSModel, y_hat_1D depends only on x_1d via fc_1d, so outputs should match
        assert torch.allclose(out_a, out_b, atol=1e-6), \
            "Type A output should not depend on held-constant 1H tensor"

    def test_type_b_hold_tensor_used(self):
        """Type B: changing hold_tensor (1D) should NOT affect output since target is 1H direct part."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        # Use asymmetric model with no cross-frequency path
        model = MockMTSModelAsymmetric(n_feat_1d=5, n_feat_1h=7)
        model.eval()
        batch, seq_1d, seq_1h = 2, 10, 24
        x_1d_a = torch.randn(batch, seq_1d, 5)
        x_1d_b = torch.randn(batch, seq_1d, 5) * 10.0
        x_1h = torch.randn(batch, seq_1h, 7)

        fn_a = _make_captum_forward_mts(model, target_freq="1H", vary_freq="1H", hold_tensor=x_1d_a)
        fn_b = _make_captum_forward_mts(model, target_freq="1H", vary_freq="1H", hold_tensor=x_1d_b)

        out_a = fn_a(x_1h)
        out_b = fn_b(x_1h)

        assert torch.allclose(out_a, out_b, atol=1e-6), \
            "Type B output (no cross-freq model) should not depend on held-constant 1D tensor"

    def test_type_c_cross_frequency_gradient_flows(self):
        """Type C: varying 1D inputs should change y_hat_1H output (cross-freq effect)."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()  # has cross-frequency path
        model.eval()
        batch, seq_1d, seq_1h, feat = 2, 10, 24, 7
        x_1h = torch.randn(batch, seq_1h, feat)

        forward_fn = _make_captum_forward_mts(model, target_freq="1H", vary_freq="1D", hold_tensor=x_1h)

        x_1d_a = torch.randn(batch, seq_1d, feat)
        x_1d_b = torch.randn(batch, seq_1d, feat) * 5.0

        out_a = forward_fn(x_1d_a)
        out_b = forward_fn(x_1d_b)

        assert not torch.allclose(out_a, out_b, atol=1e-6), \
            "Type C: different 1D inputs should produce different 1H outputs via cross-freq path"

    def test_forward_is_differentiable(self):
        """Forward wrapper output must be differentiable w.r.t. the varied input."""
        from UCB_training.ucb_captum import _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 2, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat, requires_grad=True)
        x_1h = torch.randn(batch, seq_1h, feat)

        forward_fn = _make_captum_forward_mts(model, target_freq="1D", vary_freq="1D", hold_tensor=x_1h)
        out = forward_fn(x_1d)
        loss = out.sum()
        loss.backward()

        assert x_1d.grad is not None, "Gradient should flow to varied input"
        assert x_1d.grad.shape == x_1d.shape, "Gradient shape should match input shape"


# ---------------------------------------------------------------------------
# 2. Tests for compute_ig_mts
# ---------------------------------------------------------------------------
class TestComputeIgMTS:
    """Tests for MTS Integrated Gradients computation."""

    def test_output_shape_type_a(self):
        """Type A: attributions shape matches [n_samples, seq_len_1D, n_features_1D]."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel(n_feat_1d=7, n_feat_1h=7)
        model.eval()
        batch, seq_1d, seq_1h, feat = 3, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5)

        assert attrs.shape == (batch, seq_1d, feat), \
            f"Expected shape ({batch}, {seq_1d}, {feat}), got {attrs.shape}"

    def test_output_shape_type_b(self):
        """Type B: attributions shape matches [n_samples, seq_len_1H, n_features_1H]."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel(n_feat_1d=7, n_feat_1h=7)
        model.eval()
        batch, seq_1d, seq_1h, feat = 3, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1H", vary_freq="1H", n_steps=5)

        assert attrs.shape == (batch, seq_1h, feat), \
            f"Expected shape ({batch}, {seq_1h}, {feat}), got {attrs.shape}"

    def test_output_shape_type_c(self):
        """Type C: attributions shape matches varied input [n_samples, seq_len_1D, n_features_1D]."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel(n_feat_1d=7, n_feat_1h=7)
        model.eval()
        batch, seq_1d, seq_1h, feat = 3, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1H", vary_freq="1D", n_steps=5)

        assert attrs.shape == (batch, seq_1d, feat), \
            f"Expected shape ({batch}, {seq_1d}, {feat}), got {attrs.shape}"

    def test_zero_baseline_zero_input_gives_zero_attributions(self):
        """Zero baseline + zero input = zero attributions (no path to integrate)."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 2, 10, 24, 7
        x_1d = torch.zeros(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", baseline="zero", n_steps=10)

        assert torch.allclose(attrs, torch.zeros_like(attrs), atol=1e-6), \
            "Zero input with zero baseline should produce zero attributions"

    def test_completeness_axiom(self):
        """IG completeness: sum of attributions approx equals f(x) - f(baseline) per sample."""
        from UCB_training.ucb_captum import compute_ig_mts, _make_captum_forward_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 2, 10, 24, 7

        torch.manual_seed(42)
        x_1d = torch.randn(batch, seq_1d, feat)
        x_1h = torch.randn(batch, seq_1h, feat)
        baseline_1d = torch.zeros_like(x_1d)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", baseline="zero", n_steps=100)

        # f(x) - f(baseline) for each sample
        forward_fn = _make_captum_forward_mts(model, target_freq="1D", vary_freq="1D", hold_tensor=x_1h)
        with torch.no_grad():
            f_x = forward_fn(x_1d)
            f_base = forward_fn(baseline_1d)

        expected_diff = f_x - f_base  # [batch]
        attr_sum = attrs.sum(dim=(1, 2))  # [batch]

        assert torch.allclose(attr_sum, expected_diff, atol=0.05), \
            f"Completeness violated: attr_sum={attr_sum.tolist()}, expected={expected_diff.tolist()}"

    def test_baseline_mean(self):
        """Mean baseline uses per-feature mean of the varied input."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        batch, seq_1d, seq_1h, feat = 3, 10, 24, 7
        x_1d = torch.randn(batch, seq_1d, feat) + 5.0  # offset from zero
        x_1h = torch.randn(batch, seq_1h, feat)

        # Should not raise
        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", baseline="mean", n_steps=5)

        assert attrs.shape == (batch, seq_1d, feat)
        # With mean baseline, attributions should generally be smaller than with zero baseline
        # (since baseline is closer to the input)
        attrs_zero = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", baseline="zero", n_steps=5)
        assert attrs.abs().mean() <= attrs_zero.abs().mean() * 1.5, \
            "Mean baseline attributions should not be wildly larger than zero baseline"

    def test_invalid_baseline_raises(self):
        """Unknown baseline string raises ValueError."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        with pytest.raises(ValueError, match="[Uu]nknown baseline"):
            compute_ig_mts(model, x_1d, x_1h, baseline="uniform")

    def test_attributions_are_detached(self):
        """Returned attributions should be detached (no grad tracking)."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5)
        assert not attrs.requires_grad, "Attributions should be detached"

    def test_single_sample(self):
        """IG should work with n_samples=1 (single sequence)."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(1, 10, 7)
        x_1h = torch.randn(1, 24, 7)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5)
        assert attrs.shape == (1, 10, 7)

    def test_different_feature_counts_per_frequency(self):
        """1D has 5 features, 1H has 9 features - attributions have correct feature dim."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModelAsymmetric(n_feat_1d=5, n_feat_1h=9)
        model.eval()
        batch = 2
        x_1d = torch.randn(batch, 10, 5)
        x_1h = torch.randn(batch, 24, 9)

        # Type A: vary 1D (5 features)
        attrs_a = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5)
        assert attrs_a.shape == (batch, 10, 5), f"Expected 5 features for 1D, got {attrs_a.shape}"

        # Type B: vary 1H (9 features)
        attrs_b = compute_ig_mts(model, x_1d, x_1h, target_freq="1H", vary_freq="1H", n_steps=5)
        assert attrs_b.shape == (batch, 24, 9), f"Expected 9 features for 1H, got {attrs_b.shape}"


# ---------------------------------------------------------------------------
# 3. Tests for load_model_for_ig feature_names fix
# ---------------------------------------------------------------------------
class TestLoadModelForIgFeatureNames:
    """Tests for the feature_names handling in load_model_for_ig for MTS configs."""

    def test_mts_config_returns_dict(self):
        """When dynamic_inputs is a dict, feature_names should be a dict of lists."""
        from UCB_training.ucb_captum import load_model_for_ig

        mock_cfg = MagicMock()
        mock_cfg.dynamic_inputs = {
            "1D": ["precip", "temp", "humidity"],
            "1H": ["precip", "temp", "humidity", "wind"],
        }

        with patch("UCB_training.ucb_captum.Config", return_value=mock_cfg), \
             patch("UCB_training.ucb_captum.get_model") as mock_get_model, \
             patch("UCB_training.ucb_captum.torch.load", return_value={}):

            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_get_model.return_value = mock_model

            # Create a temp run_dir with a config.yml and model file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                (tmpdir / "config.yml").touch()
                (tmpdir / "model_best.pt").touch()

                _, _, feature_names = load_model_for_ig(tmpdir)

        assert isinstance(feature_names, dict), f"Expected dict, got {type(feature_names)}"
        assert set(feature_names.keys()) == {"1D", "1H"}
        assert feature_names["1D"] == ["precip", "temp", "humidity"]
        assert feature_names["1H"] == ["precip", "temp", "humidity", "wind"]

    def test_single_freq_config_returns_list(self):
        """When dynamic_inputs is a list (single-freq), feature_names stays a list."""
        from UCB_training.ucb_captum import load_model_for_ig

        mock_cfg = MagicMock()
        mock_cfg.dynamic_inputs = ["precip", "temp", "humidity"]

        with patch("UCB_training.ucb_captum.Config", return_value=mock_cfg), \
             patch("UCB_training.ucb_captum.get_model") as mock_get_model, \
             patch("UCB_training.ucb_captum.torch.load", return_value={}):

            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_get_model.return_value = mock_model

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                (tmpdir / "config.yml").touch()
                (tmpdir / "model_best.pt").touch()

                _, _, feature_names = load_model_for_ig(tmpdir)

        assert isinstance(feature_names, list), f"Expected list, got {type(feature_names)}"
        assert feature_names == ["precip", "temp", "humidity"]


# ---------------------------------------------------------------------------
# 4. Tests for rank_features with MTS attribution tensors
# ---------------------------------------------------------------------------
class TestRankFeaturesFromMTS:
    """Verify existing rank_features() works with MTS attribution tensors."""

    def test_rank_features_with_mts_1d_attrs(self):
        """rank_features() should work with MTS attribution tensors (generic shape)."""
        from UCB_training.ucb_captum import rank_features

        n_samples, seq_len, n_feat = 5, 90, 7
        attrs = torch.randn(n_samples, seq_len, n_feat)
        feature_names = [f"feat_{i}" for i in range(n_feat)]

        df = rank_features(attrs, feature_names)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["feature", "mean_abs_attr", "rank"]
        assert len(df) == n_feat
        assert list(df["rank"]) == list(range(1, n_feat + 1))
        assert all(df["mean_abs_attr"] > 0)
        # Verify sorted descending
        assert df["mean_abs_attr"].is_monotonic_decreasing

    def test_rank_features_with_mts_1h_attrs(self):
        """rank_features() works with 1H-shaped attributions (longer seq_len)."""
        from UCB_training.ucb_captum import rank_features

        n_samples, seq_len, n_feat = 5, 336, 7
        attrs = torch.randn(n_samples, seq_len, n_feat)
        feature_names = [f"feat_{i}" for i in range(n_feat)]

        df = rank_features(attrs, feature_names)
        assert len(df) == n_feat
        assert df["rank"].iloc[0] == 1


# ---------------------------------------------------------------------------
# 5. Tests for save/load attributions with MTS metadata
# ---------------------------------------------------------------------------
class TestSaveLoadAttributionsMTS:
    """Tests for saving and loading MTS attributions with frequency metadata."""

    def test_roundtrip_with_frequency_metadata(self):
        """Save and load preserves tensor data and MTS metadata."""
        from UCB_training.ucb_captum import save_attributions, load_attributions

        attrs = torch.randn(5, 90, 7)
        feature_names = ["precip", "temp", "humidity", "wind", "solar", "et", "flow"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_attrs.pt"
            save_attributions(
                attrs, feature_names, path,
                basin="calpella",
                experiment="EXT_SEQ_A",
                phys_type="NP",
                target_freq="1D",
                vary_freq="1D",
            )

            loaded_attrs, meta = load_attributions(path)

        assert torch.allclose(loaded_attrs, attrs)
        assert meta["feature_names"] == feature_names
        assert meta["basin"] == "calpella"
        assert meta["target_freq"] == "1D"
        assert meta["vary_freq"] == "1D"

    def test_roundtrip_1h_attributions(self):
        """Save/load works for 1H-shaped attributions."""
        from UCB_training.ucb_captum import save_attributions, load_attributions

        attrs = torch.randn(5, 336, 7)
        feature_names = [f"feat_{i}" for i in range(7)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_1h.pt"
            save_attributions(attrs, feature_names, path, target_freq="1H", vary_freq="1H")
            loaded_attrs, meta = load_attributions(path)

        assert loaded_attrs.shape == (5, 336, 7)
        assert meta["target_freq"] == "1H"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge case tests for MTS IG."""

    def test_invalid_vary_freq_raises(self):
        """vary_freq not in ['1D', '1H'] should raise ValueError."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        with pytest.raises(ValueError):
            compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="2D")

    def test_invalid_target_freq_raises(self):
        """target_freq not in ['1D', '1H'] should raise ValueError."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        with pytest.raises(ValueError):
            compute_ig_mts(model, x_1d, x_1h, target_freq="invalid", vary_freq="1D")

    def test_batch_size_one(self):
        """Single sample batch should work without errors."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(1, 10, 7)
        x_1h = torch.randn(1, 24, 7)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5)
        assert attrs.shape == (1, 10, 7)

    def test_large_batch(self):
        """Larger batch size should work (tests internal_batch_size handling)."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(20, 10, 7)
        x_1h = torch.randn(20, 24, 7)

        attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1D", n_steps=5, internal_batch_size=4)
        assert attrs.shape == (20, 10, 7)

    def test_all_valid_freq_combinations_no_error(self):
        """All 3 valid (target_freq, vary_freq) combos should run without error."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        combos = [("1D", "1D"), ("1H", "1H"), ("1H", "1D")]
        for target, vary in combos:
            attrs = compute_ig_mts(model, x_1d, x_1h, target_freq=target, vary_freq=vary, n_steps=3)
            assert attrs.ndim == 3, f"combo ({target}, {vary}) returned wrong ndim"

    def test_vary_1h_target_1d_raises_or_works(self):
        """vary_freq=1H, target_freq=1D is unusual but should either work or raise cleanly."""
        from UCB_training.ucb_captum import compute_ig_mts

        model = MockMTSModel()
        model.eval()
        x_1d = torch.randn(2, 10, 7)
        x_1h = torch.randn(2, 24, 7)

        # This combo (1H->1D) is questionable but the implementation should either
        # handle it gracefully or raise a clear ValueError
        try:
            attrs = compute_ig_mts(model, x_1d, x_1h, target_freq="1D", vary_freq="1H", n_steps=3)
            # If it works, shape should match varied input (1H)
            assert attrs.shape == (2, 24, 7)
        except ValueError:
            # Explicitly disallowed - also acceptable
            pass


# ---------------------------------------------------------------------------
# 7. Integration tests (real model checkpoint)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_CHECKPOINT or not _HAS_DATA, reason="Calpella MTS checkpoint or data not found")
class TestIntegrationRealCheckpoint:
    """Integration tests using real trained MTS model checkpoint."""

    def test_run_ig_analysis_mts_type_a(self):
        """Type A (1D->1D) with real model returns valid ranked DataFrame."""
        from UCB_training.ucb_captum import run_ig_analysis_mts

        df = run_ig_analysis_mts(
            _CHECKPOINT_DIR,
            target_freq="1D",
            vary_freq="1D",
            period="test",
            n_steps=10,
            n_samples=5,
            baseline="zero",
            device="cpu",
            data_dir=_DATA_DIR,
        )

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_attr" in df.columns
        assert "rank" in df.columns
        assert len(df) == 7, f"Expected 7 features, got {len(df)}"
        assert all(df["mean_abs_attr"] > 0), "All features should have nonzero attribution"
        assert list(df["rank"]) == list(range(1, 8))

    def test_run_ig_analysis_mts_type_b(self):
        """Type B (1H->1H) with real model returns valid ranked DataFrame."""
        from UCB_training.ucb_captum import run_ig_analysis_mts

        df = run_ig_analysis_mts(
            _CHECKPOINT_DIR,
            target_freq="1H",
            vary_freq="1H",
            period="test",
            n_steps=10,
            n_samples=5,
            baseline="zero",
            device="cpu",
            data_dir=_DATA_DIR,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7
        assert all(df["mean_abs_attr"] > 0)

    def test_run_ig_analysis_mts_return_attributions(self):
        """return_attributions=True returns (DataFrame, tensor, feature_names)."""
        from UCB_training.ucb_captum import run_ig_analysis_mts

        result = run_ig_analysis_mts(
            _CHECKPOINT_DIR,
            target_freq="1D",
            vary_freq="1D",
            period="test",
            n_steps=10,
            n_samples=5,
            device="cpu",
            data_dir=_DATA_DIR,
            return_attributions=True,
        )

        assert isinstance(result, tuple) and len(result) == 3
        df, attrs, feat_names = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(attrs, torch.Tensor)
        assert isinstance(feat_names, list)
        assert attrs.shape[2] == len(feat_names), \
            f"Feature dim {attrs.shape[2]} != len(feature_names) {len(feat_names)}"
        assert attrs.shape[0] == 5, "Should have n_samples=5 samples"
        # seq_len should be 90 for 1D
        assert attrs.shape[1] == 90, f"Expected seq_len=90 for 1D, got {attrs.shape[1]}"

    def test_load_input_data_mts_shapes(self):
        """Loaded MTS data has correct shapes for both frequencies."""
        from UCB_training.ucb_captum import load_model_for_ig, load_input_data_mts

        _, cfg, _ = load_model_for_ig(_CHECKPOINT_DIR, data_dir=_DATA_DIR)
        data = load_input_data_mts(cfg, _CHECKPOINT_DIR, period="test", n_samples=10)

        assert "1D" in data and "1H" in data
        x_1d, dates_1d = data["1D"]
        x_1h, dates_1h = data["1H"]

        assert x_1d.shape[0] == x_1h.shape[0], "Both freqs must have same batch size"
        assert x_1d.shape[0] <= 10, "n_samples should limit to <= 10"
        assert x_1d.shape[1] == 90, f"Expected seq_len 1D=90, got {x_1d.shape[1]}"
        assert x_1h.shape[1] == 336, f"Expected seq_len 1H=336, got {x_1h.shape[1]}"
        assert x_1d.shape[2] == 7, f"Expected 7 features for 1D, got {x_1d.shape[2]}"
        assert x_1h.shape[2] == 7, f"Expected 7 features for 1H, got {x_1h.shape[2]}"

    def test_load_input_data_mts_all_samples(self):
        """n_samples=None returns all available samples."""
        from UCB_training.ucb_captum import load_model_for_ig, load_input_data_mts

        _, cfg, _ = load_model_for_ig(_CHECKPOINT_DIR, data_dir=_DATA_DIR)
        data_all = load_input_data_mts(cfg, _CHECKPOINT_DIR, period="test", n_samples=None)
        data_sub = load_input_data_mts(cfg, _CHECKPOINT_DIR, period="test", n_samples=5)

        n_all = data_all["1D"][0].shape[0]
        n_sub = data_sub["1D"][0].shape[0]
        assert n_all >= n_sub, "All samples should be >= subsampled count"
        assert n_sub == min(5, n_all)

    def test_load_model_for_ig_mts_feature_names(self):
        """Real MTS checkpoint returns dict feature_names."""
        from UCB_training.ucb_captum import load_model_for_ig

        model, cfg, feature_names = load_model_for_ig(_CHECKPOINT_DIR, data_dir=_DATA_DIR)
        assert isinstance(feature_names, dict), \
            f"MTS config should return dict feature_names, got {type(feature_names)}"
        assert "1D" in feature_names and "1H" in feature_names
        assert len(feature_names["1D"]) == 7
        assert len(feature_names["1H"]) == 7

    def test_type_c_cross_frequency_real(self):
        """Type C (1D->1H cross-frequency) works with real model."""
        from UCB_training.ucb_captum import run_ig_analysis_mts

        df = run_ig_analysis_mts(
            _CHECKPOINT_DIR,
            target_freq="1H",
            vary_freq="1D",
            period="test",
            n_steps=10,
            n_samples=3,
            device="cpu",
            data_dir=_DATA_DIR,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7
        # Cross-frequency attributions should exist (non-zero) due to state transfer
        assert any(df["mean_abs_attr"] > 0), "Cross-freq attributions should be nonzero"


# ---------------------------------------------------------------------------
# 8. Regression tests - existing single-freq functions unchanged
# ---------------------------------------------------------------------------
class TestRegressionSingleFreq:
    """Verify existing single-freq functions are not broken by MTS additions."""

    def test_make_captum_forward_still_works(self):
        """Original _make_captum_forward still works for single-freq CudaLSTM-like models."""
        from UCB_training.ucb_captum import _make_captum_forward

        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(7, 1, bias=False)
            def forward(self, data):
                return {"y_hat": self.fc(data["x_d"])}

        model = SimpleLSTM()
        model.eval()
        forward_fn = _make_captum_forward(model)

        x = torch.randn(3, 10, 7)
        out = forward_fn(x)
        assert out.shape == (3,)

    def test_compute_ig_still_works(self):
        """Original compute_ig still works for single-freq models."""
        from UCB_training.ucb_captum import compute_ig

        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(7, 1, bias=False)
            def forward(self, data):
                return {"y_hat": self.fc(data["x_d"])}

        model = SimpleLSTM()
        model.eval()
        x = torch.randn(3, 10, 7)

        attrs = compute_ig(model, x, baseline="zero", n_steps=5)
        assert attrs.shape == (3, 10, 7)

    def test_rank_features_unchanged(self):
        """rank_features() produces identical output to before (no behavioral change)."""
        from UCB_training.ucb_captum import rank_features

        torch.manual_seed(123)
        attrs = torch.randn(5, 10, 3)
        names = ["a", "b", "c"]

        df = rank_features(attrs, names)
        assert len(df) == 3
        assert list(df.columns) == ["feature", "mean_abs_attr", "rank"]

    def test_save_load_attributions_unchanged(self):
        """save_attributions/load_attributions roundtrip works as before."""
        from UCB_training.ucb_captum import save_attributions, load_attributions

        attrs = torch.randn(5, 10, 3)
        names = ["a", "b", "c"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_attributions(attrs, names, path, basin="test", experiment="X", phys_type="NP")
            loaded, meta = load_attributions(path)

        assert torch.allclose(loaded, attrs)
        assert meta["feature_names"] == names
        assert meta["basin"] == "test"

    def test_build_combined_df_unchanged(self):
        """build_combined_df still works with 3-tuple keys."""
        from UCB_training.ucb_captum import build_combined_df, rank_features

        attrs1 = torch.randn(5, 10, 3)
        attrs2 = torch.randn(5, 10, 3)
        names = ["a", "b", "c"]

        df1 = rank_features(attrs1, names)
        df2 = rank_features(attrs2, names)

        combined = build_combined_df({
            ("basin1", "exp1", "NP"): df1,
            ("basin2", "exp2", "NP"): df2,
        })

        assert len(combined) == 6
        assert "basin" in combined.columns
        assert "experiment" in combined.columns
        assert "phys_type" in combined.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
