"""Tests for MC Dropout uncertainty estimation.

Uses a tiny random PyTorch model — no HuggingFace downloads.
Key assertion: N stochastic passes produce non-zero variance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from uncertainty_classifier.uncertainty.mc_dropout import (
    mc_dropout_mode,
    mc_dropout_predict,
    epistemic_uncertainty,
    confidence_interval,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class TinyDropoutModel(nn.Module):
    """Tiny linear model with dropout for testing."""

    def __init__(self, input_dim: int = 16, num_classes: int = 3, p: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.drop = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


def make_inputs(batch: int = 1, dim: int = 16) -> dict:
    torch.manual_seed(99)
    return {"x": torch.randn(batch, dim)}


def _forward_fn(model: TinyDropoutModel, inputs: dict) -> torch.Tensor:
    return model(inputs["x"])


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestMcDropoutMode:
    def test_dropout_active_in_eval(self):
        """With mc_dropout_mode, outputs differ across passes."""
        model = TinyDropoutModel(p=0.5)
        inputs = make_inputs()
        outputs = []
        with mc_dropout_mode(model):
            with torch.no_grad():
                for _ in range(10):
                    out = _forward_fn(model, inputs)
                    outputs.append(out.numpy().copy())
        outputs = np.stack(outputs)
        # At least some passes must differ
        assert outputs.std(axis=0).max() > 0.0, "All MC passes identical — dropout is off"

    def test_without_mc_mode_passes_identical(self):
        """Standard model.eval() gives identical outputs (sanity check)."""
        model = TinyDropoutModel(p=0.5)
        model.eval()
        inputs = make_inputs()
        outputs = []
        with torch.no_grad():
            for _ in range(5):
                out = _forward_fn(model, inputs)
                outputs.append(out.numpy().copy())
        outputs = np.stack(outputs)
        assert outputs.std(axis=0).max() == 0.0, "Expected deterministic eval"

    def test_restores_eval_after_context(self):
        """After context manager exits, model returns to full eval mode."""
        model = TinyDropoutModel(p=0.5)
        model.eval()
        with mc_dropout_mode(model):
            pass
        # Check dropout is off
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                assert not m.training


class TestMcDropoutPredict:
    def test_returns_correct_shapes(self):
        model = TinyDropoutModel(num_classes=3, p=0.3)
        inputs = make_inputs(batch=1)
        mean_p, std_p, all_p = mc_dropout_predict(
            model, inputs, n_passes=20, forward_fn=_forward_fn
        )
        assert mean_p.shape == (3,), f"mean_probs shape {mean_p.shape}"
        assert std_p.shape == (3,)
        assert all_p.shape == (20, 3)

    def test_mean_probs_sum_to_one(self):
        model = TinyDropoutModel(num_classes=4, p=0.2)
        inputs = make_inputs()
        mean_p, _, _ = mc_dropout_predict(
            model, inputs, n_passes=15, forward_fn=_forward_fn
        )
        assert abs(mean_p.sum() - 1.0) < 1e-5

    def test_nonzero_variance(self):
        """Core MC-dropout property: std > 0 with dropout active."""
        model = TinyDropoutModel(num_classes=3, p=0.5)
        inputs = make_inputs()
        _, std_p, _ = mc_dropout_predict(
            model, inputs, n_passes=50, forward_fn=_forward_fn
        )
        assert std_p.max() > 1e-6, "Variance is zero — MC dropout is broken"

    def test_epistemic_uncertainty_positive(self):
        model = TinyDropoutModel(p=0.4)
        inputs = make_inputs()
        _, std_p, _ = mc_dropout_predict(
            model, inputs, n_passes=20, forward_fn=_forward_fn
        )
        unc = epistemic_uncertainty(std_p)
        assert unc >= 0.0

    def test_confidence_interval_ordered(self):
        model = TinyDropoutModel(num_classes=2, p=0.4)
        inputs = make_inputs()
        _, _, all_p = mc_dropout_predict(
            model, inputs, n_passes=30, forward_fn=_forward_fn
        )
        lo, hi = confidence_interval(all_p, predicted_class=0)
        assert lo <= hi, f"CI inverted: {lo} > {hi}"
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_higher_dropout_gives_higher_variance(self):
        """Higher dropout rate → higher epistemic uncertainty on average."""
        torch.manual_seed(1)
        inputs = make_inputs(batch=1, dim=32)

        def run_unc(p):
            uncs = []
            for seed in range(5):
                torch.manual_seed(seed)
                model = TinyDropoutModel(input_dim=32, p=p)
                _, std_p, _ = mc_dropout_predict(
                    model, inputs, n_passes=50, forward_fn=_forward_fn
                )
                uncs.append(epistemic_uncertainty(std_p))
            return float(np.mean(uncs))

        unc_low = run_unc(0.1)
        unc_high = run_unc(0.8)
        assert unc_high > unc_low, (
            f"Expected higher dropout to give higher uncertainty: {unc_high} vs {unc_low}"
        )
