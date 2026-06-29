"""Tests for calibration metrics and temperature scaling.

All tests use synthetic data — no model downloads.
Key assertions:
  - Temperature scaling reduces ECE on overconfident predictions.
  - ECE and Brier score are correctly computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from uncertainty_classifier.calibration.metrics import expected_calibration_error, brier_score
from uncertainty_classifier.calibration.curves import reliability_diagram_data
from uncertainty_classifier.model.temperature import TemperatureScaler


RNG = np.random.default_rng(2024)


# ------------------------------------------------------------------
# Synthetic data factories
# ------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def make_overconfident_logits(n: int, num_classes: int = 2, rng=RNG, sharpness: float = 5.0):
    """Generate logits that are overconfident (T << 1 equivalent).

    True labels are drawn from soft probabilities (T=1), but the logits
    returned are sharpened by `sharpness`, causing the model to be
    overconfident relative to its accuracy.
    """
    base_logits = rng.standard_normal((n, num_classes))
    soft_probs = _softmax(base_logits)
    # Sample labels from soft distribution
    labels = np.array([
        rng.choice(num_classes, p=soft_probs[i]) for i in range(n)
    ])
    # Return sharp logits (overconfident)
    sharp_logits = base_logits * sharpness
    return sharp_logits, labels


def make_calibrated_logits(n: int, num_classes: int = 2, rng=RNG):
    """Logits approximately calibrated (T near 1)."""
    logits = rng.standard_normal((n, num_classes))
    probs = _softmax(logits)
    labels = np.array([rng.choice(num_classes, p=probs[i]) for i in range(n)])
    return logits, labels


# ------------------------------------------------------------------
# ECE tests
# ------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration_low_ece(self):
        """For uniformly random probs + matching label frequencies, ECE is low."""
        rng = np.random.default_rng(0)
        n = 5000
        num_classes = 4
        # Create perfectly calibrated probs: confidence = accuracy by construction
        # Use the true prob as both confidence and label generation
        logits = rng.standard_normal((n, num_classes))
        probs = _softmax(logits)
        labels = np.array([rng.choice(num_classes, p=probs[i]) for i in range(n)])
        ece = expected_calibration_error(probs, labels)
        assert ece < 0.15, f"ECE={ece:.4f} too high for soft-label data"

    def test_overconfident_high_ece(self):
        """Overconfident model has high ECE."""
        rng = np.random.default_rng(1)
        logits, labels = make_overconfident_logits(2000, sharpness=10.0, rng=rng)
        probs = _softmax(logits)
        ece = expected_calibration_error(probs, labels)
        assert ece > 0.05, f"ECE={ece:.4f} unexpectedly low for overconfident model"

    def test_ece_range(self):
        rng = np.random.default_rng(3)
        probs, labels = _softmax(rng.standard_normal((100, 3))), rng.integers(0, 3, 100)
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_ece_zero_for_perfect_model(self):
        """If confidence matches accuracy perfectly, ECE = 0."""
        # Degenerate case: one-hot probs, always correct
        n = 50
        probs = np.eye(3)[np.arange(n) % 3]
        labels = np.arange(n) % 3
        ece = expected_calibration_error(probs, labels)
        assert ece < 1e-6


# ------------------------------------------------------------------
# Brier Score tests
# ------------------------------------------------------------------

class TestBrierScore:
    def test_brier_perfect_prediction(self):
        """One-hot correct predictions → Brier = 0."""
        n, nc = 20, 3
        labels = np.arange(n) % nc
        probs = np.eye(nc)[labels]
        assert brier_score(probs, labels) < 1e-8

    def test_brier_uniform_prediction(self):
        """Uniform probs → Brier = (nc-1)/nc for each sample."""
        n, nc = 100, 4
        probs = np.ones((n, nc)) / nc
        labels = np.zeros(n, dtype=int)
        expected = (nc - 1) / nc  # sum_c (1/nc - 1[y==c])^2 for nc=4
        assert abs(brier_score(probs, labels) - expected) < 1e-6

    def test_brier_range(self):
        rng = np.random.default_rng(5)
        probs = _softmax(rng.standard_normal((200, 3)))
        labels = rng.integers(0, 3, 200)
        bs = brier_score(probs, labels)
        assert 0.0 <= bs <= 2.0


# ------------------------------------------------------------------
# Temperature Scaling tests
# ------------------------------------------------------------------

class TestTemperatureScaling:
    def test_fit_finds_positive_temperature(self):
        rng = np.random.default_rng(10)
        logits, labels = make_overconfident_logits(500, rng=rng)
        ts = TemperatureScaler()
        ts.fit(logits, labels)
        assert ts.temperature > 0.0

    def test_overconfident_temperature_greater_than_one(self):
        """Overconfident model should be corrected by T > 1."""
        rng = np.random.default_rng(11)
        logits, labels = make_overconfident_logits(1000, sharpness=8.0, rng=rng)
        ts = TemperatureScaler()
        ts.fit(logits, labels)
        assert ts.temperature > 1.0, f"Expected T>1 for overconfident model, got {ts.temperature:.3f}"

    def test_temperature_scaling_reduces_ece(self):
        """Core requirement: ECE after temperature scaling < ECE before."""
        rng = np.random.default_rng(12)
        n = 3000
        logits, labels = make_overconfident_logits(n, sharpness=6.0, rng=rng)

        # Val/test split
        n_val = n // 2
        val_logits, val_labels = logits[:n_val], labels[:n_val]
        test_logits, test_labels = logits[n_val:], labels[n_val:]

        ts = TemperatureScaler()
        ts.fit(val_logits, val_labels)

        probs_before = _softmax(test_logits)
        probs_after = ts.transform_probs(test_logits)

        ece_before = expected_calibration_error(probs_before, test_labels)
        ece_after = expected_calibration_error(probs_after, test_labels)

        assert ece_after < ece_before, (
            f"Temperature scaling did not reduce ECE: before={ece_before:.4f}, after={ece_after:.4f}"
        )

    def test_transform_shape_preserved(self):
        rng = np.random.default_rng(13)
        logits = rng.standard_normal((50, 5))
        ts = TemperatureScaler()
        ts.temperature = 1.5
        scaled = ts.transform(logits)
        assert scaled.shape == logits.shape

    def test_transform_probs_sum_to_one(self):
        rng = np.random.default_rng(14)
        logits = rng.standard_normal((100, 4))
        ts = TemperatureScaler()
        ts.temperature = 2.0
        probs = ts.transform_probs(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ------------------------------------------------------------------
# Reliability diagram tests
# ------------------------------------------------------------------

class TestReliabilityDiagram:
    def test_returns_bin_data(self):
        rng = np.random.default_rng(20)
        probs = _softmax(rng.standard_normal((200, 3)))
        labels = rng.integers(0, 3, 200)
        bins = reliability_diagram_data(probs, labels)
        assert len(bins) > 0

    def test_bin_counts_sum_to_n(self):
        rng = np.random.default_rng(21)
        n = 300
        probs = _softmax(rng.standard_normal((n, 4)))
        labels = rng.integers(0, 4, n)
        bins = reliability_diagram_data(probs, labels)
        assert sum(b.count for b in bins) == n

    def test_fraction_correct_in_range(self):
        rng = np.random.default_rng(22)
        probs = _softmax(rng.standard_normal((200, 3)))
        labels = rng.integers(0, 3, 200)
        bins = reliability_diagram_data(probs, labels)
        for b in bins:
            assert 0.0 <= b.fraction_correct <= 1.0
            assert 0.0 <= b.mean_confidence <= 1.0
