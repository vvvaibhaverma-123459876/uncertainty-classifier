"""Tests for conformal prediction.

All tests use synthetic random data — no model downloads.
Key assertion: empirical coverage >= (1 - alpha) on held-out data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from uncertainty_classifier.uncertainty.conformal import ConformalPredictor


RNG = np.random.default_rng(42)


def make_probs(n: int, num_classes: int = 4, rng=RNG) -> tuple[np.ndarray, np.ndarray]:
    """Random softmax probabilities and true labels."""
    logits = rng.standard_normal((n, num_classes))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    labels = rng.integers(0, num_classes, size=n)
    return probs, labels


class TestConformalPredictor:
    def test_calibrate_returns_q_hat(self):
        cp = ConformalPredictor()
        probs, labels = make_probs(200)
        q = cp.calibrate(probs, labels, alpha=0.1)
        assert 0.0 <= q <= 1.0
        assert cp.q_hat == q

    def test_q_hat_increases_with_lower_alpha(self):
        """Lower alpha (higher coverage) → larger q_hat (more inclusive sets)."""
        cp1, cp2 = ConformalPredictor(), ConformalPredictor()
        probs, labels = make_probs(500)
        q_90 = cp1.calibrate(probs, labels, alpha=0.10)
        q_80 = cp2.calibrate(probs, labels, alpha=0.20)
        assert q_90 >= q_80

    def test_predict_set_non_empty(self):
        cp = ConformalPredictor()
        probs, labels = make_probs(300)
        cp.calibrate(probs, labels, alpha=0.1)
        test_probs, _ = make_probs(1)
        pset = cp.predict_set(test_probs[0])
        assert len(pset) >= 1

    def test_predict_set_before_calibrate_raises(self):
        cp = ConformalPredictor()
        probs, _ = make_probs(1)
        with pytest.raises(RuntimeError):
            cp.predict_set(probs[0])

    def test_empirical_coverage_at_90(self):
        """Core guarantee: marginal coverage >= 0.9 with finite-sample correction."""
        rng = np.random.default_rng(0)
        num_classes = 4
        n_cal = 1000
        n_test = 2000
        alpha = 0.10
        tol = 0.04  # allow up to 4% slack for finite test set

        cal_probs, cal_labels = make_probs(n_cal, num_classes, rng)
        test_probs, test_labels = make_probs(n_test, num_classes, rng)

        cp = ConformalPredictor()
        cp.calibrate(cal_probs, cal_labels, alpha=alpha)

        covered = 0
        for p, y in zip(test_probs, test_labels):
            pset = cp.predict_set(p)
            if y in pset:
                covered += 1

        empirical_coverage = covered / n_test
        assert empirical_coverage >= (1 - alpha) - tol, (
            f"Coverage {empirical_coverage:.4f} < {1 - alpha - tol:.4f}"
        )

    def test_empirical_coverage_at_80(self):
        """Coverage guarantee also holds at alpha=0.20."""
        rng = np.random.default_rng(7)
        num_classes = 3
        n_cal, n_test = 800, 1500
        alpha = 0.20
        tol = 0.04

        cal_probs, cal_labels = make_probs(n_cal, num_classes, rng)
        test_probs, test_labels = make_probs(n_test, num_classes, rng)

        cp = ConformalPredictor()
        cp.calibrate(cal_probs, cal_labels, alpha=alpha)

        covered = sum(
            1 for p, y in zip(test_probs, test_labels) if y in cp.predict_set(p)
        )
        empirical_coverage = covered / n_test
        assert empirical_coverage >= (1 - alpha) - tol, (
            f"Coverage {empirical_coverage:.4f} < {1 - alpha - tol:.4f}"
        )

    def test_larger_alpha_gives_smaller_average_set_size(self):
        """Higher miscoverage rate → smaller sets on average."""
        rng = np.random.default_rng(1)
        cal_probs, cal_labels = make_probs(500, 4, rng)
        test_probs, _ = make_probs(200, 4, rng)

        cp_tight = ConformalPredictor()
        cp_loose = ConformalPredictor()
        cp_tight.calibrate(cal_probs, cal_labels, alpha=0.30)
        cp_loose.calibrate(cal_probs, cal_labels, alpha=0.05)

        avg_tight = np.mean([len(cp_tight.predict_set(p)) for p in test_probs])
        avg_loose = np.mean([len(cp_loose.predict_set(p)) for p in test_probs])
        assert avg_tight <= avg_loose

    def test_invalid_inputs(self):
        cp = ConformalPredictor()
        with pytest.raises(ValueError):
            cp.calibrate(np.ones((10, 3)), np.ones(5), alpha=0.1)  # length mismatch
        with pytest.raises(ValueError):
            cp.calibrate(np.ones((10, 3)), np.ones(10), alpha=1.1)  # bad alpha
        with pytest.raises(ValueError):
            cp.calibrate(np.ones((10,)), np.ones(10), alpha=0.1)  # 1-D probs

    def test_batch_prediction(self):
        rng = np.random.default_rng(2)
        cal_probs, cal_labels = make_probs(200, 4, rng)
        test_probs, _ = make_probs(50, 4, rng)
        cp = ConformalPredictor()
        cp.calibrate(cal_probs, cal_labels, alpha=0.1)
        sets = cp.predict_sets_batch(test_probs)
        assert len(sets) == 50
        assert all(len(s) >= 1 for s in sets)
