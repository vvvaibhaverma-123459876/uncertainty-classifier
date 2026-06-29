"""Calibration metrics: ECE (Expected Calibration Error) and Brier Score."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - confidence| per bin,
    and returns the sample-weighted average.

    Args:
        probs: Softmax probabilities, shape (n, num_classes).
        labels: True integer labels, shape (n,).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    confidences = probs.max(axis=1)  # top-class probability
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include right edge in last bin
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score for multi-class classification.

    Brier = (1/n) * sum_i sum_c (p_ic - 1[y_i == c])^2

    Args:
        probs: Softmax probabilities, shape (n, num_classes).
        labels: True integer labels, shape (n,).

    Returns:
        Brier score in [0, 2]. Lower is better.
    """
    n, num_classes = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
