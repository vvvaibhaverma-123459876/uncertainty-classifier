"""Reliability diagram (calibration curve) data generation.

Returns bin data suitable for plotting: for each confidence bin,
the mean confidence and the fraction correct (accuracy).
A perfectly calibrated model sits on the diagonal y = x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BinData:
    """Data for a single bin of the reliability diagram."""
    mean_confidence: float
    fraction_correct: float
    count: int


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> List[BinData]:
    """Compute reliability diagram data.

    Args:
        probs: Softmax probabilities, shape (n, num_classes).
        labels: True integer labels, shape (n,).
        n_bins: Number of equal-width bins.

    Returns:
        List of BinData for non-empty bins.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    result: List[BinData] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        result.append(BinData(
            mean_confidence=float(confidences[mask].mean()),
            fraction_correct=float(correct[mask].mean()),
            count=int(mask.sum()),
        ))

    return result
