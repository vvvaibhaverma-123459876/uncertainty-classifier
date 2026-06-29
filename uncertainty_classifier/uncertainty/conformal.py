"""Conformal prediction for classification.

Uses split-conformal (inductive conformal prediction) with the LAC (Least
Ambiguous set-valued Classifier) nonconformity score.

For class y, the nonconformity score is:
    s(x, y) = 1 - softmax(logits)[y]

Given a calibration set of size n at level alpha, the quantile threshold is:
    q_hat = quantile(scores, ceil((n+1)*(1-alpha)) / n, method="higher")

Prediction set: C(x) = {y : 1 - p_y <= q_hat}

The finite-sample (n+1) correction guarantees marginal coverage:
    P(Y_new in C(X_new)) >= 1 - alpha
for exchangeable (calibration, test) data.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class ConformalPredictor:
    """Split-conformal predictor with guaranteed coverage at 1-alpha."""

    def __init__(self) -> None:
        self._q_hat: float | None = None
        self._n_cal: int = 0
        self._cal_scores: np.ndarray | None = None  # stored for per-request alpha

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        alpha: float = 0.1,
    ) -> float:
        """Compute q_hat from a calibration set.

        Args:
            cal_probs: Softmax probabilities, shape (n, num_classes).
            cal_labels: True integer labels, shape (n,).
            alpha: Miscoverage rate; coverage target is 1 - alpha.

        Returns:
            q_hat threshold value.
        """
        if cal_probs.ndim != 2:
            raise ValueError("cal_probs must be 2-D (n, num_classes)")
        if cal_labels.ndim != 1 or len(cal_labels) != len(cal_probs):
            raise ValueError("cal_labels must be 1-D with same length as cal_probs")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        n = len(cal_labels)
        # Nonconformity score: 1 - probability of true class
        scores = 1.0 - cal_probs[np.arange(n), cal_labels]

        # Finite-sample corrected quantile level
        level = math.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)  # clip to [0,1]

        self._q_hat = float(np.quantile(scores, level, method="higher"))
        self._n_cal = n
        return self._q_hat

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_set(self, probs: np.ndarray) -> List[int]:
        """Return prediction set for a single example.

        Args:
            probs: Softmax probabilities, shape (num_classes,).

        Returns:
            List of class indices included in the prediction set.
        """
        if self._q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_set()")
        prediction_set = [
            int(i) for i, p in enumerate(probs) if (1.0 - p) <= self._q_hat
        ]
        # Guarantee at least one class (the most probable) to avoid empty sets
        if not prediction_set:
            prediction_set = [int(np.argmax(probs))]
        return prediction_set

    def predict_sets_batch(self, probs: np.ndarray) -> List[List[int]]:
        """Return prediction sets for a batch of examples.

        Args:
            probs: Softmax probabilities, shape (n, num_classes).

        Returns:
            List of prediction sets, one per example.
        """
        return [self.predict_set(p) for p in probs]

    @property
    def q_hat(self) -> float | None:
        """Calibrated quantile threshold."""
        return self._q_hat
