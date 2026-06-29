"""Temperature scaling for post-hoc calibration.

Temperature scaling is the simplest post-hoc calibration method:
apply a single scalar T > 0 to the logits before softmax.

    p(y | x, T) = softmax(logits / T)

T > 1 softens the distribution (reduces overconfidence).
T < 1 sharpens it (increases confidence).
T = 1 is no scaling.

Optimal T is found by minimising negative log-likelihood on a validation set.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaler:
    """Post-hoc temperature scaler.

    Fit on a validation set, then call transform() to rescale logits.
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        self.temperature: float = init_temperature
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        bounds: tuple[float, float] = (0.05, 10.0),
    ) -> "TemperatureScaler":
        """Find T that minimises NLL on (logits, labels).

        Args:
            logits: Raw model logits, shape (n, num_classes).
            labels: True integer labels, shape (n,).
            bounds: Search bounds for temperature (must be positive).

        Returns:
            self (fitted)
        """
        def nll(T: float) -> float:
            scaled = logits / T
            # Log-sum-exp numerically stable
            log_probs = scaled - _logsumexp(scaled, axis=1, keepdims=True)
            nll_val = -log_probs[np.arange(len(labels)), labels].mean()
            return float(nll_val)

        result = minimize_scalar(nll, bounds=bounds, method="bounded")
        self.temperature = float(result.x)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits by 1/T.

        Args:
            logits: Raw logits, shape (..., num_classes).

        Returns:
            Scaled logits same shape.
        """
        return logits / self.temperature

    def transform_probs(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities.

        Args:
            logits: Raw logits, shape (n, num_classes).

        Returns:
            Softmax probabilities shape (n, num_classes).
        """
        scaled = self.transform(logits)
        return _softmax(scaled)

    @property
    def fitted(self) -> bool:
        return self._fitted


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    out = np.log(np.exp(x - x_max).sum(axis=axis, keepdims=keepdims))
    if not keepdims:
        out += x_max.squeeze(axis=axis)
    else:
        out += x_max
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)
