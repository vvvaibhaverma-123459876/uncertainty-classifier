"""Abstention gating based on uncertainty threshold.

Refuses to classify when epistemic or conformal uncertainty is too high,
preventing overconfident predictions on out-of-distribution inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AbstentionConfig:
    """Configuration for abstention gating.

    Attributes:
        max_std: Maximum allowed mean std of MC-dropout probs.
                 Abstain if mean_std > max_std.
        max_set_size: Maximum allowed conformal prediction set size.
                      Abstain if |C(x)| > max_set_size.
        min_top_prob: Minimum probability for predicted class.
                      Abstain if max_prob < min_top_prob.
    """
    max_std: float = 0.3
    max_set_size: int = 3
    min_top_prob: float = 0.3


def should_abstain(
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    prediction_set: List[int],
    config: Optional[AbstentionConfig] = None,
) -> tuple[bool, str]:
    """Decide whether to abstain from classification.

    Args:
        mean_probs: Mean MC-dropout probabilities, shape (num_classes,).
        std_probs: Std of MC-dropout probabilities, shape (num_classes,).
        prediction_set: Conformal prediction set (list of class indices).
        config: Abstention thresholds. Uses defaults if None.

    Returns:
        (abstained, reason) where reason is an empty string if not abstaining.
    """
    if config is None:
        config = AbstentionConfig()

    mean_std = float(std_probs.mean())
    set_size = len(prediction_set)
    top_prob = float(mean_probs.max())

    if mean_std > config.max_std:
        return True, f"High epistemic uncertainty (mean_std={mean_std:.3f} > {config.max_std})"

    if set_size > config.max_set_size:
        return True, f"Large prediction set (|C(x)|={set_size} > {config.max_set_size})"

    if top_prob < config.min_top_prob:
        return True, f"Low top probability ({top_prob:.3f} < {config.min_top_prob})"

    return False, ""
