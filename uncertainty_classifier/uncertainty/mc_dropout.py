"""Monte Carlo Dropout for epistemic uncertainty estimation.

Standard model.eval() disables dropout. This module forces dropout layers
to remain active during inference via a context manager, enabling N stochastic
forward passes whose variance captures model uncertainty.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@contextmanager
def mc_dropout_mode(model: nn.Module):
    """Context manager that enables dropout even in eval mode.

    Sets the model to eval (disables BatchNorm updates etc.) but
    re-enables all Dropout layers so that each forward pass is stochastic.
    """
    model.eval()
    # Enable dropout layers only
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
    try:
        yield model
    finally:
        model.eval()  # restore full eval (dropout off again)


def mc_dropout_predict(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    n_passes: int = 30,
    forward_fn: Callable | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N stochastic forward passes and aggregate results.

    Args:
        model: PyTorch model with dropout layers.
        inputs: Dict of input tensors (e.g. from a HuggingFace tokenizer).
        n_passes: Number of stochastic forward passes.
        forward_fn: Optional callable(model, inputs) -> logits tensor.
                    Defaults to model(**inputs).logits.

    Returns:
        mean_probs: Mean softmax probabilities, shape (num_classes,).
        std_probs:  Std of softmax probabilities, shape (num_classes,).
        all_probs:  All pass probabilities, shape (n_passes, num_classes).
    """
    if forward_fn is None:
        def forward_fn(m, inp):  # type: ignore[misc]
            return m(**inp).logits

    all_probs_list = []
    with mc_dropout_mode(model):
        with torch.no_grad():
            for _ in range(n_passes):
                logits = forward_fn(model, inputs)
                probs = torch.softmax(logits, dim=-1)
                all_probs_list.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs_list, axis=0)  # (n_passes, batch, num_classes)
    # If batch dim is 1, squeeze it
    if all_probs.ndim == 3 and all_probs.shape[1] == 1:
        all_probs = all_probs[:, 0, :]  # (n_passes, num_classes)

    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    return mean_probs, std_probs, all_probs


def epistemic_uncertainty(std_probs: np.ndarray) -> float:
    """Scalar epistemic uncertainty: mean std across classes."""
    return float(std_probs.mean())


def confidence_interval(
    all_probs: np.ndarray,
    predicted_class: int,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> Tuple[float, float]:
    """Percentile confidence interval for predicted class probability.

    Args:
        all_probs: Shape (n_passes, num_classes).
        predicted_class: Index of the predicted class.
        lower_pct: Lower percentile (default 5).
        upper_pct: Upper percentile (default 95).

    Returns:
        (lower, upper) probability bounds.
    """
    class_probs = all_probs[:, predicted_class]
    lower = float(np.percentile(class_probs, lower_pct))
    upper = float(np.percentile(class_probs, upper_pct))
    return lower, upper
