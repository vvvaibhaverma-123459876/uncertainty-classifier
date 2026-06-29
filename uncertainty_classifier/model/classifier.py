"""DistilBERT-based text classifier with Monte Carlo Dropout support.

The standard HuggingFace DistilBERT already has dropout layers.
We expose a `mc_forward` method that uses the mc_dropout context manager
to keep dropout active during inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertConfig,
)

from uncertainty_classifier.uncertainty.mc_dropout import (
    mc_dropout_predict,
    confidence_interval,
)


class UncertaintyClassifier(nn.Module):
    """DistilBERT classifier with MC-Dropout inference.

    Args:
        model_name: HuggingFace model name or path.
        num_labels: Number of output classes.
        id2label: Optional mapping from label index to label string.
        dropout_p: Dropout probability for the classification head.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label or {i: str(i) for i in range(num_labels)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        # Override dropout probability if different from default
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_p

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def predict_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Single deterministic forward pass (eval mode). Returns logits."""
        self.eval()
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.cpu().numpy()

    def mc_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_passes: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MC-Dropout inference: N stochastic forward passes.

        Returns:
            mean_probs: shape (num_classes,)
            std_probs: shape (num_classes,)
            all_probs: shape (n_passes, num_classes)
        """
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        def _forward(m, inp):
            return m.backbone(**inp).logits

        return mc_dropout_predict(
            self,
            inputs=inputs,
            n_passes=n_passes,
            forward_fn=_forward,
        )


def build_tiny_classifier(num_labels: int = 2) -> "UncertaintyClassifier":
    """Build a tiny random-weight classifier for tests (no download needed)."""
    config = DistilBertConfig(
        vocab_size=1000,
        max_position_embeddings=64,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        dropout=0.1,
        attention_dropout=0.1,
        num_labels=num_labels,
    )
    model = UncertaintyClassifier.__new__(UncertaintyClassifier)
    nn.Module.__init__(model)
    model.num_labels = num_labels
    model.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model.model_name = "tiny-random"
    model.backbone = AutoModelForSequenceClassification.from_config(config)
    # Ensure dropout is enabled at p=0.1
    for module in model.backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.1
    return model
