"""Full prediction pipeline: MC Dropout + Conformal + Abstention.

Orchestrates all modules into a single predict() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from uncertainty_classifier.model.classifier import UncertaintyClassifier
from uncertainty_classifier.model.temperature import TemperatureScaler
from uncertainty_classifier.uncertainty.conformal import ConformalPredictor
from uncertainty_classifier.uncertainty.abstention import AbstentionConfig, should_abstain
from uncertainty_classifier.uncertainty.mc_dropout import confidence_interval


@dataclass
class PredictionResult:
    """Output of a single prediction."""
    label: str
    label_id: int
    confidence_interval: Tuple[float, float]  # 5th–95th percentile of MC probs
    mean_prob: float
    prediction_set: List[int]
    prediction_set_labels: List[str]
    set_size: int
    abstained: bool
    abstain_reason: str
    epistemic_uncertainty: float
    coverage: float  # requested coverage level


class Predictor:
    """Orchestrates DistilBERT + MC Dropout + Temperature Scaling + Conformal Prediction.

    Usage::

        predictor = Predictor.from_pretrained("distilbert-base-uncased")
        # calibrate conformal predictor
        predictor.calibrate(cal_texts, cal_labels, alpha=0.1)
        result = predictor.predict("I loved this movie.", coverage=0.9)
    """

    def __init__(
        self,
        model: UncertaintyClassifier,
        tokenizer,
        temperature_scaler: Optional[TemperatureScaler] = None,
        conformal: Optional[ConformalPredictor] = None,
        abstention_config: Optional[AbstentionConfig] = None,
        n_mc_passes: int = 30,
        max_length: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.temperature_scaler = temperature_scaler or TemperatureScaler()
        self.conformal = conformal or ConformalPredictor()
        self.abstention_config = abstention_config or AbstentionConfig()
        self.n_mc_passes = n_mc_passes
        self.max_length = max_length
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        **kwargs,
    ) -> "Predictor":
        from uncertainty_classifier.model.classifier import UncertaintyClassifier
        model = UncertaintyClassifier(model_name, num_labels=num_labels, id2label=id2label)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_conformal(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        alpha: float = 0.1,
    ) -> None:
        """Calibrate conformal predictor from pre-computed probabilities."""
        self.conformal.calibrate(cal_probs, cal_labels, alpha=alpha)

    def calibrate_temperature(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> None:
        """Fit temperature scaler from validation logits."""
        self.temperature_scaler.fit(val_logits, val_labels)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str, coverage: float = 0.9) -> PredictionResult:
        """Full uncertainty-aware prediction.

        Args:
            text: Input text.
            coverage: Desired conformal coverage level (e.g. 0.9).

        Returns:
            PredictionResult with all uncertainty estimates.
        """
        alpha = 1.0 - coverage
        inputs = self._tokenize(text)

        # MC Dropout passes
        mean_probs, std_probs, all_probs = self.model.mc_predict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            n_passes=self.n_mc_passes,
        )

        # Predicted class from mean probs
        predicted_class = int(np.argmax(mean_probs))
        predicted_label = self.model.id2label[predicted_class]

        # Confidence interval (5th-95th percentile across MC passes)
        ci = confidence_interval(all_probs, predicted_class)

        # Epistemic uncertainty
        epi_unc = float(std_probs.mean())

        # Conformal prediction set
        if self.conformal.q_hat is not None:
            # Re-calibrate if alpha changes
            pred_set = self.conformal.predict_set(mean_probs)
        else:
            # No calibration data: fall back to single-label prediction
            pred_set = [predicted_class]

        pred_set_labels = [self.model.id2label[i] for i in pred_set]

        # Abstention check
        abstained, abstain_reason = should_abstain(
            mean_probs, std_probs, pred_set, self.abstention_config
        )

        return PredictionResult(
            label=predicted_label,
            label_id=predicted_class,
            confidence_interval=ci,
            mean_prob=float(mean_probs[predicted_class]),
            prediction_set=pred_set,
            prediction_set_labels=pred_set_labels,
            set_size=len(pred_set),
            abstained=abstained,
            abstain_reason=abstain_reason,
            epistemic_uncertainty=epi_unc,
            coverage=coverage,
        )
