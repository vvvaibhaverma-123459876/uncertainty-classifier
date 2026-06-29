"""Pydantic schemas for the FastAPI application."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request body for POST /classify."""

    text: str = Field(..., description="Text to classify.", min_length=1, max_length=4096)
    coverage: float = Field(
        0.9,
        description="Desired conformal coverage level (e.g. 0.9 means 90% coverage).",
        ge=0.5,
        le=0.99,
    )


class ClassifyResponse(BaseModel):
    """Response body for POST /classify."""

    label: str = Field(..., description="Predicted class label.")
    label_id: int = Field(..., description="Predicted class index.")
    confidence_interval: List[float] = Field(
        ...,
        description=(
            "5th–95th percentile band of the predicted class probability "
            "across Monte Carlo dropout passes. E.g. [0.72, 0.95]."
        ),
        min_length=2,
        max_length=2,
    )
    mean_prob: float = Field(..., description="Mean probability for the predicted class.")
    prediction_set: List[int] = Field(
        ...,
        description="Conformal prediction set: indices of all plausible labels at requested coverage.",
    )
    prediction_set_labels: List[str] = Field(
        ...,
        description="Human-readable labels in the prediction set.",
    )
    set_size: int = Field(..., description="Size of the conformal prediction set.")
    abstained: bool = Field(
        ...,
        description="True if the model abstained due to high uncertainty.",
    )
    abstain_reason: str = Field(
        "",
        description="Human-readable reason for abstention (empty if not abstained).",
    )
    epistemic_uncertainty: float = Field(
        ...,
        description="Mean standard deviation across MC dropout passes (epistemic uncertainty).",
    )
    coverage: float = Field(..., description="Requested conformal coverage level.")


class HealthResponse(BaseModel):
    status: str = "ok"
