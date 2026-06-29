"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from uncertainty_classifier.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)

router = APIRouter()


def get_predictor():
    """Dependency: retrieve the predictor from app state."""
    from uncertainty_classifier.api.app import _predictor
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call load_model() first.",
        )
    return _predictor


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    request: ClassifyRequest,
    predictor=Depends(get_predictor),
) -> ClassifyResponse:
    """Classify text with uncertainty estimation.

    Returns a conformal prediction set guaranteed to contain the true label
    with probability >= coverage, plus MC-dropout confidence intervals.
    """
    try:
        result = predictor.predict(request.text, coverage=request.coverage)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ClassifyResponse(
        label=result.label,
        label_id=result.label_id,
        confidence_interval=list(result.confidence_interval),
        mean_prob=result.mean_prob,
        prediction_set=result.prediction_set,
        prediction_set_labels=result.prediction_set_labels,
        set_size=result.set_size,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        epistemic_uncertainty=result.epistemic_uncertainty,
        coverage=result.coverage,
    )
