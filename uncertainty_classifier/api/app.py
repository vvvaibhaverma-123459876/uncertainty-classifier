"""FastAPI application factory."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from uncertainty_classifier.api.routes import router

# Module-level predictor singleton (set by load_model)
_predictor: Optional[object] = None


def create_app(predictor=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        predictor: Optional Predictor instance to attach at startup.

    Returns:
        Configured FastAPI app.
    """
    global _predictor
    if predictor is not None:
        _predictor = predictor

    app = FastAPI(
        title="Uncertainty Classifier",
        description=(
            "Uncertainty-aware text classifier using DistilBERT with MC Dropout "
            "and conformal prediction. Returns prediction sets with guaranteed coverage."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    return app


def load_model(predictor) -> None:
    """Attach a predictor to the global app state."""
    global _predictor
    _predictor = predictor
