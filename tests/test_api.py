"""Tests for FastAPI application.

Uses a mock predictor — no real model, no downloads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uncertainty_classifier.api.app import create_app
from uncertainty_classifier.predictor import PredictionResult


# ------------------------------------------------------------------
# Mock predictor
# ------------------------------------------------------------------

class MockPredictor:
    """Minimal predictor stub for API tests."""

    def predict(self, text: str, coverage: float = 0.9) -> PredictionResult:
        return PredictionResult(
            label="POSITIVE",
            label_id=1,
            confidence_interval=(0.72, 0.95),
            mean_prob=0.85,
            prediction_set=[1],
            prediction_set_labels=["POSITIVE"],
            set_size=1,
            abstained=False,
            abstain_reason="",
            epistemic_uncertainty=0.03,
            coverage=coverage,
        )


class MockAbstainPredictor:
    """Predictor that always abstains."""

    def predict(self, text: str, coverage: float = 0.9) -> PredictionResult:
        return PredictionResult(
            label="NEGATIVE",
            label_id=0,
            confidence_interval=(0.2, 0.8),
            mean_prob=0.4,
            prediction_set=[0, 1],
            prediction_set_labels=["NEGATIVE", "POSITIVE"],
            set_size=2,
            abstained=True,
            abstain_reason="High epistemic uncertainty (mean_std=0.35 > 0.3)",
            epistemic_uncertainty=0.35,
            coverage=coverage,
        )


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app(predictor=MockPredictor())
    return TestClient(app)


@pytest.fixture
def abstain_client():
    app = create_app(predictor=MockAbstainPredictor())
    return TestClient(app)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestClassifyEndpoint:
    def test_classify_basic(self, client):
        resp = client.post("/classify", json={"text": "I loved this film."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "POSITIVE"
        assert data["label_id"] == 1
        assert len(data["confidence_interval"]) == 2
        assert data["confidence_interval"][0] <= data["confidence_interval"][1]
        assert data["set_size"] == 1
        assert data["abstained"] is False

    def test_classify_custom_coverage(self, client):
        resp = client.post("/classify", json={"text": "hello", "coverage": 0.95})
        assert resp.status_code == 200
        assert resp.json()["coverage"] == 0.95

    def test_classify_abstention(self, abstain_client):
        resp = abstain_client.post("/classify", json={"text": "ambiguous input"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True
        assert len(data["abstain_reason"]) > 0

    def test_classify_empty_text_rejected(self, client):
        resp = client.post("/classify", json={"text": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_classify_coverage_out_of_range(self, client):
        resp = client.post("/classify", json={"text": "test", "coverage": 1.5})
        assert resp.status_code == 422

    def test_classify_response_fields(self, client):
        resp = client.post("/classify", json={"text": "Great product!"})
        data = resp.json()
        required_fields = {
            "label", "label_id", "confidence_interval", "mean_prob",
            "prediction_set", "prediction_set_labels", "set_size",
            "abstained", "abstain_reason", "epistemic_uncertainty", "coverage",
        }
        assert required_fields.issubset(data.keys())

    def test_no_predictor_returns_503(self):
        """When no predictor is loaded, /classify returns 503."""
        import uncertainty_classifier.api.app as app_module
        original = app_module._predictor
        app_module._predictor = None
        try:
            app = create_app()
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.post("/classify", json={"text": "test"})
            assert resp.status_code == 503
        finally:
            app_module._predictor = original
