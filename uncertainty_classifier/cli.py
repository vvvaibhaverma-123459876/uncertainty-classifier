"""Command-line interface for uncertainty-classifier.

Usage:
    python -m uncertainty_classifier classify "I loved this film." --coverage 0.9
    python -m uncertainty_classifier serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys


def classify_cmd(args: argparse.Namespace) -> None:
    """Run a single classification from the command line."""
    from uncertainty_classifier.predictor import Predictor

    print(f"Loading model {args.model!r}...", file=sys.stderr)
    predictor = Predictor.from_pretrained(
        model_name=args.model,
        n_mc_passes=args.n_passes,
    )

    result = predictor.predict(args.text, coverage=args.coverage)

    output = {
        "text": args.text,
        "label": result.label,
        "mean_prob": round(result.mean_prob, 4),
        "confidence_interval": [round(v, 4) for v in result.confidence_interval],
        "prediction_set": result.prediction_set_labels,
        "set_size": result.set_size,
        "abstained": result.abstained,
        "abstain_reason": result.abstain_reason,
        "epistemic_uncertainty": round(result.epistemic_uncertainty, 4),
        "coverage": args.coverage,
    }
    print(json.dumps(output, indent=2))


def serve_cmd(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from uncertainty_classifier.predictor import Predictor
    from uncertainty_classifier.api.app import create_app, load_model

    print(f"Loading model {args.model!r}...", file=sys.stderr)
    predictor = Predictor.from_pretrained(model_name=args.model)
    load_model(predictor)
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uncertainty_classifier",
        description="Uncertainty-aware text classification with conformal prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # classify sub-command
    classify_p = subparsers.add_parser("classify", help="Classify a single text string.")
    classify_p.add_argument("text", help="Text to classify.")
    classify_p.add_argument(
        "--coverage", type=float, default=0.9, help="Conformal coverage level (default 0.9)."
    )
    classify_p.add_argument(
        "--model", default="distilbert-base-uncased", help="HuggingFace model name or path."
    )
    classify_p.add_argument(
        "--n-passes", type=int, default=30, help="Number of MC dropout passes."
    )
    classify_p.set_defaults(func=classify_cmd)

    # serve sub-command
    serve_p = subparsers.add_parser("serve", help="Start the FastAPI server.")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--model", default="distilbert-base-uncased")
    serve_p.set_defaults(func=serve_cmd)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
