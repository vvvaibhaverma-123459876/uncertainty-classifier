"""Demo: Conformal prediction sets with uncertainty quantification."""
import numpy as np
from uncertainty_classifier.uncertainty.conformal import ConformalPredictor
from uncertainty_classifier.calibration.metrics import expected_calibration_error, brier_score

print("=== Uncertainty-Aware Classifier Demo ===\n")
np.random.seed(42)
cal_probs = np.random.dirichlet([1, 1], size=200)
cal_labels = (cal_probs[:, 1] > 0.5).astype(int)

conformal = ConformalPredictor()
conformal.calibrate(cal_probs, cal_labels, alpha=0.10)  # 90% coverage
print("Conformal predictor calibrated at 90% coverage (alpha=0.10)\n")

labels = ["Negative", "Positive"]
test_cases = [
    ([0.05, 0.95], "High confidence positive"),
    ([0.92, 0.08], "High confidence negative"),
    ([0.48, 0.52], "Borderline / uncertain"),
    ([0.30, 0.70], "Moderate confidence"),
]
print(f"  {'Description':<32} {'Max Prob':>9} {'Prediction Set':>22}")
print("  " + "-" * 67)
for probs, desc in test_cases:
    arr = np.array(probs)
    pred_set = conformal.predict_set(arr)
    set_str = "{" + ", ".join(labels[i] for i in pred_set) + "}" if pred_set else "{abstain}"
    print(f"  {desc:<32} {max(probs):>9.1%} {set_str:>22}")

print(f"\nECE: {expected_calibration_error(cal_probs, cal_labels):.4f}  |  Brier: {brier_score(cal_probs, cal_labels):.4f}")
print("\nDemo complete.")
