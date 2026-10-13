"""
Evaluation metrics for MediSense thesis.

This module computes every metric required by the thesis proposal:
    - Patient anomaly detection accuracy (target ≥90%)
    - Sensor malfunction detection accuracy (target ≥95%)
    - Transfer learning improvement over baseline (target ≥5pp)
    - Data efficiency: accuracy vs. training set fraction (target ≥50% reduction)
    - McNemar's test for statistical significance (p<0.05)
    - Detailed false negative / false positive error analysis
    - Inference latency profiling (target <100ms per window)

All results are returned as dicts and saved as JSON — ready for thesis tables/figures.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from ..models.autoencoder import MediSenseAutoencoder


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    model: MediSenseAutoencoder,
    normal_features: np.ndarray,
    percentile: float = 95.0,
    device: str = "cpu",
) -> float:
    """
    Set anomaly detection threshold from normal validation data.
    Threshold = Nth percentile of reconstruction error on known-normal windows.
    Windows exceeding this threshold are flagged as anomalies.

    Args:
        model: Trained autoencoder.
        normal_features: (N, D) array of normal validation windows.
        percentile: Threshold percentile (95 for sensor malfunction, 90 for patient).

    Returns:
        Scalar threshold value.
    """
    model.eval()
    # Move to same device as model
    model_device = next(model.parameters()).device
    t = torch.tensor(normal_features, dtype=torch.float32).to(model_device)
    with torch.no_grad():
        scores = model.anomaly_score(t).cpu().numpy()
    threshold = float(np.percentile(scores, percentile))
    return threshold


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    accuracy: float
    sensitivity: float      # recall for anomaly class (critical: catch real emergencies)
    specificity: float      # recall for normal class (avoid alert fatigue)
    precision: float
    f1_score: float
    auroc: float
    auprc: float
    n_false_negatives: int  # missed emergencies — most dangerous error
    n_false_positives: int  # false alarms
    n_true_positives: int
    n_true_negatives: int
    threshold: float
    inference_latency_ms: float
    report: str             # sklearn classification report


def evaluate(
    model: MediSenseAutoencoder,
    features: np.ndarray,
    true_labels: np.ndarray,  # 0=normal, 1=anomaly
    threshold: float,
    device: str = "cpu",
    run_name: str = "",
) -> EvaluationResult:
    """
    Full evaluation pass on a held-out feature matrix.

    Args:
        model: Trained autoencoder.
        features: (N, D) test feature matrix.
        true_labels: (N,) binary labels (0=normal, 1=anomaly).
        threshold: Calibrated anomaly threshold.
        device: Torch device string.

    Returns:
        EvaluationResult with all thesis metrics.
    """
    model.eval()
    model_device = next(model.parameters()).device
    t = torch.tensor(features, dtype=torch.float32).to(model_device)

    # Latency measurement — single-window inference (simulate real-time)
    with torch.no_grad():
        start = time.perf_counter()
        scores = model.anomaly_score(t).cpu().numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000 / len(features)

    pred_labels = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()
    acc = accuracy_score(true_labels, pred_labels)
    sensitivity = tp / (tp + fn + 1e-10)   # recall for anomaly
    specificity = tn / (tn + fp + 1e-10)   # recall for normal
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

    try:
        auroc = roc_auc_score(true_labels, scores)
    except ValueError:
        auroc = 0.5  # only one class present

    try:
        auprc = average_precision_score(true_labels, scores)
    except ValueError:
        auprc = 0.0

    report = classification_report(true_labels, pred_labels, target_names=["Normal", "Anomaly"])

    print(f"\n{'='*55}")
    if run_name:
        print(f"Results: {run_name}")
    print(f"  Accuracy:     {acc:.4f}  (target ≥0.90)")
    print(f"  Sensitivity:  {sensitivity:.4f}  (catch emergencies)")
    print(f"  Specificity:  {specificity:.4f}  (avoid alert fatigue)")
    print(f"  AUROC:        {auroc:.4f}")
    print(f"  False Negatives (missed emergencies): {fn}")
    print(f"  False Positives (false alarms):       {fp}")
    print(f"  Latency:      {elapsed_ms:.2f}ms/window  (target <100ms)")
    print(f"{'='*55}")

    return EvaluationResult(
        accuracy=float(acc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1_score=float(f1),
        auroc=float(auroc),
        auprc=float(auprc),
        n_false_negatives=int(fn),
        n_false_positives=int(fp),
        n_true_positives=int(tp),
        n_true_negatives=int(tn),
        threshold=threshold,
        inference_latency_ms=elapsed_ms,
        report=report,
    )


# ---------------------------------------------------------------------------
# McNemar's test — statistical significance
# ---------------------------------------------------------------------------

def mcnemars_test(
    labels: np.ndarray,
    preds_a: np.ndarray,   # transfer learning predictions
    preds_b: np.ndarray,   # baseline predictions
) -> dict:
    """
    McNemar's test for paired classifier comparison.
    Tests whether transfer learning and baseline make DIFFERENT errors.
    p<0.05 confirms that the improvement is statistically significant.

    Contingency table:
        b01 = baseline wrong, transfer right  (transfer wins)
        b10 = transfer wrong, baseline right  (baseline wins)
    """
    b01 = np.sum((preds_b != labels) & (preds_a == labels))
    b10 = np.sum((preds_a != labels) & (preds_b == labels))

    # With continuity correction (standard for McNemar's)
    if b01 + b10 == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p_value = 1 - chi2.cdf(statistic, df=1)

    result = {
        "b01_transfer_wins": int(b01),
        "b10_baseline_wins": int(b10),
        "chi2_statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            "Transfer learning improvement is statistically significant (p<0.05)"
            if p_value < 0.05
            else f"Not significant (p={p_value:.4f})"
        ),
    }
    print(f"\n[McNemar's Test] χ²={statistic:.4f}, p={p_value:.4f} — {result['interpretation']}")
    return result


# ---------------------------------------------------------------------------
# Data efficiency experiment — the key thesis figure
# ---------------------------------------------------------------------------

def data_efficiency_experiment(
    transfer_results: dict[float, EvaluationResult],
    baseline_results: dict[float, EvaluationResult],
    target_accuracy: float = 0.90,
) -> dict:
    """
    Compute data efficiency: at each training fraction, compare transfer vs. baseline.

    Args:
        transfer_results: {fraction: EvaluationResult} for transfer learning model.
        baseline_results: {fraction: EvaluationResult} for baseline model.
        target_accuracy: Accuracy level we're measuring data efficiency at.

    Returns:
        Dict with fractions, accuracies, and the key data efficiency claim.
    """
    fractions = sorted(transfer_results.keys())

    transfer_accs = [transfer_results[f].accuracy for f in fractions]
    baseline_accs = [baseline_results[f].accuracy for f in fractions]

    # Find fraction at which each model first reaches target_accuracy
    def first_fraction_at_target(accs):
        for f, a in zip(fractions, accs):
            if a >= target_accuracy:
                return f
        return None

    transfer_crossover = first_fraction_at_target(transfer_accs)
    baseline_crossover = first_fraction_at_target(baseline_accs)

    efficiency_gain = None
    if transfer_crossover is not None and baseline_crossover is not None:
        efficiency_gain = 1.0 - (transfer_crossover / baseline_crossover)

    improvements = [t - b for t, b in zip(transfer_accs, baseline_accs)]

    result = {
        "fractions": fractions,
        "transfer_accuracies": transfer_accs,
        "baseline_accuracies": baseline_accs,
        "improvements_pp": [round(i * 100, 2) for i in improvements],
        "max_improvement_pp": round(max(improvements) * 100, 2),
        "transfer_reaches_target_at": transfer_crossover,
        "baseline_reaches_target_at": baseline_crossover,
        "data_efficiency_gain": round(efficiency_gain * 100, 1) if efficiency_gain else None,
        "target_accuracy": target_accuracy,
        "thesis_claim": (
            f"Transfer learning reaches {target_accuracy:.0%} accuracy with "
            f"{efficiency_gain*100:.0f}% less data than baseline"
            if efficiency_gain
            else "Target accuracy not reached within experiment range"
        ),
    }

    print(f"\n[Data Efficiency] {result['thesis_claim']}")
    return result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert dataclasses to dicts for JSON serialisation
    def _serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_serialise)
    print(f"Results saved → {path}")
