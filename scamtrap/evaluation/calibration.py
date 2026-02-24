"""Expected Calibration Error (ECE) and reliability diagram computation."""

import numpy as np


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error.

    Args:
        probs: np.array of predicted probabilities (for the positive class
               or the predicted class confidence)
        labels: np.array of ground truth (binary or matching the probs)
        n_bins: number of equal-width bins

    Returns:
        dict with ece, mce, and reliability diagram data
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_midpoints = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = mask.sum()
        bin_counts.append(int(count))
        bin_midpoints.append(float((lo + hi) / 2))

        if count > 0:
            bin_acc = float(labels[mask].mean())
            bin_conf = float(probs[mask].mean())
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    # ECE: weighted average of |acc - conf| per bin
    total = len(probs)
    ece = sum(
        (bin_counts[i] / max(total, 1)) * abs(bin_accuracies[i] - bin_confidences[i])
        for i in range(n_bins)
    )

    # MCE: max calibration error
    mce = max(
        abs(bin_accuracies[i] - bin_confidences[i])
        for i in range(n_bins)
        if bin_counts[i] > 0
    ) if any(c > 0 for c in bin_counts) else 0.0

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "reliability_diagram": {
            "bin_midpoints": bin_midpoints,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        },
    }


def compute_multiclass_ece(probs, labels, n_bins=15):
    """Compute ECE for multi-class classification.

    Uses the confidence of the predicted class.

    Args:
        probs: np.array [N, C] of class probabilities (softmax output)
        labels: np.array [N] of ground truth class indices

    Returns:
        dict with ece, mce, per_class_ece, and reliability diagram data
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    overall = compute_ece(confidences, correct, n_bins=n_bins)

    # Per-class ECE
    per_class_ece = {}
    n_classes = probs.shape[1]
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_conf = probs[mask, c]
            class_correct = (labels[mask] == c).astype(float)
            per_class_ece[int(c)] = compute_ece(class_conf, class_correct, n_bins=n_bins)["ece"]

    overall["per_class_ece"] = per_class_ece
    return overall
