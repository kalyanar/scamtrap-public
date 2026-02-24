"""Few-shot linear probe evaluation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


def evaluate_fewshot(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    label_fractions: list[float] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> dict:
    """Few-shot classification via linear probe on frozen embeddings.

    For each fraction, sample that fraction of training data,
    fit LogisticRegression, evaluate on full test set.
    Repeat n_trials times, report mean +/- std.
    """
    label_fractions = label_fractions or [0.01, 0.05, 0.10, 1.0]
    results = {}

    for frac in label_fractions:
        trial_metrics = {"accuracy": [], "f1_macro": [], "f1_weighted": [],
                         "precision_macro": [], "recall_macro": []}

        for trial in range(n_trials):
            if frac >= 1.0:
                X_sub, y_sub = train_embeddings, train_labels
            else:
                n_samples = max(int(len(train_labels) * frac), 2)
                # Stratified subsample
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples,
                                                 random_state=seed + trial)
                    idx, _ = next(sss.split(train_embeddings, train_labels))
                    X_sub = train_embeddings[idx]
                    y_sub = train_labels[idx]
                except ValueError:
                    # Fallback: random sample if stratification fails
                    rng = np.random.RandomState(seed + trial)
                    idx = rng.choice(len(train_labels), size=n_samples, replace=False)
                    X_sub = train_embeddings[idx]
                    y_sub = train_labels[idx]

            # Fit linear probe
            clf = LogisticRegression(max_iter=1000, random_state=seed + trial)
            clf.fit(X_sub, y_sub)
            preds = clf.predict(test_embeddings)

            trial_metrics["accuracy"].append(accuracy_score(test_labels, preds))
            trial_metrics["f1_macro"].append(f1_score(test_labels, preds, average="macro", zero_division=0))
            trial_metrics["f1_weighted"].append(f1_score(test_labels, preds, average="weighted", zero_division=0))
            trial_metrics["precision_macro"].append(precision_score(test_labels, preds, average="macro", zero_division=0))
            trial_metrics["recall_macro"].append(recall_score(test_labels, preds, average="macro", zero_division=0))

        results[str(frac)] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in trial_metrics.items()
        }

    return results
