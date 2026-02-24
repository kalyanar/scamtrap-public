"""Audit weak supervision label quality — confidence, ambiguity, coverage."""

import numpy as np
import pandas as pd
from collections import defaultdict

from scamtrap.data.intent_labeler import KeywordIntentLabeler


def audit_label_quality(df: pd.DataFrame):
    """Compute label quality metrics for weak supervision audit.

    Args:
        df: DataFrame with 'text', 'binary_label', 'intent_label' columns

    Returns:
        dict with overall and per-intent audit metrics
    """
    labeler = KeywordIntentLabeler()

    # Only audit scam samples (ham is deterministic from binary_label)
    scam_df = df[df["binary_label"] == 1].copy()
    n_scam = len(scam_df)

    confidences = []
    ambiguous_count = 0
    multi_match_count = 0
    generic_fallback_count = 0
    per_intent = defaultdict(lambda: {
        "count": 0,
        "confidences": [],
        "ambiguous": 0,
        "confusions": defaultdict(int),
    })

    for _, row in scam_df.iterrows():
        intent, scores = labeler.label_with_scores(row["text"], row["binary_label"])

        # Filter to intents with score > 0
        matched = {k: v for k, v in scores.items() if v > 0}
        n_matched = len(matched)

        # Multi-match rate
        if n_matched >= 2:
            multi_match_count += 1

        # Generic fallback rate
        if intent == "generic_scam":
            generic_fallback_count += 1

        # Confidence: best / second_best (higher = more confident)
        sorted_scores = sorted(scores.values(), reverse=True)
        best = sorted_scores[0] if sorted_scores else 0
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0

        if best > 0 and second > 0:
            confidence = best / second
        elif best > 0:
            confidence = float(best)  # no competition
        else:
            confidence = 0.0

        confidences.append(confidence)

        # Ambiguity: top-2 scores within 1 match of each other
        is_ambiguous = (best > 0 and second > 0 and (best - second) <= 1)
        if is_ambiguous:
            ambiguous_count += 1

        # Per-intent tracking
        per_intent[intent]["count"] += 1
        per_intent[intent]["confidences"].append(confidence)
        if is_ambiguous:
            per_intent[intent]["ambiguous"] += 1

        # Track confusion: which other intents also matched?
        if n_matched >= 2:
            for other_intent in matched:
                if other_intent != intent:
                    per_intent[intent]["confusions"][other_intent] += 1

    # Compile results
    results = {
        "overall": {
            "n_scam_samples": n_scam,
            "generic_fallback_count": generic_fallback_count,
            "generic_fallback_rate": generic_fallback_count / max(n_scam, 1),
            "multi_match_count": multi_match_count,
            "multi_match_rate": multi_match_count / max(n_scam, 1),
            "ambiguity_count": ambiguous_count,
            "ambiguity_rate": ambiguous_count / max(n_scam, 1),
            "coverage_rate": 1.0 - (generic_fallback_count / max(n_scam, 1)),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "median_confidence": float(np.median(confidences)) if confidences else 0.0,
        },
        "per_intent": {},
    }

    for intent_name in sorted(per_intent.keys()):
        info = per_intent[intent_name]
        n = info["count"]
        confs = info["confidences"]
        top_confusions = sorted(
            info["confusions"].items(), key=lambda x: x[1], reverse=True,
        )[:3]
        results["per_intent"][intent_name] = {
            "count": n,
            "fraction": n / max(n_scam, 1),
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "ambiguity_rate": info["ambiguous"] / max(n, 1),
            "top_confusions": [
                {"intent": c[0], "count": c[1]} for c in top_confusions
            ],
        }

    return results
