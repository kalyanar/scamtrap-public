"""Trajectory evaluation: stage prediction and escalation forecasting."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, precision_score, recall_score,
)


def evaluate_stage_prediction(model, test_loader, device="cuda"):
    """Evaluate per-turn stage prediction accuracy.

    Returns:
        dict with accuracy, f1_macro, f1_weighted, per_stage_report
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["length"].to(device)
            stage_labels = batch["stage_labels"]
            mask = batch["mask"]

            stage_logits, _, _ = model(embs, lengths)
            preds = stage_logits.argmax(dim=-1).cpu()  # [B, T]

            # Only collect predictions for valid timesteps
            for b in range(preds.shape[0]):
                T = int(lengths[b].item())
                valid_preds = preds[b, :T].numpy()
                valid_labels = stage_labels[b, :T].numpy()
                # Filter out -1 padding labels
                valid_mask = valid_labels >= 0
                all_preds.extend(valid_preds[valid_mask])
                all_labels.extend(valid_labels[valid_mask])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    stage_names = [
        "hook", "trust_building", "urgency",
        "info_request", "payment_attempt", "escalation",
    ]
    present_labels = sorted(set(all_labels))
    target_names = [stage_names[i] for i in present_labels if i < len(stage_names)]

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(
            all_labels, all_preds, average="macro", zero_division=0,
        )),
        "f1_weighted": float(f1_score(
            all_labels, all_preds, average="weighted", zero_division=0,
        )),
        "classification_report": classification_report(
            all_labels, all_preds,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_escalation_forecast(model, test_loader, device="cuda"):
    """Evaluate escalation prediction (binary: will payment/escalation happen?).

    Returns:
        dict with auroc, brier_score, accuracy, precision, recall
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["length"].to(device)
            esc_labels = batch["escalation_labels"]
            mask = batch["mask"]

            _, esc_logits, _ = model(embs, lengths)
            esc_probs = torch.sigmoid(esc_logits.squeeze(-1)).cpu()  # [B, T]

            for b in range(esc_probs.shape[0]):
                T = int(lengths[b].item())
                all_probs.extend(esc_probs[b, :T].numpy())
                all_labels.extend(esc_labels[b, :T].numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    # Brier score: mean (prob - label)^2
    brier = float(np.mean((all_probs - all_labels) ** 2))

    # AUROC (needs both classes present)
    if len(set(all_labels)) >= 2:
        auroc = float(roc_auc_score(all_labels, all_probs))
    else:
        auroc = float("nan")

    return {
        "auroc": auroc,
        "brier_score": brier,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(np.sum((all_preds == 1) & (all_labels == 1))
                           / max(np.sum(all_preds == 1), 1)),
        "recall": float(np.sum((all_preds == 1) & (all_labels == 1))
                        / max(np.sum(all_labels == 1), 1)),
    }


def evaluate_early_warning(model, test_loader, device="cuda", horizon=3):
    """Evaluate early warning: can the model predict escalation K turns early?

    For each conversation that escalates, check if the model predicted
    escalation_prob > 0.5 at least `horizon` turns before the actual escalation.

    Returns:
        dict with early_detection_rate, avg_lead_time
    """
    model.eval()
    model.to(device)

    detection_results = []

    with torch.no_grad():
        for batch in test_loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["length"].to(device)
            esc_labels = batch["escalation_labels"]

            _, esc_logits, _ = model(embs, lengths)
            esc_probs = torch.sigmoid(esc_logits.squeeze(-1)).cpu()

            for b in range(esc_probs.shape[0]):
                T = int(lengths[b].item())
                labels = esc_labels[b, :T].numpy()
                probs = esc_probs[b, :T].numpy()

                # Find first escalation turn
                esc_turns = np.where(labels >= 0.5)[0]
                if len(esc_turns) == 0:
                    continue  # No escalation in this conversation

                first_esc = esc_turns[0]

                # Find first prediction turn (prob > 0.5)
                pred_turns = np.where(probs >= 0.5)[0]
                if len(pred_turns) == 0:
                    detection_results.append({
                        "detected": False,
                        "lead_time": 0,
                    })
                    continue

                first_pred = pred_turns[0]
                lead_time = first_esc - first_pred

                detection_results.append({
                    "detected": lead_time >= horizon,
                    "lead_time": max(0, lead_time),
                })

    if not detection_results:
        return {"early_detection_rate": 0.0, "avg_lead_time": 0.0}

    return {
        "early_detection_rate": float(np.mean([
            r["detected"] for r in detection_results
        ])),
        "avg_lead_time": float(np.mean([
            r["lead_time"] for r in detection_results
        ])),
        "total_escalating_conversations": len(detection_results),
    }


def evaluate_early_warning_sweep(model, test_loader, device="cuda",
                                 horizons=None, thresholds=None):
    """Evaluate early warning across multiple horizons and thresholds.

    Args:
        model: trained world model
        test_loader: DataLoader for test set
        device: torch device
        horizons: list of lead-time horizons to test (default: [1,2,3,5,7])
        thresholds: list of probability thresholds (default: [0.3,0.5,0.7])

    Returns:
        dict keyed by horizon with precision, recall, f1, lead-time distribution
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 7]
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    model.eval()
    model.to(device)

    # Collect per-conversation data
    conversations = []
    with torch.no_grad():
        for batch in test_loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["length"].to(device)
            esc_labels = batch["escalation_labels"]

            _, esc_logits, _ = model(embs, lengths)
            esc_probs = torch.sigmoid(esc_logits.squeeze(-1)).cpu()

            for b in range(esc_probs.shape[0]):
                T = int(lengths[b].item())
                conversations.append({
                    "probs": esc_probs[b, :T].numpy(),
                    "labels": esc_labels[b, :T].numpy(),
                })

    results = {}
    for horizon in horizons:
        horizon_results = {}
        for threshold in thresholds:
            detected = 0
            missed = 0
            false_alarms = 0
            lead_times = []

            for conv in conversations:
                labels = conv["labels"]
                probs = conv["probs"]

                esc_turns = np.where(labels >= 0.5)[0]
                has_escalation = len(esc_turns) > 0
                first_esc = esc_turns[0] if has_escalation else None

                pred_turns = np.where(probs >= threshold)[0]
                has_prediction = len(pred_turns) > 0
                first_pred = pred_turns[0] if has_prediction else None

                if has_escalation:
                    if has_prediction and (first_esc - first_pred) >= horizon:
                        detected += 1
                        lead_times.append(int(first_esc - first_pred))
                    else:
                        missed += 1
                        if has_prediction:
                            lead_times.append(max(0, int(first_esc - first_pred)))
                else:
                    if has_prediction:
                        false_alarms += 1

            total_escalating = detected + missed
            total_predicted = detected + false_alarms

            precision = detected / max(total_predicted, 1)
            recall = detected / max(total_escalating, 1)
            f1 = (2 * precision * recall / max(precision + recall, 1e-9)
                   if (precision + recall) > 0 else 0.0)

            lead_arr = np.array(lead_times) if lead_times else np.array([0])
            horizon_results[str(threshold)] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "detected": detected,
                "missed": missed,
                "false_alarms": false_alarms,
            }

        # Best threshold (by F1 at this horizon)
        best_thr = max(thresholds, key=lambda t: horizon_results[str(t)]["f1"])
        best = horizon_results[str(best_thr)]

        lead_arr = np.array(lead_times) if lead_times else np.array([0])
        results[str(horizon)] = {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "best_threshold": float(best_thr),
            "lead_time_p25": float(np.percentile(lead_arr, 25)),
            "lead_time_p50": float(np.percentile(lead_arr, 50)),
            "lead_time_p75": float(np.percentile(lead_arr, 75)),
            "early_detection_rate": best["recall"],
            "per_threshold": horizon_results,
        }

    return results
