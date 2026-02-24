"""Trajectory robustness: evaluate GRU vs baselines under perturbation."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def perturb_stage_noise(stages, prob):
    """Replace stage labels with random stages at probability p."""
    num_stages = max(stages) + 1 if stages else 6
    perturbed = []
    for s in stages:
        if np.random.random() < prob:
            perturbed.append(np.random.randint(0, num_stages))
        else:
            perturbed.append(s)
    return perturbed


def perturb_turn_dropout(embeddings, stages, prob):
    """Randomly drop fraction of turns."""
    n = len(stages)
    keep_mask = np.random.random(n) >= prob
    # Always keep at least 2 turns
    if keep_mask.sum() < 2:
        keep_mask[:2] = True
    new_embs = embeddings[keep_mask]
    new_stages = [s for s, k in zip(stages, keep_mask) if k]
    return new_embs, new_stages


def perturb_non_monotonic(stages, prob):
    """Inject backward stage transitions at probability p."""
    perturbed = list(stages)
    for i in range(1, len(perturbed)):
        if np.random.random() < prob and perturbed[i] > 0:
            # Go back 1-2 stages
            step_back = np.random.randint(1, 3)
            perturbed[i] = max(0, perturbed[i] - step_back)
    return perturbed


def perturb_combined(embeddings, stages, prob):
    """Apply all perturbations at moderate rates."""
    # Stage noise at prob/2
    stages = perturb_stage_noise(stages, prob / 2)
    # Non-monotonic at prob/2
    stages = perturb_non_monotonic(stages, prob / 2)
    # Turn dropout at prob/3
    embeddings, stages = perturb_turn_dropout(embeddings, stages, prob / 3)
    return embeddings, stages


def evaluate_model_on_perturbed(model, trajectories, perturbation_fn,
                                perturbation_level, device="cpu",
                                max_turns=30):
    """Evaluate a PyTorch world model on perturbed trajectories.

    Returns stage accuracy and escalation AUROC.
    """
    model.eval()
    model.to(device)

    all_stage_preds = []
    all_stage_labels = []
    all_esc_probs = []
    all_esc_labels = []

    for traj in trajectories:
        embs = traj["embeddings"].copy()
        stages = list(traj["stages"])

        # Apply perturbation
        if perturbation_fn == "stage_noise":
            stages = perturb_stage_noise(stages, perturbation_level)
        elif perturbation_fn == "turn_dropout":
            embs, stages = perturb_turn_dropout(embs, stages, perturbation_level)
        elif perturbation_fn == "non_monotonic":
            stages = perturb_non_monotonic(stages, perturbation_level)
        elif perturbation_fn == "combined":
            embs, stages = perturb_combined(embs, stages, perturbation_level)

        T = min(len(stages), max_turns)
        if T < 1:
            continue

        # Prepare input
        padded_emb = np.zeros((1, max_turns, embs.shape[1]), dtype=np.float32)
        padded_emb[0, :T] = embs[:T]
        emb_tensor = torch.tensor(padded_emb, device=device)
        length_tensor = torch.tensor([T], device=device)

        with torch.no_grad():
            stage_logits, esc_logits, _ = model(emb_tensor, length_tensor)
            stage_preds = stage_logits[0, :T].argmax(dim=-1).cpu().numpy()
            esc_probs = torch.sigmoid(esc_logits[0, :T, 0]).cpu().numpy()

        all_stage_preds.extend(stage_preds)
        all_stage_labels.extend(stages[:T])
        all_esc_probs.extend(esc_probs)
        all_esc_labels.extend([1.0 if s >= 4 else 0.0 for s in stages[:T]])

    stage_acc = float(accuracy_score(all_stage_labels, all_stage_preds))

    esc_labels_arr = np.array(all_esc_labels)
    esc_probs_arr = np.array(all_esc_probs)
    if len(set(esc_labels_arr)) >= 2:
        esc_auroc = float(roc_auc_score(esc_labels_arr, esc_probs_arr))
    else:
        esc_auroc = float("nan")

    return {"stage_accuracy": stage_acc, "escalation_auroc": esc_auroc}


def evaluate_markov_on_perturbed(markov_model, trajectories, perturbation_fn,
                                 perturbation_level):
    """Evaluate Markov chain baseline on perturbed trajectories."""
    all_stage_preds = []
    all_stage_labels = []
    all_esc_probs = []
    all_esc_labels = []

    for traj in trajectories:
        embs = traj["embeddings"].copy()
        stages = list(traj["stages"])

        if perturbation_fn == "stage_noise":
            stages = perturb_stage_noise(stages, perturbation_level)
        elif perturbation_fn == "turn_dropout":
            embs, stages = perturb_turn_dropout(embs, stages, perturbation_level)
        elif perturbation_fn == "non_monotonic":
            stages = perturb_non_monotonic(stages, perturbation_level)
        elif perturbation_fn == "combined":
            embs, stages = perturb_combined(embs, stages, perturbation_level)

        for i in range(len(stages) - 1):
            current = stages[i]
            if 0 <= current < markov_model.num_stages:
                pred = np.argmax(markov_model.transition_matrix[current])
                all_stage_preds.append(pred)
                all_stage_labels.append(stages[i + 1])

        for i in range(len(stages)):
            current = stages[i]
            if 0 <= current < markov_model.num_stages:
                prob = sum(
                    markov_model.transition_matrix[current, s]
                    for s in range(4, markov_model.num_stages)
                )
            else:
                prob = 0.0
            all_esc_probs.append(prob)
            all_esc_labels.append(1.0 if stages[i] >= 4 else 0.0)

    stage_acc = float(accuracy_score(all_stage_labels, all_stage_preds)) if all_stage_labels else 0.0

    esc_labels_arr = np.array(all_esc_labels)
    esc_probs_arr = np.array(all_esc_probs)
    if len(set(esc_labels_arr)) >= 2:
        esc_auroc = float(roc_auc_score(esc_labels_arr, esc_probs_arr))
    else:
        esc_auroc = float("nan")

    return {"stage_accuracy": stage_acc, "escalation_auroc": esc_auroc}


def evaluate_logreg_on_perturbed(logreg_model, trajectories, perturbation_fn,
                                 perturbation_level):
    """Evaluate LogReg baseline on perturbed trajectories."""
    all_embs = []
    all_stages = []
    all_esc = []

    for traj in trajectories:
        embs = traj["embeddings"].copy()
        stages = list(traj["stages"])

        if perturbation_fn == "stage_noise":
            stages = perturb_stage_noise(stages, perturbation_level)
        elif perturbation_fn == "turn_dropout":
            embs, stages = perturb_turn_dropout(embs, stages, perturbation_level)
        elif perturbation_fn == "non_monotonic":
            stages = perturb_non_monotonic(stages, perturbation_level)
        elif perturbation_fn == "combined":
            embs, stages = perturb_combined(embs, stages, perturbation_level)

        for i in range(len(stages)):
            if i < embs.shape[0]:
                all_embs.append(embs[i])
                all_stages.append(stages[i])
                all_esc.append(1 if stages[i] >= 4 else 0)

    if not all_embs:
        return {"stage_accuracy": 0.0, "escalation_auroc": float("nan")}

    X = np.array(all_embs)
    stage_labels = np.array(all_stages)
    esc_labels = np.array(all_esc)

    stage_preds = logreg_model.stage_clf.predict(X)
    stage_acc = float(accuracy_score(stage_labels, stage_preds))

    if len(set(esc_labels)) >= 2:
        esc_probs = logreg_model.esc_clf.predict_proba(X)[:, 1]
        esc_auroc = float(roc_auc_score(esc_labels, esc_probs))
    else:
        esc_auroc = float("nan")

    return {"stage_accuracy": stage_acc, "escalation_auroc": esc_auroc}
