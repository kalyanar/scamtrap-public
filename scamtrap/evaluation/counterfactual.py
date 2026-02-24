"""Counterfactual analysis: what if we intervene at turn t?"""

import numpy as np
import torch


def simulate_intervention(model, trajectory_embeddings, lengths,
                          intervention_turn, device="cuda"):
    """Simulate intervention by zeroing out embeddings from turn t onward.

    This simulates "what would happen if the conversation stopped at turn t?"
    by replacing future turn embeddings with zeros.

    Args:
        model: trained ScamWorldModel
        trajectory_embeddings: [B, T, 768] original embeddings
        lengths: [B] actual lengths
        intervention_turn: int, turn at which to intervene

    Returns:
        dict with original and intervened escalation probabilities
    """
    model.eval()
    model.to(device)

    embs = trajectory_embeddings.to(device)
    lengths_dev = lengths.to(device)

    with torch.no_grad():
        # Original predictions
        _, orig_esc_logits, _ = model(embs, lengths_dev)
        orig_probs = torch.sigmoid(orig_esc_logits.squeeze(-1)).cpu()

        # Intervened: zero out embeddings from intervention_turn onward
        # Keep original lengths so output shapes match for delta computation
        intervened_embs = embs.clone()
        intervened_embs[:, intervention_turn:, :] = 0.0

        _, int_esc_logits, _ = model(intervened_embs, lengths_dev)
        int_probs = torch.sigmoid(int_esc_logits.squeeze(-1)).cpu()

    # Ensure same shape for delta (trim to shorter if needed)
    min_len = min(orig_probs.shape[1], int_probs.shape[1])
    orig_trimmed = orig_probs[:, :min_len]
    int_trimmed = int_probs[:, :min_len]

    return {
        "original_probs": orig_trimmed.numpy(),
        "intervened_probs": int_trimmed.numpy(),
        "delta": (orig_trimmed - int_trimmed).numpy(),
    }


def analyze_intervention_impact(model, test_loader, device="cuda",
                                intervention_turns=None):
    """Run counterfactual analysis across the test set.

    For each intervention turn, compute the average reduction in
    escalation probability.

    Args:
        model: trained world model
        test_loader: DataLoader with trajectory data
        intervention_turns: list of turns to test (default: [3, 5, 7, 10])

    Returns:
        dict mapping intervention_turn -> avg_prob_reduction
    """
    if intervention_turns is None:
        intervention_turns = [3, 5, 7, 10]

    model.eval()
    model.to(device)

    results = {}

    for t in intervention_turns:
        all_deltas = []

        with torch.no_grad():
            for batch in test_loader:
                embs = batch["embeddings"].to(device)
                lengths = batch["length"].to(device)
                esc_labels = batch["escalation_labels"]

                # Only analyze conversations that escalate and are long enough
                for b in range(embs.shape[0]):
                    length = int(lengths[b].item())
                    if length <= t:
                        continue
                    labels = esc_labels[b, :length].numpy()
                    if labels.max() < 0.5:
                        continue  # No escalation

                    # Single sample intervention
                    single_emb = embs[b:b+1]
                    single_len = lengths[b:b+1]

                    result = simulate_intervention(
                        model, single_emb, single_len, t, device,
                    )

                    # Average delta over valid timesteps after intervention
                    delta = result["delta"][0]
                    valid_delta = delta[t:length]
                    if len(valid_delta) > 0:
                        all_deltas.append(float(np.mean(valid_delta)))

        results[f"turn_{t}"] = {
            "avg_prob_reduction": float(np.mean(all_deltas)) if all_deltas else 0.0,
            "num_conversations": len(all_deltas),
        }

    return results
