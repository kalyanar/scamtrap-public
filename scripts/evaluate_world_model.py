"""Evaluate world model (Stage C) -- stage prediction, escalation, counterfactual."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.utils.io import load_checkpoint, save_results
from scamtrap.models.world_model import ScamWorldModel, TransformerWorldModel
from scamtrap.data.trajectory_dataset import create_trajectory_dataloaders
from scamtrap.evaluation.trajectory import (
    evaluate_stage_prediction,
    evaluate_escalation_forecast,
    evaluate_early_warning,
    evaluate_early_warning_sweep,
)
from scamtrap.evaluation.counterfactual import analyze_intervention_impact
from scamtrap.evaluation.calibration import compute_ece, compute_multiclass_ece
from scamtrap.baselines.trajectory_baselines import (
    MarkovChainBaseline,
    LogRegBaseline,
)


def load_trajectories(config):
    """Load pre-computed trajectory embeddings and metadata."""
    conv_dir = Path(config.data.output_dir) / "conversations"

    with open(conv_dir / "metadata.json") as f:
        meta = json.load(f)

    npz = np.load(str(conv_dir / "embeddings.npz"))

    trajectories = []
    for i, traj_meta in enumerate(meta["trajectories"]):
        trajectories.append({
            "embeddings": npz[f"arr_{i}"],
            "stages": traj_meta["stages"],
            "is_scam": traj_meta["is_scam"],
            "split": traj_meta["split"],
        })

    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate world model (Stage C)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-type", choices=["gru", "transformer"],
                        default="gru")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(
        args.output_dir or config.evaluation.results_dir,
    ).resolve() / "world_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trajectories
    print("Loading trajectory data...")
    trajectories = load_trajectories(config)
    loaders = create_trajectory_dataloaders(trajectories, config)

    # Load model
    if args.model_type == "transformer":
        model = TransformerWorldModel(config)
    else:
        model = ScamWorldModel(config)

    ckpt_path = Path(config.stage_c.checkpoint_dir) / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(str(ckpt_path), model)
        print(f"Loaded world model from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}")

    all_results = {"model": f"world_model_{args.model_type}"}

    # 1. Stage prediction
    print("\n--- Stage Prediction ---")
    stage_results = evaluate_stage_prediction(model, loaders["test"], device)
    all_results["stage_prediction"] = stage_results
    print(f"  Accuracy: {stage_results['accuracy']:.3f}")
    print(f"  F1-macro: {stage_results['f1_macro']:.3f}")

    # 2. Escalation forecast
    print("\n--- Escalation Forecast ---")
    esc_results = evaluate_escalation_forecast(model, loaders["test"], device)
    all_results["escalation_forecast"] = esc_results
    print(f"  AUROC: {esc_results['auroc']:.3f}")
    print(f"  Brier: {esc_results['brier_score']:.4f}")
    print(f"  Accuracy: {esc_results['accuracy']:.3f}")

    # 3. Early warning (original horizon=3)
    print("\n--- Early Warning ---")
    early_results = evaluate_early_warning(
        model, loaders["test"], device, horizon=3,
    )
    all_results["early_warning"] = early_results
    print(f"  Detection rate (3-turn lead): "
          f"{early_results['early_detection_rate']:.3f}")
    print(f"  Avg lead time: {early_results['avg_lead_time']:.1f} turns")

    # 3b. Early warning sweep (multiple horizons)
    print("\n--- Early Warning Sweep ---")
    horizons = [1, 2, 3, 5, 7]
    early_sweep = evaluate_early_warning_sweep(
        model, loaders["test"], device, horizons=horizons,
    )
    all_results["early_warning_sweep"] = early_sweep
    for h, info in early_sweep.items():
        print(f"  Horizon={h}: detection={info['early_detection_rate']:.3f}, "
              f"precision={info['precision']:.3f}, recall={info['recall']:.3f}, "
              f"f1={info['f1']:.3f}")

    # 3c. Calibration (ECE) for escalation probabilities
    print("\n--- Calibration (ECE) ---")
    # Collect escalation probs and labels from test set
    model.eval()
    model.to(device)
    all_esc_probs = []
    all_esc_labels = []
    all_stage_probs = []
    all_stage_labels = []

    with torch.no_grad():
        for batch in loaders["test"]:
            embs = batch["embeddings"].to(device)
            lengths = batch["length"].to(device)
            esc_labels = batch["escalation_labels"]
            stage_labels = batch["stage_labels"]

            stage_logits, esc_logits, _ = model(embs, lengths)
            esc_probs = torch.sigmoid(esc_logits.squeeze(-1)).cpu()
            stage_probs_batch = torch.softmax(stage_logits, dim=-1).cpu()

            for b in range(esc_probs.shape[0]):
                T = int(lengths[b].item())
                all_esc_probs.extend(esc_probs[b, :T].numpy())
                all_esc_labels.extend(esc_labels[b, :T].numpy())
                # Stage probs — filter out padding
                valid_stage = stage_labels[b, :T].numpy()
                valid_mask = valid_stage >= 0
                if valid_mask.any():
                    all_stage_probs.append(stage_probs_batch[b, :T][valid_mask].numpy())
                    all_stage_labels.extend(valid_stage[valid_mask])

    esc_probs_arr = np.array(all_esc_probs)
    esc_labels_arr = np.array(all_esc_labels)
    esc_ece = compute_ece(esc_probs_arr, esc_labels_arr)
    all_results["calibration_escalation"] = {
        "ece": esc_ece["ece"],
        "mce": esc_ece["mce"],
    }
    print(f"  Escalation ECE: {esc_ece['ece']:.4f}, MCE: {esc_ece['mce']:.4f}")

    if all_stage_probs:
        stage_probs_arr = np.vstack(all_stage_probs)
        stage_labels_arr = np.array(all_stage_labels)
        stage_ece = compute_multiclass_ece(stage_probs_arr, stage_labels_arr)
        all_results["calibration_stage"] = {
            "ece": stage_ece["ece"],
            "mce": stage_ece["mce"],
        }
        print(f"  Stage ECE: {stage_ece['ece']:.4f}, MCE: {stage_ece['mce']:.4f}")

    # 4. Counterfactual analysis
    print("\n--- Counterfactual Analysis ---")
    cf_results = analyze_intervention_impact(
        model, loaders["test"], device,
        intervention_turns=[3, 5, 7, 10],
    )
    all_results["counterfactual"] = cf_results
    for turn, info in cf_results.items():
        print(f"  {turn}: avg prob reduction = "
              f"{info['avg_prob_reduction']:.3f} "
              f"(n={info['num_conversations']})")

    # 5. Baselines
    print("\n--- Baselines ---")
    train_trajs = [t for t in trajectories if t["split"] == "train"]
    test_trajs = [t for t in trajectories if t["split"] == "test"]

    # Markov chain
    markov = MarkovChainBaseline(num_stages=config.stage_c.num_stages)
    markov.fit(train_trajs)
    markov_results = markov.evaluate(test_trajs)
    all_results["baselines"] = {"markov_chain": markov_results}
    print(f"  Markov - Stage Acc: {markov_results['stage_accuracy']:.3f}, "
          f"Esc AUROC: {markov_results['escalation_auroc']:.3f}")

    # LogReg (single turn)
    logreg = LogRegBaseline(num_stages=config.stage_c.num_stages)
    logreg.fit(train_trajs)
    logreg_results = logreg.evaluate(test_trajs)
    all_results["baselines"]["logreg_single_turn"] = logreg_results
    print(f"  LogReg - Stage Acc: {logreg_results['stage_accuracy']:.3f}, "
          f"Esc AUROC: {logreg_results['escalation_auroc']:.3f}")

    # Save
    results_path = output_dir / "world_model_results.json"
    save_results(all_results, str(results_path))
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
