"""Evaluate trajectory prediction robustness under perturbation."""

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
from scamtrap.models.world_model import ScamWorldModel
from scamtrap.baselines.trajectory_baselines import MarkovChainBaseline, LogRegBaseline
from scamtrap.evaluation.robustness_trajectory import (
    evaluate_model_on_perturbed,
    evaluate_markov_on_perturbed,
    evaluate_logreg_on_perturbed,
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
        description="Evaluate trajectory robustness under perturbation",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=5)
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
    train_trajs = [t for t in trajectories if t["split"] == "train"]
    test_trajs = [t for t in trajectories if t["split"] == "test"]

    # Load GRU model
    model = ScamWorldModel(config)
    ckpt_path = Path(config.stage_c.checkpoint_dir) / "best_model_gru.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(config.stage_c.checkpoint_dir) / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(str(ckpt_path), model)
        print(f"Loaded GRU model from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found")

    # Fit baselines
    markov = MarkovChainBaseline(num_stages=config.stage_c.num_stages)
    markov.fit(train_trajs)
    logreg = LogRegBaseline(num_stages=config.stage_c.num_stages)
    logreg.fit(train_trajs)

    perturbation_types = ["stage_noise", "turn_dropout", "non_monotonic", "combined"]
    perturbation_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    n_trials = args.n_trials

    all_results = {}

    for ptype in perturbation_types:
        print(f"\n--- Perturbation: {ptype} ---")
        all_results[ptype] = {}

        for level in perturbation_levels:
            gru_accs = []
            gru_aurocs = []
            markov_accs = []
            markov_aurocs = []
            logreg_accs = []
            logreg_aurocs = []

            for trial in range(n_trials):
                np.random.seed(config.seed + trial)

                # GRU
                gru_res = evaluate_model_on_perturbed(
                    model, test_trajs, ptype, level, device=device,
                    max_turns=config.stage_c.max_turns,
                )
                gru_accs.append(gru_res["stage_accuracy"])
                gru_aurocs.append(gru_res["escalation_auroc"])

                # Markov
                markov_res = evaluate_markov_on_perturbed(
                    markov, test_trajs, ptype, level,
                )
                markov_accs.append(markov_res["stage_accuracy"])
                markov_aurocs.append(markov_res["escalation_auroc"])

                # LogReg
                logreg_res = evaluate_logreg_on_perturbed(
                    logreg, test_trajs, ptype, level,
                )
                logreg_accs.append(logreg_res["stage_accuracy"])
                logreg_aurocs.append(logreg_res["escalation_auroc"])

            all_results[ptype][str(level)] = {
                "gru": {
                    "stage_acc_mean": float(np.mean(gru_accs)),
                    "stage_acc_std": float(np.std(gru_accs)),
                    "esc_auroc_mean": float(np.nanmean(gru_aurocs)),
                    "esc_auroc_std": float(np.nanstd(gru_aurocs)),
                },
                "markov": {
                    "stage_acc_mean": float(np.mean(markov_accs)),
                    "stage_acc_std": float(np.std(markov_accs)),
                    "esc_auroc_mean": float(np.nanmean(markov_aurocs)),
                    "esc_auroc_std": float(np.nanstd(markov_aurocs)),
                },
                "logreg": {
                    "stage_acc_mean": float(np.mean(logreg_accs)),
                    "stage_acc_std": float(np.std(logreg_accs)),
                    "esc_auroc_mean": float(np.nanmean(logreg_aurocs)),
                    "esc_auroc_std": float(np.nanstd(logreg_aurocs)),
                },
            }

            gru_info = all_results[ptype][str(level)]["gru"]
            markov_info = all_results[ptype][str(level)]["markov"]
            print(f"  p={level:.1f}: GRU acc={gru_info['stage_acc_mean']:.3f}±"
                  f"{gru_info['stage_acc_std']:.3f}, "
                  f"Markov acc={markov_info['stage_acc_mean']:.3f}±"
                  f"{markov_info['stage_acc_std']:.3f}")

    results_path = output_dir / "robustness_results.json"
    save_results(all_results, str(results_path))
    print(f"\nSaved robustness results -> {results_path}")


if __name__ == "__main__":
    main()
