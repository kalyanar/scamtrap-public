"""Generate UMAP visualizations of embeddings."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scamtrap.utils.config import load_config
from scamtrap.evaluation.visualization import plot_umap


def visualize_embeddings(emb_dir, output_dir, id_to_intent, holdout_ids,
                         prefix, seed):
    """Generate UMAP plots for a set of embeddings."""
    figures_generated = 0

    for split in ["test_seen", "test_unseen"]:
        emb_path = emb_dir / f"{split}_embeddings.npy"
        label_path = emb_dir / f"{split}_labels.npy"
        if emb_path.exists() and label_path.exists():
            embs = np.load(str(emb_path))
            labels = np.load(str(label_path))
            plot_umap(
                embs, labels, id_to_intent,
                title=f"{prefix} - {split}",
                save_path=str(output_dir / f"umap_{prefix.lower().replace(' ', '_')}_{split}.png"),
                highlight_unseen=holdout_ids if split == "test_unseen" else None,
                seed=seed,
            )
            figures_generated += 1

    # Combined plot
    seen_emb = emb_dir / "test_seen_embeddings.npy"
    unseen_emb = emb_dir / "test_unseen_embeddings.npy"
    if seen_emb.exists() and unseen_emb.exists():
        all_embs = np.vstack([np.load(str(seen_emb)), np.load(str(unseen_emb))])
        all_labels = np.concatenate([
            np.load(str(emb_dir / "test_seen_labels.npy")),
            np.load(str(emb_dir / "test_unseen_labels.npy")),
        ])
        plot_umap(
            all_embs, all_labels, id_to_intent,
            title=f"{prefix} - Seen + Unseen Intents",
            save_path=str(output_dir / f"umap_{prefix.lower().replace(' ', '_')}_combined.png"),
            highlight_unseen=holdout_ids,
            seed=seed,
        )
        figures_generated += 1

    return figures_generated


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP plots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/figures")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir or config.evaluation.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config.data.output_dir).resolve()
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    id_to_intent = meta["id_to_intent"]

    # Determine holdout intent IDs
    holdout_ids = []
    for intent in config.data.holdout_intents:
        for k, v in meta["intent_to_id"].items():
            if k == intent:
                holdout_ids.append(v)

    total_figures = 0

    # 1. Stage A (SupCon) embeddings
    emb_dir = results_dir / "embeddings"
    if emb_dir.exists():
        n = visualize_embeddings(
            emb_dir, output_dir, id_to_intent, holdout_ids,
            "ScamTrap SupCon", config.seed,
        )
        total_figures += n
    else:
        print(f"WARNING: Stage A embeddings not found at {emb_dir}")

    # 2. Stage B (CLIP) embeddings
    clip_emb_dir = results_dir / "clip" / "embeddings"
    if clip_emb_dir.exists():
        n = visualize_embeddings(
            clip_emb_dir, output_dir, id_to_intent, holdout_ids,
            "ScamTrap CLIP", config.seed,
        )
        total_figures += n
    else:
        print(f"INFO: Stage B embeddings not found at {clip_emb_dir}")

    print(f"Generated {total_figures} figures in {output_dir}")


if __name__ == "__main__":
    main()
