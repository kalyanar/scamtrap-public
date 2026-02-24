"""UMAP embedding visualization."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from pathlib import Path


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    id_to_intent: dict,
    title: str = "ScamTrap Embeddings",
    save_path: str = "umap.png",
    highlight_unseen: list[int] = None,
    seed: int = 42,
):
    """Generate UMAP 2D projection colored by intent.

    Unseen intents (if specified) are drawn with star markers.
    """
    highlight_unseen = highlight_unseen or []

    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette("husl", n_colors=len(unique_labels))
    color_map = {l: palette[i] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for label_id in unique_labels:
        mask = labels == label_id
        name = id_to_intent.get(str(label_id), id_to_intent.get(label_id, f"class_{label_id}"))
        marker = "*" if label_id in highlight_unseen else "o"
        size = 100 if label_id in highlight_unseen else 20
        alpha = 0.9 if label_id in highlight_unseen else 0.6

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[label_id]],
            marker=marker, s=size, alpha=alpha,
            label=f"{name} ({mask.sum()})",
        )

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # Also save PDF for LaTeX
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP plot -> {save_path}")
