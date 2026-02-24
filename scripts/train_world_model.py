"""Train world model for scam trajectory prediction (Stage C)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.models.world_model import ScamWorldModel, TransformerWorldModel
from scamtrap.data.trajectory_dataset import create_trajectory_dataloaders
from scamtrap.training.world_model_trainer import WorldModelTrainer


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
        description="Train world model (Stage C)",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-type", choices=["gru", "transformer"],
                        default="gru", help="World model architecture")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # Load pre-computed trajectories
    print("Loading trajectory data...")
    trajectories = load_trajectories(config)
    print(f"  Total: {len(trajectories)} trajectories")

    # Create dataloaders
    loaders = create_trajectory_dataloaders(trajectories, config)
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset)} trajectories, "
              f"{len(loader)} batches")

    # Build model
    if args.model_type == "transformer":
        model = TransformerWorldModel(config)
        print("Using Transformer world model")
    else:
        model = ScamWorldModel(config)
        print("Using GRU world model")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = WorldModelTrainer(
        model, loaders["train"], loaders["val"], config,
    )
    history = trainer.train()

    # Save history
    ckpt_dir = Path(config.stage_c.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
