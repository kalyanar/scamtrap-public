"""Train the contrastive ScamTrap model."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.utils.io import save_results
from scamtrap.models.scamtrap_model import ScamTrapModel
from scamtrap.data.dataloader import create_dataloaders
from scamtrap.training.trainer import ContrastiveTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ScamTrap contrastive model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    data_dir = Path(args.data_dir or config.data.output_dir)
    output_dir = Path(args.output_dir or config.training.checkpoint_dir)

    # Load processed data
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {len(splits[name])} samples")

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    intent_to_id = meta["intent_to_id"]

    # Create dataloaders
    loaders = create_dataloaders(splits, config, intent_to_id)

    # Build model
    model = ScamTrapModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable:,} trainable")

    # Update checkpoint dir
    config.training.checkpoint_dir = str(output_dir)

    # Train
    trainer = ContrastiveTrainer(model, loaders["train"], loaders["val"], config)
    history = trainer.train()

    # Save training history
    save_results(history, str(output_dir / "training_history.json"))
    print(f"\nTraining complete. Checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
