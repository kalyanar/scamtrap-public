"""Train CLIP-style intent alignment model (Stage B)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.models.clip_model import CLIPScamModel
from scamtrap.data.clip_dataloader import create_clip_dataloaders
from scamtrap.data.intent_descriptions import INTENT_DESCRIPTIONS, SEEN_INTENTS
from scamtrap.training.clip_trainer import CLIPTrainer


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model (Stage B)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (for ablation)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # Load processed data
    data_dir = Path(config.data.output_dir)
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)
            print(f"Loaded {name}: {len(splits[name])} samples")

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    intent_to_id = meta["intent_to_id"]

    # Build model
    model = CLIPScamModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally warm-start from Stage A
    if config.stage_b.message_encoder_init == "stage_a":
        ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
        if ckpt_path.exists():
            model.load_stage_a_encoder(str(ckpt_path))
        else:
            print(f"WARNING: Stage A checkpoint not found at {ckpt_path}")

    # Create dataloaders
    loaders = create_clip_dataloaders(
        splits, config, intent_to_id, augment=args.augment,
    )

    # Filter descriptions to seen intents only for training
    seen_descriptions = {
        k: v for k, v in INTENT_DESCRIPTIONS.items() if k in SEEN_INTENTS
    }

    # Build seen_intent_to_id (only seen intents)
    seen_intent_to_id = {
        k: v for k, v in intent_to_id.items() if k in SEEN_INTENTS
    }

    # Train
    trainer = CLIPTrainer(
        model, loaders["train"], loaders["val"], config,
        seen_descriptions, seen_intent_to_id,
    )
    history = trainer.train()

    # Save history
    ckpt_dir = Path(config.stage_b.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
