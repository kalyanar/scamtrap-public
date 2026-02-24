"""Download, preprocess, label, and split datasets."""

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.data.datasets import load_and_merge
from scamtrap.data.intent_labeler import label_intents
from scamtrap.data.splits import create_splits, encode_labels


def main():
    parser = argparse.ArgumentParser(description="Prepare ScamTrap datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # Load and merge datasets
    df = load_and_merge(config)

    # Apply intent labeling
    df = label_intents(df, config)

    # Create splits
    splits = create_splits(
        df,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        holdout_intents=config.data.holdout_intents,
        min_samples_per_intent=config.data.min_samples_per_intent,
        seed=config.seed,
    )

    # Encode labels
    intent_to_id, id_to_intent = encode_labels(splits)

    # Save
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        path = output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {name}: {len(split_df)} samples -> {path}")

    # Save label mappings
    meta = {
        "intent_to_id": intent_to_id,
        "id_to_intent": {str(k): v for k, v in id_to_intent.items()},
        "num_intents": len(intent_to_id),
        "holdout_intents": config.data.holdout_intents,
        "splits": {name: len(df) for name, df in splits.items()},
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {meta_path}")

    # Save per-intent x per-split distribution report
    results_dir = Path(config.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    distribution = {}
    for name, split_df in splits.items():
        distribution[name] = split_df["intent_label"].value_counts().to_dict()
    # Totals across splits
    all_combined = pd.concat(splits.values(), ignore_index=True)
    distribution["total"] = all_combined["intent_label"].value_counts().to_dict()
    # Source breakdown
    distribution["source_breakdown"] = df["source"].value_counts().to_dict()
    # Dedup stats (from the merged df before splitting)
    distribution["dedup_stats"] = {
        "after_dedup": len(df),
    }
    dist_path = results_dir / "data_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(distribution, f, indent=2, default=str)
    print(f"Saved data distribution -> {dist_path}")


if __name__ == "__main__":
    main()
