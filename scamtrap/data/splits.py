"""Train/val/test splitting with open-set holdout."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    holdout_intents: list[str] = None,
    min_samples_per_intent: int = 20,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits with open-set holdout.

    Holdout intents are removed entirely from train/val and placed
    only in test_unseen. Remaining intents are stratified across splits.

    Returns dict with keys: train, val, test_seen, test_unseen
    """
    holdout_intents = holdout_intents or []

    # Filter out intents with too few samples
    intent_counts = df["intent_label"].value_counts()
    small_intents = intent_counts[intent_counts < min_samples_per_intent].index.tolist()
    if small_intents:
        print(f"Merging small intents into generic_scam: {small_intents}")
        df = df.copy()
        df.loc[
            df["intent_label"].isin(small_intents) & (df["binary_label"] == 1),
            "intent_label",
        ] = "generic_scam"

    # Separate holdout intents
    holdout_mask = df["intent_label"].isin(holdout_intents)
    df_holdout = df[holdout_mask].copy()
    df_main = df[~holdout_mask].copy()

    print(f"\nOpen-set holdout intents: {holdout_intents}")
    print(f"  Holdout samples: {len(df_holdout)}")
    print(f"  Main samples: {len(df_main)}")

    # Stratified split of main data
    # First split: train+val vs test_seen
    df_trainval, df_test_seen = train_test_split(
        df_main,
        test_size=test_size,
        stratify=df_main["intent_label"],
        random_state=seed,
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction,
        stratify=df_trainval["intent_label"],
        random_state=seed,
    )

    splits = {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test_seen": df_test_seen.reset_index(drop=True),
        "test_unseen": df_holdout.reset_index(drop=True),
    }

    for name, split_df in splits.items():
        print(f"  {name}: {len(split_df)} samples")

    return splits


def encode_labels(splits: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """Create label-to-int mappings from training data.

    Returns:
        intent_to_id: dict mapping intent string to int
        id_to_intent: dict mapping int to intent string
    """
    all_intents = sorted(splits["train"]["intent_label"].unique())
    # Add holdout intents at the end
    for split_name in ["test_unseen"]:
        for intent in sorted(splits[split_name]["intent_label"].unique()):
            if intent not in all_intents:
                all_intents.append(intent)

    intent_to_id = {intent: i for i, intent in enumerate(all_intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}

    # Add encoded column to all splits
    for name in splits:
        splits[name] = splits[name].copy()
        splits[name]["intent_id"] = splits[name]["intent_label"].map(intent_to_id)

    return intent_to_id, id_to_intent
