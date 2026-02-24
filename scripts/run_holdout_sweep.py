"""Run open-set holdout sweep: 4 holdout configurations."""

import argparse
import json
import copy
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.utils.io import save_results
from scamtrap.data.datasets import load_and_merge
from scamtrap.data.intent_labeler import label_intents
from scamtrap.data.splits import create_splits, encode_labels
from scamtrap.data.intent_descriptions import (
    INTENT_DESCRIPTIONS, get_seen_and_holdout,
)
from scamtrap.data.clip_dataloader import create_clip_dataloaders
from scamtrap.models.clip_model import CLIPScamModel
from scamtrap.training.clip_trainer import CLIPTrainer
from scamtrap.evaluation.zeroshot import evaluate_zeroshot
from scamtrap.evaluation.openset import evaluate_openset


HOLDOUT_CONFIGS = [
    {"name": "crypto_romance", "holdout": ["crypto", "romance"]},
    {"name": "job_prize", "holdout": ["job_offer", "prize_lottery"]},
    {"name": "bank_delivery", "holdout": ["bank_alert", "delivery"]},
    {"name": "credential_generic", "holdout": ["credential_theft", "generic_scam"]},
]


def extract_embeddings(model, texts, tokenizer, max_length=128,
                       batch_size=64, device="cpu"):
    """Extract encoder embeddings (768d)."""
    model.eval()
    model.to(device)
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            embs = model.get_embeddings(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser(
        description="Run holdout sweep for open-set evaluation",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(
        args.output_dir or config.evaluation.results_dir,
    ).resolve() / "holdout_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # Load full dataset (before splitting)
    print("Loading full dataset...")
    df = load_and_merge(config)
    df = label_intents(df, config)

    all_results = {}

    for holdout_cfg in HOLDOUT_CONFIGS:
        name = holdout_cfg["name"]
        holdout_intents = holdout_cfg["holdout"]
        seen_intents, _ = get_seen_and_holdout(holdout_intents)

        print(f"\n{'='*60}")
        print(f"Holdout config: {name}")
        print(f"  Holdout: {holdout_intents}")
        print(f"  Seen: {seen_intents}")
        print(f"{'='*60}")

        # Re-split with this holdout
        splits = create_splits(
            df,
            test_size=config.data.test_size,
            val_size=config.data.val_size,
            holdout_intents=holdout_intents,
            min_samples_per_intent=config.data.min_samples_per_intent,
            seed=config.seed,
        )

        # Check if holdout has enough samples
        n_unseen = len(splits["test_unseen"])
        if n_unseen < 10:
            print(f"  WARNING: Only {n_unseen} unseen samples, skipping")
            all_results[name] = {"skipped": True, "reason": f"only {n_unseen} unseen"}
            continue

        intent_to_id, id_to_intent = encode_labels(splits)

        # Build seen descriptions and ID mapping
        seen_descriptions = {
            k: v for k, v in INTENT_DESCRIPTIONS.items() if k in seen_intents
        }
        seen_intent_to_id = {
            k: v for k, v in intent_to_id.items() if k in seen_intents
        }

        # Create dataloaders
        loaders = create_clip_dataloaders(
            splits, config, intent_to_id, augment=False,
        )

        # Build and train model
        set_seed(config.seed)
        run_config = copy.deepcopy(config)
        run_config["stage_b"]["checkpoint_dir"] = f"checkpoints/clip_holdout/{name}"

        model = CLIPScamModel(run_config)

        # Warm-start from Stage A
        if run_config.stage_b.message_encoder_init == "stage_a":
            ckpt_path = Path(run_config.training.checkpoint_dir) / "best_model.pt"
            if ckpt_path.exists():
                model.load_stage_a_encoder(str(ckpt_path))

        trainer = CLIPTrainer(
            model, loaders["train"], loaders["val"], run_config,
            seen_descriptions, seen_intent_to_id,
        )
        history = trainer.train()

        # Evaluate
        intent_to_id_all = {k: int(v) for k, v in intent_to_id.items()}

        zs_seen = evaluate_zeroshot(
            model, splits["test_seen"]["text"].tolist(),
            splits["test_seen"]["intent_id"].values,
            INTENT_DESCRIPTIONS, intent_to_id_all,
            config.model.encoder_name, max_length=config.data.max_length,
            device=device,
        )

        zs_unseen = evaluate_zeroshot(
            model, splits["test_unseen"]["text"].tolist(),
            splits["test_unseen"]["intent_id"].values,
            INTENT_DESCRIPTIONS, intent_to_id_all,
            config.model.encoder_name, max_length=config.data.max_length,
            device=device,
        )

        # Open-set eval
        embeddings = {}
        for split_name in ["train", "test_seen", "test_unseen"]:
            embeddings[split_name] = extract_embeddings(
                model, splits[split_name]["text"].tolist(),
                tokenizer, max_length=config.data.max_length, device=device,
            )

        openset = evaluate_openset(
            embeddings["train"], splits["train"]["intent_id"].values,
            embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
            embeddings["test_unseen"], splits["test_unseen"]["intent_id"].values,
            seed=config.seed,
        )

        run_results = {
            "holdout": holdout_intents,
            "n_train": len(splits["train"]),
            "n_test_seen": len(splits["test_seen"]),
            "n_test_unseen": len(splits["test_unseen"]),
            "seen_accuracy": zs_seen.get("accuracy", 0.0),
            "seen_f1": zs_seen.get("f1_macro", 0.0),
            "unseen_accuracy": zs_unseen.get("accuracy", 0.0),
            "unseen_f1": zs_unseen.get("f1_macro", 0.0),
            "novelty_auroc": openset.get("novelty_auroc", 0.0),
            "unseen_nmi": openset.get("unseen_clustering_nmi", 0.0),
        }

        all_results[name] = run_results
        print(f"\n  Seen Acc: {run_results['seen_accuracy']:.3f}")
        print(f"  Unseen Acc: {run_results['unseen_accuracy']:.3f}")
        print(f"  Novelty AUROC: {run_results['novelty_auroc']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("HOLDOUT SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Holdout':<25} {'Seen':>6} {'Unseen':>8} {'Unseen-F1':>10} {'AUROC':>8}")
    print("-" * 60)

    valid_results = {k: v for k, v in all_results.items() if not v.get("skipped")}
    for name, res in valid_results.items():
        print(f"{name:<25} {res['seen_accuracy']:>6.3f} "
              f"{res['unseen_accuracy']:>8.3f} "
              f"{res['unseen_f1']:>10.3f} "
              f"{res['novelty_auroc']:>8.3f}")

    if valid_results:
        means = {
            "seen_accuracy": np.mean([r["seen_accuracy"] for r in valid_results.values()]),
            "unseen_accuracy": np.mean([r["unseen_accuracy"] for r in valid_results.values()]),
            "unseen_f1": np.mean([r["unseen_f1"] for r in valid_results.values()]),
            "novelty_auroc": np.mean([r["novelty_auroc"] for r in valid_results.values()]),
        }
        stds = {
            "seen_accuracy": np.std([r["seen_accuracy"] for r in valid_results.values()]),
            "unseen_accuracy": np.std([r["unseen_accuracy"] for r in valid_results.values()]),
            "unseen_f1": np.std([r["unseen_f1"] for r in valid_results.values()]),
            "novelty_auroc": np.std([r["novelty_auroc"] for r in valid_results.values()]),
        }
        all_results["_summary"] = {"mean": means, "std": stds}
        print(f"\nMean±std across configs:")
        print(f"  Seen:   {means['seen_accuracy']:.3f}±{stds['seen_accuracy']:.3f}")
        print(f"  Unseen: {means['unseen_accuracy']:.3f}±{stds['unseen_accuracy']:.3f}")
        print(f"  AUROC:  {means['novelty_auroc']:.3f}±{stds['novelty_auroc']:.3f}")

    results_path = output_dir / "sweep_results.json"
    save_results(all_results, str(results_path))
    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
