"""Run freeze ablation: 7 configurations of description encoder freezing."""

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
from scamtrap.models.clip_model import CLIPScamModel
from scamtrap.data.clip_dataloader import create_clip_dataloaders
from scamtrap.data.intent_descriptions import INTENT_DESCRIPTIONS, SEEN_INTENTS
from scamtrap.training.clip_trainer import CLIPTrainer
from scamtrap.evaluation.zeroshot import evaluate_zeroshot
from scamtrap.evaluation.openset import evaluate_openset


ABLATION_CONFIGS = [
    {"name": "fully_frozen", "freeze_layers": 6, "l2sp_alpha": 0.0},
    {"name": "top2_unfrozen", "freeze_layers": 4, "l2sp_alpha": 0.0},
    {"name": "top2_unfrozen_l2sp", "freeze_layers": 4, "l2sp_alpha": 0.01},
    {"name": "top4_unfrozen", "freeze_layers": 2, "l2sp_alpha": 0.0},
    {"name": "top4_unfrozen_l2sp", "freeze_layers": 2, "l2sp_alpha": 0.01},
    {"name": "fully_unfrozen", "freeze_layers": 0, "l2sp_alpha": 0.0},
    {"name": "fully_unfrozen_l2sp", "freeze_layers": 0, "l2sp_alpha": 0.01},
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
        description="Run freeze ablation for description encoder",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(
        args.output_dir or config.evaluation.results_dir,
    ).resolve() / "freeze_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data_dir = Path(config.data.output_dir)
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    intent_to_id = meta["intent_to_id"]
    id_to_intent = meta["id_to_intent"]

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    seen_descriptions = {
        k: v for k, v in INTENT_DESCRIPTIONS.items() if k in SEEN_INTENTS
    }
    seen_intent_to_id = {
        k: v for k, v in intent_to_id.items() if k in SEEN_INTENTS
    }

    all_results = {}

    for ablation in ABLATION_CONFIGS:
        name = ablation["name"]
        print(f"\n{'='*60}")
        print(f"Config: {name} (freeze={ablation['freeze_layers']}, "
              f"L2-SP={ablation['l2sp_alpha']})")
        print(f"{'='*60}")

        # Modify config for this run
        run_config = copy.deepcopy(config)
        run_config["stage_b"]["freeze_description_layers"] = ablation["freeze_layers"]
        run_config["stage_b"]["l2sp_alpha"] = ablation["l2sp_alpha"]
        run_config["stage_b"]["checkpoint_dir"] = f"checkpoints/clip_ablation/{name}"

        # Build fresh model
        set_seed(config.seed)
        model = CLIPScamModel(run_config)

        # Warm-start from Stage A
        if run_config.stage_b.message_encoder_init == "stage_a":
            ckpt_path = Path(run_config.training.checkpoint_dir) / "best_model.pt"
            if ckpt_path.exists():
                model.load_stage_a_encoder(str(ckpt_path))

        # Create dataloaders
        loaders = create_clip_dataloaders(
            splits, run_config, intent_to_id, augment=False,
        )

        # Train
        trainer = CLIPTrainer(
            model, loaders["train"], loaders["val"], run_config,
            seen_descriptions, seen_intent_to_id,
        )
        history = trainer.train()

        # Evaluate: zero-shot on seen + unseen
        intent_to_id_all = {k: int(v) for k, v in intent_to_id.items()}

        zs_seen = evaluate_zeroshot(
            model, splits["test_seen"]["text"].tolist(),
            splits["test_seen"]["intent_id"].values,
            INTENT_DESCRIPTIONS, intent_to_id_all,
            config.model.encoder_name, max_length=config.data.max_length,
            device=device,
        )

        zs_unseen = {}
        if len(splits.get("test_unseen", [])) > 0:
            zs_unseen = evaluate_zeroshot(
                model, splits["test_unseen"]["text"].tolist(),
                splits["test_unseen"]["intent_id"].values,
                INTENT_DESCRIPTIONS, intent_to_id_all,
                config.model.encoder_name, max_length=config.data.max_length,
                device=device,
            )

        # Open-set novelty detection
        embeddings = {}
        for split_name in ["train", "test_seen", "test_unseen"]:
            if split_name in splits:
                embeddings[split_name] = extract_embeddings(
                    model, splits[split_name]["text"].tolist(),
                    tokenizer, max_length=config.data.max_length, device=device,
                )

        openset = {}
        if "test_unseen" in embeddings:
            openset = evaluate_openset(
                embeddings["train"], splits["train"]["intent_id"].values,
                embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
                embeddings["test_unseen"], splits["test_unseen"]["intent_id"].values,
                seed=config.seed,
            )

        run_results = {
            "config": ablation,
            "zeroshot_seen_accuracy": zs_seen.get("accuracy", 0.0),
            "zeroshot_seen_f1": zs_seen.get("f1_macro", 0.0),
            "zeroshot_unseen_accuracy": zs_unseen.get("accuracy", 0.0),
            "zeroshot_unseen_f1": zs_unseen.get("f1_macro", 0.0),
            "novelty_auroc": openset.get("novelty_auroc", 0.0),
            "training_epochs": len(history.get("train_loss", [])),
            "best_val_loss": min(history.get("val_loss", [float("inf")])),
        }

        all_results[name] = run_results
        print(f"\n  Seen ZS Acc: {run_results['zeroshot_seen_accuracy']:.3f}")
        print(f"  Unseen ZS Acc: {run_results['zeroshot_unseen_accuracy']:.3f}")
        print(f"  Novelty AUROC: {run_results['novelty_auroc']:.3f}")

    # Summary table
    print(f"\n{'='*60}")
    print("FREEZE ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Seen-ZS':>8} {'Unseen-ZS':>10} {'Nov-AUROC':>10}")
    print("-" * 55)
    for name, res in all_results.items():
        print(f"{name:<25} {res['zeroshot_seen_accuracy']:>8.3f} "
              f"{res['zeroshot_unseen_accuracy']:>10.3f} "
              f"{res['novelty_auroc']:>10.3f}")

    # Save
    results_path = output_dir / "freeze_ablation_results.json"
    save_results(all_results, str(results_path))
    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
