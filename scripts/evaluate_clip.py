"""Evaluate CLIP model (Stage B) -- all Stage A metrics + zero-shot."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.utils.io import load_checkpoint, save_results
from scamtrap.models.clip_model import CLIPScamModel
from scamtrap.evaluation.fewshot import evaluate_fewshot
from scamtrap.evaluation.openset import evaluate_openset
from scamtrap.evaluation.retrieval import evaluate_retrieval
from scamtrap.evaluation.clustering import evaluate_clustering
from scamtrap.evaluation.robustness import evaluate_robustness
from scamtrap.evaluation.zeroshot import evaluate_zeroshot
from scamtrap.evaluation.calibration import compute_multiclass_ece
from scamtrap.evaluation.report import save_report
from scamtrap.data.intent_descriptions import INTENT_DESCRIPTIONS


def extract_embeddings(model, texts, tokenizer, max_length=128,
                       batch_size=64, device="cuda"):
    """Extract encoder embeddings (768d) for all texts."""
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP model (Stage B)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    data_dir = Path(args.data_dir or config.data.output_dir)
    model_dir = Path(args.model_dir or config.stage_b.checkpoint_dir)
    output_dir = Path(args.output_dir or config.evaluation.results_dir) / "clip"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    intent_to_id = meta["intent_to_id"]
    id_to_intent = meta["id_to_intent"]

    # Load model
    model = CLIPScamModel(config)
    ckpt_path = model_dir / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(str(ckpt_path), model)
        print(f"Loaded CLIP model from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}, using untrained model")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # Extract 768d embeddings (compatible with all Stage A eval code)
    print("Extracting embeddings...")
    embeddings = {}
    for name, df in splits.items():
        embeddings[name] = extract_embeddings(
            model, df["text"].tolist(), tokenizer,
            max_length=config.data.max_length, device=device,
        )
        print(f"  {name}: {embeddings[name].shape}")

    all_results = {"model": "clip_alignment"}

    # 1. Few-shot evaluation
    print("\n--- Few-shot Evaluation ---")
    fewshot_results = evaluate_fewshot(
        embeddings["train"], splits["train"]["intent_id"].values,
        embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
        label_fractions=config.evaluation.fewshot_fractions,
        n_trials=config.evaluation.fewshot_trials,
        seed=config.seed,
    )
    all_results["fewshot"] = fewshot_results
    for frac, metrics in fewshot_results.items():
        print(f"  {frac}: F1={metrics['f1_macro']['mean']:.3f} "
              f"+/- {metrics['f1_macro']['std']:.3f}")

    # 2. Open-set evaluation
    print("\n--- Open-Set Evaluation ---")
    if len(splits.get("test_unseen", [])) > 0:
        openset_results = evaluate_openset(
            embeddings["train"], splits["train"]["intent_id"].values,
            embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
            embeddings["test_unseen"], splits["test_unseen"]["intent_id"].values,
            seed=config.seed,
        )
        all_results["openset"] = openset_results
        for k, v in openset_results.items():
            print(f"  {k}: {v}")

    # 3. Retrieval evaluation
    print("\n--- Retrieval Evaluation ---")
    retrieval_results = evaluate_retrieval(
        embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
        k_values=config.evaluation.retrieval_k,
    )
    all_results["retrieval"] = retrieval_results
    for k, v in retrieval_results.items():
        print(f"  {k}: {v:.3f}")

    # 4. Clustering evaluation
    print("\n--- Clustering Evaluation ---")
    clustering_results = evaluate_clustering(
        embeddings["test_seen"], splits["test_seen"]["intent_id"].values,
        n_runs=config.evaluation.clustering_seeds, seed=config.seed,
    )
    all_results["clustering"] = clustering_results
    for k, v in clustering_results.items():
        print(f"  {k}: {v['mean']:.3f} +/- {v['std']:.3f}")

    # 5. Robustness evaluation
    print("\n--- Robustness Evaluation ---")
    robustness_results = evaluate_robustness(
        model, splits["test_seen"]["text"].tolist(),
        splits["test_seen"]["intent_id"].values,
        tokenizer, max_length=config.data.max_length, device=device,
        seed=config.seed,
    )
    all_results["robustness"] = robustness_results
    for k, v in robustness_results.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['mean']:.3f} +/- {v['std']:.3f}")

    # 6. Zero-shot evaluation (NEW for Stage B)
    print("\n--- Zero-Shot Evaluation ---")

    # Build full intent_to_id including holdout intents
    intent_to_id_all = {k: int(v) for k, v in intent_to_id.items()}

    # Zero-shot on test_seen
    zs_seen = evaluate_zeroshot(
        model, splits["test_seen"]["text"].tolist(),
        splits["test_seen"]["intent_id"].values,
        INTENT_DESCRIPTIONS, intent_to_id_all,
        config.model.encoder_name, max_length=config.data.max_length,
        device=device,
    )
    all_results["zeroshot_seen"] = zs_seen
    print(f"  Seen - Accuracy: {zs_seen['accuracy']:.3f}, "
          f"F1-macro: {zs_seen['f1_macro']:.3f}")

    # Zero-shot on test_unseen (key test -- crypto + romance)
    if len(splits.get("test_unseen", [])) > 0:
        zs_unseen = evaluate_zeroshot(
            model, splits["test_unseen"]["text"].tolist(),
            splits["test_unseen"]["intent_id"].values,
            INTENT_DESCRIPTIONS, intent_to_id_all,
            config.model.encoder_name, max_length=config.data.max_length,
            device=device,
        )
        all_results["zeroshot_unseen"] = zs_unseen
        print(f"  Unseen - Accuracy: {zs_unseen['accuracy']:.3f}, "
              f"F1-macro: {zs_unseen['f1_macro']:.3f}")

    # 7. Calibration (ECE) for zero-shot classifier
    print("\n--- Calibration (ECE) ---")
    # Compute softmax probabilities for test_seen via CLIP forward pass
    model.eval()
    model.to(device)
    all_zs_probs = []
    all_zs_labels = []

    # Tokenize all 9 intent descriptions for full zero-shot
    all_desc_texts = []
    all_desc_intents = []
    for intent_name in sorted(INTENT_DESCRIPTIONS.keys()):
        all_desc_texts.append(INTENT_DESCRIPTIONS[intent_name])
        all_desc_intents.append(intent_name)

    desc_enc = tokenizer(
        all_desc_texts, max_length=config.data.max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    desc_ids = desc_enc["input_ids"].to(device)
    desc_mask = desc_enc["attention_mask"].to(device)

    # Build intent_name -> column index mapping
    desc_intent_to_col = {name: i for i, name in enumerate(all_desc_intents)}

    for split_name in ["test_seen"]:
        df_split = splits[split_name]
        texts = df_split["text"].tolist()
        labels = df_split["intent_id"].values
        # Map intent_id -> intent_name -> desc column
        id_to_intent_map = {int(k): v for k, v in id_to_intent.items()}

        batch_size_eval = 64
        split_probs = []
        for i in range(0, len(texts), batch_size_eval):
            batch_texts = texts[i:i + batch_size_eval]
            enc = tokenizer(
                batch_texts, max_length=config.data.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                _, logits = model(
                    enc["input_ids"].to(device), enc["attention_mask"].to(device),
                    desc_ids, desc_mask,
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                split_probs.append(probs)

        split_probs = np.vstack(split_probs)
        # Remap labels to desc column indices
        remapped_labels = np.array([
            desc_intent_to_col[id_to_intent_map[int(l)]]
            for l in labels
        ])
        ece_results = compute_multiclass_ece(split_probs, remapped_labels)
        all_results["calibration_test_seen"] = {
            "ece": ece_results["ece"],
            "mce": ece_results["mce"],
        }
        print(f"  Test-Seen ECE: {ece_results['ece']:.4f}, MCE: {ece_results['mce']:.4f}")

    # Save results
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "clip_results.json"
    save_results(all_results, str(results_path))
    print(f"\nSaved clip_results.json -> {results_path}")

    # Save embeddings for visualization
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for name, emb in embeddings.items():
        np.save(str(emb_dir / f"{name}_embeddings.npy"), emb)
        np.save(str(emb_dir / f"{name}_labels.npy"),
                splits[name]["intent_id"].values)

    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
