"""Run full evaluation suite on trained models."""

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
from scamtrap.models.scamtrap_model import ScamTrapModel
from scamtrap.evaluation.fewshot import evaluate_fewshot
from scamtrap.evaluation.openset import evaluate_openset
from scamtrap.evaluation.retrieval import evaluate_retrieval
from scamtrap.evaluation.clustering import evaluate_clustering
from scamtrap.evaluation.robustness import evaluate_robustness
from scamtrap.evaluation.report import save_report


def extract_embeddings(model, texts, tokenizer, max_length=128, batch_size=64, device="cuda"):
    """Extract encoder embeddings for all texts."""
    model.eval()
    model.to(device)
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
        with torch.no_grad():
            embs = model.get_embeddings(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
        all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ScamTrap models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    data_dir = Path(args.data_dir or config.data.output_dir)
    model_dir = Path(args.model_dir or config.training.checkpoint_dir)
    output_dir = Path(args.output_dir or config.evaluation.results_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    # Load model
    model = ScamTrapModel(config)
    ckpt_path = model_dir / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(str(ckpt_path), model)
        print(f"Loaded model from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}, using untrained model")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}
    for name, df in splits.items():
        embeddings[name] = extract_embeddings(
            model, df["text"].tolist(), tokenizer,
            max_length=config.data.max_length, device=device,
        )
        print(f"  {name}: {embeddings[name].shape}")

    all_results = {"model": "scamtrap_supcon"}

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
        print(f"  {frac}: F1={metrics['f1_macro']['mean']:.3f} +/- {metrics['f1_macro']['std']:.3f}")

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
    test_embs = embeddings["test_seen"]
    test_labels = splits["test_seen"]["intent_id"].values
    retrieval_results = evaluate_retrieval(
        test_embs, test_labels, k_values=config.evaluation.retrieval_k,
    )
    all_results["retrieval"] = retrieval_results
    for k, v in retrieval_results.items():
        print(f"  {k}: {v:.3f}")

    # 4. Clustering evaluation
    print("\n--- Clustering Evaluation ---")
    clustering_results = evaluate_clustering(
        test_embs, test_labels, n_runs=config.evaluation.clustering_seeds, seed=config.seed,
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

    # Save — use absolute paths to avoid CWD issues
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_report({"scamtrap_supcon": all_results}, str(output_dir))

    results_path = output_dir / "scamtrap_results.json"
    save_results(all_results, str(results_path))
    print(f"Saved scamtrap_results.json -> {results_path}")
    assert results_path.exists(), f"FAILED to write {results_path}"

    # Save embeddings for visualization
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for name, emb in embeddings.items():
        np.save(str(emb_dir / f"{name}_embeddings.npy"), emb)
        np.save(str(emb_dir / f"{name}_labels.npy"), splits[name]["intent_id"].values)
        print(f"Saved {name} embeddings -> {emb_dir / f'{name}_embeddings.npy'}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
