"""Train all baseline models."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.utils.io import save_results
from scamtrap.baselines.tfidf_logreg import TfIdfLogRegBaseline
from scamtrap.baselines.finetuned_bert import FineTunedBertBaseline
from scamtrap.baselines.sbert_linear import SBERTLinearBaseline
from scamtrap.evaluation.fewshot import evaluate_fewshot
from scamtrap.evaluation.openset import evaluate_openset
from scamtrap.evaluation.retrieval import evaluate_retrieval
from scamtrap.evaluation.clustering import evaluate_clustering


def evaluate_baseline(name, train_embs, train_labels, test_seen_embs,
                      test_seen_labels, test_unseen_embs, test_unseen_labels, config):
    """Run standard evaluation on a baseline's embeddings."""
    results = {"model": name}

    # Few-shot
    results["fewshot"] = evaluate_fewshot(
        train_embs, train_labels, test_seen_embs, test_seen_labels,
        label_fractions=config.evaluation.fewshot_fractions,
        n_trials=config.evaluation.fewshot_trials,
        seed=config.seed,
    )

    # Open-set
    if len(test_unseen_embs) > 0:
        results["openset"] = evaluate_openset(
            train_embs, train_labels,
            test_seen_embs, test_seen_labels,
            test_unseen_embs, test_unseen_labels,
            seed=config.seed,
        )

    # Retrieval
    results["retrieval"] = evaluate_retrieval(
        test_seen_embs, test_seen_labels,
        k_values=config.evaluation.retrieval_k,
    )

    # Clustering
    results["clustering"] = evaluate_clustering(
        test_seen_embs, test_seen_labels,
        n_runs=config.evaluation.clustering_seeds,
        seed=config.seed,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/baselines")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    data_dir = Path(args.data_dir or config.data.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    splits = {}
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = pd.read_parquet(path)

    all_baseline_results = {}

    # --- TF-IDF + LogReg ---
    print("=" * 50)
    print("Training TF-IDF + LogReg baseline...")
    tfidf = TfIdfLogRegBaseline(
        max_features=config.baselines.tfidf.max_features,
        ngram_range=tuple(config.baselines.tfidf.ngram_range),
        seed=config.seed,
    )
    tfidf.fit(
        splits["train"]["text"].tolist(),
        splits["train"]["intent_id"].values,
    )

    tfidf_embs = {
        name: tfidf.get_embeddings(splits[name]["text"].tolist())
        for name in splits
    }

    results = evaluate_baseline(
        "tfidf_logreg",
        tfidf_embs["train"], splits["train"]["intent_id"].values,
        tfidf_embs["test_seen"], splits["test_seen"]["intent_id"].values,
        tfidf_embs.get("test_unseen", np.array([])),
        splits.get("test_unseen", pd.DataFrame()).get("intent_id", pd.Series(dtype=int)).values,
        config,
    )
    all_baseline_results["tfidf_logreg"] = results
    print(f"  Few-shot 100%: F1={results['fewshot']['1.0']['f1_macro']['mean']:.3f}")

    # --- Fine-tuned BERT ---
    print("=" * 50)
    print("Training Fine-tuned DistilBERT baseline...")
    num_classes = len(splits["train"]["intent_id"].unique())
    bert = FineTunedBertBaseline(
        model_name=config.baselines.finetuned_bert.encoder_name,
        num_classes=num_classes,
        seed=config.seed,
    )
    bert.fit(
        splits["train"]["text"].tolist(),
        splits["train"]["intent_id"].values,
        epochs=config.baselines.finetuned_bert.epochs,
        lr=config.baselines.finetuned_bert.lr,
    )

    bert_embs = {
        name: bert.extract_all_embeddings(splits[name]["text"].tolist())
        for name in splits
    }

    results = evaluate_baseline(
        "finetuned_bert",
        bert_embs["train"], splits["train"]["intent_id"].values,
        bert_embs["test_seen"], splits["test_seen"]["intent_id"].values,
        bert_embs.get("test_unseen", np.array([])),
        splits.get("test_unseen", pd.DataFrame()).get("intent_id", pd.Series(dtype=int)).values,
        config,
    )
    all_baseline_results["finetuned_bert"] = results
    print(f"  Few-shot 100%: F1={results['fewshot']['1.0']['f1_macro']['mean']:.3f}")

    # --- SBERT + Linear ---
    print("=" * 50)
    print("Training SBERT + Linear baseline...")
    sbert = SBERTLinearBaseline(
        model_name=config.baselines.sbert.model_name,
        seed=config.seed,
    )
    sbert.fit(
        splits["train"]["text"].tolist(),
        splits["train"]["intent_id"].values,
    )

    sbert_embs = {
        name: sbert.get_embeddings(splits[name]["text"].tolist())
        for name in splits
    }

    results = evaluate_baseline(
        "sbert_linear",
        sbert_embs["train"], splits["train"]["intent_id"].values,
        sbert_embs["test_seen"], splits["test_seen"]["intent_id"].values,
        sbert_embs.get("test_unseen", np.array([])),
        splits.get("test_unseen", pd.DataFrame()).get("intent_id", pd.Series(dtype=int)).values,
        config,
    )
    all_baseline_results["sbert_linear"] = results
    print(f"  Few-shot 100%: F1={results['fewshot']['1.0']['f1_macro']['mean']:.3f}")

    # Save all baseline results
    save_results(all_baseline_results, str(output_dir / "baseline_results.json"))

    # Save embeddings
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for method, embs_dict in [("tfidf", tfidf_embs), ("bert", bert_embs), ("sbert", sbert_embs)]:
        for split_name, embs in embs_dict.items():
            if isinstance(embs, np.ndarray) and len(embs) > 0:
                np.save(str(emb_dir / f"{method}_{split_name}.npy"), embs)

    print(f"\nAll baseline results saved to {output_dir}")


if __name__ == "__main__":
    main()
