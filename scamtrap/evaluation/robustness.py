"""Robustness evaluation under text obfuscation."""

import numpy as np
import torch
from tqdm import tqdm

from scamtrap.data.augmentations import ScamAugmenter


def evaluate_robustness(
    model,
    texts: list[str],
    labels: np.ndarray,
    tokenizer,
    max_length: int = 128,
    n_augmentations: int = 5,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Measure embedding stability under augmentation.

    For each text, generate augmented versions, compute cosine similarity
    between original and augmented embeddings.
    """
    augmenter = ScamAugmenter(seed=seed)
    model.eval()
    model.to(device)

    # Map strategy names to actual method names on ScamAugmenter
    strategy_methods = {
        'homoglyph': 'homoglyph_substitute',
        'leetspeak': 'leetspeak',
        'random_spacing': 'random_spacing',
        'synonym_swap': 'synonym_swap',
    }
    cosine_sims = {s: [] for s in strategy_methods}
    overall_sims = []

    with torch.no_grad():
        for text in tqdm(texts[:500], desc="Robustness eval"):  # Cap at 500 for speed
            # Original embedding
            enc = tokenizer(text, max_length=max_length, padding="max_length",
                           truncation=True, return_tensors="pt")
            orig_emb = model.get_embeddings(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            ).cpu().numpy().flatten()

            for strategy in cosine_sims.keys():
                aug_func = getattr(augmenter, strategy_methods[strategy])
                aug_text = aug_func(text)

                enc_aug = tokenizer(aug_text, max_length=max_length, padding="max_length",
                                   truncation=True, return_tensors="pt")
                aug_emb = model.get_embeddings(
                    enc_aug["input_ids"].to(device),
                    enc_aug["attention_mask"].to(device),
                ).cpu().numpy().flatten()

                # Cosine similarity
                sim = np.dot(orig_emb, aug_emb) / (
                    np.linalg.norm(orig_emb) * np.linalg.norm(aug_emb) + 1e-9
                )
                cosine_sims[strategy].append(sim)
                overall_sims.append(sim)

    results = {
        "overall_cosine_sim": {
            "mean": float(np.mean(overall_sims)),
            "std": float(np.std(overall_sims)),
        }
    }
    for strategy, sims in cosine_sims.items():
        results[f"{strategy}_cosine_sim"] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
        }

    return results
