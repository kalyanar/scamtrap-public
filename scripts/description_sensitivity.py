"""Sensitivity analysis for intent description wording.

Tests how much Stage B results depend on the specific wording of
intent descriptions by training with 3 different paraphrase variants
and reporting variance in key metrics.

Addresses reviewer question: "How sensitive are Stage B results to
the choice and quality of intent descriptions?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scamtrap.utils.config import load_config
from scamtrap.data.intent_description_variants import (
    DESCRIPTION_VARIANTS,
    VARIANT_NAMES,
    get_descriptions_for_variant,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sensitivity to intent description wording"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode", choices=["analyze", "compare"],
        default="analyze",
        help="'analyze' prints description variants, "
             "'compare' compares pre-computed results across variants"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "analyze":
        print("=" * 70)
        print("INTENT DESCRIPTION VARIANTS")
        print("=" * 70)

        for intent in sorted(DESCRIPTION_VARIANTS.keys()):
            print(f"\n{'='*50}")
            print(f"Intent: {intent}")
            print(f"{'='*50}")
            for variant in VARIANT_NAMES:
                desc = DESCRIPTION_VARIANTS[intent][variant]
                words = len(desc.split())
                print(f"\n  [{variant}] ({words} words):")
                print(f"    {desc}")

        # Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for variant in VARIANT_NAMES:
            descs = get_descriptions_for_variant(variant)
            word_counts = [len(d.split()) for d in descs.values()]
            print(f"  {variant}: avg {np.mean(word_counts):.0f} words "
                  f"(range {min(word_counts)}-{max(word_counts)})")

        print(f"\nTo run training with a specific variant:")
        print(f"  python scripts/train_clip.py --config {args.config} "
              f"--description-variant v2")

    elif args.mode == "compare":
        # Compare pre-computed results across variants
        results_dir = Path(config.evaluation.results_dir)
        all_results = {}

        for variant in VARIANT_NAMES:
            result_file = results_dir / f"clip_results_{variant}.json"
            if result_file.exists():
                with open(result_file) as f:
                    all_results[variant] = json.load(f)
            else:
                print(f"  Missing results for {variant}: {result_file}")

        if len(all_results) < 2:
            print("Need at least 2 variants with results to compare.")
            print("Run training with each variant first:")
            for v in VARIANT_NAMES:
                print(f"  python scripts/train_clip.py --config {args.config} "
                      f"--description-variant {v}")
            return

        # Compare key metrics
        metrics = [
            "fewshot_f1_1pct", "fewshot_f1_100pct",
            "clustering_nmi", "novelty_auroc",
            "zeroshot_seen_accuracy", "zeroshot_unseen_accuracy",
        ]

        print(f"\n{'='*70}")
        print("DESCRIPTION SENSITIVITY RESULTS")
        print(f"{'='*70}")
        print(f"\n{'Metric':<30} " + " ".join(f"{v:>10}" for v in all_results.keys())
              + f" {'Std':>10}")

        all_stds = []
        for metric in metrics:
            values = []
            row = f"{metric:<30}"
            for variant, results in all_results.items():
                val = results.get(metric, None)
                if val is not None:
                    values.append(val)
                    row += f" {val:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
            if values:
                std = np.std(values)
                row += f" {std:>10.4f}"
                all_stds.append(std)
            print(row)

        if all_stds:
            max_std = max(all_stds)
            print(f"\nOverall observation: "
                  f"{'HIGH' if max_std > 0.05 else 'LOW'} sensitivity "
                  f"to description wording (max std = {max_std:.4f})")


if __name__ == "__main__":
    main()
