"""Audit weak supervision label quality — driver script."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scamtrap.utils.config import load_config
from scamtrap.evaluation.audit_labels import audit_label_quality


def main():
    parser = argparse.ArgumentParser(description="Audit weak label quality")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = Path(config.data.output_dir)
    output_dir = Path(args.output_dir or config.evaluation.results_dir) / "label_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all splits and combine for full audit
    all_dfs = []
    for name in ["train", "val", "test_seen", "test_unseen"]:
        path = data_dir / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["split"] = name
            all_dfs.append(df)

    if not all_dfs:
        print("ERROR: No data found. Run prepare_data.py first.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} total samples across {len(all_dfs)} splits")

    # Run audit
    print("\n--- Label Quality Audit ---")
    results = audit_label_quality(full_df)

    # Print summary
    overall = results["overall"]
    print(f"\nOverall ({overall['n_scam_samples']} scam samples):")
    print(f"  Coverage rate (non-generic): {overall['coverage_rate']:.1%}")
    print(f"  Generic fallback rate:       {overall['generic_fallback_rate']:.1%}")
    print(f"  Multi-match rate:            {overall['multi_match_rate']:.1%}")
    print(f"  Ambiguity rate:              {overall['ambiguity_rate']:.1%}")
    print(f"  Mean confidence ratio:       {overall['mean_confidence']:.2f}")

    print("\nPer-intent breakdown:")
    print(f"  {'Intent':<20} {'N':>6} {'Frac':>6} {'Conf':>6} {'Ambig%':>7} {'Top Confusion'}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*20}")
    for intent, info in results["per_intent"].items():
        top_conf = info["top_confusions"][0]["intent"] if info["top_confusions"] else "-"
        print(f"  {intent:<20} {info['count']:>6} "
              f"{info['fraction']:>5.1%} "
              f"{info['mean_confidence']:>6.2f} "
              f"{info['ambiguity_rate']:>6.1%} "
              f"{top_conf}")

    # Save
    results_path = output_dir / "audit_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved audit results -> {results_path}")


if __name__ == "__main__":
    main()
