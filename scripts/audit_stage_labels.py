"""Audit stage labels for deterministic artifacts in MASC dataset.

This script addresses reviewer concern about Markov chain's perfect
AUROC=1.0, which suggests deterministic escalation patterns. We test:

1. Position-stage correlation: are stages trivially recoverable from
   turn position alone?
2. Keyword-only vs position-only labeling agreement rate
3. Mutual information between position and stage label
4. Escalation label leakage: does the presence of any escalation-stage
   turn perfectly predict the conversation-level escalation label?
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    accuracy_score,
    classification_report,
)

from scamtrap.utils.config import load_config


def load_conversation_metadata(config):
    """Load conversation metadata with stage labels."""
    conv_dir = Path(config.data.output_dir) / "conversations"
    meta_path = conv_dir / "metadata.json"

    if not meta_path.exists():
        print(f"No conversation metadata found at {meta_path}")
        print("Run prepare_conversations.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    return meta["trajectories"]


def analyze_position_correlation(trajectories):
    """Test if stage labels are trivially recoverable from turn position."""
    print("\n" + "=" * 60)
    print("1. POSITION-STAGE CORRELATION")
    print("=" * 60)

    positions = []
    stages = []
    total_turns = []

    for traj in trajectories:
        n_turns = len(traj["stages"])
        total_turns.append(n_turns)
        for i, stage in enumerate(traj["stages"]):
            positions.append(i)
            stages.append(stage)

    positions = np.array(positions)
    stages = np.array(stages)

    # Pearson correlation
    corr = np.corrcoef(positions, stages)[0, 1]
    print(f"  Pearson correlation (position, stage): {corr:.4f}")

    # NMI between position bins and stage labels
    # Bin positions into 6 equal bins
    pos_bins = np.digitize(positions, bins=np.linspace(0, max(positions), 7)[1:])
    nmi = normalized_mutual_info_score(stages, pos_bins)
    mi = mutual_info_score(stages, pos_bins)
    print(f"  MI(position_bins, stage): {mi:.4f}")
    print(f"  NMI(position_bins, stage): {nmi:.4f}")

    # Per-position stage distribution
    print("\n  Position → most common stage:")
    for pos in range(min(10, max(positions) + 1)):
        mask = positions == pos
        if mask.sum() > 0:
            stage_counts = Counter(stages[mask])
            most_common = stage_counts.most_common(1)[0]
            total = mask.sum()
            print(f"    Position {pos}: stage {most_common[0]} "
                  f"({most_common[1]}/{total} = {most_common[1]/total:.1%})")

    # Can a simple position-based rule recover stages?
    # Predict stage = position (clipped to [0, 5])
    position_pred = np.clip(positions, 0, 5)
    pos_acc = accuracy_score(stages, position_pred)
    print(f"\n  Position-only prediction accuracy (stage=clip(pos, 0, 5)): {pos_acc:.4f}")

    return corr, nmi


def analyze_escalation_leakage(trajectories):
    """Test if escalation is trivially recoverable."""
    print("\n" + "=" * 60)
    print("2. ESCALATION LABEL LEAKAGE")
    print("=" * 60)

    escalation_stage = 5  # Stage index for "escalation"

    has_esc_stage = []
    is_escalating = []

    for traj in trajectories:
        stages = traj["stages"]
        has_any_esc = any(s >= 4 for s in stages)  # payment_attempt or escalation
        has_esc_stage.append(has_any_esc)
        is_escalating.append(traj.get("is_scam", True))

    has_esc_stage = np.array(has_esc_stage)
    is_escalating = np.array(is_escalating)

    # Perfect prediction check
    agreement = (has_esc_stage == is_escalating).mean()
    print(f"  Agreement between 'has stage>=4' and 'is_scam' label: {agreement:.4f}")

    # Count conversations with escalation stages
    n_with_esc = has_esc_stage.sum()
    n_total = len(trajectories)
    print(f"  Conversations with stage >= 4: {n_with_esc}/{n_total} ({n_with_esc/n_total:.1%})")

    # Stage transition patterns in escalating vs non-escalating
    print("\n  Stage transition patterns:")
    for label, name in [(True, "Escalating"), (False, "Non-escalating")]:
        trajs = [t for t, e in zip(trajectories, is_escalating) if e == label]
        if not trajs:
            continue
        # Most common transition sequences (first 3 stages)
        prefixes = []
        for t in trajs:
            stages = t["stages"][:4]
            prefixes.append(tuple(stages))
        prefix_counts = Counter(prefixes)
        print(f"\n  {name} conversations ({len(trajs)} total):")
        for prefix, count in prefix_counts.most_common(5):
            print(f"    {' → '.join(str(s) for s in prefix)}: "
                  f"{count} ({count/len(trajs):.1%})")

    return agreement


def analyze_stage_monotonicity(trajectories):
    """Check if stage labels are monotonically increasing."""
    print("\n" + "=" * 60)
    print("3. STAGE MONOTONICITY")
    print("=" * 60)

    monotonic = 0
    non_monotonic = 0
    reversals = []

    for traj in trajectories:
        stages = traj["stages"]
        is_mono = all(stages[i] <= stages[i+1] for i in range(len(stages)-1))
        if is_mono:
            monotonic += 1
        else:
            non_monotonic += 1
            for i in range(len(stages)-1):
                if stages[i] > stages[i+1]:
                    reversals.append((stages[i], stages[i+1]))

    total = monotonic + non_monotonic
    print(f"  Monotonically increasing: {monotonic}/{total} ({monotonic/total:.1%})")
    print(f"  Non-monotonic: {non_monotonic}/{total} ({non_monotonic/total:.1%})")

    if reversals:
        reversal_counts = Counter(reversals)
        print("\n  Most common reversals (from → to):")
        for (s_from, s_to), count in reversal_counts.most_common(5):
            print(f"    Stage {s_from} → {s_to}: {count}")


def analyze_stage_distribution(trajectories):
    """Report overall stage label distribution."""
    print("\n" + "=" * 60)
    print("4. STAGE DISTRIBUTION")
    print("=" * 60)

    stage_names = [
        "hook", "trust_building", "urgency",
        "info_request", "payment_attempt", "escalation",
    ]

    all_stages = []
    for traj in trajectories:
        all_stages.extend(traj["stages"])

    stage_counts = Counter(all_stages)
    total = len(all_stages)
    for stage_idx in sorted(stage_counts.keys()):
        name = stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage_{stage_idx}"
        count = stage_counts[stage_idx]
        print(f"  {name} (stage {stage_idx}): {count} ({count/total:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Audit stage labels for artifacts")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    trajectories = load_conversation_metadata(config)

    print(f"Loaded {len(trajectories)} trajectories")

    # Run all analyses
    corr, nmi = analyze_position_correlation(trajectories)
    agreement = analyze_escalation_leakage(trajectories)
    analyze_stage_monotonicity(trajectories)
    analyze_stage_distribution(trajectories)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Position-stage correlation: {corr:.4f}")
    print(f"  Position-stage NMI: {nmi:.4f}")
    print(f"  Escalation leakage agreement: {agreement:.4f}")
    if agreement > 0.99:
        print("  WARNING: Escalation labels are near-perfectly recoverable")
        print("  from stage labels. Markov AUROC=1.0 is a data artifact.")
    if corr > 0.8:
        print("  WARNING: Stages are highly correlated with position.")
        print("  Stage labels may encode positional information.")

    # Save results
    results = {
        "position_stage_correlation": float(corr),
        "position_stage_nmi": float(nmi),
        "escalation_leakage_agreement": float(agreement),
        "n_trajectories": len(trajectories),
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stage_label_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'stage_label_audit.json'}")


if __name__ == "__main__":
    main()
