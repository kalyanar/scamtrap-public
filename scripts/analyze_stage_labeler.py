"""Analyze stage labeler: position vs keyword contribution ablation."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.data.stage_labeler import ScamStageLabeler, STAGE_NAMES


def load_conversations(config):
    """Load conversation data from processed output."""
    conv_dir = Path(config.data.output_dir) / "conversations"
    meta_path = conv_dir / "metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    return meta.get("trajectories", [])


def build_mock_turns(trajectory):
    """Build mock turn dicts from trajectory metadata.

    The trajectory metadata stores stage labels but may not store raw text.
    If 'turns' field exists, use it; otherwise create position-only stubs.
    """
    if "turns" in trajectory and trajectory["turns"]:
        return trajectory["turns"]

    # If no text available, create dummy turns with empty text
    # (useful only for position-only mode analysis)
    n = len(trajectory["stages"])
    return [{"text": ""} for _ in range(n)]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stage labeler: position vs keyword",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(
        args.output_dir or config.evaluation.results_dir,
    ) / "stage_labeler_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load conversations
    conversations = load_conversations(config)
    if not conversations:
        print("No conversation data found. Generating synthetic conversations "
              "for analysis...")
        conversations = generate_synthetic_conversations(config)

    labeler = ScamStageLabeler()
    modes = ["hybrid", "keyword_only", "position_only"]

    # Label all conversations under each mode
    all_labels = {mode: [] for mode in modes}
    n_conversations = 0
    n_turns = 0

    for conv in conversations:
        if not conv.get("is_scam", True):
            continue

        turns = build_mock_turns(conv)
        if not turns:
            continue

        n_conversations += 1
        n_turns += len(turns)

        for mode in modes:
            labels = labeler.label_turns(turns, is_scam=True, mode=mode)
            all_labels[mode].append(labels)

    print(f"Analyzed {n_conversations} scam conversations ({n_turns} turns)")

    if n_conversations == 0:
        print("No scam conversations found. Exiting.")
        return

    # Compare modes
    hybrid_flat = [l for conv in all_labels["hybrid"] for l in conv]
    keyword_flat = [l for conv in all_labels["keyword_only"] for l in conv]
    position_flat = [l for conv in all_labels["position_only"] for l in conv]

    hybrid_arr = np.array(hybrid_flat)
    keyword_arr = np.array(keyword_flat)
    position_arr = np.array(position_flat)

    # Agreement rates
    kw_agreement = float(np.mean(hybrid_arr == keyword_arr))
    pos_agreement = float(np.mean(hybrid_arr == position_arr))
    kw_pos_agreement = float(np.mean(keyword_arr == position_arr))

    print(f"\nAgreement rates:")
    print(f"  Hybrid vs Keyword-only: {kw_agreement:.1%}")
    print(f"  Hybrid vs Position-only: {pos_agreement:.1%}")
    print(f"  Keyword-only vs Position-only: {kw_pos_agreement:.1%}")

    # Per-stage analysis
    per_stage = {}
    for stage_id, name in enumerate(STAGE_NAMES):
        hybrid_mask = hybrid_arr == stage_id
        n_hybrid = int(hybrid_mask.sum())
        if n_hybrid == 0:
            per_stage[name] = {
                "count": 0,
                "keyword_agreement": 0.0,
                "position_agreement": 0.0,
            }
            continue

        kw_agrees = int((keyword_arr[hybrid_mask] == stage_id).sum())
        pos_agrees = int((position_arr[hybrid_mask] == stage_id).sum())

        per_stage[name] = {
            "count": n_hybrid,
            "keyword_agreement": kw_agrees / max(n_hybrid, 1),
            "position_agreement": pos_agrees / max(n_hybrid, 1),
            "primarily_keyword": kw_agrees > pos_agrees,
        }

    print(f"\nPer-stage agreement with hybrid labels:")
    print(f"  {'Stage':<20} {'N':>6} {'KW Agree':>10} {'Pos Agree':>10} {'Primary':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for name, info in per_stage.items():
        primary = "keyword" if info.get("primarily_keyword", False) else "position"
        print(f"  {name:<20} {info['count']:>6} "
              f"{info['keyword_agreement']:>9.1%} "
              f"{info['position_agreement']:>9.1%} "
              f"{primary:>10}")

    # Confusion matrix: hybrid vs keyword-only
    n_stages = len(STAGE_NAMES)
    confusion_kw = np.zeros((n_stages, n_stages), dtype=int)
    confusion_pos = np.zeros((n_stages, n_stages), dtype=int)
    for h, k, p in zip(hybrid_flat, keyword_flat, position_flat):
        if 0 <= h < n_stages and 0 <= k < n_stages:
            confusion_kw[h, k] += 1
        if 0 <= h < n_stages and 0 <= p < n_stages:
            confusion_pos[h, p] += 1

    # Fraction of labels determined primarily by position
    # (where keyword-only disagrees but position-only agrees)
    pos_determines = int(np.sum(
        (keyword_arr != hybrid_arr) & (position_arr == hybrid_arr)
    ))
    kw_determines = int(np.sum(
        (keyword_arr == hybrid_arr) & (position_arr != hybrid_arr)
    ))
    both_agree = int(np.sum(
        (keyword_arr == hybrid_arr) & (position_arr == hybrid_arr)
    ))
    neither_agree = int(np.sum(
        (keyword_arr != hybrid_arr) & (position_arr != hybrid_arr)
    ))
    total = len(hybrid_flat)

    print(f"\nLabel determination breakdown:")
    print(f"  Keyword primary:  {kw_determines:>6} ({kw_determines/max(total,1):.1%})")
    print(f"  Position primary: {pos_determines:>6} ({pos_determines/max(total,1):.1%})")
    print(f"  Both agree:       {both_agree:>6} ({both_agree/max(total,1):.1%})")
    print(f"  Neither agrees:   {neither_agree:>6} ({neither_agree/max(total,1):.1%})")

    results = {
        "n_conversations": n_conversations,
        "n_turns": n_turns,
        "agreement": {
            "hybrid_vs_keyword": kw_agreement,
            "hybrid_vs_position": pos_agreement,
            "keyword_vs_position": kw_pos_agreement,
        },
        "determination": {
            "keyword_primary": kw_determines,
            "keyword_primary_rate": kw_determines / max(total, 1),
            "position_primary": pos_determines,
            "position_primary_rate": pos_determines / max(total, 1),
            "both_agree": both_agree,
            "both_agree_rate": both_agree / max(total, 1),
            "neither_agrees": neither_agree,
            "neither_agrees_rate": neither_agree / max(total, 1),
        },
        "per_stage": per_stage,
        "confusion_hybrid_vs_keyword": confusion_kw.tolist(),
        "confusion_hybrid_vs_position": confusion_pos.tolist(),
    }

    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis -> {results_path}")


def generate_synthetic_conversations(config, n_conversations=100):
    """Generate synthetic conversations for analysis when real data is missing."""
    import random
    random.seed(config.seed)

    scam_phrases = {
        0: ["hello", "hi there", "this is calling from", "good morning", "my name is"],
        1: ["our records show", "for your protection", "we detected", "official notice",
            "don't worry", "trust me", "government agency"],
        2: ["immediately", "right now", "urgent", "suspended", "limited time",
            "act now", "within 24 hours", "last chance"],
        3: ["social security", "verify your account", "need your password",
            "provide your address", "confirm your credit card", "date of birth"],
        4: ["transfer the payment", "gift card", "send money via wire",
            "bitcoin wallet", "processing fee", "pay with zelle"],
        5: ["legal action", "you will be arrested", "final warning",
            "warrant for your arrest", "consequences", "goodbye"],
    }

    conversations = []
    for _ in range(n_conversations):
        n_turns = random.randint(5, 20)
        turns = []
        for t in range(n_turns):
            # Pick stage roughly by position (with noise)
            expected_stage = min(5, int(t / max(n_turns - 1, 1) * 5.5))
            stage = max(0, min(5, expected_stage + random.randint(-1, 1)))
            phrases = scam_phrases[stage]
            text = random.choice(phrases) + " " + f"turn {t} filler text"
            turns.append({"text": text})

        conversations.append({
            "turns": turns,
            "stages": list(range(min(6, n_turns))),  # placeholder
            "is_scam": True,
        })

    return conversations


if __name__ == "__main__":
    main()
