"""Prepare conversation data for Stage C world model training.

1. Download BothBosu/multi-agent-scam-conversation from HuggingFace
2. Parse dialogues into turns
3. Label each turn with scam stage (0-5)
4. Load frozen encoder (Stage B or A checkpoint)
5. Encode every turn -> 768d embedding
6. Save trajectories + embeddings to data/processed/conversations/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.data.conversation_loader import load_scam_conversations, parse_turns
from scamtrap.data.stage_labeler import ScamStageLabeler


def load_encoder(config):
    """Load the best available encoder (Stage B > Stage A > pretrained)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try Stage B checkpoint first
    clip_ckpt = Path(config.stage_b.checkpoint_dir) / "best_model.pt"
    if clip_ckpt.exists():
        from scamtrap.models.clip_model import CLIPScamModel
        model = CLIPScamModel(config)
        ckpt = torch.load(str(clip_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Stage B encoder from {clip_ckpt}")
        model.eval().to(device)
        return model, device

    # Try Stage A checkpoint
    stage_a_ckpt = Path(config.training.checkpoint_dir) / "best_model.pt"
    if stage_a_ckpt.exists():
        from scamtrap.models.scamtrap_model import ScamTrapModel
        model = ScamTrapModel(config)
        ckpt = torch.load(str(stage_a_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Stage A encoder from {stage_a_ckpt}")
        model.eval().to(device)
        return model, device

    # Fallback: pretrained encoder only
    from scamtrap.models.encoder import TextEncoder
    model = TextEncoder(
        model_name=config.model.encoder_name,
        pooling=config.model.pooling,
    )
    print("WARNING: No trained checkpoint found, using pretrained encoder")

    # Wrap in a simple object with get_embeddings method
    class EncoderWrapper:
        def __init__(self, encoder, dev):
            self.encoder = encoder.eval().to(dev)
            self._device = dev

        def get_embeddings(self, input_ids, attention_mask):
            return self.encoder(input_ids, attention_mask)

        def eval(self):
            return self

        def to(self, device):
            self.encoder.to(device)
            return self

    return EncoderWrapper(model, device), device


def encode_turns(model, turn_texts, tokenizer, max_length, device,
                 batch_size=64):
    """Encode a list of turn texts into embeddings."""
    all_embs = []

    for i in range(0, len(turn_texts), batch_size):
        batch = turn_texts[i:i + batch_size]
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
        description="Prepare conversation data for Stage C",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # 1. Load conversations
    print("Loading conversations from HuggingFace...")
    conversations = load_scam_conversations()
    print(f"  Loaded {len(conversations)} conversations")

    # 2. Parse turns and label stages
    print("Parsing turns and labeling stages...")
    labeler = ScamStageLabeler()
    parsed = []

    for conv in tqdm(conversations, desc="Parsing"):
        turns = parse_turns(conv["dialogue"])
        if len(turns) < 2:
            continue

        is_scam = conv["labels"] == 1
        stages = labeler.label_turns(turns, is_scam=is_scam)

        parsed.append({
            "turns": turns,
            "stages": stages,
            "is_scam": is_scam,
            "type": conv["type"],
            "hf_split": conv["split"],
        })

    print(f"  Parsed {len(parsed)} conversations "
          f"({sum(1 for p in parsed if p['is_scam'])} scam, "
          f"{sum(1 for p in parsed if not p['is_scam'])} non-scam)")

    # 3. Load encoder
    print("Loading encoder...")
    model, device = load_encoder(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # 4. Encode turns
    print("Encoding turns...")
    trajectories = []

    for item in tqdm(parsed, desc="Encoding"):
        turn_texts = [t["text"] for t in item["turns"]]
        embs = encode_turns(
            model, turn_texts, tokenizer,
            max_length=config.data.max_length, device=device,
        )

        trajectories.append({
            "embeddings": embs,           # [T, 768]
            "stages": item["stages"],     # [T]
            "is_scam": item["is_scam"],
            "type": item["type"],
            "hf_split": item["hf_split"],
            "num_turns": len(item["turns"]),
        })

    # 5. Split: HF train -> our train + val, HF test -> our test
    # Use 80/20 split of HF train for train/val
    hf_train = [t for t in trajectories if t["hf_split"] == "train"]
    hf_test = [t for t in trajectories if t["hf_split"] != "train"]

    np.random.seed(config.seed)
    indices = np.random.permutation(len(hf_train))
    split_idx = int(len(hf_train) * 0.8)

    for i in indices[:split_idx]:
        hf_train[i]["split"] = "train"
    for i in indices[split_idx:]:
        hf_train[i]["split"] = "val"
    for t in hf_test:
        t["split"] = "test"

    all_trajectories = hf_train + hf_test

    # 6. Save
    output_dir = Path(config.data.output_dir) / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectories (embeddings as .npz, metadata as .json)
    embeddings_list = []
    metadata_list = []

    for i, traj in enumerate(all_trajectories):
        embeddings_list.append(traj["embeddings"])
        metadata_list.append({
            "stages": traj["stages"],
            "is_scam": traj["is_scam"],
            "type": traj["type"],
            "split": traj["split"],
            "num_turns": traj["num_turns"],
        })

    np.savez_compressed(
        str(output_dir / "embeddings.npz"),
        *embeddings_list,
    )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "num_trajectories": len(all_trajectories),
            "trajectories": metadata_list,
            "splits": {
                "train": sum(1 for t in all_trajectories if t["split"] == "train"),
                "val": sum(1 for t in all_trajectories if t["split"] == "val"),
                "test": sum(1 for t in all_trajectories if t["split"] == "test"),
            },
        }, f, indent=2)

    print(f"\nSaved {len(all_trajectories)} trajectories to {output_dir}")
    split_counts = {}
    for t in all_trajectories:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1
    for split, count in split_counts.items():
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
