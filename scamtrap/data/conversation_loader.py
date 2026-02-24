"""Load and parse multi-turn scam conversations."""

import re
from datasets import load_dataset


def load_scam_conversations(dataset_name="BothBosu/multi-agent-scam-conversation"):
    """Load conversation dataset from HuggingFace.

    Returns list of dicts:
        {"dialogue": str, "type": str, "labels": int, "personality": str, "split": str}
    """
    ds = load_dataset(dataset_name)
    conversations = []
    for split_name in ds:
        for item in ds[split_name]:
            conversations.append({
                "dialogue": item["dialogue"],
                "type": item.get("type", "unknown"),
                "labels": item["labels"],  # 0=non-scam, 1=scam
                "personality": item.get("personality", ""),
                "split": split_name,
            })
    return conversations


def parse_turns(dialogue):
    """Parse a dialogue string into individual turns.

    Expected format: "Innocent: Hello. Suspect: Hi, I'm calling from..."

    Returns list of {"speaker": "innocent"|"suspect", "text": str}
    """
    # Split on speaker labels
    pattern = r'(Innocent|Suspect):\s*'
    parts = re.split(pattern, dialogue)

    turns = []
    i = 1  # Skip any text before first speaker label
    while i < len(parts) - 1:
        speaker = parts[i].lower()
        text = parts[i + 1].strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
        i += 2

    return turns
