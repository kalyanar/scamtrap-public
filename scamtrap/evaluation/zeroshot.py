"""Zero-shot intent classification via nearest prototype."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer


def evaluate_zeroshot(
    model, test_texts, test_labels, all_intent_descriptions,
    intent_to_id, tokenizer_name, max_length=128, device="cuda",
):
    """Zero-shot classification: classify messages by nearest description.

    Args:
        model: CLIPScamModel
        test_texts: list of test messages
        test_labels: numpy array of intent_id labels
        all_intent_descriptions: dict with ALL 9 intents (including holdout)
        intent_to_id: dict mapping intent name -> id (must include holdout)
        tokenizer_name: HuggingFace tokenizer name
        max_length: max token length
        device: torch device string

    Returns:
        dict with accuracy, f1_macro, f1_weighted, classification_report
    """
    model.eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 1. Encode all intent descriptions (including holdout)
    K = len(all_intent_descriptions)
    id_to_intent = {v: k for k, v in intent_to_id.items()}

    # Build ordered description list by intent_id
    ordered_descs = [""] * K
    for name, desc in all_intent_descriptions.items():
        if name in intent_to_id:
            ordered_descs[intent_to_id[name]] = desc

    desc_enc = tokenizer(
        ordered_descs, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        desc_z = model.encode_descriptions(
            desc_enc["input_ids"].to(device),
            desc_enc["attention_mask"].to(device),
        )  # [K, proj_dim]

    # 2. Encode test messages in batches, classify by nearest prototype
    all_preds = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            _, z_msg = model.encode_messages(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )  # [B, proj_dim]

            # Cosine similarity to all prototypes
            sims = torch.matmul(z_msg, desc_z.T)  # [B, K]
            preds = sims.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    # Determine which intent ids are actually present in test_labels
    present_ids = sorted(set(int(l) for l in test_labels))
    target_names = [id_to_intent.get(i, f"intent_{i}") for i in present_ids]

    return {
        "accuracy": float(accuracy_score(test_labels, all_preds)),
        "f1_macro": float(f1_score(
            test_labels, all_preds, average="macro", zero_division=0,
        )),
        "f1_weighted": float(f1_score(
            test_labels, all_preds, average="weighted", zero_division=0,
        )),
        "classification_report": classification_report(
            test_labels, all_preds,
            labels=present_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }
