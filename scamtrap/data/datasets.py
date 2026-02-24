"""Load and merge HuggingFace scam/phishing datasets."""

import re
import pandas as pd
from datasets import load_dataset


def preprocess_text(text: str) -> str:
    """Normalize text: collapse whitespace, replace URLs/phones with tokens."""
    if not isinstance(text, str):
        return ""
    # Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # Replace phone numbers with token
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', '[PHONE]', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_sms_spam() -> pd.DataFrame:
    """Load ucirvine/sms_spam dataset."""
    ds = load_dataset("ucirvine/sms_spam", split="train")
    df = ds.to_pandas()
    df = df.rename(columns={"sms": "text", "label": "binary_label"})
    df["source"] = "sms_spam"
    return df[["text", "binary_label", "source"]]


def load_phishing_texts() -> pd.DataFrame:
    """Load ealvaradob/phishing-dataset texts.json directly from HuggingFace."""
    url = (
        "https://huggingface.co/datasets/ealvaradob/phishing-dataset"
        "/resolve/main/texts.json"
    )
    print(f"  Downloading texts.json from HuggingFace...")
    df = pd.read_json(url)
    df = df.rename(columns={"label": "binary_label"})
    if "text" not in df.columns:
        for col in ["email_text", "message", "content", "body"]:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break
    df["source"] = "phishing"
    return df[["text", "binary_label", "source"]]


def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge, deduplicate, and preprocess datasets."""
    merged = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing text
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].str.strip().str.len() > 0]

    # Preprocess
    merged["text"] = merged["text"].apply(preprocess_text)

    # Drop very short texts (likely noise)
    merged = merged[merged["text"].str.len() >= 10]

    # Deduplicate on text
    merged = merged.drop_duplicates(subset=["text"], keep="first")

    # Ensure binary_label is int
    merged["binary_label"] = merged["binary_label"].astype(int)

    return merged.reset_index(drop=True)


def load_and_merge(config) -> pd.DataFrame:
    """Full pipeline: load all datasets, merge, preprocess."""
    dfs = []
    print("Loading ucirvine/sms_spam...")
    dfs.append(load_sms_spam())
    print(f"  -> {len(dfs[-1])} samples")

    print("Loading ealvaradob/phishing-dataset...")
    dfs.append(load_phishing_texts())
    print(f"  -> {len(dfs[-1])} samples")

    print("Merging and preprocessing...")
    merged = merge_datasets(dfs)
    print(f"  -> {len(merged)} samples after dedup/filter")
    print(f"  -> ham: {(merged['binary_label'] == 0).sum()}, "
          f"scam: {(merged['binary_label'] == 1).sum()}")

    return merged
