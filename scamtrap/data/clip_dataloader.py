"""Dataset and dataloaders for CLIP-style intent alignment training."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from scamtrap.data.augmentations import ScamAugmenter


class CLIPScamDataset(Dataset):
    """Simple dataset for message classification against prototypes.

    Each item returns tokenized message + intent_id label.
    No multi-view -- augmentation is optional (applied once if enabled).
    """

    def __init__(self, texts, intent_ids, tokenizer_name, max_length,
                 augmenter=None):
        self.texts = texts
        self.intent_ids = intent_ids
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.augmenter = augmenter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augmenter is not None:
            text = self.augmenter.apply_random(text)

        enc = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_id": torch.tensor(self.intent_ids[idx], dtype=torch.long),
        }


def create_clip_dataloaders(splits, config, intent_to_id, augment=False):
    """Create train/val/test dataloaders for CLIP training.

    No contrastive batch sampler needed -- standard shuffled batching.
    """
    augmenter = None
    if augment:
        augmenter = ScamAugmenter(
            strategies=config.augmentations.strategies,
            application_prob=config.augmentations.application_prob,
        )

    def make_loader(split_df, shuffle=False, aug=None):
        ds = CLIPScamDataset(
            texts=split_df["text"].tolist(),
            intent_ids=split_df["intent_id"].tolist(),
            tokenizer_name=config.model.encoder_name,
            max_length=config.data.max_length,
            augmenter=aug,
        )
        return DataLoader(
            ds, batch_size=config.stage_b.batch_size,
            shuffle=shuffle, num_workers=4, pin_memory=True,
        )

    loaders = {
        "train": make_loader(splits["train"], shuffle=True, aug=augmenter),
        "val": make_loader(splits["val"]),
        "test_seen": make_loader(splits["test_seen"]),
    }

    if "test_unseen" in splits and len(splits["test_unseen"]) > 0:
        loaders["test_unseen"] = make_loader(splits["test_unseen"])
    else:
        loaders["test_unseen"] = None

    return loaders
