"""PyTorch Dataset and contrastive batch sampler."""

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from transformers import AutoTokenizer
from scamtrap.data.augmentations import ScamAugmenter


class ScamDataset(Dataset):
    """PyTorch dataset for contrastive scam text training.

    Each __getitem__ returns n_views augmented tokenizations of the same text,
    plus labels. During eval, n_views=1 and no augmentation is applied.
    """

    def __init__(
        self,
        texts: list[str],
        intent_ids: list[int],
        binary_labels: list[int],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        n_views: int = 2,
        augmenter: ScamAugmenter = None,
    ):
        self.texts = texts
        self.intent_ids = intent_ids
        self.binary_labels = binary_labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.n_views = n_views
        self.augmenter = augmenter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        intent_id = self.intent_ids[idx]
        binary_label = self.binary_labels[idx]

        if self.n_views > 1 and self.augmenter is not None:
            # Generate n_views augmented versions
            views_input_ids = []
            views_attention_mask = []
            for _ in range(self.n_views):
                aug_text = self.augmenter.apply_random(text)
                enc = self.tokenizer(
                    aug_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                views_input_ids.append(enc["input_ids"].squeeze(0))
                views_attention_mask.append(enc["attention_mask"].squeeze(0))

            return {
                "input_ids": torch.stack(views_input_ids),       # [n_views, seq_len]
                "attention_mask": torch.stack(views_attention_mask),
                "intent_id": torch.tensor(intent_id, dtype=torch.long),
                "binary_label": torch.tensor(binary_label, dtype=torch.long),
            }
        else:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "intent_id": torch.tensor(intent_id, dtype=torch.long),
                "binary_label": torch.tensor(binary_label, dtype=torch.long),
            }


class ContrastiveBatchSampler(Sampler):
    """Batch sampler ensuring multiple samples per class in each batch.

    SupCon loss requires at least 2 samples per class. This sampler
    builds batches by selecting a subset of classes, then sampling
    samples_per_class from each to fill the batch to batch_size.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int = 64,
        min_per_class: int = 2,
        drop_last: bool = True,
    ):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.drop_last = drop_last

        # Group indices by label
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            self.class_indices.setdefault(label, [])
            self.class_indices[label].append(idx)

        # Filter classes with enough samples
        self.valid_classes = [
            c for c, idxs in self.class_indices.items()
            if len(idxs) >= min_per_class
        ]

        # Compute how many classes per batch and samples per class
        n_classes = len(self.valid_classes)
        self.classes_per_batch = min(n_classes, batch_size // min_per_class)
        self.samples_per_class = batch_size // self.classes_per_batch

        # Total batches: based on largest class to ensure full epoch coverage
        self._num_batches = len(self.labels) // self.batch_size

    def __iter__(self):
        # Shuffle indices within each class; use infinite cycling
        shuffled = {}
        for c in self.valid_classes:
            perm = np.random.permutation(self.class_indices[c]).tolist()
            shuffled[c] = perm
        pointers = {c: 0 for c in self.valid_classes}

        def _sample_from_class(c, n):
            """Sample n indices from class c, cycling if exhausted."""
            indices = []
            for _ in range(n):
                if pointers[c] >= len(shuffled[c]):
                    # Reshuffle and reset pointer
                    shuffled[c] = np.random.permutation(
                        self.class_indices[c]
                    ).tolist()
                    pointers[c] = 0
                indices.append(shuffled[c][pointers[c]])
                pointers[c] += 1
            return indices

        batches = []
        for _ in range(self._num_batches):
            batch = []
            # Select a random subset of classes for this batch
            chosen = np.random.choice(
                self.valid_classes,
                size=self.classes_per_batch,
                replace=False,
            ).tolist()

            for c in chosen:
                batch.extend(_sample_from_class(c, self.samples_per_class))

            # Trim to exact batch_size (rounding may overshoot)
            batch = batch[:self.batch_size]
            np.random.shuffle(batch)
            batches.append(batch)

        np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return self._num_batches


def create_dataloaders(splits, config, intent_to_id):
    """Create train/val/test DataLoaders from splits."""
    augmenter = ScamAugmenter(
        strategies=config.augmentations.strategies,
        application_prob=config.augmentations.application_prob,
    )

    # Training: multi-view with augmentation + contrastive sampler
    train_dataset = ScamDataset(
        texts=splits["train"]["text"].tolist(),
        intent_ids=splits["train"]["intent_id"].tolist(),
        binary_labels=splits["train"]["binary_label"].tolist(),
        tokenizer_name=config.model.encoder_name,
        max_length=config.data.max_length,
        n_views=config.augmentations.n_views,
        augmenter=augmenter,
    )

    train_sampler = ContrastiveBatchSampler(
        labels=splits["train"]["intent_id"].tolist(),
        batch_size=config.training.batch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Val: multi-view with augmentation so val loss is comparable to train loss
    val_dataset = ScamDataset(
        texts=splits["val"]["text"].tolist(),
        intent_ids=splits["val"]["intent_id"].tolist(),
        binary_labels=splits["val"]["binary_label"].tolist(),
        tokenizer_name=config.model.encoder_name,
        max_length=config.data.max_length,
        n_views=config.augmentations.n_views,
        augmenter=augmenter,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # Test: single view, no augmentation, standard batching
    def make_eval_loader(split_df):
        ds = ScamDataset(
            texts=split_df["text"].tolist(),
            intent_ids=split_df["intent_id"].tolist(),
            binary_labels=split_df["binary_label"].tolist(),
            tokenizer_name=config.model.encoder_name,
            max_length=config.data.max_length,
            n_views=1,
            augmenter=None,
        )
        return DataLoader(ds, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_seen_loader = make_eval_loader(splits["test_seen"])
    test_unseen_loader = make_eval_loader(splits["test_unseen"]) if len(splits["test_unseen"]) > 0 else None

    return {
        "train": train_loader,
        "val": val_loader,
        "test_seen": test_seen_loader,
        "test_unseen": test_unseen_loader,
    }
