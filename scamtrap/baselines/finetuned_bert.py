"""Fine-tuned DistilBERT classifier baseline."""

import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class FineTunedBertBaseline(nn.Module):
    """DistilBERT + linear classification head trained with cross-entropy.

    The [CLS] token representation is used for embedding-based evaluation.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_classes=7, seed=42):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seed = seed

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return cls_output, logits

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

    def fit(self, texts, labels, epochs=10, lr=2e-5, batch_size=32, max_length=128):
        """Train the classifier."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        enc = self.tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        dataset = TensorDataset(
            enc["input_ids"], enc["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"BERT Epoch {epoch+1}"):
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                _, logits = self(batch_ids, batch_mask)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

        return self

    def extract_all_embeddings(self, texts, max_length=128, batch_size=64):
        """Extract embeddings for all texts."""
        device = next(self.parameters()).device
        self.eval()
        all_embs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch_texts, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                embs = self.get_embeddings(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)
