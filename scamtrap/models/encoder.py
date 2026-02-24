"""Text encoder wrapping HuggingFace transformers."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    """Transformer-based text encoder.

    Wraps a pretrained model (DistilBERT/BERT/RoBERTa) and returns
    pooled representations. Designed for reuse in Stage B (dual encoder)
    and Stage C (per-turn embedding).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        pooling: str = "cls",
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        self.pooling = pooling
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n_layers: int):
        """Freeze the bottom n_layers of the transformer."""
        if n_layers <= 0:
            return
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        # Freeze transformer layers
        if hasattr(self.backbone, "transformer"):
            layers = self.backbone.transformer.layer
        elif hasattr(self.backbone, "encoder"):
            layers = self.backbone.encoder.layer
        else:
            return
        for layer in layers[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to pooled representation.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            [batch_size, hidden_dim] pooled representation
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return hidden_states.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
