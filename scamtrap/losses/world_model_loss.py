"""Combined loss for world model: stage prediction + escalation forecasting."""

import torch
import torch.nn as nn


class WorldModelLoss(nn.Module):
    """Joint loss for stage classification and escalation prediction.

    Combines:
    - Stage CE loss (ignore_index=-1 for padded timesteps)
    - Escalation BCE loss (masked to valid timesteps only)

    Alpha controls the balance: loss = alpha * stage + (1-alpha) * escalation
    """

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.stage_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.escalation_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, stage_logits, escalation_logits,
                stage_labels, escalation_labels, mask):
        """Compute combined loss.

        Args:
            stage_logits: [B, T, num_stages]
            escalation_logits: [B, T, 1]
            stage_labels: [B, T] with -1 for padding
            escalation_labels: [B, T] float
            mask: [B, T] boolean
        """
        # Stage loss: [B, T, C] vs [B, T], ignore padding (-1)
        # Use reshape instead of view — GRU pad_packed_sequence output may be non-contiguous
        B, T, C = stage_logits.shape
        sl = self.stage_loss(stage_logits.reshape(-1, C), stage_labels.reshape(-1))

        # Escalation loss: only on valid (masked) timesteps
        el = self.escalation_loss(
            escalation_logits.squeeze(-1), escalation_labels,
        )
        el = (el * mask.float()).sum() / mask.float().sum().clamp(min=1)

        return self.alpha * sl + (1 - self.alpha) * el
