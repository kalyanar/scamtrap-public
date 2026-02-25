"""Generalized Cross-Entropy (GCE) loss for noise-robust training.

Implements the truncated loss from Zhang & Sabuncu (NeurIPS 2018):
"Generalized Cross Entropy Loss for Training Deep Neural Networks
with Noisy Labels."

GCE interpolates between CE (q→0) and MAE (q=1). The truncated
variant clips the loss at a threshold to reduce the influence of
noisy samples. This is particularly relevant for ScamTrap's
keyword-based weak supervision, where ~65% of scam samples fall
to the generic_scam fallback category.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    """GCE loss: L_q(f(x), y) = (1 - f_y(x)^q) / q

    Args:
        q: Box-Cox parameter (0 < q <= 1). Lower q → closer to CE.
           q=0.7 is recommended by Zhang & Sabuncu (2018).
        k: Truncation parameter. Samples with loss > (1-k^q)/q are
           down-weighted. Set to 0 to disable truncation.
        reduction: 'mean' or 'sum'
    """

    def __init__(self, q: float = 0.7, k: float = 0.5, reduction: str = "mean"):
        super().__init__()
        assert 0 < q <= 1, f"q must be in (0, 1], got {q}"
        self.q = q
        self.k = k
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute GCE loss.

        Args:
            logits: [B, C] raw logits (pre-softmax)
            labels: [B] integer class labels

        Returns:
            scalar loss
        """
        probs = F.softmax(logits, dim=1)
        # Gather probability of true class
        p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        # Clamp to avoid numerical issues
        p_true = p_true.clamp(min=1e-7, max=1.0)

        # GCE loss
        loss = (1.0 - p_true ** self.q) / self.q

        # Truncation: down-weight samples with very high loss
        if self.k > 0:
            threshold = (1.0 - self.k ** self.q) / self.q
            weight = (loss <= threshold).float()
            loss = loss * weight

        if self.reduction == "mean":
            if self.k > 0:
                # Mean over non-truncated samples only
                return loss.sum() / weight.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
