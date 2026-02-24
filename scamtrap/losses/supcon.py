"""Supervised Contrastive Loss following Khosla et al. (NeurIPS 2020).

This is the core loss replicating ConRo's approach, adapted from
the HobbitLong/SupContrast reference implementation.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Multiple positives per anchor (all samples with same label).
    When labels=None, degenerates to SimCLR self-supervised loss.

    Args:
        temperature: scaling factor for cosine similarity
        base_temperature: for loss normalization
        contrast_mode: 'all' uses every view as anchor, 'one' uses first only
    """

    def __init__(self, temperature=0.07, base_temperature=0.07, contrast_mode="all"):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None):
        """Compute SupCon loss.

        Args:
            features: [batch_size, n_views, proj_dim] L2-normalized
            labels: [batch_size] class labels (optional)

        Returns:
            scalar loss
        """
        device = features.device

        if features.dim() < 3:
            raise ValueError(f"features must be [B, V, D], got shape {features.shape}")

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Labels length must match batch size")
            # Positive mask: same label = positive pair
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Self-supervised: only augmented views of same sample are positive
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # Tile mask for multi-view: [B*V, B*V]
        mask = mask.repeat(n_views, n_views)

        # Flatten views: [B*V, D]
        contrast_features = features.reshape(batch_size * n_views, -1)

        if self.contrast_mode == "one":
            anchor_features = features[:, 0]  # [B, D]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_features = contrast_features  # [B*V, D]
            anchor_count = n_views
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute pairwise similarities
        anchor_dot_contrast = torch.matmul(anchor_features, contrast_features.T) / self.temperature

        # Numerical stability: subtract max
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Build masks
        total_anchors = batch_size * anchor_count
        total_contrast = batch_size * n_views

        if self.contrast_mode == "one":
            mask = mask[:batch_size, :]  # [B, B*V]

        # Self-contrast mask: exclude diagonal (don't contrast with yourself)
        if self.contrast_mode == "all":
            logits_mask = 1 - torch.eye(total_anchors, total_contrast, device=device)
        else:
            logits_mask = torch.ones(total_anchors, total_contrast, device=device)
            # Exclude exact same view
            logits_mask[:batch_size, :batch_size] -= torch.eye(batch_size, device=device)

        mask = mask * logits_mask

        # Compute log softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positives
        # Avoid division by zero for anchors with no positives
        positives_per_anchor = mask.sum(dim=1)
        positives_per_anchor = positives_per_anchor.clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives_per_anchor

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
