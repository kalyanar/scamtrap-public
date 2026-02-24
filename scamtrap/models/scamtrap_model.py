"""Combined encoder + projection model for contrastive training."""

import torch
import torch.nn as nn
from scamtrap.models.encoder import TextEncoder
from scamtrap.models.projection import ProjectionHead


class ScamTrapModel(nn.Module):
    """Full contrastive model: text encoder + projection head.

    During training: returns (h, z) where z is for SupCon loss.
    During evaluation: returns (h, None) where h is for downstream tasks.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = TextEncoder(
            model_name=config.model.encoder_name,
            pooling=config.model.pooling,
            freeze_layers=config.model.freeze_encoder_layers,
        )
        self.projector = ProjectionHead(
            input_dim=self.encoder.hidden_dim,
            hidden_dim=config.model.proj_hidden_dim,
            output_dim=config.model.proj_output_dim,
            num_layers=config.model.proj_layers,
        )

    def forward(self, input_ids, attention_mask, return_projection=True):
        """Forward pass.

        Args:
            input_ids: [B, seq_len] or [B, n_views, seq_len]
            attention_mask: [B, seq_len] or [B, n_views, seq_len]
            return_projection: if True, also compute projection z

        Returns:
            h: encoder embeddings [B, hidden_dim] or [B, n_views, hidden_dim]
            z: projected embeddings or None
        """
        # Handle multi-view input
        if input_ids.dim() == 3:
            B, V, L = input_ids.shape
            input_ids = input_ids.view(B * V, L)
            attention_mask = attention_mask.view(B * V, L)
            h = self.encoder(input_ids, attention_mask)  # [B*V, hidden_dim]
            h = h.view(B, V, -1)  # [B, V, hidden_dim]
            if return_projection:
                z = self.projector(h.view(B * V, -1))  # [B*V, proj_dim]
                z = z.view(B, V, -1)  # [B, V, proj_dim]
                return h, z
            return h, None
        else:
            h = self.encoder(input_ids, attention_mask)
            if return_projection:
                z = self.projector(h)
                return h, z
            return h, None

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask):
        """Extract encoder embeddings only (for evaluation)."""
        return self.encoder(input_ids, attention_mask)
