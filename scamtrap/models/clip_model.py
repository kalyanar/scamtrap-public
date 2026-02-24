"""Dual-encoder CLIP-style model for scam intent alignment."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scamtrap.models.encoder import TextEncoder


class CLIPScamModel(nn.Module):
    """Aligns scam messages with natural language intent descriptions.

    Two separate DistilBERT encoders:
    - message_encoder: encodes scam/ham messages
    - description_encoder: encodes intent descriptions

    Each has a 2-layer projection MLP mapping 768d -> proj_dim (256d).
    Temperature is a learnable parameter.
    """

    def __init__(self, config):
        super().__init__()
        # Message encoder (can be warm-started from Stage A)
        self.message_encoder = TextEncoder(
            model_name=config.model.encoder_name,
            pooling=config.model.pooling,
        )
        # Description encoder (separate, always from pretrained)
        self.description_encoder = TextEncoder(
            model_name=config.model.encoder_name,
            pooling="mean",  # Mean pooling better for descriptions
        )

        hidden = self.message_encoder.hidden_dim  # 768
        proj_dim = config.stage_b.proj_dim  # 256

        self.message_proj = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.description_proj = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Learnable temperature
        init_temp = config.stage_b.initial_temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temp))
        )

    def encode_messages(self, input_ids, attention_mask):
        """Encode messages -> L2-normalized proj_dim vectors."""
        h = self.message_encoder(input_ids, attention_mask)  # [B, 768]
        z = F.normalize(self.message_proj(h), dim=1)  # [B, proj_dim]
        return h, z

    def encode_descriptions(self, input_ids, attention_mask):
        """Encode intent descriptions -> L2-normalized proj_dim vectors."""
        h = self.description_encoder(input_ids, attention_mask)  # [K, 768]
        z = F.normalize(self.description_proj(h), dim=1)  # [K, proj_dim]
        return z

    def forward(self, msg_input_ids, msg_attention_mask,
                desc_input_ids, desc_attention_mask):
        """Compute message-to-prototype logits.

        Args:
            msg_input_ids: [B, seq_len]
            msg_attention_mask: [B, seq_len]
            desc_input_ids: [K, seq_len]  (K intent descriptions)
            desc_attention_mask: [K, seq_len]

        Returns:
            h_msg: [B, 768] encoder embeddings (for downstream eval)
            logits: [B, K] similarity scores / temperature
        """
        h_msg, z_msg = self.encode_messages(msg_input_ids, msg_attention_mask)
        z_desc = self.encode_descriptions(desc_input_ids, desc_attention_mask)

        temperature = self.log_temperature.exp()
        logits = torch.matmul(z_msg, z_desc.T) / temperature  # [B, K]

        return h_msg, logits

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask):
        """Extract message encoder embeddings (for evaluation).

        Returns 768d encoder output -- compatible with ALL Stage A
        evaluation code (fewshot, retrieval, clustering, etc.).
        """
        return self.message_encoder(input_ids, attention_mask)

    def load_stage_a_encoder(self, checkpoint_path):
        """Load Stage A encoder weights into message_encoder.

        Stage A checkpoint contains ScamTrapModel with .encoder attribute.
        We extract just the encoder state_dict.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Extract encoder.* keys, remove "encoder." prefix
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v

        self.message_encoder.load_state_dict(encoder_state)
        print(f"Loaded Stage A encoder from {checkpoint_path}")
