"""World models for scam trajectory prediction."""

import torch
import torch.nn as nn


class ScamWorldModel(nn.Module):
    """GRU-based world model.

    Input: pre-computed frozen encoder embeddings [B, T, 768]
    Output: stage predictions [B, T, 6] + escalation logits [B, T, 1]
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.stage_c.gru_hidden_dim  # 256

        self.input_proj = nn.Linear(768, hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=config.stage_c.gru_layers,
            batch_first=True, dropout=config.stage_c.gru_dropout if config.stage_c.gru_layers > 1 else 0.0,
        )
        self.stage_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, config.stage_c.num_stages),
        )
        self.escalation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, embeddings, lengths=None):
        """Forward pass.

        Args:
            embeddings: [B, T, 768] pre-computed encoder outputs
            lengths: [B] actual sequence lengths (for packing)

        Returns:
            stage_logits: [B, T, num_stages]
            escalation_logits: [B, T, 1]
            gru_out: [B, T, hidden] GRU hidden states
        """
        x = self.input_proj(embeddings)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
        gru_out, _ = self.gru(x)
        if lengths is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True,
            )

        return self.stage_head(gru_out), self.escalation_head(gru_out), gru_out


class TransformerWorldModel(nn.Module):
    """Transformer-based world model (ablation).
    Same inputs/outputs as ScamWorldModel."""

    def __init__(self, config):
        super().__init__()
        hidden = config.stage_c.gru_hidden_dim
        self.input_proj = nn.Linear(768, hidden)
        self.pos_encoding = nn.Embedding(config.stage_c.max_turns, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=config.stage_c.transformer_heads,
            dim_feedforward=hidden * 4, dropout=config.stage_c.gru_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.stage_c.transformer_layers,
        )
        self.stage_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, config.stage_c.num_stages),
        )
        self.escalation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, embeddings, lengths=None):
        """Forward pass with causal masking.

        Args:
            embeddings: [B, T, 768]
            lengths: [B] actual sequence lengths

        Returns:
            stage_logits: [B, T, num_stages]
            escalation_logits: [B, T, 1]
            hidden_states: [B, T, hidden]
        """
        B, T, _ = embeddings.shape
        x = self.input_proj(embeddings)
        positions = torch.arange(T, device=embeddings.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoding(positions)

        # Causal mask: can only attend to past/current turns
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=embeddings.device,
        )

        # Padding mask
        padding_mask = None
        if lengths is not None:
            padding_mask = (
                torch.arange(T, device=embeddings.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )

        out = self.transformer(
            x, mask=causal_mask, src_key_padding_mask=padding_mask,
        )
        return self.stage_head(out), self.escalation_head(out), out
