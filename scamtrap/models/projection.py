"""MLP projection head for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder outputs to contrastive space.

    Following SimCLR/SupCon: 2-layer MLP with BN and ReLU.
    Output is L2-normalized. This head is discarded after training;
    downstream tasks use the encoder output directly.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
        self.head = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            h: [batch_size, input_dim]
        Returns:
            z: [batch_size, output_dim] L2-normalized
        """
        return F.normalize(self.head(h), dim=1)
