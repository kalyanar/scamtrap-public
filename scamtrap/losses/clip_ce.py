"""CLIP-style cross-entropy loss for message-to-prototype alignment.

Design note: We use message-to-prototype CE rather than symmetric CLIP loss.
With only K=7 distinct intent descriptions, many batch messages share the same
intent. A symmetric [B, B] CLIP matrix would have ambiguous positive assignments
(multiple rows mapping to the same column). Message-to-prototype CE naturally
handles this by scoring each message against all K prototypes independently.
"""

import torch.nn as nn
import torch.nn.functional as F


class CLIPCrossEntropyLoss(nn.Module):
    """Cross-entropy over message-to-prototype similarity logits.

    Input:  logits [B, K] -- cosine similarity / temperature
    Labels: [B] -- intent index (0..K-1)
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        return F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing,
        )
