"""PyTorch Dataset for trajectory training on pre-computed embeddings."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """Dataset of pre-computed embedding trajectories.

    Each item:
        embeddings: [max_turns, 768] -- frozen encoder outputs (padded)
        stage_labels: [max_turns] -- ground truth stage (padded with -1)
        escalation_labels: [max_turns] -- 1 from first payment/escalation turn onward
        length: actual number of turns
        mask: [max_turns] boolean mask
    """

    def __init__(self, trajectories, max_turns=30):
        """
        Args:
            trajectories: list of dicts with keys:
                "embeddings": np.array [T, 768]
                "stages": list[int] of length T
                "is_scam": bool
        """
        self.max_turns = max_turns
        self.data = []

        for traj in trajectories:
            embs = traj["embeddings"]  # [T, 768]
            stages = traj["stages"]
            T = min(len(stages), max_turns)

            # Pad embeddings
            padded_emb = np.zeros((max_turns, embs.shape[1]), dtype=np.float32)
            padded_emb[:T] = embs[:T]

            # Pad stage labels with -1 (ignored in loss)
            padded_stages = np.full(max_turns, -1, dtype=np.int64)
            padded_stages[:T] = stages[:T]

            # Escalation labels: 1 from first stage >= 4 onward
            escalation = np.zeros(max_turns, dtype=np.float32)
            escalated = False
            for t in range(T):
                if stages[t] >= 4:
                    escalated = True
                if escalated:
                    escalation[t] = 1.0

            # Mask
            mask = np.zeros(max_turns, dtype=bool)
            mask[:T] = True

            self.data.append({
                "embeddings": padded_emb,
                "stage_labels": padded_stages,
                "escalation_labels": escalation,
                "length": T,
                "mask": mask,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "embeddings": torch.tensor(d["embeddings"]),
            "stage_labels": torch.tensor(d["stage_labels"]),
            "escalation_labels": torch.tensor(d["escalation_labels"]),
            "length": torch.tensor(d["length"]),
            "mask": torch.tensor(d["mask"]),
        }


def create_trajectory_dataloaders(trajectories, config):
    """Split trajectories and create DataLoaders."""
    train_ds = TrajectoryDataset(
        [t for t in trajectories if t["split"] == "train"],
        max_turns=config.stage_c.max_turns,
    )
    val_ds = TrajectoryDataset(
        [t for t in trajectories if t["split"] == "val"],
        max_turns=config.stage_c.max_turns,
    )
    test_ds = TrajectoryDataset(
        [t for t in trajectories if t["split"] == "test"],
        max_turns=config.stage_c.max_turns,
    )

    return {
        "train": DataLoader(
            train_ds, batch_size=config.stage_c.batch_size, shuffle=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=config.stage_c.batch_size,
        ),
        "test": DataLoader(
            test_ds, batch_size=config.stage_c.batch_size,
        ),
    }
