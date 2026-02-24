"""Checkpoint and results I/O utilities."""

import json
import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, loss, path: str):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    """Load model checkpoint. Returns epoch and loss."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


def save_results(results: dict, path: str):
    """Save results dict as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: str) -> dict:
    """Load results from JSON."""
    with open(path) as f:
        return json.load(f)
