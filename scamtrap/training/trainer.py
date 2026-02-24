"""Contrastive training loop with mixed precision and gradient accumulation."""

import time
import math
import torch
import torch.nn as nn
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm

from scamtrap.losses.supcon import SupConLoss
from scamtrap.utils.io import save_checkpoint


def get_scheduler(optimizer, config, num_training_steps):
    """Warmup + cosine annealing schedule."""
    warmup_steps = int(num_training_steps * config.training.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ContrastiveTrainer:
    """Training loop for SupCon contrastive learning."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = SupConLoss(
            temperature=config.loss.temperature,
            base_temperature=config.loss.base_temperature,
            contrast_mode=config.loss.contrast_mode,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Materialize actual batch count for accurate scheduling.
        # len() on a DataLoader with batch_sampler returns batch_sampler.__len__().
        self.num_batches_per_epoch = len(train_loader)
        num_steps = self.num_batches_per_epoch * config.training.epochs
        self.scheduler = get_scheduler(self.optimizer, config, num_steps)

        self.use_amp = config.training.fp16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp) if self.use_amp else None
        self.grad_accum = config.training.gradient_accumulation_steps

        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        """One epoch of contrastive training."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["intent_id"].to(self.device)

            amp_ctx = torch.amp.autocast(self.device.type) if self.use_amp else nullcontext()
            with amp_ctx:
                _, z = self.model(input_ids, attention_mask, return_projection=True)
                # z shape: [B, n_views, proj_dim]
                loss = self.criterion(z, labels)
                loss = loss / self.grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.grad_accum
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        """Compute validation loss.

        Uses the same multi-view augmented loader as training so that
        val loss is directly comparable to train loss.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["intent_id"].to(self.device)

            _, z = self.model(input_ids, attention_mask, return_projection=True)
            if z.dim() == 2:
                z = z.unsqueeze(1)

            loss = self.criterion(z, labels)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self):
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = self.config.training.early_stopping_patience
        history = {"train_loss": [], "val_loss": []}

        print(f"Training on {self.device} | AMP: {self.use_amp}")
        print(f"Effective batch size: {self.config.training.batch_size * self.grad_accum}")

        for epoch in range(1, self.config.training.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch}/{self.config.training.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    str(self.checkpoint_dir / "best_model.pt"),
                )
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        # Save final model
        save_checkpoint(
            self.model, self.optimizer, epoch, val_loss,
            str(self.checkpoint_dir / "final_model.pt"),
        )

        return history
