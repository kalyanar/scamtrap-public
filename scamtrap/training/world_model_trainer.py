"""Training loop for world model (Stage C)."""

import time
import math
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from scamtrap.losses.world_model_loss import WorldModelLoss
from scamtrap.utils.io import save_checkpoint


def get_scheduler(optimizer, config, num_training_steps):
    """Warmup + cosine annealing schedule."""
    warmup_steps = int(num_training_steps * config.stage_c.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class WorldModelTrainer:
    """Training loop for GRU/Transformer world model."""

    def __init__(self, model, train_loader, val_loader, config,
                 model_type="gru"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_type = model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = WorldModelLoss(alpha=config.stage_c.loss_alpha)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.stage_c.lr,
            weight_decay=config.stage_c.weight_decay,
        )

        num_steps = len(train_loader) * config.stage_c.epochs
        self.scheduler = get_scheduler(self.optimizer, config, num_steps)

        self.checkpoint_dir = Path(config.stage_c.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        """One epoch of world model training."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            embs = batch["embeddings"].to(self.device)
            stage_labels = batch["stage_labels"].to(self.device)
            esc_labels = batch["escalation_labels"].to(self.device)
            lengths = batch["length"].to(self.device)
            mask = batch["mask"].to(self.device)

            stage_logits, esc_logits, _ = self.model(embs, lengths)

            # Truncate to match packed sequence output length
            T_out = stage_logits.shape[1]
            loss = self.criterion(
                stage_logits, esc_logits,
                stage_labels[:, :T_out], esc_labels[:, :T_out],
                mask[:, :T_out],
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            embs = batch["embeddings"].to(self.device)
            stage_labels = batch["stage_labels"].to(self.device)
            esc_labels = batch["escalation_labels"].to(self.device)
            lengths = batch["length"].to(self.device)
            mask = batch["mask"].to(self.device)

            stage_logits, esc_logits, _ = self.model(embs, lengths)

            T_out = stage_logits.shape[1]
            loss = self.criterion(
                stage_logits, esc_logits,
                stage_labels[:, :T_out], esc_labels[:, :T_out],
                mask[:, :T_out],
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self):
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = self.config.stage_c.early_stopping_patience
        history = {"train_loss": [], "val_loss": []}

        print(f"Training world model on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.config.stage_c.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch}/{self.config.stage_c.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_name = f"best_model_{self.model_type}.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    str(self.checkpoint_dir / ckpt_name),
                )
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break

        final_name = f"final_model_{self.model_type}.pt"
        save_checkpoint(
            self.model, self.optimizer, epoch, val_loss,
            str(self.checkpoint_dir / final_name),
        )

        return history
