"""Training loop for CLIP-style intent alignment."""

import time
import math
import torch
import torch.nn as nn
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from scamtrap.losses.clip_ce import CLIPCrossEntropyLoss
from scamtrap.utils.io import save_checkpoint


def get_scheduler(optimizer, config, num_training_steps):
    """Warmup + cosine annealing schedule."""
    warmup_steps = int(num_training_steps * config.stage_b.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CLIPTrainer:
    """Training loop for CLIP-style message-to-prototype alignment."""

    def __init__(self, model, train_loader, val_loader, config,
                 seen_descriptions, seen_intent_to_id):
        """
        Args:
            seen_descriptions: dict {intent_name: description_text} for SEEN intents only
            seen_intent_to_id: dict {intent_name: int} mapping used in training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Freeze description encoder — layer-level control
        freeze_layers = getattr(config.stage_b, "freeze_description_layers", -1)
        if freeze_layers == -1:
            # Use boolean flag (backward compat)
            freeze_desc = getattr(config.stage_b, "freeze_description_encoder", True)
            if freeze_desc:
                for param in model.description_encoder.parameters():
                    param.requires_grad = False
                print("Froze description encoder (preserves pretrained semantics for zero-shot)")
        else:
            self._freeze_description_layers(model.description_encoder, freeze_layers)

        # L2-SP: snapshot pretrained params before training
        self.l2sp_alpha = getattr(config.stage_b, "l2sp_alpha", 0.0)
        self.pretrained_params = {}
        if self.l2sp_alpha > 0:
            for name, param in model.description_encoder.named_parameters():
                if param.requires_grad:
                    self.pretrained_params[name] = param.data.clone()
            print(f"L2-SP enabled: alpha={self.l2sp_alpha}, "
                  f"regularizing {len(self.pretrained_params)} params")

        self.model.to(self.device)

        self.criterion = CLIPCrossEntropyLoss()

        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.stage_b.lr,
            weight_decay=config.training.weight_decay,
        )

        num_steps = len(train_loader) * config.stage_b.epochs
        self.scheduler = get_scheduler(self.optimizer, config, num_steps)

        self.use_amp = config.stage_b.fp16 and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            if self.use_amp else None
        )
        self.grad_accum = config.stage_b.gradient_accumulation_steps

        self.checkpoint_dir = Path(config.stage_b.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Pre-tokenize intent descriptions (ordered by intent_id)
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

        # Build ordered list: index i holds the description for intent_id i
        # Only include seen intents (those present in seen_intent_to_id)
        max_id = max(seen_intent_to_id.values()) + 1
        ordered_descs = [""] * max_id
        for name, desc in seen_descriptions.items():
            if name in seen_intent_to_id:
                ordered_descs[seen_intent_to_id[name]] = desc

        # Filter out empty slots (holdout intents not in training)
        # We need a contiguous mapping: training uses only seen intent ids
        self.seen_ids = sorted(seen_intent_to_id.values())
        contiguous_descs = [ordered_descs[i] for i in self.seen_ids]

        enc = tokenizer(
            contiguous_descs, max_length=config.data.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        self.desc_input_ids = enc["input_ids"].to(self.device)       # [K, seq_len]
        self.desc_attention_mask = enc["attention_mask"].to(self.device)

        # Build mapping from original intent_id -> contiguous index
        self.id_to_contiguous = {
            orig_id: idx for idx, orig_id in enumerate(self.seen_ids)
        }

    @staticmethod
    def _freeze_description_layers(encoder, n_freeze):
        """Freeze the bottom n_freeze transformer layers of the description encoder.

        With DistilBERT (6 layers), n_freeze=4 means top 2 layers are unfrozen.
        n_freeze=0 means fully unfrozen, n_freeze=6 means fully frozen.
        """
        if n_freeze <= 0:
            print("Description encoder: fully unfrozen")
            return

        # Freeze embeddings
        for param in encoder.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze bottom n_freeze transformer layers
        if hasattr(encoder.backbone, "transformer"):
            layers = encoder.backbone.transformer.layer
        elif hasattr(encoder.backbone, "encoder"):
            layers = encoder.backbone.encoder.layer
        else:
            print("WARNING: Could not find transformer layers to freeze")
            return

        total_layers = len(layers)
        actual_freeze = min(n_freeze, total_layers)
        for layer in layers[:actual_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        unfrozen = total_layers - actual_freeze
        print(f"Description encoder: froze {actual_freeze}/{total_layers} layers "
              f"({unfrozen} unfrozen)")

    def _compute_l2sp_loss(self):
        """Compute L2-SP regularization: alpha * sum((theta - theta_0)^2)."""
        if self.l2sp_alpha <= 0 or not self.pretrained_params:
            return 0.0
        l2sp = 0.0
        for name, param in self.model.description_encoder.named_parameters():
            if name in self.pretrained_params:
                pretrained = self.pretrained_params[name].to(param.device)
                l2sp += ((param - pretrained) ** 2).sum()
        return self.l2sp_alpha * l2sp

    def _remap_labels(self, labels):
        """Remap original intent_ids to contiguous 0..K-1 for CE loss."""
        return torch.tensor(
            [self.id_to_contiguous[l.item()] for l in labels],
            dtype=torch.long, device=self.device,
        )

    def train_epoch(self, epoch):
        """One epoch of CLIP training."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            msg_ids = batch["input_ids"].to(self.device)
            msg_mask = batch["attention_mask"].to(self.device)
            labels = self._remap_labels(batch["intent_id"])

            amp_ctx = (
                torch.amp.autocast(self.device.type)
                if self.use_amp else nullcontext()
            )
            with amp_ctx:
                _, logits = self.model(
                    msg_ids, msg_mask,
                    self.desc_input_ids, self.desc_attention_mask,
                )
                loss = self.criterion(logits, labels)
                l2sp_loss = self._compute_l2sp_loss()
                if l2sp_loss != 0.0:
                    loss = loss + l2sp_loss
                loss = loss / self.grad_accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.grad_accum == 0:
                if self.scaler:
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
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            msg_ids = batch["input_ids"].to(self.device)
            msg_mask = batch["attention_mask"].to(self.device)
            labels = self._remap_labels(batch["intent_id"])

            _, logits = self.model(
                msg_ids, msg_mask,
                self.desc_input_ids, self.desc_attention_mask,
            )
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self):
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        patience = self.config.stage_b.early_stopping_patience
        history = {"train_loss": [], "val_loss": []}

        print(f"Training on {self.device} | AMP: {self.use_amp}")
        print(f"Effective batch size: {self.config.stage_b.batch_size * self.grad_accum}")

        for epoch in range(1, self.config.stage_b.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}/{self.config.stage_b.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")

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

        save_checkpoint(
            self.model, self.optimizer, epoch, val_loss,
            str(self.checkpoint_dir / "final_model.pt"),
        )

        return history
