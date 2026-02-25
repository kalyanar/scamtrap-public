# ScamTrap Stages B & C — Learning Guide

This guide teaches you **how** and **why** Stages B and C work, building on the Stage A foundation you already have. Read each section in order. Each section explains the concept, walks through the actual code, and tells you how to verify it.

---

## Prerequisites: What You Already Have (Stage A)

Before diving in, let's be clear about what Stage A built:

```
Stage A produced:
  1. A trained DistilBERT encoder (768d embeddings)
  2. ~15K scam/ham messages labeled with 7 intent types
  3. Open-set splits: crypto + romance held out entirely
  4. Evaluation suite: fewshot, retrieval, clustering, robustness, open-set
```

**Stage A's limitation**: It groups messages by numeric label ID. To detect a *new* scam type (like crypto), you'd need to collect examples, label them, and retrain. That's where Stage B comes in.

---

# STAGE B: CLIP-like Intent Alignment

## The Core Idea (Read This First)

Imagine you have a new scam type — say "crypto scams." With Stage A, you can't detect it because the model was never trained on crypto examples. But what if instead of matching messages to label *numbers*, you matched them to label *descriptions*?

```
Stage A thinks:               Stage B thinks:
  message → label #3            message → "This message impersonates a bank..."
  message → label #5            message → "This message offers fraudulent employment..."
```

If you train the model to understand *what scam descriptions mean*, then at test time you can write a new description like "This message promotes fraudulent cryptocurrency..." and the model can match messages to it — **zero-shot, no retraining**.

This is exactly how CLIP (OpenAI, 2021) works, but adapted from image-text to message-description.

---

## B1: Intent Descriptions

**File**: `scamtrap/data/intent_descriptions.py`

This is the simplest file — it's just a dictionary. But the *content* matters a lot.

### What's in it

```python
INTENT_DESCRIPTIONS = {
    "ham": "This is a legitimate, non-malicious message...",
    "credential_theft": "This message impersonates a legitimate service to steal login credentials...",
    "delivery": "This message impersonates a delivery service like UPS, FedEx...",
    "bank_alert": "This message impersonates a bank or financial institution...",
    "generic_scam": "This message is a scam or spam that uses deceptive tactics...",
    "job_offer": "This message offers a fraudulent employment opportunity...",
    "prize_lottery": "This message falsely claims the recipient has won a prize...",
    # HOLDOUT — only used at test time:
    "crypto": "This message promotes a fraudulent cryptocurrency investment...",
    "romance": "This message uses emotional manipulation and fake romantic interest...",
}

SEEN_INTENTS = ["ham", "credential_theft", "delivery", "bank_alert",
                "generic_scam", "job_offer", "prize_lottery"]  # 7 for training
HOLDOUT_INTENTS = ["crypto", "romance"]                        # 2 for zero-shot test
```

### Helper variables and functions

The file also provides utilities used by the ablation scripts:

```python
# All scam intents (excludes ham)
ALL_SCAM_INTENTS = [
    "credential_theft", "delivery", "bank_alert", "generic_scam",
    "job_offer", "prize_lottery", "crypto", "romance",
]

def get_seen_and_holdout(holdout_list):
    """Derive seen/holdout intent lists from any holdout specification.
    Returns (seen_intents, holdout_intents) where seen always includes 'ham'.
    """
```

`get_seen_and_holdout()` is used by `run_holdout_sweep.py` to test different holdout configurations beyond the default crypto+romance.

### Why descriptions matter

Each description describes the **mechanism** of the scam, not just the topic:
- Bad: "A crypto scam" (too vague, the model can't generalize)
- Good: "promotes a fraudulent cryptocurrency investment, trading opportunity, or wallet service designed to steal funds" (describes *how* it works)

This is important because at zero-shot time, the model needs enough semantic signal in the description to match it against messages it has never seen.

### Verify it

```python
from scamtrap.data.intent_descriptions import (
    INTENT_DESCRIPTIONS, SEEN_INTENTS, HOLDOUT_INTENTS,
    ALL_SCAM_INTENTS, get_seen_and_holdout,
)
assert len(INTENT_DESCRIPTIONS) == 9   # 7 seen + 2 holdout
assert len(SEEN_INTENTS) == 7
assert len(HOLDOUT_INTENTS) == 2
assert "crypto" not in SEEN_INTENTS    # Never trained on
assert len(ALL_SCAM_INTENTS) == 8      # All scam types (no ham)

# Dynamic holdout: hold out job_offer and prize_lottery instead
seen, holdout = get_seen_and_holdout(["job_offer", "prize_lottery"])
assert "ham" in seen
assert "job_offer" not in seen
```

---

## B2: The Loss Function — Why Not Standard CLIP?

**File**: `scamtrap/losses/clip_ce.py`

This is a critical design decision. Let me explain *why* we don't use the standard symmetric CLIP loss.

### Standard CLIP loss (what OpenAI uses)

In CLIP, you have N images paired with N *unique* captions. You build an N×N similarity matrix and push each row/column to match its diagonal:

```
                caption_1  caption_2  caption_3
  image_1         HIGH       low        low      ← row 1 matches col 1
  image_2         low        HIGH       low      ← row 2 matches col 2
  image_3         low        low        HIGH     ← row 3 matches col 3
```

### Our problem: captions aren't unique

In our case, many messages share the *same* intent description. A batch of 64 messages might have 20 "ham" messages, 15 "credential_theft", etc. — all pointing to the same description. The N×N matrix breaks down:

```
                ham_desc  cred_desc  delivery_desc
  msg_1 (ham)     HIGH      low         low
  msg_2 (ham)     HIGH      low         low      ← same target as msg_1!
  msg_3 (cred)    low       HIGH        low
```

Rows 1 and 2 both want column 1. The symmetric loss gets confused about which row "owns" which column.

### Our solution: Message-to-Prototype CE

Instead of an N×N matrix, we compute an N×K matrix where K=7 (number of intent types). Each message is scored against all 7 prototypes, and we use standard cross-entropy:

```python
class CLIPCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # logits: [B, K] — similarity of each message to each prototype
        # labels: [B]    — which prototype is correct (0..K-1)
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
```

The simplicity is the point — the complexity is in the *model*, not the loss.

**Label smoothing** (`label_smoothing` parameter) is supported but defaults to 0.0. When enabled, it softens the target distribution to prevent overconfidence — useful in ablation studies.

### Verify it

```python
import torch
from scamtrap.losses.clip_ce import CLIPCrossEntropyLoss

loss_fn = CLIPCrossEntropyLoss()
logits = torch.randn(4, 7)  # 4 messages, 7 prototypes
labels = torch.tensor([0, 2, 5, 1])  # correct prototype for each
loss = loss_fn(logits, labels)
print(f"Loss: {loss.item():.4f}")  # Should be ~1.9 (random logits)
```

---

## B3: The CLIP Model — How Two Encoders Become One System

**File**: `scamtrap/models/clip_model.py`

This is the heart of Stage B. Let me walk through it piece by piece.

### Architecture overview

```
Message path:                Description path:
  "Click here to verify"       "This message impersonates a legitimate service..."
        ↓                              ↓
  [DistilBERT #1]                [DistilBERT #2]
        ↓ (768d)                       ↓ (768d)
  [Message MLP]                  [Description MLP]
        ↓ (256d)                       ↓ (256d)
  [L2 normalize]                 [L2 normalize]
        ↓                              ↓
     z_msg ─────── cosine sim ──────── z_desc
                      ↓
              logits [B, K] / temperature
                      ↓
              cross-entropy loss
```

### Two separate encoders — why?

Messages and descriptions are fundamentally different kinds of text:
- Messages: short, informal, often obfuscated ("Ur acc0unt has been susp3nded")
- Descriptions: long, formal, descriptive ("This message impersonates a bank...")

Using separate encoders lets each specialize. The message encoder uses CLS pooling (same as Stage A). The description encoder uses mean pooling (better for sentences where meaning is spread across all tokens).

### The projection MLPs

Each encoder outputs 768d. We project down to 256d with a 2-layer MLP:

```python
self.message_proj = nn.Sequential(
    nn.Linear(768, 256),  # Compress
    nn.ReLU(),            # Non-linearity
    nn.Linear(256, 256),  # Refine
)
```

Then L2-normalize so cosine similarity = dot product:

```python
z = F.normalize(self.message_proj(h), dim=1)  # unit vectors
```

### Learnable temperature

Temperature controls how "peaked" the softmax is:
- Low temperature (0.01): model is very confident, small differences become large
- High temperature (1.0): model is uncertain, all prototypes look similar

We make it learnable so the model finds the right sharpness:

```python
self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
# In forward:
temperature = self.log_temperature.exp()  # always positive
logits = (z_msg @ z_desc.T) / temperature
```

### The forward pass (training)

```python
def forward(self, msg_input_ids, msg_attention_mask,
            desc_input_ids, desc_attention_mask):
    # 1. Encode messages
    h_msg, z_msg = self.encode_messages(msg_input_ids, msg_attention_mask)
    # h_msg: [B, 768]  (raw encoder output — for eval)
    # z_msg: [B, 256]  (projected, normalized — for training)

    # 2. Encode descriptions
    z_desc = self.encode_descriptions(desc_input_ids, desc_attention_mask)
    # z_desc: [K, 256]  (K=7 prototype vectors)

    # 3. Compute similarity logits
    temperature = self.log_temperature.exp()
    logits = torch.matmul(z_msg, z_desc.T) / temperature
    # logits: [B, K]  — each message scored against each prototype

    return h_msg, logits
```

### get_embeddings() — backward compatibility

This is crucial. It returns 768d vectors from the message encoder only — *exactly the same interface as Stage A's ScamTrapModel*. This means ALL your Stage A evaluation code works unchanged:

```python
@torch.no_grad()
def get_embeddings(self, input_ids, attention_mask):
    return self.message_encoder(input_ids, attention_mask)  # [B, 768]
```

### Warm-starting from Stage A

The message encoder can load weights from your trained Stage A model. The Stage A checkpoint has keys like `encoder.backbone.transformer.layer.0...`, so we strip the `encoder.` prefix:

```python
def load_stage_a_encoder(self, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    encoder_state = {k[len("encoder."):]: v
                     for k, v in state_dict.items()
                     if k.startswith("encoder.")}
    self.message_encoder.load_state_dict(encoder_state)
```

### Verify it

```python
import torch
from scamtrap.utils.config import load_config
from scamtrap.models.clip_model import CLIPScamModel

config = load_config("configs/default.yaml")
model = CLIPScamModel(config)

# Simulate forward pass
msg_ids = torch.randint(0, 100, (4, 128))   # 4 messages
msg_mask = torch.ones(4, 128, dtype=torch.long)
desc_ids = torch.randint(0, 100, (7, 128))  # 7 descriptions
desc_mask = torch.ones(7, 128, dtype=torch.long)

h_msg, logits = model(msg_ids, msg_mask, desc_ids, desc_mask)
print(f"h_msg: {h_msg.shape}")   # [4, 768]
print(f"logits: {logits.shape}") # [4, 7]

# Verify eval interface matches Stage A
embs = model.get_embeddings(msg_ids, msg_mask)
print(f"embs: {embs.shape}")    # [4, 768]  — same as ScamTrapModel!
```

---

## B4: CLIP Dataloader — Why It's Simpler Than Stage A

**File**: `scamtrap/data/clip_dataloader.py`

### Stage A dataloader was complex because:
1. **Multi-view**: Each sample produced 2 augmented copies (for SupCon)
2. **Contrastive batch sampler**: Guaranteed >=2 samples per class per batch

### Stage B doesn't need any of that:
1. **Single view**: Each sample is tokenized once (no augmentation by default)
2. **Standard shuffled batching**: Cross-entropy works with any batch composition

```python
class CLIPScamDataset(Dataset):
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augmenter is not None:
            text = self.augmenter.apply_random(text)  # optional single augmentation
        enc = self.tokenizer(text, max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),      # [128]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [128]
            "intent_id": torch.tensor(self.intent_ids[idx]),    # scalar
        }
```

### Verify it

```python
import pandas as pd
from scamtrap.utils.config import load_config
from scamtrap.data.clip_dataloader import CLIPScamDataset

config = load_config("configs/default.yaml")
df = pd.read_parquet("data/processed/train.parquet")

ds = CLIPScamDataset(
    texts=df["text"].tolist()[:100],
    intent_ids=df["intent_id"].tolist()[:100],
    tokenizer_name="distilbert-base-uncased",
    max_length=128,
)
item = ds[0]
print(f"input_ids: {item['input_ids'].shape}")      # [128]
print(f"intent_id: {item['intent_id']}")             # scalar tensor
```

---

## B5: The Trainer — What's Different From Stage A

**File**: `scamtrap/training/clip_trainer.py`

The training loop follows the exact same pattern as Stage A's `ContrastiveTrainer`. The key difference is *what happens in each forward pass*.

### Stage A forward pass:
```
batch → model → z [B, 2_views, 128d] → SupConLoss(z, labels) → loss
```

### Stage B forward pass:
```
batch → model(batch, descriptions) → logits [B, 7] → CrossEntropy(logits, labels) → loss
```

### Label remapping

The intent_to_id mapping from Stage A might assign non-contiguous IDs when holdout intents are included. The trainer's `_remap_labels()` method converts original intent_ids to contiguous 0..K-1 indices for the cross-entropy loss:

```python
def _remap_labels(self, labels):
    """Remap original intent_ids to contiguous 0..K-1 for CE loss."""
    return torch.tensor(
        [self.id_to_contiguous[l.item()] for l in labels],
        dtype=torch.long, device=self.device,
    )
```

### Pre-tokenized descriptions

The 7 intent descriptions are tokenized once in `__init__` and stored on GPU. Every forward pass re-encodes them through the description encoder (because that encoder's weights change each step):

```python
# In __init__:
self.desc_input_ids = enc["input_ids"].to(device)       # [7, 128] — stays on GPU
self.desc_attention_mask = enc["attention_mask"].to(device)

# In train_epoch:
_, logits = self.model(msg_ids, msg_mask,
                       self.desc_input_ids, self.desc_attention_mask)
```

### Description encoder freezing

By default, the description encoder is **frozen** (`freeze_description_encoder: true` in config). This preserves DistilBERT's pretrained semantics, which are important for zero-shot generalization.

For ablation studies, finer-grained control is available:

```python
# Boolean flag (default behavior)
freeze_description_encoder: true     # Freeze entire encoder

# Layer-level control (overrides boolean when >= 0)
freeze_description_layers: 4         # Freeze bottom 4 of 6 DistilBERT layers
                                     # Top 2 layers are fine-tuned
```

The `_freeze_description_layers(encoder, n_freeze)` method freezes the bottom N transformer layers plus the embedding layer. With DistilBERT's 6 layers:
- `n_freeze=6`: fully frozen (same as boolean True)
- `n_freeze=4`: top 2 layers unfrozen
- `n_freeze=0`: fully unfrozen

### L2-SP Regularization

L2-SP (L2 Structural Penalty) penalizes deviation from pretrained weights when unfreezing layers. This prevents catastrophic forgetting:

```python
# In config:
l2sp_alpha: 0.01  # 0.0 = disabled (default)

# In training:
loss = clip_ce_loss + alpha * Σ(θ - θ₀)²
```

The trainer snapshots pretrained parameters in `__init__` and computes the L2-SP penalty each training step:

```python
def _compute_l2sp_loss(self):
    l2sp = 0.0
    for name, param in self.model.description_encoder.named_parameters():
        if name in self.pretrained_params:
            pretrained = self.pretrained_params[name].to(param.device)
            l2sp += ((param - pretrained) ** 2).sum()
    return self.l2sp_alpha * l2sp
```

This is used by the freeze ablation study (`run_freeze_ablation.py`) to compare frozen vs. unfrozen vs. unfrozen+L2-SP configurations.

---

## B6: Zero-Shot Evaluation — The Payoff

**File**: `scamtrap/evaluation/zeroshot.py`

This is why we built Stage B. The function does this:

```
1. Encode ALL 9 intent descriptions → 9 prototype vectors (256d each)
   (Including crypto + romance that were NEVER seen during training!)

2. For each test message:
   a. Encode it → 256d vector
   b. Compute cosine similarity to all 9 prototypes
   c. Predict = argmax(similarity)  ← whichever description is closest

3. Compare predictions to ground truth labels
```

### The zero-shot magic

The model was trained to push "credential_theft" messages toward the "credential_theft" description. In doing so, it learned a *general* mapping between scam messages and scam descriptions.

At test time, when we add the "crypto" description, the model can match crypto scam messages to it — even though it never saw a single crypto example during training. It learned the *concept* of matching messages to descriptions.

### What we measure

```python
return {
    "accuracy": ...,        # Overall correct classification
    "f1_macro": ...,        # F1 balanced across all intents
    "f1_weighted": ...,     # F1 weighted by support
    "classification_report": ...,  # Per-intent breakdown
}
```

The key result is the **unseen F1**: how well can the model classify crypto and romance messages using *only* their descriptions?

---

## B7: Calibration Evaluation

**File**: `scamtrap/evaluation/calibration.py`

Beyond accuracy, we also measure how **calibrated** the model's probabilities are. A model that says "80% confident" should be correct 80% of the time.

### Expected Calibration Error (ECE)

```python
def compute_ece(probs, labels, n_bins=15):
    """Bin predictions by confidence, compare to actual accuracy per bin.
    Returns dict with ece, mce, and reliability diagram data."""
```

ECE divides predictions into 15 equal-width bins by confidence, then computes the weighted average of |accuracy - confidence| per bin. A perfectly calibrated model has ECE = 0.

### Multi-class ECE

```python
def compute_multiclass_ece(probs, labels, n_bins=15):
    """ECE for multi-class: uses confidence of the predicted class.
    Also returns per_class_ece for each intent."""
```

This is used in `evaluate_clip.py` to measure calibration of zero-shot predictions — how trustworthy are the model's confidence scores when classifying unseen scam types?

MCE (Maximum Calibration Error) is also computed — the worst-case bin calibration gap.

---

## B8 & B9: Running Stage B

### Train (B8)

```bash
cd code/
python scripts/train_clip.py --config configs/default.yaml
# or: make train-clip
```

What this does:
1. Loads your processed data (same parquet files from Stage A)
2. Builds `CLIPScamModel` — two DistilBERT encoders + projections
3. Warm-starts the message encoder from your Stage A checkpoint
4. Freezes the description encoder (default) or applies layer-level control
5. Trains with CLIP CE loss (+ optional L2-SP regularization) for 15 epochs
6. Saves to `checkpoints/clip/best_model.pt`

The `--augment` flag enables single-view augmentation during training (for ablation studies).

### Evaluate (B9)

```bash
python scripts/evaluate_clip.py --config configs/default.yaml
# or: make evaluate-clip
```

What this does:
1. Loads the trained CLIP model
2. Runs ALL Stage A evaluations (fewshot, retrieval, clustering, etc.)
   - Uses `model.get_embeddings()` → 768d — same interface!
3. Runs zero-shot evaluation on seen and unseen intents
4. Computes **calibration** (ECE/MCE) for zero-shot predictions
5. Saves to `results/clip/clip_results.json`

### What to look for in results

```
--- Zero-Shot Evaluation ---
  Seen - Accuracy: ~0.70, F1-macro: ~0.55     ← decent on known intents
  Unseen - Accuracy: ~0.45, F1-macro: ~0.40   ← can detect crypto/romance!

--- Calibration ---
  ECE: ~0.10  (lower is better)
  MCE: ~0.25  (worst bin gap)
```

Even 40% F1 on unseen intents is remarkable — the model never saw a single crypto or romance example during training.

---

# STAGE C: World Model for Scam Trajectory Prediction

## The Core Idea

Stage A and B look at individual messages in isolation. But real scams are **conversations** — they unfold over time through predictable stages:

```
Turn 1:  "Hello, this is Agent Smith from the IRS"     → Stage 0: Hook
Turn 3:  "We've detected suspicious activity"          → Stage 1: Trust Building
Turn 5:  "You must act immediately"                    → Stage 2: Urgency
Turn 7:  "Please provide your Social Security number"  → Stage 3: Info Request
Turn 9:  "Send payment via gift cards"                 → Stage 4: Payment Attempt
Turn 11: "If you don't pay, you'll be arrested"        → Stage 5: Escalation
```

Stage C builds a **world model** — a neural network that reads the conversation history and predicts:
1. **What stage comes next?** (stage prediction)
2. **Will the scam escalate to a payment request?** (escalation forecasting)

This lets us warn victims *before* the payment request happens.

---

## C1: Loading Conversations

**File**: `scamtrap/data/conversation_loader.py`

We use a HuggingFace dataset of multi-turn scam conversations. Each conversation is a single string with speaker labels:

```
"Innocent: Hello? Suspect: Hi, this is the IRS. Innocent: Oh really?"
```

`parse_turns()` splits this into structured turns:

```python
turns = parse_turns(dialogue)
# [
#   {"speaker": "innocent", "text": "Hello?"},
#   {"speaker": "suspect",  "text": "Hi, this is the IRS."},
#   {"speaker": "innocent", "text": "Oh really?"},
# ]
```

### Verify it

```python
from scamtrap.data.conversation_loader import parse_turns

dialogue = "Innocent: Hello? Suspect: Hi, this is the IRS. Innocent: Really?"
turns = parse_turns(dialogue)
for t in turns:
    print(f"  [{t['speaker']}] {t['text']}")
# [innocent] Hello?
# [suspect] Hi, this is the IRS.
# [innocent] Really?
```

---

## C2: Stage Labeling — How We Create Ground Truth

**File**: `scamtrap/data/stage_labeler.py`

This is weak supervision (just like `intent_labeler.py` in Stage A), but for conversation *stages* instead of message *intents*.

### The 6-stage lifecycle

```
Stage 0: Hook              — "Hello, this is calling from..."
Stage 1: Trust Building     — "For your safety, we noticed..."
Stage 2: Urgency           — "You must act immediately..."
Stage 3: Info Request      — "Please provide your SSN..."
Stage 4: Payment Attempt   — "Send gift cards to..."
Stage 5: Escalation        — "You'll be arrested if..."
```

### How labeling works: keyword + position

Each stage has a set of keywords AND a position weight. The position weight captures the intuition that certain stages are more likely at certain points in the conversation:

```python
# Stage 0 (Hook) — most likely at the start
"position_weight": lambda pos: 1.0 if pos < 0.2 else 0.3

# Stage 4 (Payment) — most likely in the second half
"position_weight": lambda pos: 1.0 if pos > 0.4 else 0.3
```

For each turn, we:
1. Count keyword matches for each stage
2. Multiply by position weight
3. Pick the highest-scoring stage
4. If no keywords match, assign by position alone (first 20% → Hook, etc.)

### Three labeling modes

The `mode` parameter controls how keyword and position signals combine:

```python
labeler.label_turns(turns, is_scam=True, mode="hybrid")       # default: keywords × position
labeler.label_turns(turns, is_scam=True, mode="keyword_only")  # ignore position weights
labeler.label_turns(turns, is_scam=True, mode="position_only") # keywords gate, position weights score
```

These modes are used by `analyze_stage_labeler.py` to ablate how much each signal contributes.

### Soft monotonicity smoothing

Real scams don't jump backward — a scammer at the "payment" stage doesn't go back to "hook." We enforce this softly:

```python
# Don't allow more than 1-step backward
if stages[i] < smoothed[-1] - 1:
    smoothed.append(smoothed[-1])  # Keep previous stage
else:
    smoothed.append(stages[i])     # Allow current (even if 1 step back)
```

### Verify it

```python
from scamtrap.data.stage_labeler import ScamStageLabeler, STAGE_NAMES

labeler = ScamStageLabeler()
turns = [
    {"speaker": "suspect", "text": "Hello, this is calling from the IRS"},
    {"speaker": "suspect", "text": "We noticed suspicious activity on your account"},
    {"speaker": "suspect", "text": "You need to act immediately"},
    {"speaker": "suspect", "text": "Please provide your social security number"},
    {"speaker": "suspect", "text": "Send payment via gift card"},
]
stages = labeler.label_turns(turns, is_scam=True)
for turn, stage in zip(turns, stages):
    print(f"  Stage {stage} ({STAGE_NAMES[stage]}): {turn['text'][:50]}")
# Should show stages generally increasing from 0 toward 4-5
```

---

## C3: Trajectory Dataset — Pre-computed Embeddings

**File**: `scamtrap/data/trajectory_dataset.py`

### Why pre-compute?

The world model (GRU) is tiny and fast. DistilBERT is large and slow. If we ran DistilBERT inside the training loop, each epoch would take 10x longer. Instead:

```
Offline (once):
  For each conversation:
    For each turn:
      turn_text → DistilBERT → 768d embedding
    Save [T, 768] array to disk

Training (fast, many epochs):
  Load pre-computed [T, 768] arrays
  Feed through GRU → predictions
  No DistilBERT at training time!
```

### Padding and masking

Conversations have different lengths (5-30 turns). We pad all to `max_turns=30`:

```python
# Embeddings: pad with zeros
padded_emb = np.zeros((30, 768))
padded_emb[:T] = actual_embeddings[:T]

# Stage labels: pad with -1 (ignored by CrossEntropyLoss)
padded_stages = np.full(30, -1)
padded_stages[:T] = actual_stages[:T]

# Mask: True for real turns, False for padding
mask = np.zeros(30, dtype=bool)
mask[:T] = True
```

### Escalation labels

For each timestep, the escalation label is 1 if the scam has reached stage >= 4 at that point or any future point. This is the target for the escalation prediction head:

```python
escalation = np.zeros(30)
escalated = False
for t in range(T):
    if stages[t] >= 4:
        escalated = True
    if escalated:
        escalation[t] = 1.0

# Example: stages = [0, 1, 2, 3, 4, 5]
# escalation =      [0, 0, 0, 0, 1, 1]
```

---

## C4: Preparing Conversations

**File**: `scripts/prepare_conversations.py`

This is the data pipeline script. Run it once, then training is fast.

```bash
python scripts/prepare_conversations.py --config configs/default.yaml
# or: make prepare-conversations
```

What it does:
1. Downloads `BothBosu/multi-agent-scam-conversation` from HuggingFace
2. Parses each dialogue into turns
3. Labels each turn with a scam stage (0-5)
4. Loads your trained encoder (prefers Stage B > Stage A > pretrained)
5. Encodes every turn into a 768d embedding
6. Splits HuggingFace train (80/20) into our train/val, keeps HF test as test
7. Saves embeddings (`.npz`) and metadata (`.json`) to `data/processed/conversations/`

### Output structure

```
data/processed/conversations/
├── embeddings.npz    # All trajectory embeddings (compressed numpy)
└── metadata.json     # Stages, splits, scam labels per trajectory
```

---

## C5: The World Model — GRU Architecture

**File**: `scamtrap/models/world_model.py`

### Why a GRU?

Conversations are sequential — each turn depends on what came before. A GRU (Gated Recurrent Unit) is perfect for this:
- It processes turns left-to-right
- Its hidden state accumulates context from earlier turns
- It's lightweight (training takes minutes, not hours)

### Architecture

```
Input: [B, T, 768] pre-computed embeddings (frozen, from DistilBERT)
         ↓
  input_proj: Linear(768 → 256)     ← compress to GRU input size
         ↓
  GRU: 2 layers, hidden_dim=256     ← sequential processing
         ↓
  gru_out: [B, T, 256]              ← hidden state at each turn
      ↓                     ↓
  stage_head             escalation_head
  Linear(256→256)→ReLU   Linear(256→128)→ReLU
  Linear(256→6)          Linear(128→1)
      ↓                     ↓
  [B, T, 6]              [B, T, 1]
  stage logits            escalation logit
```

### Two prediction heads

**Stage head**: At each turn t, predict which of the 6 stages the conversation is currently in. This is a 6-way classification task.

**Escalation head**: At each turn t, predict the probability that the conversation will reach stage 4+ (payment/escalation). This is a binary prediction — the key output for early warning.

### Packed sequences for efficiency

Conversations have different lengths. Instead of wasting compute on padded zeros, we use PyTorch's `pack_padded_sequence`:

```python
if lengths is not None:
    x = nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False)
gru_out, _ = self.gru(x)
if lengths is not None:
    gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
```

### Transformer ablation

We also provide `TransformerWorldModel` with the same interface, for ablation studies. It uses causal masking (each turn can only attend to past turns) — important because at prediction time, you can't see the future.

### Verify it

```python
import torch
from scamtrap.utils.config import load_config
from scamtrap.models.world_model import ScamWorldModel

config = load_config("configs/default.yaml")
model = ScamWorldModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# ~600K — tiny compared to DistilBERT's 66M

embs = torch.randn(4, 15, 768)  # 4 conversations, 15 turns each
lengths = torch.tensor([15, 10, 8, 12])

stage_logits, esc_logits, hidden = model(embs, lengths)
print(f"Stage logits: {stage_logits.shape}")      # [4, 15, 6]
print(f"Escalation logits: {esc_logits.shape}")    # [4, 15, 1]
print(f"Hidden states: {hidden.shape}")            # [4, 15, 256]
```

---

## C6: Combined Loss

**File**: `scamtrap/losses/world_model_loss.py`

Two tasks, one loss:

```python
loss = alpha * stage_CE + (1 - alpha) * escalation_BCE
```

- **Stage CE**: Cross-entropy with `ignore_index=-1` (padded timesteps get -1)
- **Escalation BCE**: Binary cross-entropy, masked to valid timesteps only
- **alpha=0.7**: Stage prediction gets more weight (it's the harder task)

```python
class WorldModelLoss(nn.Module):
    def forward(self, stage_logits, escalation_logits,
                stage_labels, escalation_labels, mask):
        # Stage: reshape [B,T,6] → [B*T, 6] for CE
        sl = self.stage_loss(stage_logits.view(-1, C), stage_labels.view(-1))

        # Escalation: mask out padding
        el = self.escalation_loss(escalation_logits.squeeze(-1), escalation_labels)
        el = (el * mask.float()).sum() / mask.float().sum().clamp(min=1)

        return 0.7 * sl + 0.3 * el
```

---

## C7: Training the World Model

**File**: `scamtrap/training/world_model_trainer.py`

Same pattern as Stages A and B trainers, but simpler:
- No AMP (model is tiny, FP32 is fine)
- No gradient accumulation (small model, small batches)
- Higher learning rate (1e-3 vs 2e-5) because we're training a small GRU, not fine-tuning a pretrained transformer
- More epochs (50 vs 15-20) because the model is small and data is limited

```bash
python scripts/train_world_model.py --config configs/default.yaml
# or: make train-world-model
```

---

## C8: Trajectory Evaluation

**File**: `scamtrap/evaluation/trajectory.py`

Four evaluation functions:

### 1. Stage Prediction (`evaluate_stage_prediction`)

For every turn in every test conversation, compare predicted stage to ground truth:

```
Metrics:
  - accuracy: % of turns correctly classified
  - f1_macro: F1 balanced across all 6 stages
  - classification_report: per-stage precision/recall/F1
```

### 2. Escalation Forecast (`evaluate_escalation_forecast`)

At every turn, check if the model correctly predicts whether escalation will happen:

```
Metrics:
  - AUROC: how well the probability ranking separates escalating from non-escalating
  - Brier score: calibration — how close probabilities are to 0/1
  - accuracy: at threshold 0.5
  - precision, recall
```

### 3. Early Warning (`evaluate_early_warning`)

The most practically important metric. For each conversation that *does* escalate:
- Find the turn where escalation actually happens
- Find the turn where the model first predicts escalation (prob > 0.5)
- If the model predicted >= 3 turns before the actual escalation, that's a successful early warning

```
Metrics:
  - early_detection_rate: % of escalations predicted >= 3 turns early
  - avg_lead_time: average number of turns of advance warning
```

### 4. Early Warning Sweep (`evaluate_early_warning_sweep`)

A more comprehensive evaluation that tests multiple horizons and probability thresholds:

```python
evaluate_early_warning_sweep(
    model, test_loader,
    horizons=[1, 2, 3, 5, 7],        # lead-time requirements
    thresholds=[0.3, 0.5, 0.7],      # prediction confidence thresholds
)
```

For each horizon/threshold combination, it computes precision, recall, F1, and reports the best threshold. This produces the early warning table in the paper, showing detection rates at different lead-time requirements.

Returns per-horizon results including:
- Best threshold (by F1)
- Lead-time percentiles (P25, P50, P75)
- Per-threshold breakdown with detected/missed/false alarm counts

---

## C9: Counterfactual Analysis

**File**: `scamtrap/evaluation/counterfactual.py`

This answers: **"If we interrupted the scam at turn t, how much would escalation probability drop?"**

The function:
1. Runs the conversation through the GRU normally → get escalation probability
2. Zeros out all embeddings from turn t onward (simulating intervention)
3. Runs the truncated conversation → get new escalation probability
4. Reports the delta (drop in probability)

This provides actionable insight for the paper: "Intervention at turn 5 reduces escalation probability by X%."

---

## C10: Calibration for Stage C

**File**: `scamtrap/evaluation/calibration.py` (shared with Stage B)

The same ECE/MCE calibration module is used for Stage C:
- **Escalation calibration**: `compute_ece()` on escalation probabilities — are the model's escalation risk scores trustworthy?
- **Stage prediction calibration**: `compute_multiclass_ece()` on stage softmax — how well-calibrated are per-stage confidences?

Both are computed in `evaluate_world_model.py` and saved in the results JSON.

---

## C11: Baselines

**File**: `scamtrap/baselines/trajectory_baselines.py`

Two baselines to compare against the GRU:

### Markov Chain
- Learns a 6×6 transition matrix from training data
- Predicts next stage = most likely transition from current stage
- No embeddings, no neural network — just counting transitions
- **Limitation**: Ignores the actual content of each turn

### Logistic Regression (Single Turn)
- Trains a classifier on individual turn embeddings
- Predicts stage and escalation from a single turn in isolation
- **Limitation**: No conversation history — can't see patterns across turns

The GRU should beat both because it combines embedding content (unlike Markov) with sequential history (unlike LogReg).

---

## C12 & C13: Running Stage C

### Prepare data (C12)

```bash
python scripts/prepare_conversations.py --config configs/default.yaml
# Downloads conversations, labels stages, pre-computes embeddings
# Output: data/processed/conversations/
```

### Train (C12)

```bash
python scripts/train_world_model.py --config configs/default.yaml
# Trains GRU on pre-computed embeddings (fast — minutes on CPU)
# Output: checkpoints/world_model/best_model.pt
```

For the Transformer ablation:
```bash
python scripts/train_world_model.py --config configs/default.yaml --model-type transformer
```

### Evaluate (C13)

```bash
python scripts/evaluate_world_model.py --config configs/default.yaml
# Runs: stage prediction, escalation forecast, early warning,
#        early warning sweep, counterfactual analysis,
#        calibration (ECE/MCE), Markov + LogReg baselines
# Output: results/world_model/world_model_results.json
```

---

# Ablation Studies

These scripts are used to produce specific results tables in the paper.

## Freeze Ablation — Description Encoder Study

**File**: `scripts/run_freeze_ablation.py`

Tests 7 configurations of the description encoder in Stage B:

| Configuration | Description |
|---------------|-------------|
| `fully_frozen` | All layers frozen (default) |
| `top2_unfrozen` | Bottom 4 layers frozen, top 2 fine-tuned |
| `top2_unfrozen_l2sp` | Same as above + L2-SP regularization |
| `top4_unfrozen` | Bottom 2 layers frozen, top 4 fine-tuned |
| `top4_unfrozen_l2sp` | Same as above + L2-SP regularization |
| `fully_unfrozen` | All layers fine-tuned |
| `fully_unfrozen_l2sp` | All layers fine-tuned + L2-SP regularization |

```bash
python scripts/run_freeze_ablation.py --config configs/default.yaml
# Output: results/freeze_ablation/freeze_ablation_results.json
```

## Holdout Sweep — Open-Set Generalization

**File**: `scripts/run_holdout_sweep.py`

Tests 4 different holdout configurations to validate that zero-shot generalization isn't specific to crypto+romance:

| Configuration | Held-out intents |
|---------------|-----------------|
| `crypto_romance` | crypto, romance (default) |
| `job_prize` | job_offer, prize_lottery |
| `bank_delivery` | bank_alert, delivery |
| `credential_generic` | credential_theft, generic_scam |

```bash
python scripts/run_holdout_sweep.py --config configs/default.yaml
# Output: results/holdout_sweep/sweep_results.json
```

Uses `get_seen_and_holdout()` from `intent_descriptions.py` to dynamically configure splits.

## Label Quality Audit

**File**: `scripts/audit_labels.py`

Validates weak supervision quality for the intent labeler (Stage A data):

```bash
python scripts/audit_labels.py --config configs/default.yaml
# Output: results/label_audit/audit_results.json
```

## Stage Labeler Analysis

**File**: `scripts/analyze_stage_labeler.py`

Compares the three labeling modes (hybrid, keyword_only, position_only) for Stage C's stage labeler. If real conversation data isn't available, generates synthetic conversations for analysis.

```bash
python scripts/analyze_stage_labeler.py --config configs/default.yaml
```

## Trajectory Robustness

**File**: `scripts/evaluate_trajectory_robustness.py`

Tests Stage C model robustness under embedding perturbation. Uses perturbation functions from `scamtrap/evaluation/robustness_trajectory.py`:
- `perturb_stage_noise()` — Gaussian noise to embeddings
- `perturb_turn_dropout()` — randomly zero out turns
- `perturb_non_monotonic()` — shuffle stage ordering
- `perturb_combined()` — all perturbations together

Compares GRU vs. Markov vs. LogReg baselines under perturbation.

```bash
python scripts/evaluate_trajectory_robustness.py --config configs/default.yaml
```

## Inference Latency Benchmark

**File**: `scripts/benchmark_latency.py`

Measures CPU inference latency and throughput for all three stages:
- Stage A: DistilBERT encoder
- Stage B: CLIP dual encoder
- Stage C: GRU world model

Reports latency percentiles (P50, P95, P99) for edge/mobile deployment analysis.

```bash
python scripts/benchmark_latency.py --config configs/default.yaml
# Output: results/benchmark/latency_results.json
```

---

# Full Pipeline — Running Everything

## End-to-end commands

```bash
cd code/

# Stage A (already done, but for reference):
make data                    # Download + preprocess + split
make train                   # Train SupCon model
make evaluate                # Evaluate Stage A

# Stage B:
make train-clip              # Train CLIP alignment model
make evaluate-clip           # Evaluate (all Stage A metrics + zero-shot + calibration)

# Stage C:
make prepare-conversations   # Download conversations + pre-compute embeddings
make train-world-model       # Train GRU world model
make evaluate-world-model    # Evaluate trajectory prediction + calibration

# Reports:
make visualize               # UMAP plots (Stage A + B)
make tables                  # LaTeX tables (all stages)

# Or run everything at once:
make all-stages
```

## What gets produced

```
results/
├── scamtrap_results.json           # Stage A results
├── clip/
│   ├── clip_results.json           # Stage B results (includes zero-shot + calibration)
│   └── embeddings/                 # CLIP embeddings for UMAP
├── world_model/
│   └── world_model_results.json    # Stage C results (+ calibration + sweep)
├── baselines/
│   └── baseline_results.json       # All Stage A baseline metrics
├── figures/
│   ├── umap_scamtrap_supcon_*.png  # Stage A UMAP plots
│   └── umap_scamtrap_clip_*.png    # Stage B UMAP plots
├── tables/
│   ├── table_comparison.tex        # Stage A vs B vs baselines
│   ├── table_zeroshot.tex          # Zero-shot results (Stage B)
│   ├── table_world_model.tex       # Trajectory prediction (Stage C)
│   ├── table_early_warning.tex     # Early warning sweep
│   ├── table_calibration.tex       # ECE/MCE for Stages B + C
│   ├── table_freeze_ablation.tex   # Description encoder study
│   ├── table_holdout_sweep.tex     # Open-set generalization
│   ├── table_robustness.tex        # Augmentation + perturbation robustness
│   ├── table_latency.tex           # Inference benchmarks
│   └── table_label_audit.tex       # Weak supervision quality
├── freeze_ablation/                # Freeze ablation results
├── holdout_sweep/                  # Holdout sweep results
├── label_audit/                    # Label quality audit
└── benchmark/                      # Latency benchmarks
```

---

# Configuration Reference

All Stage B and C settings live in `configs/default.yaml`:

```yaml
# Stage B: CLIP-style intent alignment
stage_b:
  proj_dim: 256                        # Projection dimension (256d vectors)
  initial_temperature: 0.07            # Learnable temperature init
  batch_size: 64                       # Standard batching (no contrastive sampler)
  gradient_accumulation_steps: 2       # Effective batch = 128
  epochs: 15
  lr: 2.0e-5                          # Same as Stage A (fine-tuning DistilBERT)
  warmup_ratio: 0.1
  fp16: true
  early_stopping_patience: 5
  checkpoint_dir: "checkpoints/clip"
  message_encoder_init: "stage_a"      # Warm-start from Stage A
  freeze_description_encoder: true     # Boolean: freeze entire description encoder
  freeze_description_layers: -1        # Layer-level control (-1 = use boolean flag)
                                       # 0 = fully unfrozen, 4 = top 2 unfrozen, 6 = fully frozen
  l2sp_alpha: 0.0                     # L2-SP regularization (0 = disabled)

# Stage C: World model for trajectory prediction
stage_c:
  max_turns: 30                        # Max conversation length
  num_stages: 6                        # Scam lifecycle stages (0-5)
  gru_hidden_dim: 256                  # GRU hidden dimension
  gru_layers: 2                        # Number of GRU layers
  gru_dropout: 0.2                     # Between GRU layers
  transformer_heads: 4                 # For ablation model
  transformer_layers: 2
  batch_size: 32
  epochs: 50                           # More epochs (tiny model)
  lr: 1.0e-3                          # Higher LR (small model, not pretrained)
  weight_decay: 0.01
  warmup_ratio: 0.05
  early_stopping_patience: 10
  loss_alpha: 0.7                      # 70% stage + 30% escalation
  checkpoint_dir: "checkpoints/world_model"
```

---

# File Map

## Stage B files (9 total)

| # | File | What it does |
|---|------|-------------|
| B1 | `scamtrap/data/intent_descriptions.py` | 9 text descriptions (7 seen + 2 holdout) + `get_seen_and_holdout()` |
| B2 | `scamtrap/losses/clip_ce.py` | Cross-entropy over similarity logits (+ label smoothing) |
| B3 | `scamtrap/models/clip_model.py` | Dual-encoder (message + description) + projections |
| B4 | `scamtrap/data/clip_dataloader.py` | Simple dataset — no multi-view, no contrastive sampler |
| B5 | `scamtrap/training/clip_trainer.py` | Training loop with freezing, L2-SP, pre-tokenized descriptions |
| B6 | `scamtrap/evaluation/zeroshot.py` | Classify by nearest prototype (tests crypto/romance) |
| B7 | `scamtrap/evaluation/calibration.py` | ECE/MCE calibration (shared with Stage C) |
| B8 | `scripts/train_clip.py` | Orchestration: load data, build model, train |
| B9 | `scripts/evaluate_clip.py` | All Stage A evals + zero-shot + calibration |

## Stage C files (13 total)

| # | File | What it does |
|---|------|-------------|
| C1 | `scamtrap/data/conversation_loader.py` | Load HuggingFace conversations, parse turns |
| C2 | `scamtrap/data/stage_labeler.py` | 6-stage lifecycle labeling (hybrid/keyword/position modes) |
| C3 | `scamtrap/data/trajectory_dataset.py` | PyTorch Dataset for pre-computed trajectories |
| C4 | `scripts/prepare_conversations.py` | Full pipeline: download → parse → label → encode → save |
| C5 | `scamtrap/models/world_model.py` | GRU + Transformer world models |
| C6 | `scamtrap/losses/world_model_loss.py` | Combined stage CE + escalation BCE |
| C7 | `scamtrap/training/world_model_trainer.py` | Training loop for world models |
| C8 | `scamtrap/evaluation/trajectory.py` | Stage accuracy, escalation AUROC, early warning + sweep |
| C9 | `scamtrap/evaluation/counterfactual.py` | "What-if" intervention analysis |
| C10 | `scamtrap/evaluation/calibration.py` | ECE/MCE calibration (shared with Stage B) |
| C11 | `scamtrap/baselines/trajectory_baselines.py` | Markov chain + single-turn LogReg |
| C12 | `scripts/train_world_model.py` | Train GRU/Transformer |
| C13 | `scripts/evaluate_world_model.py` | Full eval + baselines + counterfactual + calibration + sweep |

## Ablation scripts (6 total)

| File | What it does |
|------|-------------|
| `scripts/run_freeze_ablation.py` | 7 description encoder freezing configurations |
| `scripts/run_holdout_sweep.py` | 4 open-set holdout configurations |
| `scripts/audit_labels.py` | Weak supervision label quality audit |
| `scripts/analyze_stage_labeler.py` | Stage labeling mode comparison (hybrid/keyword/position) |
| `scripts/evaluate_trajectory_robustness.py` | World model robustness under perturbation |
| `scripts/benchmark_latency.py` | CPU inference latency benchmarks |

## Additional evaluation modules

| File | What it does |
|------|-------------|
| `scamtrap/evaluation/robustness_trajectory.py` | Perturbation functions for trajectory robustness testing |

## Modified files from Stage A

| File | What changed |
|------|-------------|
| `configs/default.yaml` | Added `stage_b:` and `stage_c:` sections |
| `Makefile` | Added 6 new targets + `all-stages` |
| `scripts/generate_tables.py` | Added zero-shot, world model, early warning, calibration, freeze, holdout, latency, and label audit LaTeX tables |
| `scripts/visualize.py` | Added CLIP embedding UMAP plots |
