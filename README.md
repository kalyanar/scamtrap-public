# ScamTrap — Contrastive Scam Embeddings with Zero-Shot Detection and Trajectory Prediction

Proof-of-concept implementation for the IEEE Access paper. ScamTrap uses a staged approach (A → B → C) to detect scams, generalize to unseen scam types zero-shot, and predict scam escalation before it happens.

| Stage | What it does | Key technique |
|-------|-------------|---------------|
| **A** | Learn intent-aware scam embeddings | Supervised contrastive learning (SupCon) |
| **B** | Zero-shot detection of unseen scam types | CLIP-like message-to-description alignment |
| **C** | Predict scam escalation from conversation history | GRU world model over turn embeddings |

## Quick Start

```bash
cd code/

# Install
pip install -e . && pip install -r requirements.txt

# Run the full pipeline (all three stages)
make all-stages

# Or run just Stage A
make all
```

`make all-stages` runs every step end-to-end: data preparation, training, baselines, evaluation, visualization, and LaTeX table generation across all three stages.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- GPU recommended (T4 or better). CPU works but training is slow.
- ~4 GB disk for datasets, checkpoints, and results

All Python dependencies are in `requirements.txt`:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
faiss-cpu>=1.7.4
umap-learn>=0.5.3
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
nltk>=3.8.0
```

## Setup

```bash
cd code/
pip install -e .
pip install -r requirements.txt
# or: make setup
```

This installs the `scamtrap` package in editable mode so all imports work.

## Replicating the Paper Results

### Step-by-step execution

All commands are run from the `code/` directory. Each step depends on the previous one.

```bash
# ──────────────── Stage A: Contrastive Embeddings ────────────────

# 1. Download datasets from HuggingFace, apply weak supervision intent labels,
#    create train/val/test_seen/test_unseen splits with open-set holdout
make data                       # ~2 min

# 2. Train SupCon model (DistilBERT encoder + projection head)
make train                      # ~20-30 min on T4 GPU

# 3. Train baselines (TF-IDF+LogReg, fine-tuned DistilBERT, SBERT+Linear)
make train-baselines            # ~30 min on T4 GPU

# 4. Evaluate Stage A (few-shot, open-set, retrieval, clustering, robustness)
make evaluate                   # ~5 min

# ──────────────── Stage B: Zero-Shot Intent Alignment ────────────────

# 5. Train CLIP-like dual encoder (warm-started from Stage A)
make train-clip                 # ~15-20 min on T4 GPU

# 6. Evaluate Stage B (all Stage A metrics + zero-shot on unseen intents)
make evaluate-clip              # ~5 min

# ──────────────── Stage C: World Model ────────────────

# 7. Download scam conversations, label stages, pre-compute turn embeddings
make prepare-conversations      # ~5 min

# 8. Train GRU world model on trajectory embeddings
make train-world-model          # ~5 min (tiny model, fast)

# 9. Evaluate Stage C (stage prediction, escalation forecast, early warning,
#    counterfactual analysis, Markov + LogReg baselines)
make evaluate-world-model       # ~2 min

# ──────────────── Reports ────────────────

# 10. Generate UMAP visualizations (Stage A + B embeddings)
make visualize                  # ~1 min

# 11. Generate LaTeX tables for the paper
make tables                     # instant
```

### One-command full pipeline

```bash
make all-stages
```

This runs steps 1-11 sequentially. Total wall time: ~60-90 min on a T4 GPU.

## Ablation Studies and Additional Experiments

These scripts produce results referenced in specific paper sections:

```bash
# Label quality audit — validates weak supervision accuracy
python scripts/audit_labels.py --config configs/default.yaml
# Output: results/label_audit/audit_results.json

# Freeze ablation — tests description encoder freezing strategies for Stage B
python scripts/run_freeze_ablation.py --config configs/default.yaml
# Output: results/freeze_ablation/freeze_ablation_results.json

# Holdout sweep — open-set evaluation across 4 different holdout configurations
python scripts/run_holdout_sweep.py --config configs/default.yaml
# Output: results/holdout_sweep/sweep_results.json

# Inference latency benchmark — CPU throughput for all three stages
python scripts/benchmark_latency.py --config configs/default.yaml
# Output: results/benchmark/latency_results.json

# Stage labeler analysis — quality metrics for conversation stage labels
python scripts/analyze_stage_labeler.py --config configs/default.yaml

# Trajectory robustness — embedding perturbation study for Stage C
python scripts/evaluate_trajectory_robustness.py --config configs/default.yaml
```

## Output Structure

After `make all-stages`, you'll have:

```
data/processed/
├── train.parquet                  # Training split (~70%)
├── val.parquet                    # Validation split (~15%)
├── test_seen.parquet              # Test split — seen intents (~15%)
├── test_unseen.parquet            # Test split — unseen intents (crypto + romance)
├── metadata.json                  # Dataset statistics and label maps
└── conversations/                 # Stage C data
    ├── embeddings.npz             # Pre-computed turn embeddings
    └── metadata.json              # Stages, splits, scam labels per trajectory

checkpoints/
├── best_model.pt                  # Stage A encoder (best validation loss)
├── final_model.pt                 # Stage A encoder (final epoch)
├── training_history.json          # Stage A loss curves
├── clip/
│   ├── best_model.pt              # Stage B CLIP model
│   └── training_history.json
└── world_model/
    ├── best_model.pt              # Stage C GRU model
    └── training_history.json

results/
├── scamtrap_results.json          # Stage A metrics
├── baselines/
│   └── baseline_results.json      # All baseline metrics
├── clip/
│   └── clip_results.json          # Stage B metrics (includes zero-shot)
├── world_model/
│   └── world_model_results.json   # Stage C metrics
├── embeddings/                    # .npy files for all models
├── figures/                       # UMAP PNGs + PDFs
│   ├── umap_scamtrap_supcon_*.png
│   └── umap_scamtrap_clip_*.png
└── tables/                        # LaTeX table fragments
    ├── table_comparison.tex       # Stage A vs baselines
    ├── table_zeroshot.tex         # Zero-shot results (Stage B)
    ├── table_world_model.tex      # Trajectory prediction (Stage C)
    ├── table_early_warning.tex
    ├── table_calibration.tex
    ├── table_freeze_ablation.tex
    ├── table_holdout_sweep.tex
    ├── table_robustness.tex
    ├── table_latency.tex
    └── table_label_audit.tex
```

LaTeX tables can be included directly in the paper with `\input{}`.

## Architecture Overview

### Stage A: Supervised Contrastive Learning

Replicates and extends ConRo (IEEE BigData 2023) from tabular to text-based scam detection.

```
Text Input → [DistilBERT Encoder (768d)] → [Projection MLP (128d)] → SupCon Loss
                      ↓
              Encoder output h (768d) ← used for all downstream evaluation
```

- **Training**: SupCon loss with multi-view augmentation (homoglyph, leetspeak, spacing, synonyms)
- **Evaluation**: Few-shot (1/5/10/100%), open-set generalization, retrieval, clustering, robustness
- **Open-set holdout**: Crypto and romance intents are removed entirely from training. The model never sees them until test time.
- **Baselines**: TF-IDF+LogReg, fine-tuned DistilBERT (cross-entropy), SBERT+Linear

### Stage B: CLIP-like Intent Alignment

Adds zero-shot capability — detect scam types from text descriptions alone, no examples needed.

```
Message path:                 Description path:
  "Click here to verify"        "This message impersonates a service..."
        ↓                              ↓
  [DistilBERT #1]                [DistilBERT #2]
  (warm-started from A)          (frozen or fine-tuned)
        ↓ (768d)                       ↓ (768d)
  [Message MLP → 256d]          [Description MLP → 256d]
        ↓                              ↓
     z_msg ───── cosine sim ────── z_desc  →  logits [B, K] / τ  →  CE loss
```

- **7 seen intent descriptions** used during training; **2 holdout** (crypto, romance) used only at zero-shot test time
- Message encoder warm-started from Stage A checkpoint
- Learnable temperature parameter

### Stage C: World Model for Scam Trajectory Prediction

Predicts scam escalation from conversation history using pre-computed turn embeddings.

```
Conversation turns → [Frozen Encoder → 768d per turn] → [GRU (2-layer, 256d)]
                                                              ↓
                                                    ┌─────────┴─────────┐
                                                    ↓                   ↓
                                              Stage Head          Escalation Head
                                              [B, T, 6]          [B, T, 1]
                                           (which stage?)     (will it escalate?)
```

- **6 scam stages**: Hook → Trust Building → Urgency → Info Request → Payment Attempt → Escalation
- **Data**: Multi-turn scam conversations from HuggingFace (`BothBosu/multi-agent-scam-conversation`)
- **Stage labeling**: Keyword + position-based weak supervision with monotonicity smoothing
- **Evaluation**: Stage accuracy/F1, escalation AUROC/Brier, early warning detection rate, counterfactual intervention analysis
- **Baselines**: Markov chain (transition matrix), single-turn LogReg

## Data Pipeline

### Datasets (all public, auto-downloaded from HuggingFace)

| Dataset | Used in | Content |
|---------|---------|---------|
| `ucirvine/sms_spam` | Stage A, B | 5,574 SMS messages (ham/spam) |
| `ealvaradob/phishing-dataset` | Stage A, B | Phishing text corpus |
| `BothBosu/multi-agent-scam-conversation` | Stage C | Multi-turn scam dialogues |

### Intent Labels (Weak Supervision)

Raw datasets only have binary labels (ham/scam). We assign 7 intent labels using keyword/regex rules:

| Intent | Examples |
|--------|----------|
| `credential_theft` | "verify your account", "confirm identity" |
| `delivery` | "package tracking", "delivery failed" |
| `bank_alert` | "account suspended", "unauthorized transaction" |
| `job_offer` | "work from home", "hiring" |
| `crypto` | "bitcoin", "crypto investment" (holdout) |
| `romance` | "dear love", "lonely" (holdout) |
| `prize_lottery` | "you've won", "claim your prize" |

### Splits

| Split | Contents | Used for |
|-------|----------|----------|
| Train (~70%) | No crypto, no romance | Model training |
| Val (~15%) | No crypto, no romance | Early stopping |
| Test Seen (~15%) | Same intents as train | Standard evaluation |
| Test Unseen | All crypto + romance | Open-set evaluation |

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Value | Section |
|-----------|-------|---------|
| Encoder | `distilbert-base-uncased` | `model` |
| Max length | 128 tokens | `data` |
| SupCon temperature | 0.07 | `loss` |
| Batch size | 64 (effective 256 with grad accum) | `training` |
| Learning rate | 2e-5 (Stages A, B), 1e-3 (Stage C) | `training` / `stage_c` |
| Epochs | 20 (A), 15 (B), 50 (C) | `training` / `stage_b` / `stage_c` |
| Open-set holdout | crypto, romance | `data` |
| CLIP projection dim | 256 | `stage_b` |
| GRU hidden dim | 256, 2 layers | `stage_c` |
| Loss alpha (Stage C) | 0.7 stage + 0.3 escalation | `stage_c` |

## File Map

```
code/
├── configs/
│   └── default.yaml                         # All hyperparameters
│
├── scamtrap/                                # Python package
│   ├── data/
│   │   ├── datasets.py                      # HF dataset loading + merging
│   │   ├── intent_labeler.py                # Keyword-based intent assignment
│   │   ├── splits.py                        # Train/val/test with open-set holdout
│   │   ├── augmentations.py                 # Scam-specific text augmentations
│   │   ├── dataloader.py                    # Multi-view dataset + contrastive batch sampler
│   │   ├── intent_descriptions.py           # 9 intent descriptions for Stage B
│   │   ├── clip_dataloader.py               # Simple dataset for Stage B (no multi-view)
│   │   ├── conversation_loader.py           # Parse HF scam conversations into turns
│   │   ├── stage_labeler.py                 # 6-stage lifecycle labeling
│   │   └── trajectory_dataset.py            # PyTorch Dataset for pre-computed trajectories
│   │
│   ├── models/
│   │   ├── encoder.py                       # DistilBERT wrapper (reused in all stages)
│   │   ├── projection.py                    # MLP projection head
│   │   ├── scamtrap_model.py                # Stage A: encoder + projection + multi-view
│   │   ├── clip_model.py                    # Stage B: dual encoder + learnable temperature
│   │   └── world_model.py                   # Stage C: GRU + Transformer world models
│   │
│   ├── losses/
│   │   ├── supcon.py                        # Supervised contrastive loss (Stage A)
│   │   ├── clip_ce.py                       # Cross-entropy over similarity logits (Stage B)
│   │   └── world_model_loss.py              # Combined stage CE + escalation BCE (Stage C)
│   │
│   ├── training/
│   │   ├── trainer.py                       # Stage A: AMP, grad accum, early stopping
│   │   ├── clip_trainer.py                  # Stage B: pre-tokenized descriptions
│   │   └── world_model_trainer.py           # Stage C: higher LR, more epochs
│   │
│   ├── evaluation/
│   │   ├── fewshot.py                       # Few-shot linear probe (1/5/10/100%)
│   │   ├── openset.py                       # Seen accuracy + unseen clustering + novelty AUROC
│   │   ├── retrieval.py                     # Recall@1,5,10,20
│   │   ├── clustering.py                    # K-means NMI/ARI/silhouette
│   │   ├── robustness.py                    # Augmentation robustness
│   │   ├── visualization.py                 # UMAP scatter plots
│   │   ├── zeroshot.py                      # Zero-shot classify by nearest prototype (Stage B)
│   │   ├── trajectory.py                    # Stage prediction, escalation, early warning (Stage C)
│   │   ├── counterfactual.py                # Intervention impact analysis (Stage C)
│   │   ├── calibration.py                   # ECE/MCE for calibration (Stages B, C)
│   │   └── report.py                        # JSON results → LaTeX table fragments
│   │
│   ├── baselines/
│   │   ├── tfidf_logreg.py                  # TF-IDF + Logistic Regression
│   │   ├── finetuned_bert.py                # DistilBERT + cross-entropy
│   │   ├── sbert_linear.py                  # Frozen SBERT + linear head
│   │   └── trajectory_baselines.py          # Markov chain + single-turn LogReg (Stage C)
│   │
│   └── utils/
│       ├── config.py                        # YAML config with dot-access
│       ├── seed.py                          # Reproducibility (seed=42)
│       └── io.py                            # Checkpoint save/load, JSON I/O
│
├── scripts/
│   ├── prepare_data.py                      # Step 1: download → preprocess → split
│   ├── train.py                             # Step 2: train SupCon model
│   ├── train_baselines.py                   # Step 3: train all baselines
│   ├── evaluate.py                          # Step 4: evaluate Stage A
│   ├── train_clip.py                        # Step 5: train CLIP model
│   ├── evaluate_clip.py                     # Step 6: evaluate Stage B
│   ├── prepare_conversations.py             # Step 7: prepare conversation data
│   ├── train_world_model.py                 # Step 8: train world model
│   ├── evaluate_world_model.py              # Step 9: evaluate Stage C
│   ├── visualize.py                         # Step 10: UMAP plots
│   ├── generate_tables.py                   # Step 11: LaTeX tables
│   ├── audit_labels.py                      # Ablation: label quality audit
│   ├── run_freeze_ablation.py               # Ablation: description encoder freeze study
│   ├── run_holdout_sweep.py                 # Ablation: open-set holdout configs
│   ├── benchmark_latency.py                 # Ablation: inference latency
│   ├── analyze_stage_labeler.py             # Ablation: stage labeling quality
│   └── evaluate_trajectory_robustness.py    # Ablation: trajectory perturbation
│
├── Makefile                                 # Task automation
├── setup.py                                 # Package install
├── requirements.txt                         # Python dependencies
└── GUIDE_STAGES_B_AND_C.md                  # Detailed walkthrough of Stages B and C
```

## Makefile Targets

| Target | What it does |
|--------|-------------|
| `make setup` | Install package + dependencies |
| `make data` | Download and preprocess datasets |
| `make train` | Train Stage A SupCon model |
| `make train-baselines` | Train all three baselines |
| `make evaluate` | Evaluate Stage A |
| `make train-clip` | Train Stage B CLIP model |
| `make evaluate-clip` | Evaluate Stage B (+ zero-shot) |
| `make prepare-conversations` | Prepare Stage C conversation data |
| `make train-world-model` | Train Stage C GRU model |
| `make evaluate-world-model` | Evaluate Stage C |
| `make visualize` | Generate UMAP plots |
| `make tables` | Generate LaTeX tables |
| `make all` | Stage A only (data → train → baselines → evaluate → visualize → tables) |
| `make all-stages` | Full pipeline (all three stages + reports) |
| `make clean` | Delete all generated data, checkpoints, and results |

## Validating Paper Claims

The evaluation validates these paper sections:

| Paper Section | What's validated | Script |
|---------------|-----------------|--------|
| Section 4.2 | Contrastive embedding quality (few-shot, retrieval, clustering) | `evaluate.py` |
| Section 4.3 | Open-set generalization to unseen scam types | `evaluate.py` |
| Section 5.1 | Zero-shot detection via intent descriptions | `evaluate_clip.py` |
| Section 5.2 | Description encoder freeze ablation | `run_freeze_ablation.py` |
| Section 6.1 | Scam stage prediction accuracy | `evaluate_world_model.py` |
| Section 6.2 | Escalation forecasting (AUROC, early warning) | `evaluate_world_model.py` |
| Section 6.3 | Counterfactual intervention analysis | `evaluate_world_model.py` |
| Section 7 | Comparison against baselines across all stages | `train_baselines.py`, `evaluate_world_model.py` |

## Running on Google Cloud (GPU)

```bash
# 1. Create a VM with a T4 GPU (~$0.35/hr) or use Colab
# 2. SSH in and clone the repo

cd scamtrap/code
make setup
make all-stages    # Full replication (~60-90 min on T4)
```

Results will be in `results/` — JSON metrics files and LaTeX tables ready for the paper.

## Key References

- **ConRo**: Vinay M.S. et al., "Robust Fraud Detection via Supervised Contrastive Learning", IEEE BigData 2023
- **SupCon Loss**: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **AI-in-the-Loop** (comparison target): Hossain et al., arXiv 2509.05362, 2025
