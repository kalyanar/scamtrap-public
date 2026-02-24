"""Benchmark latency, throughput, and memory footprint on CPU."""

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer

from scamtrap.utils.config import load_config
from scamtrap.utils.seed import set_seed
from scamtrap.models.encoder import TextEncoder
from scamtrap.models.clip_model import CLIPScamModel
from scamtrap.models.world_model import ScamWorldModel


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    """Estimate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def benchmark_component(fn, n_warmup=5, n_runs=50):
    """Benchmark a callable, return latency stats in ms."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency/throughput")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-runs", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    output_dir = Path(args.output_dir or config.evaluation.results_dir) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"  # Edge/mobile simulation

    results = {"device": device, "n_runs": args.n_runs}

    # Sample text for benchmarking
    sample_text = (
        "Congratulations! You have won a free iPhone 15. "
        "Click here to claim your prize now. Offer expires in 24 hours."
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

    # --- Tokenization ---
    print("Benchmarking tokenization...")
    def tokenize_fn():
        tokenizer(
            [sample_text], max_length=config.data.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
    results["tokenization"] = benchmark_component(tokenize_fn, n_runs=args.n_runs)
    print(f"  Tokenization: {results['tokenization']['mean_ms']:.2f}ms")

    # --- Stage A Encoder ---
    print("\nBenchmarking Stage A encoder...")
    encoder = TextEncoder(
        model_name=config.model.encoder_name,
        pooling=config.model.pooling,
    ).to(device).eval()

    total_params, _ = count_params(encoder)
    results["stage_a_encoder"] = {
        "params": total_params,
        "params_M": total_params / 1e6,
        "size_mb": model_size_mb(encoder),
    }

    enc_input = tokenizer(
        [sample_text], max_length=config.data.max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    ids_1 = enc_input["input_ids"].to(device)
    mask_1 = enc_input["attention_mask"].to(device)

    for batch_size in [1, 8, 32]:
        ids_b = ids_1.repeat(batch_size, 1)
        mask_b = mask_1.repeat(batch_size, 1)

        def encode_fn(ids=ids_b, mask=mask_b):
            with torch.no_grad():
                encoder(ids, mask)

        latency = benchmark_component(encode_fn, n_runs=args.n_runs)
        throughput = batch_size / (latency["mean_ms"] / 1000)
        results["stage_a_encoder"][f"batch_{batch_size}"] = latency
        results["stage_a_encoder"][f"throughput_b{batch_size}"] = throughput
        print(f"  B={batch_size}: {latency['mean_ms']:.2f}ms, "
              f"{throughput:.1f} samples/s")

    # --- Stage B CLIP ---
    print("\nBenchmarking Stage B CLIP matching...")
    clip_model = CLIPScamModel(config).to(device).eval()

    total_params_clip, _ = count_params(clip_model)
    results["stage_b_clip"] = {
        "params": total_params_clip,
        "params_M": total_params_clip / 1e6,
        "size_mb": model_size_mb(clip_model),
    }

    # Pre-tokenize 9 intent descriptions
    from scamtrap.data.intent_descriptions import INTENT_DESCRIPTIONS
    desc_texts = [INTENT_DESCRIPTIONS[k] for k in sorted(INTENT_DESCRIPTIONS.keys())]
    desc_enc = tokenizer(
        desc_texts, max_length=config.data.max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    desc_ids = desc_enc["input_ids"].to(device)
    desc_mask = desc_enc["attention_mask"].to(device)

    for batch_size in [1, 8, 32]:
        ids_b = ids_1.repeat(batch_size, 1)
        mask_b = mask_1.repeat(batch_size, 1)

        def clip_fn(ids=ids_b, mask=mask_b):
            with torch.no_grad():
                clip_model(ids, mask, desc_ids, desc_mask)

        latency = benchmark_component(clip_fn, n_runs=args.n_runs)
        throughput = batch_size / (latency["mean_ms"] / 1000)
        results["stage_b_clip"][f"batch_{batch_size}"] = latency
        results["stage_b_clip"][f"throughput_b{batch_size}"] = throughput
        print(f"  B={batch_size}: {latency['mean_ms']:.2f}ms, "
              f"{throughput:.1f} samples/s")

    # --- Stage C GRU ---
    print("\nBenchmarking Stage C GRU step...")
    gru_model = ScamWorldModel(config).to(device).eval()

    total_params_gru, _ = count_params(gru_model)
    results["stage_c_gru"] = {
        "params": total_params_gru,
        "params_M": total_params_gru / 1e6,
        "size_mb": model_size_mb(gru_model),
    }

    # Simulate single-turn GRU step (1 turn of 768-dim embedding)
    for seq_len in [1, 5, 15, 30]:
        fake_embs = torch.randn(1, seq_len, 768, device=device)
        fake_lengths = torch.tensor([seq_len])

        def gru_fn(e=fake_embs, l=fake_lengths):
            with torch.no_grad():
                gru_model(e, l)

        latency = benchmark_component(gru_fn, n_runs=args.n_runs)
        results["stage_c_gru"][f"seq_{seq_len}"] = latency
        print(f"  T={seq_len}: {latency['mean_ms']:.2f}ms")

    # --- Summary table ---
    print("\n--- Summary ---")
    print(f"{'Component':<25} {'Params(M)':>10} {'Size(MB)':>10} "
          f"{'B=1(ms)':>10} {'B=32(ms)':>10}")
    print("-" * 70)
    for component, key in [
        ("Stage A Encoder", "stage_a_encoder"),
        ("Stage B CLIP", "stage_b_clip"),
        ("Stage C GRU", "stage_c_gru"),
    ]:
        info = results[key]
        b1 = info.get("batch_1", info.get("seq_1", {})).get("mean_ms", 0)
        b32 = info.get("batch_32", info.get("seq_30", {})).get("mean_ms", 0)
        print(f"{component:<25} {info['params_M']:>10.1f} {info['size_mb']:>10.1f} "
              f"{b1:>10.2f} {b32:>10.2f}")

    # Save
    results_path = output_dir / "latency_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved benchmark results -> {results_path}")


if __name__ == "__main__":
    main()
