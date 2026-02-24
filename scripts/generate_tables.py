"""Generate LaTeX tables from evaluation results."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scamtrap.utils.config import load_config


def format_metric(val, fmt=".3f"):
    """Format a metric value, handling dict {mean, std} or plain float."""
    if isinstance(val, dict):
        if "mean" in val:
            return f"{val['mean']:{fmt}} $\\pm$ {val['std']:{fmt}}"
        return str(val)
    elif isinstance(val, (int, float)):
        if val != val:  # NaN check
            return "-"
        return f"{val:{fmt}}"
    return str(val)


def generate_comparison_table(scamtrap_results, baseline_results, output_path,
                              clip_results=None):
    """Generate a comprehensive comparison table."""
    methods = {}

    # ScamTrap
    if scamtrap_results:
        methods["ScamTrap (SupCon)"] = scamtrap_results

    # CLIP (Stage B)
    if clip_results:
        methods["ScamTrap (CLIP)"] = clip_results

    # Baselines
    name_map = {
        "tfidf_logreg": "TF-IDF + LogReg",
        "finetuned_bert": "DistilBERT (fine-tuned)",
        "sbert_linear": "SBERT + Linear",
    }
    for key, res in baseline_results.items():
        methods[name_map.get(key, key)] = res

    lines = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\caption{Comparison of ScamTrap with Baselines}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{|l|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Method} & \\textbf{F1@1\\%} & \\textbf{F1@5\\%} & "
        "\\textbf{F1@100\\%} & \\textbf{R@5} & \\textbf{NMI} & "
        "\\textbf{Novelty AUC} \\\\",
        "\\hline",
    ]

    for name, res in methods.items():
        row = [name]
        # Few-shot F1
        fs = res.get("fewshot", {})
        for frac in ["0.01", "0.05", "1.0"]:
            if frac in fs:
                row.append(format_metric(fs[frac].get("f1_macro", "")))
            else:
                row.append("-")
        # Retrieval
        ret = res.get("retrieval", {})
        row.append(format_metric(ret.get("recall@5", "-")))
        # Clustering
        cl = res.get("clustering", {})
        row.append(format_metric(cl.get("nmi", "-")))
        # Open-set
        os_res = res.get("openset", {})
        row.append(format_metric(os_res.get("novelty_auroc", "-")))

        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved comparison table -> {output_path}")


def generate_zeroshot_table(clip_results, output_path):
    """Generate zero-shot evaluation table (Stage B specific)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Zero-Shot Intent Classification (Stage B)}",
        "\\label{tab:zeroshot}",
        "\\begin{tabular}{|l|c|c|c|}",
        "\\hline",
        "\\textbf{Split} & \\textbf{Accuracy} & "
        "\\textbf{F1-macro} & \\textbf{F1-weighted} \\\\",
        "\\hline",
    ]

    for split_key, split_name in [("zeroshot_seen", "Seen Intents"),
                                   ("zeroshot_unseen", "Unseen Intents")]:
        zs = clip_results.get(split_key, {})
        if zs:
            row = [
                split_name,
                format_metric(zs.get("accuracy", "-")),
                format_metric(zs.get("f1_macro", "-")),
                format_metric(zs.get("f1_weighted", "-")),
            ]
            lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved zero-shot table -> {output_path}")


def generate_world_model_table(wm_results, output_path):
    """Generate world model evaluation table (Stage C)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{World Model Trajectory Prediction (Stage C)}",
        "\\label{tab:worldmodel}",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Stage Acc} & "
        "\\textbf{Stage F1} & \\textbf{Esc. AUROC} & "
        "\\textbf{Esc. Brier} \\\\",
        "\\hline",
    ]

    # World model results
    sp = wm_results.get("stage_prediction", {})
    ef = wm_results.get("escalation_forecast", {})
    row = [
        "GRU World Model",
        format_metric(sp.get("accuracy", "-")),
        format_metric(sp.get("f1_macro", "-")),
        format_metric(ef.get("auroc", "-")),
        format_metric(ef.get("brier_score", "-"), fmt=".4f"),
    ]
    lines.append(" & ".join(row) + " \\\\")

    # Baselines
    baselines = wm_results.get("baselines", {})
    name_map = {
        "markov_chain": "Markov Chain",
        "logreg_single_turn": "LogReg (single turn)",
    }
    for key, res in baselines.items():
        row = [
            name_map.get(key, key),
            format_metric(res.get("stage_accuracy", "-")),
            format_metric(res.get("stage_f1_macro", "-")),
            format_metric(res.get("escalation_auroc", "-")),
            "-",
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved world model table -> {output_path}")


# ── Reviewer response tables ────────────────────────────────────


def generate_label_audit_table(audit_results, output_path):
    """Table: Weak Supervision Label Quality (Item 1)."""
    per_intent = audit_results.get("per_intent", {})
    if not per_intent:
        return

    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Weak Supervision Label Quality}",
        "\\label{tab:label_audit}",
        "\\begin{tabular}{|l|r|r|r|r|l|}",
        "\\hline",
        "\\textbf{Intent} & \\textbf{N} & \\textbf{Frac(\\%)} & "
        "\\textbf{Confidence} & \\textbf{Ambig(\\%)} & "
        "\\textbf{Top Confusion} \\\\",
        "\\hline",
    ]

    for intent, info in per_intent.items():
        top_conf = info["top_confusions"][0]["intent"] if info.get("top_confusions") else "-"
        row = [
            intent.replace("_", "\\_"),
            str(info["count"]),
            f"{info['fraction']*100:.1f}",
            f"{info['mean_confidence']:.2f}",
            f"{info['ambiguity_rate']*100:.1f}",
            top_conf.replace("_", "\\_"),
        ]
        lines.append(" & ".join(row) + " \\\\")

    overall = audit_results.get("overall", {})
    lines.append("\\hline")
    lines.append(
        f"\\textbf{{Overall}} & {overall.get('n_scam_samples', '-')} & "
        f"- & {overall.get('mean_confidence', 0):.2f} & "
        f"{overall.get('ambiguity_rate', 0)*100:.1f} & "
        f"coverage: {overall.get('coverage_rate', 0)*100:.1f}\\% \\\\"
    )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved label audit table -> {output_path}")


def generate_freeze_ablation_table(results, output_path):
    """Table: Description Encoder Freeze Ablation (Item 7)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Description Encoder Freeze Ablation (Stage B)}",
        "\\label{tab:freeze_ablation}",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Config} & \\textbf{Freeze} & "
        "\\textbf{Seen ZS Acc} & \\textbf{Unseen ZS Acc} & "
        "\\textbf{Novelty AUROC} \\\\",
        "\\hline",
    ]

    for name, res in results.items():
        if name.startswith("_"):
            continue
        cfg = res.get("config", {})
        row = [
            name.replace("_", "\\_"),
            str(cfg.get("freeze_layers", "-")),
            format_metric(res.get("zeroshot_seen_accuracy", "-")),
            format_metric(res.get("zeroshot_unseen_accuracy", "-")),
            format_metric(res.get("novelty_auroc", "-")),
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved freeze ablation table -> {output_path}")


def generate_holdout_sweep_table(results, output_path):
    """Table: Open-Set Holdout Sweep (Item 8)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Open-Set Holdout Sweep (Stage B)}",
        "\\label{tab:holdout_sweep}",
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Holdout} & \\textbf{Seen Acc} & "
        "\\textbf{Unseen Acc} & \\textbf{Unseen F1} & "
        "\\textbf{AUROC} \\\\",
        "\\hline",
    ]

    for name, res in results.items():
        if name.startswith("_") or res.get("skipped"):
            continue
        holdout_str = ", ".join(res.get("holdout", [name.split("_")]))
        row = [
            holdout_str.replace("_", "\\_"),
            format_metric(res.get("seen_accuracy", "-")),
            format_metric(res.get("unseen_accuracy", "-")),
            format_metric(res.get("unseen_f1", "-")),
            format_metric(res.get("novelty_auroc", "-")),
        ]
        lines.append(" & ".join(row) + " \\\\")

    # Mean±std
    summary = results.get("_summary", {})
    if summary:
        m = summary.get("mean", {})
        s = summary.get("std", {})
        lines.append("\\hline")
        row = [
            "\\textbf{Mean $\\pm$ std}",
            f"{m.get('seen_accuracy',0):.3f} $\\pm$ {s.get('seen_accuracy',0):.3f}",
            f"{m.get('unseen_accuracy',0):.3f} $\\pm$ {s.get('unseen_accuracy',0):.3f}",
            f"{m.get('unseen_f1',0):.3f} $\\pm$ {s.get('unseen_f1',0):.3f}",
            f"{m.get('novelty_auroc',0):.3f} $\\pm$ {s.get('novelty_auroc',0):.3f}",
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved holdout sweep table -> {output_path}")


def generate_robustness_table(results, output_path):
    """Table: Trajectory Robustness Under Perturbation (Item 3)."""
    lines = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\caption{Trajectory Robustness Under Perturbation (Stage C)}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{|l|c|cc|cc|cc|}",
        "\\hline",
        " & & \\multicolumn{2}{c|}{\\textbf{GRU}} & "
        "\\multicolumn{2}{c|}{\\textbf{Markov}} & "
        "\\multicolumn{2}{c|}{\\textbf{LogReg}} \\\\",
        "\\textbf{Perturbation} & \\textbf{Level} & "
        "\\textbf{Acc} & \\textbf{AUROC} & "
        "\\textbf{Acc} & \\textbf{AUROC} & "
        "\\textbf{Acc} & \\textbf{AUROC} \\\\",
        "\\hline",
    ]

    for ptype in ["stage_noise", "turn_dropout", "non_monotonic", "combined"]:
        levels = results.get(ptype, {})
        for level_str in ["0.0", "0.1", "0.2", "0.3", "0.5"]:
            data = levels.get(level_str, {})
            if not data:
                continue
            gru = data.get("gru", {})
            mk = data.get("markov", {})
            lr = data.get("logreg", {})
            row = [
                ptype.replace("_", "\\_") if level_str == "0.0" else "",
                level_str,
                f"{gru.get('stage_acc_mean', 0):.3f}",
                f"{gru.get('esc_auroc_mean', 0):.3f}",
                f"{mk.get('stage_acc_mean', 0):.3f}",
                f"{mk.get('esc_auroc_mean', 0):.3f}",
                f"{lr.get('stage_acc_mean', 0):.3f}",
                f"{lr.get('esc_auroc_mean', 0):.3f}",
            ]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")

    lines.extend(["\\end{tabular}", "\\end{table*}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved robustness table -> {output_path}")


def generate_early_warning_table(wm_results, output_path):
    """Table: Early Warning Sweep (Item 4)."""
    sweep = wm_results.get("early_warning_sweep", {})
    if not sweep:
        return

    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Early Warning Detection by Horizon (Stage C)}",
        "\\label{tab:early_warning}",
        "\\begin{tabular}{|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Horizon} & \\textbf{Threshold} & "
        "\\textbf{Precision} & \\textbf{Recall} & "
        "\\textbf{F1} & \\textbf{Med. Lead} \\\\",
        "\\hline",
    ]

    for h in ["1", "2", "3", "5", "7"]:
        info = sweep.get(h, {})
        if not info:
            continue
        row = [
            h,
            format_metric(info.get("best_threshold", "-")),
            format_metric(info.get("precision", "-")),
            format_metric(info.get("recall", "-")),
            format_metric(info.get("f1", "-")),
            format_metric(info.get("lead_time_p50", "-"), fmt=".1f"),
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved early warning table -> {output_path}")


def generate_calibration_table(clip_results, wm_results, output_path):
    """Table: Calibration (ECE) across stages (Item 5)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Calibration: Expected Calibration Error (ECE)}",
        "\\label{tab:calibration}",
        "\\begin{tabular}{|l|c|c|}",
        "\\hline",
        "\\textbf{Component} & \\textbf{ECE} & \\textbf{MCE} \\\\",
        "\\hline",
    ]

    # Stage B
    cal_b = clip_results.get("calibration_test_seen", {})
    if cal_b:
        lines.append(f"Stage B Zero-Shot (seen) & "
                     f"{format_metric(cal_b.get('ece', '-'), fmt='.4f')} & "
                     f"{format_metric(cal_b.get('mce', '-'), fmt='.4f')} \\\\")

    # Stage C - escalation
    cal_esc = wm_results.get("calibration_escalation", {})
    if cal_esc:
        lines.append(f"Stage C Escalation & "
                     f"{format_metric(cal_esc.get('ece', '-'), fmt='.4f')} & "
                     f"{format_metric(cal_esc.get('mce', '-'), fmt='.4f')} \\\\")

    # Stage C - stage prediction
    cal_st = wm_results.get("calibration_stage", {})
    if cal_st:
        lines.append(f"Stage C Stage Pred. & "
                     f"{format_metric(cal_st.get('ece', '-'), fmt='.4f')} & "
                     f"{format_metric(cal_st.get('mce', '-'), fmt='.4f')} \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved calibration table -> {output_path}")


def generate_latency_table(bench_results, output_path):
    """Table: Latency and Throughput (Item 6)."""
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Inference Latency and Model Size (CPU)}",
        "\\label{tab:latency}",
        "\\begin{tabular}{|l|r|r|r|r|r|}",
        "\\hline",
        "\\textbf{Component} & \\textbf{Params(M)} & \\textbf{Size(MB)} & "
        "\\textbf{B=1(ms)} & \\textbf{B=32(ms)} & \\textbf{Thru.(s/s)} \\\\",
        "\\hline",
    ]

    for component, key in [
        ("Stage A Encoder", "stage_a_encoder"),
        ("Stage B CLIP", "stage_b_clip"),
        ("Stage C GRU", "stage_c_gru"),
    ]:
        info = bench_results.get(key, {})
        if not info:
            continue
        b1 = info.get("batch_1", info.get("seq_1", {})).get("mean_ms", 0)
        b32 = info.get("batch_32", info.get("seq_30", {})).get("mean_ms", 0)
        thru = info.get("throughput_b32", info.get("throughput_b1", 0))
        row = [
            component,
            f"{info.get('params_M', 0):.1f}",
            f"{info.get('size_mb', 0):.1f}",
            f"{b1:.1f}",
            f"{b32:.1f}",
            f"{thru:.0f}",
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved latency table -> {output_path}")


def load_json_if_exists(path):
    """Load JSON file if it exists, else return empty dict."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--baselines-dir", type=str, default="results/baselines")
    parser.add_argument("--output-dir", type=str, default="results/tables")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir or config.evaluation.results_dir).resolve()
    baselines_dir = Path(args.baselines_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all result files ──

    scamtrap_results = load_json_if_exists(results_dir / "scamtrap_results.json")
    baseline_results = load_json_if_exists(baselines_dir / "baseline_results.json")
    clip_results = load_json_if_exists(results_dir / "clip" / "clip_results.json")
    wm_results = load_json_if_exists(results_dir / "world_model" / "world_model_results.json")

    # Reviewer response results
    audit_results = load_json_if_exists(results_dir / "label_audit" / "audit_results.json")
    freeze_results = load_json_if_exists(results_dir / "freeze_ablation" / "freeze_ablation_results.json")
    holdout_results = load_json_if_exists(results_dir / "holdout_sweep" / "sweep_results.json")
    robustness_results = load_json_if_exists(results_dir / "world_model" / "robustness_results.json")
    bench_results = load_json_if_exists(results_dir / "benchmark" / "latency_results.json")

    loaded = sum(1 for r in [scamtrap_results, baseline_results, clip_results,
                              wm_results, audit_results, freeze_results,
                              holdout_results, robustness_results, bench_results] if r)
    print(f"Loaded {loaded}/9 result files")

    # ── Original tables ──

    generate_comparison_table(
        scamtrap_results, baseline_results,
        str(output_dir / "table_comparison.tex"),
        clip_results=clip_results,
    )

    if clip_results:
        generate_zeroshot_table(
            clip_results,
            str(output_dir / "table_zeroshot.tex"),
        )

    if wm_results:
        generate_world_model_table(
            wm_results,
            str(output_dir / "table_world_model.tex"),
        )

    # ── Reviewer response tables ──

    if audit_results:
        generate_label_audit_table(
            audit_results,
            str(output_dir / "table_label_audit.tex"),
        )

    if freeze_results:
        generate_freeze_ablation_table(
            freeze_results,
            str(output_dir / "table_freeze_ablation.tex"),
        )

    if holdout_results:
        generate_holdout_sweep_table(
            holdout_results,
            str(output_dir / "table_holdout_sweep.tex"),
        )

    if robustness_results:
        generate_robustness_table(
            robustness_results,
            str(output_dir / "table_robustness.tex"),
        )

    if wm_results and wm_results.get("early_warning_sweep"):
        generate_early_warning_table(
            wm_results,
            str(output_dir / "table_early_warning.tex"),
        )

    if clip_results or wm_results:
        generate_calibration_table(
            clip_results, wm_results,
            str(output_dir / "table_calibration.tex"),
        )

    if bench_results:
        generate_latency_table(
            bench_results,
            str(output_dir / "table_latency.tex"),
        )

    print(f"\nAll tables saved to {output_dir}")


if __name__ == "__main__":
    main()
