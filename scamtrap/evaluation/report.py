"""Generate JSON reports and LaTeX table fragments."""

import json
from pathlib import Path


def generate_latex_table(rows: list[dict], columns: list[str], caption: str,
                         label: str) -> str:
    """Generate a LaTeX table string."""
    col_spec = "|l|" + "c|" * (len(columns) - 1)
    lines = [
        f"\\begin{{table}}[h!]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{c}}}" for c in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def generate_fewshot_table(results: dict, method_name: str = "ScamTrap") -> str:
    """Generate few-shot results LaTeX table."""
    rows = [{"Method": method_name}]
    columns = ["Method"]
    for frac in sorted(results.keys(), key=float):
        col = f"{float(frac)*100:.0f}%"
        columns.append(col)
        m = results[frac]["f1_macro"]["mean"]
        s = results[frac]["f1_macro"]["std"]
        rows[0][col] = f"{m:.3f}$\\pm${s:.3f}"
    return generate_latex_table(rows, columns, "Few-Shot Classification (F1-macro)", "tab:fewshot")


def save_report(all_results: dict, output_dir: str):
    """Save full results and LaTeX table fragments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # LaTeX tables directory
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    if "fewshot" in all_results:
        for method, res in all_results["fewshot"].items():
            tex = generate_fewshot_table(res, method)
            with open(tables_dir / f"table_fewshot_{method}.tex", "w") as f:
                f.write(tex)

    print(f"Report saved -> {output_dir}")
