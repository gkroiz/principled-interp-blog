"""Create visualization of metrics across steps."""

import json
import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))
import config


def main():
    """Create a plot of metric vs step number for the configured run."""

    run_id = config.RUN_TO_PROCESS.replace("state-", "")
    # Extract just the run name (e.g., "run1" from "gpt5/run1")
    run_name = Path(run_id).name
    summary_file = config.SUMMARY_DIR / f"{run_name}_metrics.json"

    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        print("Run compute_anchor_metrics.py first!")
        return

    with open(summary_file) as f:
        results = json.load(f)

    # Filter out any results with errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if not valid_results:
        print("No valid results to plot")
        return

    steps = sorted([int(k) for k in valid_results])
    metrics = [valid_results[str(s)]["metric"] for s in steps]
    hint_counts = [valid_results[str(s)]["hint_count"] for s in steps]
    hack_counts = [valid_results[str(s)]["hack_count"] for s in steps]

    # Compute differences (step N+1 - step N)
    differences = [metrics[i + 1] - metrics[i] for i in range(len(metrics) - 1)]
    diff_steps = steps[1:]  # Start from step 1 onwards

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Top plot: Metric over time
    ax1.plot(
        steps,
        metrics,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="hint / (hint + hack)",
    )

    ax1.set_xlabel("Step Number", fontsize=13)
    ax1.set_ylabel("Metric Value", fontsize=13)
    ax1.set_title(
        f"Hint Usage vs Reward Hacking Across Steps - {run_name}", fontsize=14, pad=20
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    # Set x-axis limits to align with bottom plot
    ax1.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax1.legend(fontsize=11)

    # Add annotations for extremes on top plot
    for step, metric in zip(steps, metrics, strict=True):
        if metric == 0 or metric == 1:  # Extremes
            ax1.annotate(
                f"{metric:.2f}",
                xy=(step, metric),
                xytext=(0, 10 if metric == 1 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                alpha=0.7,
            )

    # Bottom plot: Causal effect of each step
    # Bar at Step N shows the effect of the action taken at Step N
    colors = ["#D4524A" if d < 0 else "#4CAF50" for d in differences]
    ax2.bar(
        diff_steps,
        differences,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add zero line
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)

    ax2.set_xlabel("Step Number", fontsize=13)
    ax2.set_ylabel("Δ Metric (causal effect of step)", fontsize=13)
    ax2.set_title(
        "Causal Effect: Bar at Step N = Effect of Action at Step N",
        fontsize=12,
        pad=15,
    )
    ax2.grid(True, alpha=0.3, axis="y")
    # Set x-axis limits to align with top plot
    ax2.set_xlim(min(steps) - 0.5, max(steps) + 0.5)

    # Highlight the most important steps (largest absolute changes)
    abs_diffs = [(abs(d), i, d) for i, d in enumerate(differences)]
    abs_diffs.sort(reverse=True)
    top_3 = abs_diffs[:3]

    for abs_diff, idx, diff in top_3:
        if abs_diff > 0.01:  # Only annotate if change is meaningful
            step = diff_steps[idx]
            ax2.annotate(
                f"Step {step}\nΔ={diff:+.3f}",
                xy=(step, diff),
                xytext=(0, 15 if diff > 0 else -25),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1),
            )

    plt.tight_layout()
    output_file = config.SUMMARY_DIR / f"{run_name}_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")

    print(f"✅ Plot saved to: {output_file}")

    # Print summary stats
    print(f"\n{'=' * 70}")
    print(f"SUMMARY STATISTICS - {run_name}")
    print(f"{'=' * 70}")
    print(f"Total steps analyzed: {len(steps)}")
    print(f"Steps range: {min(steps)}-{max(steps)}")
    print(f"Mean metric: {sum(metrics) / len(metrics):.3f}")
    print(f"Min metric: {min(metrics):.3f} (step {steps[metrics.index(min(metrics))]})")
    print(f"Max metric: {max(metrics):.3f} (step {steps[metrics.index(max(metrics))]})")
    print(f"Total hints: {sum(hint_counts)}")
    print(f"Total hacks: {sum(hack_counts)}")
    print(
        f"Overall hint/(hint+hack): {sum(hint_counts) / (sum(hint_counts) + sum(hack_counts)):.3f}"
    )
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    fire.Fire(main)
