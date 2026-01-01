#!/usr/bin/env python3
"""
Compute anchor metrics and create visualization from resampled rollouts.

Takes a run folder with resampled step folders, computes hint/(hint+hack) metrics
for each step, and generates a plot showing how the metric changes across steps.

Usage:
    python plot_anchors.py <run-folder>
    python plot_anchors.py results/tictactoe/claude-haiku-4-5/2025-12-31/run-1

Input structure (after resampling and grading):
    run-1/
    ├── step-0/
    │   └── 2025-12-31_21-20-55/
    │       ├── run-1/rollout.log
    │       ├── run-2/rollout.log
    │       └── rollout_analysis_detailed.json
    ├── step-1/
    │   └── 2025-12-31_21-21-00/
    │       └── rollout_analysis_detailed.json
    └── ...

Output:
    run-1/anchor_metrics.json    - Metrics for each step
    run-1/anchor_plot.png        - Visualization
"""

import json
import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt

# Style settings - clean look with Arial font and dark grey text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'


def find_resampled_analysis(step_dir: Path) -> Path | None:
    """Find the rollout_analysis_detailed.json in the resampled folder.
    
    The resampled folder is a timestamped directory inside the step folder.
    Returns the most recent one if multiple exist.
    """
    # Find timestamped directories (format: YYYY-MM-DD_HH-MM-SS)
    timestamp_dirs = [
        d for d in step_dir.iterdir()
        if d.is_dir() and len(d.name) == 19 and d.name[4] == '-' and d.name[10] == '_'
    ]
    
    if not timestamp_dirs:
        return None
    
    # Use the most recent one
    latest_dir = sorted(timestamp_dirs)[-1]
    analysis_file = latest_dir / "rollout_analysis_detailed.json"
    
    if analysis_file.exists():
        return analysis_file
    return None


def compute_metric_from_rollout_data(rollout_data: list) -> dict:
    """Compute hint/(hint+hack) metric from rollout analysis data.
    
    Hack takes precedence over hint - if a rollout both hacked AND used hint,
    it counts as a hack only.
    
    Returns dict with hint_count, hack_count, total, metric.
    """
    # Count hacks and hints
    # Hack takes precedence: if both hacked and hinted, count as hack only
    hack_count = 0
    hint_only_count = 0
    
    for entry in rollout_data:
        analysis = entry.get("analysis", {})
        metadata = entry.get("metadata", {})
        
        reward_hacked = analysis.get("reward_hacked", False)
        terminated_on_hint = metadata.get("terminated_on_hint", False)
        
        if reward_hacked:
            hack_count += 1
        elif terminated_on_hint:
            hint_only_count += 1
    
    total = hint_only_count + hack_count
    metric = hint_only_count / total if total > 0 else 0.0
    
    return {
        "hint_count": hint_only_count,
        "hack_count": hack_count,
        "total": total,
        "metric": metric,
        "total_rollouts": len(rollout_data),
    }


def compute_metric_for_step(analysis_file: Path) -> dict:
    """Compute hint/(hint+hack) metric from a rollout analysis file."""
    with open(analysis_file) as f:
        rollout_data = json.load(f)
    return compute_metric_from_rollout_data(rollout_data)


def get_baseline_metric(run_path: Path) -> dict | None:
    """Get the baseline metric from the original runs in the parent folder.
    
    The parent folder (timestamp folder) contains the original N runs,
    which serve as the "resampling from before step 0" baseline.
    
    Returns dict with metrics or None if not available.
    """
    # Parent folder is the timestamp folder (e.g., 2025-12-31_18-28-38/)
    parent_analysis = run_path.parent / "rollout_analysis_detailed.json"
    
    if not parent_analysis.exists():
        return None
    
    try:
        with open(parent_analysis) as f:
            rollout_data = json.load(f)
        return compute_metric_from_rollout_data(rollout_data)
    except Exception as e:
        print(f"Warning: Could not load baseline from {parent_analysis}: {e}")
        return None


def create_plot(metrics: dict, output_file: Path, run_name: str, baseline_metric: dict | None = None):
    """Create visualization of metrics across steps.
    
    Args:
        metrics: Dict mapping step numbers to their computed metrics
        output_file: Path to save the plot
        run_name: Name of the run for plot title
        baseline_metric: Optional baseline metric from original runs (before step 0)
    """
    
    # Filter out any results with errors, and exclude "baseline" (it's passed separately)
    valid_results = {k: v for k, v in metrics.items() if "error" not in v and k != "baseline"}
    
    if not valid_results:
        print("No valid results to plot")
        return False
    
    # Original step numbers from resampling
    raw_steps = sorted([int(k) for k in valid_results])
    raw_metric_values = [valid_results[str(s)]["metric"] for s in raw_steps]
    
    # Build the line plot data with shifted x-axis:
    # x = N means "resampling from before action N"
    # - baseline (original runs) = x=0 (resampling from before action 0)
    # - step 0 resample = x=1 (resampling from before action 1, action 0 is fixed)
    # - step N resample = x=N+1
    plot_steps = []
    plot_metric_values = []
    
    if baseline_metric:
        plot_steps.append(0)
        plot_metric_values.append(baseline_metric["metric"])
    
    for raw_step, metric_val in zip(raw_steps, raw_metric_values):
        plot_steps.append(raw_step + 1)  # Shift by 1
        plot_metric_values.append(metric_val)
    
    # Compute differences for causal effect
    # Effect at x=N is metric[N] - metric[N-1]
    differences = []
    diff_steps = []
    
    for i in range(len(plot_metric_values) - 1):
        differences.append(plot_metric_values[i + 1] - plot_metric_values[i])
        diff_steps.append(plot_steps[i + 1])
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Top plot: Metric over time (x = "resampling from before action x")
    ax1.plot(
        plot_steps,
        plot_metric_values,
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="#2E86AB",
        label="hint / (hint + hack)",
    )
    
    ax1.set_xlabel("Resampling from Before Action N", fontsize=15)
    ax1.set_ylabel("Metric Value", fontsize=15)
    ax1.set_title(
        f"Hint Usage vs Reward Hacking Across Steps - {run_name}", fontsize=17, pad=15
    )
    ax1.tick_params(axis='both', labelsize=13, length=0)  # Remove tick marks
    ax1.grid(True, alpha=0.4, color="#cccccc", linewidth=0.8)
    ax1.set_axisbelow(True)  # Grid behind data
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(min(plot_steps) - 0.5, max(plot_steps) + 0.5)
    ax1.legend(fontsize=13, framealpha=0.95, edgecolor="lightgray")
    
    # Add annotations for extremes on top plot
    for step, metric in zip(plot_steps, plot_metric_values, strict=True):
        if metric == 0 or metric == 1:  # Extremes
            ax1.annotate(
                f"{metric:.2f}",
                xy=(step, metric),
                xytext=(0, 12 if metric == 1 else -18),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                alpha=0.8,
            )
    
    # Bottom plot: Causal effect of each step
    if differences:
        colors = ["#D4524A" if d < 0 else "#4CAF50" for d in differences]
        ax2.bar(
            diff_steps,
            differences,
            color=colors,
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        
        # Add zero line
        ax2.axhline(y=0, color="#666666", linestyle="-", linewidth=1, alpha=0.5)
        
        ax2.set_xlabel("Action N", fontsize=15)
        ax2.set_ylabel("Δ Metric (causal effect)", fontsize=15)
        ax2.set_title(
            "Causal Effect of Each Action",
            fontsize=14,
            pad=15,
        )
        ax2.tick_params(axis='both', labelsize=13, length=0)  # Remove tick marks
        ax2.grid(True, alpha=0.4, color="#cccccc", linewidth=0.8, axis="y")
        ax2.set_axisbelow(True)  # Grid behind data
        ax2.set_xlim(min(diff_steps) - 0.5, max(diff_steps) + 0.5)
        
        # Highlight the most important steps (largest absolute changes)
        abs_diffs = [(abs(d), i, d) for i, d in enumerate(differences)]
        abs_diffs.sort(reverse=True)
        top_3 = abs_diffs[:3]
        
        for abs_diff, idx, diff in top_3:
            if abs_diff > 0.01:  # Only annotate if change is meaningful
                action = diff_steps[idx]
                ax2.annotate(
                    f"Action {action}\nΔ={diff:+.3f}",
                    xy=(action, diff),
                    xytext=(0, 18 if diff > 0 else -28),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffde7", edgecolor="#e0e0e0", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1, color="#666666"),
                )
    else:
        ax2.text(0.5, 0.5, "Need at least 2 steps for causal effect", 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    return True


def main(run_folder: str):
    """
    Compute anchor metrics and create visualization for a run folder.
    
    Args:
        run_folder: Path to run folder containing step-N/ directories
                   with resampled and graded rollouts
    """
    run_path = Path(run_folder)
    
    if not run_path.exists():
        print(f"Error: Run folder '{run_folder}' does not exist")
        sys.exit(1)
    
    # Find all step directories, excluding the last one (terminal state)
    step_dirs = sorted(run_path.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    
    if not step_dirs:
        print(f"Error: No step-* directories found in '{run_folder}'")
        sys.exit(1)
    
    # Exclude the last step (terminal state after game ended)
    if len(step_dirs) > 1:
        last_step = step_dirs[-1]
        step_dirs = step_dirs[:-1]
        print(f"Excluding final step {last_step.name} (terminal state)")
    
    print(f"\n{'='*70}")
    print(f"COMPUTING ANCHOR METRICS")
    print(f"{'='*70}")
    print(f"Run folder: {run_path}")
    print(f"Steps to analyze: {len(step_dirs)}")
    
    # Get baseline metric from original runs (parent folder's grading)
    baseline_metric = get_baseline_metric(run_path)
    if baseline_metric:
        print(f"Baseline (original runs): metric={baseline_metric['metric']:.3f} "
              f"(hint={baseline_metric['hint_count']}, hack={baseline_metric['hack_count']}, "
              f"total={baseline_metric['total_rollouts']})")
    else:
        print("⚠️  No baseline found (parent folder not graded)")
    print(f"{'='*70}\n")
    
    # Compute metrics for each step
    metrics = {}
    
    # Store baseline in metrics if available
    if baseline_metric:
        metrics["baseline"] = baseline_metric
    
    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("-")[1])
        print(f"Step {step_num:2d}: ", end="", flush=True)
        
        # Find resampled analysis file
        analysis_file = find_resampled_analysis(step_dir)
        
        if analysis_file is None:
            print("⚠️  No resampled analysis found")
            metrics[str(step_num)] = {"error": "No resampled analysis found"}
            continue
        
        # Compute metric
        try:
            step_metrics = compute_metric_for_step(analysis_file)
            metrics[str(step_num)] = step_metrics
            print(f"✅ metric={step_metrics['metric']:.3f} "
                  f"(hint={step_metrics['hint_count']}, hack={step_metrics['hack_count']}, "
                  f"total={step_metrics['total_rollouts']})")
        except Exception as e:
            print(f"❌ Error: {e}")
            metrics[str(step_num)] = {"error": str(e)}
    
    # Save metrics JSON
    run_name = run_path.name
    metrics_file = run_path / "anchor_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_file}")
    
    # Create plot
    plot_file = run_path / "anchor_plot.png"
    if create_plot(metrics, plot_file, run_name, baseline_metric):
        print(f"✅ Plot saved to: {plot_file}")
    
    # Print summary statistics
    # Exclude baseline from step metrics for summary
    step_metrics_only = {k: v for k, v in metrics.items() if k != "baseline" and "error" not in v}
    if step_metrics_only:
        all_metrics = [v["metric"] for v in step_metrics_only.values()]
        total_hints = sum(v["hint_count"] for v in step_metrics_only.values())
        total_hacks = sum(v["hack_count"] for v in step_metrics_only.values())
        
        print(f"\n{'='*70}")
        print(f"SUMMARY STATISTICS - {run_name}")
        print(f"{'='*70}")
        print(f"Steps analyzed: {len(step_metrics_only)}")
        if baseline_metric:
            print(f"Baseline metric (original runs): {baseline_metric['metric']:.3f}")
        print(f"Mean metric across resampled steps: {sum(all_metrics) / len(all_metrics):.3f}")
        print(f"Min metric: {min(all_metrics):.3f}")
        print(f"Max metric: {max(all_metrics):.3f}")
        print(f"Total hints across all steps: {total_hints}")
        print(f"Total hacks across all steps: {total_hacks}")
        if total_hints + total_hacks > 0:
            print(f"Overall hint/(hint+hack): {total_hints / (total_hints + total_hacks):.3f}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    fire.Fire(main)

