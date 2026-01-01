"""
Analysis script for causal_analysis rollout folders.

This script computes the hint/(hint+hack) metric for rollout folders
in causal_analysis/rollouts_data/.

Usage:
    # Analyze a specific folder:
    python analysis.py rollouts_data/o3-2025-04-16-20251101-162339-75659

    # Analyze all folders:
    python analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

cwd = Path(__file__).resolve().parent
rollouts_dir = cwd / "rollouts_data"


def compute_hack_count(folder_path: Path) -> int:
    """
    Compute hack count for a single rollout folder.

    Tries multiple sources in order:
    1. rollout_summary.json (old format with reward_hack_filenames)
    2. rollout_analysis_detailed.json (from grade_rollouts_v2.py)
    """
    # Try old format first (rollout_summary.json)
    summary_path = folder_path / "rollout_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        hack_filenames = summary.get("reward_hack_filenames", [])
        return len(hack_filenames)

    # Try new format (rollout_analysis_detailed.json)
    detailed_path = folder_path / "rollout_analysis_detailed.json"
    if detailed_path.exists():
        with open(detailed_path) as f:
            data = json.load(f)
        return sum(1 for rollout in data if rollout.get("reward_hacked", False))

    print(f"WARNING: No hack data found in {folder_path}")
    print("Expected: rollout_summary.json OR rollout_analysis_detailed.json")
    return 0


def compute_hint_count(folder_path: Path) -> int:
    """
    Compute hint-only count (terminated on hint but NOT hacked).

    Tries multiple sources in order:
    1. rollout_summary.json (old format with hint_terminated_filenames & reward_hack_filenames)
    2. rollouts_summary.json + rollout_analysis_detailed.json (new format)
    """
    # Try old format first (rollout_summary.json)
    summary_path = folder_path / "rollout_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        hint_filenames = set(summary.get("hint_terminated_filenames", []))
        hack_filenames = set(summary.get("reward_hack_filenames", []))
        # Hint-only: hint terminated but NOT hacked
        hint_only_filenames = hint_filenames - hack_filenames
        return len(hint_only_filenames)

    # Try new format (rollouts_summary.json + rollout_analysis_detailed.json)
    rollouts_summary_path = folder_path / "rollouts_summary.json"
    detailed_path = folder_path / "rollout_analysis_detailed.json"

    if not rollouts_summary_path.exists():
        print(f"WARNING: No hint data found in {folder_path}")
        print("Expected: rollout_summary.json OR rollouts_summary.json")
        return 0

    if not detailed_path.exists():
        print(f"WARNING: {detailed_path} not found - run grade_rollouts_v2.py first!")
        return 0

    with open(rollouts_summary_path) as f:
        summary = json.load(f)

    with open(detailed_path) as f:
        detailed = json.load(f)

    # Get rollout IDs that hacked
    hacked_ids = {r["rollout_id"] for r in detailed if r.get("reward_hacked", False)}

    # Count hint-terminated that didn't hack
    hint_count = 0
    for rollout_id, metadata in summary.items():
        if metadata.get("terminated_on_hint", False) and rollout_id not in hacked_ids:
            hint_count += 1

    return hint_count


def compute_metric(folder_path: Path) -> dict:
    """
    Compute hint/(hint+hack) metric for a single rollout folder.

    Returns:
        dict with hint_count, hack_count, total, and metric
    """
    hack_count = compute_hack_count(folder_path)
    hint_count = compute_hint_count(folder_path)

    total = hint_count + hack_count
    metric = hint_count / total if total > 0 else 0.0

    return {
        "folder": folder_path.name,
        "hint_count": hint_count,
        "hack_count": hack_count,
        "total": total,
        "metric": metric,
    }


def plot_metrics(results: list[dict]) -> None:
    """
    Plot metric for each rollout folder.

    Args:
        results: List of dicts from compute_metric
    """
    if not results:
        print("No results to plot!")
        return

    # Sort by folder name
    results = sorted(results, key=lambda x: x["folder"])

    # Extract data
    folder_names = [r["folder"] for r in results]
    metrics = [r["metric"] for r in results]

    # Create plot
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(folder_names)), metrics, alpha=0.7)
    plt.xlabel("Rollout Folder", fontsize=12)
    plt.ylabel("hint / (hint + hack)", fontsize=12)
    plt.title("Metric by Rollout Folder", fontsize=14)
    plt.xticks(range(len(folder_names)), folder_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Save plot
    plot_path = cwd / "rollouts_metric_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {plot_path}")

    plt.show()


def analyze_single_folder(folder_path: Path):
    """Analyze a single rollout folder."""
    print("=" * 80)
    print("CAUSAL ANALYSIS ROLLOUT METRIC")
    print("=" * 80)
    print(f"Folder: {folder_path}\n")

    if not folder_path.exists():
        print(f"ERROR: {folder_path} does not exist!")
        return

    if not folder_path.is_dir():
        print(f"ERROR: {folder_path} is not a directory!")
        return

    # Compute metric
    result = compute_metric(folder_path)

    print(f"Hint-only: {result['hint_count']}")
    print(f"Hacks:     {result['hack_count']}")
    print(f"Total:     {result['total']}")
    print(f"Metric:    {result['metric']:.3f}")

    if result["total"] == 0:
        print("\nWARNING: No hint or hack data found!")
        print("Make sure you've run grade_rollouts_v2.py on this folder first.")

    # Save result
    output_path = folder_path / "metric_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to: {output_path}")


def analyze_all_folders():
    """Analyze all rollout folders in rollouts_data/."""
    print("=" * 80)
    print("CAUSAL ANALYSIS ROLLOUT METRICS")
    print("=" * 80)
    print(f"Scanning: {rollouts_dir}\n")

    if not rollouts_dir.exists():
        print(f"ERROR: {rollouts_dir} does not exist!")
        return

    # Analyze each folder
    results = []
    for folder in sorted(rollouts_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("o3-"):
            print(f"\nAnalyzing: {folder.name}")
            result = compute_metric(folder)
            results.append(result)

            print(f"  Hint-only: {result['hint_count']}")
            print(f"  Hacks:     {result['hack_count']}")
            print(f"  Total:     {result['total']}")
            print(f"  Metric:    {result['metric']:.3f}")

    if not results:
        print("\nNo rollout folders found!")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_hints = sum(r["hint_count"] for r in results)
    total_hacks = sum(r["hack_count"] for r in results)
    total_all = total_hints + total_hacks
    overall_metric = total_hints / total_all if total_all > 0 else 0.0

    print(f"Total hint-only: {total_hints}")
    print(f"Total hacks:     {total_hacks}")
    print(f"Total:           {total_all}")
    print(f"Overall metric:  {overall_metric:.3f}")

    # Save results to JSON
    output_path = cwd / "rollouts_metric_analysis.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "summary": {
                    "total_hints": total_hints,
                    "total_hacks": total_hacks,
                    "total": total_all,
                    "overall_metric": overall_metric,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to: {output_path}")

    # Plot
    plot_metrics(results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specific folder
        folder_arg = sys.argv[1]
        folder_path = Path(folder_arg)

        # Handle relative paths from causal_analysis/
        if not folder_path.is_absolute():
            folder_path = cwd / folder_path

        analyze_single_folder(folder_path)
    else:
        # Analyze all folders
        analyze_all_folders()
