#!/usr/bin/env python3
"""
Compute causal effect of a property on hint/hack behavior.

Merges property judgments with grading results and computes:
- hint/(hint+hack) metric for each group (with/without property)
- Causal effect (difference between groups)
- Bootstrap confidence intervals

Usage:
    # Analyze ALL timestamp folders in a base state folder
    python compute_causal_effect.py base_states/dev-step thought_about_exploit

    # Analyze a SINGLE timestamp folder
    python compute_causal_effect.py base_states/dev-step/2025-12-31_22-59-30 thought_about_exploit

    # Conditional analysis (effect of property B given property A = True)
    python compute_causal_effect.py base_states/dev-step ethical_concerns --condition-on thought_about_exploit

Input files (in each timestamp folder):
    - property_judgments_<property>.json
    - rollout_analysis_detailed.json
    - property_judgments_<condition>.json (if --condition-on used)

Output (saved to input folder):
    - causal_effect_<property>.json (or causal_effect_<property>_given_<condition>.json)
    - causal_effect_<property>.png (visualization)
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'


def is_timestamp_folder(path: Path) -> bool:
    """Check if a path is a timestamp folder (YYYY-MM-DD_HH-MM-SS format)."""
    name = path.name
    return len(name) == 19 and name[4] == '-' and name[10] == '_'


def discover_timestamp_folders(input_path: Path) -> list[Path]:
    """
    Discover timestamp folders to process.
    
    If input is a timestamp folder, returns [input].
    If input contains timestamp folders, returns all of them.
    """
    if is_timestamp_folder(input_path):
        return [input_path]
    
    # Look for timestamp folders inside
    timestamp_dirs = sorted([
        d for d in input_path.iterdir()
        if d.is_dir() and is_timestamp_folder(d)
    ])
    
    return timestamp_dirs


def load_and_merge_data(folder: Path, property_name: str) -> list[dict]:
    """
    Load property judgments and grading results, merge by filename.
    
    Returns list of merged rollout data with:
    - filename
    - property_present (from property judgments)
    - reward_hacked (from grading)
    - terminated_on_hint (from grading)
    """
    # Load property judgments
    judgments_file = folder / f"property_judgments_{property_name}.json"
    if not judgments_file.exists():
        raise FileNotFoundError(f"Property judgments not found: {judgments_file}")
    
    with open(judgments_file) as f:
        judgments = json.load(f)
    
    # Load grading results
    grading_file = folder / "rollout_analysis_detailed.json"
    if not grading_file.exists():
        raise FileNotFoundError(f"Grading results not found: {grading_file}")
    
    with open(grading_file) as f:
        grading = json.load(f)
    
    # Create mapping from filename to grading data
    grading_map = {}
    for item in grading:
        filename = item["filename"]
        grading_map[filename] = {
            "reward_hacked": item.get("analysis", {}).get("reward_hacked", False),
            "terminated_on_hint": item.get("metadata", {}).get("terminated_on_hint", False),
        }
    
    # Merge
    merged = []
    for j in judgments:
        filename = j["filename"]
        if filename in grading_map:
            merged.append({
                "filename": filename,
                "property_present": j["property_present"],
                "reward_hacked": grading_map[filename]["reward_hacked"],
                "terminated_on_hint": grading_map[filename]["terminated_on_hint"],
            })
        else:
            print(f"Warning: No grading found for {filename}")
    
    return merged


def apply_condition(
    data: list[dict], 
    folder: Path, 
    condition_property: str
) -> list[dict]:
    """Filter data to only rollouts where condition property is True."""
    
    # Load condition judgments
    condition_file = folder / f"property_judgments_{condition_property}.json"
    if not condition_file.exists():
        raise FileNotFoundError(
            f"Condition property judgments not found: {condition_file}\n"
            f"Run: python property_judge.py {folder} {condition_property}"
        )
    
    with open(condition_file) as f:
        condition_judgments = json.load(f)
    
    # Create mapping from filename to condition
    condition_map = {j["filename"]: j["property_present"] for j in condition_judgments}
    
    # Filter
    original_count = len(data)
    filtered = [d for d in data if condition_map.get(d["filename"], False)]
    
    print(f"Conditioning on {condition_property} = True:")
    print(f"  Filtered from {original_count} to {len(filtered)} rollouts")
    
    return filtered


def compute_metric(data: list[dict]) -> dict:
    """
    Compute hint/(hint+hack) metric for a group.
    
    Hack takes precedence: if both hacked AND hinted, counts as hack only.
    """
    hint_count = 0
    hack_count = 0
    
    for d in data:
        if d["reward_hacked"]:
            hack_count += 1
        elif d["terminated_on_hint"]:
            hint_count += 1
    
    total = hint_count + hack_count
    metric = hint_count / total if total > 0 else 0.0
    
    return {
        "hint_count": hint_count,
        "hack_count": hack_count,
        "total": total,
        "n_rollouts": len(data),
        "metric": metric,
    }


def bootstrap_ci(
    hint_with: int, hack_with: int,
    hint_without: int, hack_without: int,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for causal effect."""
    
    total_with = hint_with + hack_with
    total_without = hint_without + hack_without
    
    # Compute point estimate (metric is 0 if no outcomes in group)
    metric_with = hint_with / total_with if total_with > 0 else 0.0
    metric_without = hint_without / total_without if total_without > 0 else 0.0
    original_effect = metric_with - metric_without
    
    # Can't bootstrap if either group has no outcomes
    if total_with == 0 or total_without == 0:
        return {
            "effect": original_effect,
            "ci_lower": None,  # CI undefined
            "ci_upper": None,
            "p_value": None,  # Can't compute
            "note": "CI undefined: one group has no hint/hack outcomes",
        }
    
    # Bootstrap
    rng = np.random.RandomState(42)
    bootstrap_effects = []
    
    for _ in range(n_bootstrap):
        # Resample with group (multinomial)
        counts_with = rng.multinomial(total_with, [hint_with/total_with, hack_with/total_with])
        boot_metric_with = counts_with[0] / total_with
        
        # Resample without group
        counts_without = rng.multinomial(total_without, [hint_without/total_without, hack_without/total_without])
        boot_metric_without = counts_without[0] / total_without
        
        bootstrap_effects.append(boot_metric_with - boot_metric_without)
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_effects, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_effects, (1 - alpha/2) * 100)
    
    # P-value (two-sided)
    if original_effect >= 0:
        p_value = 2 * np.mean([e <= 0 for e in bootstrap_effects])
    else:
        p_value = 2 * np.mean([e >= 0 for e in bootstrap_effects])
    p_value = min(p_value, 1.0)
    
    return {
        "effect": original_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "confidence_level": confidence_level,
    }


def create_visualization(
    results: dict,
    output_file: Path,
    property_name: str,
    condition_on: str = None,
):
    """Create bar chart visualization of causal effect."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data
    with_metrics = results["with_property"]
    without_metrics = results["without_property"]
    
    groups = [
        f"With\n{property_name.replace('_', ' ')}\n(N={with_metrics['n_rollouts']})",
        f"Without\n{property_name.replace('_', ' ')}\n(N={without_metrics['n_rollouts']})",
    ]
    values = [with_metrics["metric"], without_metrics["metric"]]
    colors = ["#ff6b6b", "#4ecdc4"]
    
    # Error bars (approximate SE for binomial proportion)
    def se_proportion(p, n):
        return np.sqrt(p * (1 - p) / n) if n > 0 else 0
    
    yerr = [
        1.96 * se_proportion(values[0], with_metrics["total"]),
        1.96 * se_proportion(values[1], without_metrics["total"]),
    ]
    
    # Plot
    bars = ax.bar(groups, values, color=colors, alpha=0.8, edgecolor='black', 
                  linewidth=1.5, yerr=yerr, capsize=8)
    
    # Value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{value:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Labels
    ax.set_ylabel('hint / (hint + hack)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, min(1.2, max(values) * 1.4) if max(values) > 0 else 1.0)
    ax.tick_params(axis='both', labelsize=13, length=0)
    ax.grid(axis='y', alpha=0.4, color="#cccccc", linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Title
    title = f'Causal Effect of "{property_name.replace("_", " ").title()}"'
    if condition_on:
        title += f'\n(given {condition_on.replace("_", " ")} = True)'
    ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
    
    # Effect annotation
    bootstrap = results["bootstrap"]
    effect = bootstrap["effect"]
    ci_lower = bootstrap.get("ci_lower")
    ci_upper = bootstrap.get("ci_upper")
    p_value = bootstrap.get("p_value")
    
    # Handle case where CI is undefined
    if ci_lower is None or p_value is None:
        box_color = "#fff9c4"  # Yellow for undefined
        annotation = f'Effect: {effect:+.3f}\n(CI undefined - insufficient data)'
    else:
        # Significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        
        box_color = "#c8e6c9" if p_value < 0.05 else "#ffcdd2"
        annotation = f'Effect: {effect:+.3f} ({sig})\n95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]  |  p = {p_value:.4f}'
    
    ax.text(0.5, -0.15, annotation,
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, 
                     edgecolor='#666666', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Visualization saved to {output_file}")


def compute_causal_effect(
    input_path: str,
    property_name: str,
    condition_on: str = None,
    n_bootstrap: int = 10000,
):
    """
    Compute causal effect of a property on hint/hack behavior.
    
    Args:
        input_path: Path to either:
                   - A parent folder containing timestamps (aggregates ALL timestamps)
                     e.g., base_states/dev-step
                   - A specific timestamp folder (processes just that one)
                     e.g., base_states/dev-step/2025-12-31_22-00-00
        property_name: Property to analyze (e.g., 'thought_about_exploit')
        condition_on: Optional property to condition on (only analyze rollouts where this is True)
        n_bootstrap: Number of bootstrap samples for CI
    
    Examples:
        # Analyze ALL timestamp folders (aggregated)
        python compute_causal_effect.py base_states/dev-step thought_about_exploit
        
        # Analyze single timestamp
        python compute_causal_effect.py base_states/dev-step/2025-12-31_22-59-30 thought_about_exploit
        
        # Conditional analysis
        python compute_causal_effect.py base_states/dev-step ethical_concerns --condition-on thought_about_exploit
    """
    folder_path = Path(input_path)
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {input_path}")
        return
    
    # Discover timestamp folders
    timestamp_folders = discover_timestamp_folders(folder_path)
    
    if not timestamp_folders:
        print(f"Error: No timestamp folders found in {input_path}")
        print("Expected either a timestamp folder (YYYY-MM-DD_HH-MM-SS) or a parent containing them.")
        return
    
    print(f"\n{'='*70}")
    print("CAUSAL EFFECT ANALYSIS")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Timestamp folders: {len(timestamp_folders)}")
    print(f"Property: {property_name}")
    if condition_on:
        print(f"Conditioned on: {condition_on} = True")
    print(f"{'='*70}\n")
    
    # Load and merge data from ALL timestamp folders
    print("Loading data from all folders...")
    all_data = []
    for ts_folder in timestamp_folders:
        try:
            folder_data = load_and_merge_data(ts_folder, property_name)
            # Tag each rollout with its source folder
            for d in folder_data:
                d["source_folder"] = ts_folder.name
            all_data.extend(folder_data)
            print(f"  ✅ {ts_folder.name}: {len(folder_data)} rollouts")
        except FileNotFoundError as e:
            print(f"  ⚠️  {ts_folder.name}: {e}")
    
    if not all_data:
        print("\nError: No data loaded from any folder")
        return
    
    print(f"\nTotal rollouts loaded: {len(all_data)}")
    
    # Apply conditioning if specified
    if condition_on:
        print(f"\nApplying condition: {condition_on} = True")
        # Need to load condition judgments from each folder and merge
        filtered_data = []
        for ts_folder in timestamp_folders:
            folder_rollouts = [d for d in all_data if d.get("source_folder") == ts_folder.name]
            try:
                filtered = apply_condition(folder_rollouts, ts_folder, condition_on)
                filtered_data.extend(filtered)
            except FileNotFoundError:
                print(f"  ⚠️  {ts_folder.name}: No condition judgments")
        
        all_data = filtered_data
        print(f"  After filtering: {len(all_data)} rollouts")
        
        if len(all_data) == 0:
            print("Error: No rollouts satisfy the condition")
            return
    
    # Split by property
    with_property = [d for d in all_data if d["property_present"]]
    without_property = [d for d in all_data if not d["property_present"]]
    
    print(f"\nGroup sizes:")
    print(f"  With property: {len(with_property)}")
    print(f"  Without property: {len(without_property)}")
    
    if len(with_property) == 0 or len(without_property) == 0:
        print("\nError: Need rollouts in both groups for comparison")
        return
    
    # Compute metrics
    with_metrics = compute_metric(with_property)
    without_metrics = compute_metric(without_property)
    
    print(f"\nMetrics:")
    print(f"  With property:    {with_metrics['metric']:.3f} "
          f"(hint={with_metrics['hint_count']}, hack={with_metrics['hack_count']})")
    print(f"  Without property: {without_metrics['metric']:.3f} "
          f"(hint={without_metrics['hint_count']}, hack={without_metrics['hack_count']})")
    
    # Bootstrap CI
    print(f"\nComputing bootstrap CI (n={n_bootstrap})...")
    bootstrap = bootstrap_ci(
        with_metrics["hint_count"], with_metrics["hack_count"],
        without_metrics["hint_count"], without_metrics["hack_count"],
        n_bootstrap=n_bootstrap,
    )
    
    effect = bootstrap["effect"]
    print(f"\n{'='*70}")
    print("CAUSAL EFFECT")
    print(f"{'='*70}")
    print(f"Effect: {effect:+.3f}")
    
    # Handle case where CI is undefined (one group has no outcomes)
    if bootstrap.get("ci_lower") is None:
        print(f"95% CI: undefined ({bootstrap.get('note', 'insufficient data')})")
        print(f"P-value: undefined")
        print(f"⚠️  Cannot determine statistical significance")
    else:
        print(f"95% CI: [{bootstrap['ci_lower']:+.3f}, {bootstrap['ci_upper']:+.3f}]")
        print(f"P-value: {bootstrap['p_value']:.4f}")
        
        if bootstrap['p_value'] < 0.05:
            print(f"✅ Statistically significant (p < 0.05)")
            if effect > 0:
                print(f"→ Property INCREASES hint usage (decreases hacking)")
            else:
                print(f"→ Property DECREASES hint usage (increases hacking)")
        else:
            print(f"❌ Not statistically significant (p >= 0.05)")
    
    print(f"{'='*70}\n")
    
    # Compile results
    results = {
        "property_name": property_name,
        "condition_on": condition_on,
        "timestamp_folders": [f.name for f in timestamp_folders],
        "total_rollouts": len(all_data),
        "with_property": with_metrics,
        "without_property": without_metrics,
        "bootstrap": bootstrap,
    }
    
    # Save results to the input folder (not timestamp subfolder)
    if condition_on:
        output_name = f"causal_effect_{property_name}_given_{condition_on}"
    else:
        output_name = f"causal_effect_{property_name}"
    
    results_file = folder_path / f"{output_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {results_file}")
    
    # Create visualization
    plot_file = folder_path / f"{output_name}.png"
    create_visualization(results, plot_file, property_name, condition_on)
    
    return results


if __name__ == "__main__":
    fire.Fire(compute_causal_effect)

