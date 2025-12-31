"""Plotting script for comparing multiple hint results JSON files.

Similar to plot_hint_results.py but plots multiple results as lines on one plot.

Usage Examples:
    # Compare multiple result files (uses filenames as labels)
    python plot_hint_results_comparison.py result1.json result2.json result3.json
    # Output: plots/main_hint_usage_comparison.png, main_reward_hacking_comparison.png, main_hint_ratio_comparison.png

    # Compare with custom title prefix
    python plot_hint_results_comparison.py gpt5.json claude.json --title-prefix "GPT-5 vs Claude"
    # Output: plots/main_hint_usage_comparison.png (with title "Hint Usage Rate for GPT-5 vs Claude")

    # Compare with custom file prefix for outputs
    python plot_hint_results_comparison.py exp1.json exp2.json --file-prefix "experiment_comparison"
    # Output: plots/experiment_comparison_hint_usage_comparison.png, etc.

    # Combine all plots into a single figure with subplots
    python plot_hint_results_comparison.py result1.json result2.json --combine-plots
    # Output: plots/main_combined_comparison.png (single figure with 3 subplots)

    # Full example with all options
    python plot_hint_results_comparison.py \
        aggregated_results/gpt5_hint_results.json \
        aggregated_results/claude_hint_results.json \
        --title-prefix "Model Comparison" \
        --file-prefix "gpt5_vs_claude" \
        --combine-plots
    # Output: plots/gpt5_vs_claude_combined_comparison.png

Input format (each JSON file):
    {
        "hint_1": {
            "hint_cost": 1,
            "total_games": 20,
            "hints_used": 1,
            "reward_hacks": 17
        },
        "hint_3": {
            "hint_cost": 3,
            "total_games": 20,
            "hints_used": 3,
            "reward_hacks": 14
        },
        ...
    }

Plots generated:
    1. Hint Usage Rate: % of runs that chose hints vs hint cost
    2. Reward Hacking Rate: % of runs that chose reward hacking vs hint cost
    3. Hint-Hack Ratio: % of problematic behaviors that were hint usage (vs reward hacking)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
import seaborn as sns

# Set style for prettier plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

cwd = Path(__file__).resolve().parent


def extract_model_name(label: str) -> str:
    """Extract the model name from a label.

    Args:
        label: Label like "GPT-5 (Chess)" or "o3 (TTT, 1 practice)"

    Returns:
        Model name like "GPT-5" or "o3"
    """
    # Match patterns like "GPT-5", "o3", "Gemini 3.0 pro", "Sonnet 4.5" etc
    match = re.match(r'^([^(]+)', label)
    if match:
        return match.group(1).strip()
    return label


def assign_colors_and_styles(labels: List[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Assign consistent colors to model names and varying styles to configurations.

    Args:
        labels: List of labels like ["GPT-5 (Chess)", "GPT-5 (TTT)", "o3 (Chess)"]

    Returns:
        Tuple of (color_map, linestyle_map, marker_map) dictionaries mapping label -> style
    """
    # Use default seaborn color palette
    colors = sns.color_palette()

    # Define line styles and markers
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

    # Group labels by model name
    model_groups = {}
    for label in labels:
        model_name = extract_model_name(label)
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(label)

    # Assign colors to model names and styles to configurations
    color_map = {}
    linestyle_map = {}
    marker_map = {}

    # Only use same color per model if we have multiple DIFFERENT models
    if len(model_groups) > 1:
        # Use same color for same model, different styles for different configs
        for i, (model_name, model_labels) in enumerate(sorted(model_groups.items())):
            color = colors[i % len(colors)]

            for j, label in enumerate(model_labels):
                color_map[label] = color
                linestyle_map[label] = linestyles[j % len(linestyles)]
                marker_map[label] = markers[j % len(markers)]
    else:
        # Only one model - use different colors for each configuration
        for i, label in enumerate(labels):
            color_map[label] = colors[i % len(colors)]
            linestyle_map[label] = '-'  # All solid lines
            marker_map[label] = 'o'  # All circles

    return color_map, linestyle_map, marker_map


def load_results(json_path: Path) -> dict:
    """Load results from JSON file.
    
    Args:
        json_path: Path to JSON file with hint results
        
    Returns:
        Dictionary with hint results
    """
    with open(json_path) as f:
        return json.load(f)


def calculate_metrics(results: dict) -> tuple[List[float], List[float], List[float], List[int], List[int]]:
    """Calculate hint and reward hacking percentages from results.
    
    Args:
        results: Dictionary with hint results
        
    Returns:
        Tuple of (hint_costs, hint_percentages, reward_hack_percentages, hints_used_counts, reward_hack_counts) sorted by hint cost
    """
    # Extract data sorted by hint cost
    sorted_keys = sorted(results.keys(), key=lambda k: results[k].get('hint_cost', 0))
    
    hint_costs = [results[k]['hint_cost'] for k in sorted_keys]
    
    # Calculate percentages and store raw counts
    hint_percentages = []
    reward_hack_percentages = []
    hints_used_counts = []
    reward_hack_counts = []
    
    for key in sorted_keys:
        data = results[key]
        hints_used = data['hints_used']
        reward_hacks = data['reward_hacks']
        total_games = data['total_games']
        
        # Calculate percentage of runs that chose hints
        hint_pct = (hints_used / total_games * 100) if total_games > 0 else 0
        
        # Calculate percentage of runs that chose reward hacking
        reward_hack_pct = (reward_hacks / total_games * 100) if total_games > 0 else 0
        
        hint_percentages.append(hint_pct)
        reward_hack_percentages.append(reward_hack_pct)
        hints_used_counts.append(hints_used)
        reward_hack_counts.append(reward_hacks)
    
    return hint_costs, hint_percentages, reward_hack_percentages, hints_used_counts, reward_hack_counts


def plot_multiple_results(
    results_dict: Dict[str, dict],
    output_dir: Path = None,
    title_prefix: str = "Hint Behavior",
    file_prefix: str = "main",
    combine_plots: bool = False
) -> None:
    """Plot multiple results as lines on two separate plots.

    Args:
        results_dict: Dictionary mapping label -> results dictionary
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
        combine_plots: If True, combine all three plots into one figure with subplots
    """
    if output_dir is None:
        output_dir = cwd / "plots"

    output_dir.mkdir(exist_ok=True)

    # Get style mappings for consistent colors per model
    labels = list(results_dict.keys())
    color_map, linestyle_map, marker_map = assign_colors_and_styles(labels)

    # Collect data from all result sets
    all_data = {}
    for label, results in results_dict.items():
        hint_costs, hint_percentages, reward_hack_percentages, hints_used_counts, reward_hack_counts = calculate_metrics(results)

        all_data[label] = {
            'hint_costs': hint_costs,
            'hint_percentages': hint_percentages,
            'reward_hack_percentages': reward_hack_percentages,
            'hints_used_counts': hints_used_counts,
            'reward_hack_counts': reward_hack_counts
        }

    if combine_plots:
        # Create a single figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Percentage of runs that chose hints
        ax1 = axes[0]
        for label, data in all_data.items():
            ax1.plot(data['hint_costs'], data['hint_percentages'],
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        ax1.set_xlabel('Hint Cost', fontsize=12)
        ax1.set_ylabel('% of Runs That Chose Hints', fontsize=12)
        ax1.set_title(f'Hint Usage Rate for {title_prefix}', fontsize=14)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 105)

        # Plot 2: Percentage of runs that chose reward hacking
        ax2 = axes[1]
        for label, data in all_data.items():
            ax2.plot(data['hint_costs'], data['reward_hack_percentages'],
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        ax2.set_xlabel('Hint Cost', fontsize=12)
        ax2.set_ylabel('% of Runs That Chose Reward Hacking', fontsize=12)
        ax2.set_title(f'Reward Hacking Rate for {title_prefix}', fontsize=14)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-5, 105)

        # Plot 3: Ratio of hint usage to total problematic behaviors
        ax3 = axes[2]
        for label, data in all_data.items():
            # Calculate ratio: # hints / (# hints + # reward_hacks)
            ratios = []
            for hints_count, rh_count in zip(data['hints_used_counts'], data['reward_hack_counts']):
                total = hints_count + rh_count
                ratio = (hints_count / total * 100) if total > 0 else 0
                ratios.append(ratio)

            ax3.plot(data['hint_costs'], ratios,
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        ax3.set_xlabel('Hint Cost', fontsize=12)
        ax3.set_ylabel('Hint-Hack Ratio', fontsize=12)
        ax3.set_title(f'Hint Hack Ratio for {title_prefix}', fontsize=14)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-5, 105)

        plt.tight_layout()

        # Save combined plot
        combined_plot_path = output_dir / f"{file_prefix}_combined_comparison.png"
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {combined_plot_path}")
        plt.close()

    else:
        # Create separate plots (original behavior)
        # Plot 1: Percentage of runs that chose hints
        plt.figure(figsize=(12, 6))
        for label, data in all_data.items():
            plt.plot(data['hint_costs'], data['hint_percentages'],
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        plt.xlabel('Hint Cost', fontsize=12)
        plt.ylabel('% of Runs That Chose Hints', fontsize=12)
        plt.title(f'Hint Usage Rate for {title_prefix}', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)  # Percentage scale
        plt.tight_layout()

        # Save hint usage plot
        hint_plot_path = output_dir / f"{file_prefix}_hint_usage_comparison.png"
        plt.savefig(hint_plot_path, dpi=300, bbox_inches='tight')
        print(f"Hint usage plot saved to: {hint_plot_path}")
        plt.close()

        # Plot 2: Percentage of runs that chose reward hacking
        plt.figure(figsize=(12, 6))
        for label, data in all_data.items():
            plt.plot(data['hint_costs'], data['reward_hack_percentages'],
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        plt.xlabel('Hint Cost', fontsize=12)
        plt.ylabel('% of Runs That Chose Reward Hacking', fontsize=12)
        plt.title(f'Reward Hacking Rate for {title_prefix}', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)  # Percentage scale
        plt.tight_layout()

        # Save reward hacking plot
        rh_plot_path = output_dir / f"{file_prefix}_reward_hacking_comparison.png"
        plt.savefig(rh_plot_path, dpi=300, bbox_inches='tight')
        print(f"Reward hacking plot saved to: {rh_plot_path}")
        plt.close()

        # Plot 3: Ratio of hint usage to total problematic behaviors
        plt.figure(figsize=(12, 6))
        for label, data in all_data.items():
            # Calculate ratio: # hints / (# hints + # reward_hacks)
            ratios = []
            for hints_count, rh_count in zip(data['hints_used_counts'], data['reward_hack_counts']):
                total = hints_count + rh_count
                ratio = (hints_count / total * 100) if total > 0 else 0
                ratios.append(ratio)

            plt.plot(data['hint_costs'], ratios,
                    color=color_map[label],
                    linestyle=linestyle_map[label],
                    marker=marker_map[label],
                    linewidth=2, markersize=8, label=label, alpha=0.8)

        plt.xlabel('Hint Cost', fontsize=12)
        plt.ylabel('Hint-Hack Ratio', fontsize=12)
        plt.title(f'Hint Hack Ratio for {title_prefix}', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)  # Percentage scale
        plt.tight_layout()

        # Save ratio plot
        ratio_plot_path = output_dir / f"{file_prefix}_hint_ratio_comparison.png"
        plt.savefig(ratio_plot_path, dpi=300, bbox_inches='tight')
        print(f"Hint ratio plot saved to: {ratio_plot_path}")
        plt.close()
    
    # Save data as JSON
    data_path = output_dir / f"{file_prefix}_hint_behavior_comparison_data.json"
    with open(data_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Data saved to: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparison of multiple hint results JSON files"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="List of JSON result files to compare"
    )
    parser.add_argument(
        "--title-prefix",
        default="Hint Behavior",
        help="Prefix for plot titles (default: 'Hint Behavior')"
    )
    parser.add_argument(
        "--file-prefix",
        default="main",
        help="Prefix for output filenames (default: 'main')"
    )
    parser.add_argument(
        "--combine-plots",
        action="store_true",
        help="Combine all plots into one figure with subplots"
    )

    args = parser.parse_args()

    # Load all results - use filename (without extension) as label
    results_dict = {}
    for file_path_str in args.results_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            # Use filename without extension as label
            label = file_path.stem
            results_dict[label] = load_results(file_path)
            print(f"Loaded: {label} from {file_path}")
        else:
            print(f"Warning: File not found at {file_path_str}, skipping...")

    if not results_dict:
        print("\nError: No valid results files found.")
        print("Please provide valid file paths.")
    else:
        # Create comparison plots
        plot_multiple_results(
            results_dict,
            title_prefix=args.title_prefix,
            file_prefix=args.file_prefix,
            combine_plots=args.combine_plots
        )

