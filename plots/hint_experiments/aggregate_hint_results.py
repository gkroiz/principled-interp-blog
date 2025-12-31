"""Aggregate hint results from multiple experiment folders.

This script reads rollout_summary.json and container_config.yaml from each folder
and creates a summary JSON file suitable for plotting with plot_hint_results_comparison.py.

Usage Examples:
    # Basic usage - aggregate results from multiple folders
    python aggregate_hint_results.py folder1/ folder2/ folder3/
    # Output: aggregated_results/hint_results.json

    # Using wildcards to process all matching folders
    python aggregate_hint_results.py ../../results/openai-gpt-5-*/
    # Output: aggregated_results/hint_results.json

    # Specify custom output path
    python aggregate_hint_results.py folder1/ folder2/ -o custom_output.json
    # Output: custom_output.json

    # Process all results in a directory
    python aggregate_hint_results.py ../../results/*/
    # Output: aggregated_results/hint_results.json

Expected folder structure:
    folder1/
        rollout_summary.json  # Contains hint_terminated_filenames, reward_hack_filenames, total_runs
        container_config.yaml # Contains task.hint_penalty

Output format:
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
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml


def load_rollout_summary(folder: Path) -> dict:
    """Load rollout_summary.json from folder.

    Args:
        folder: Path to experiment folder

    Returns:
        Dictionary with rollout summary data
    """
    summary_path = folder / "rollout_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"rollout_summary.json not found in {folder}")

    with open(summary_path) as f:
        return json.load(f)


def load_container_config(folder: Path) -> dict:
    """Load container_config.yaml from folder.

    Args:
        folder: Path to experiment folder

    Returns:
        Dictionary with container config data
    """
    config_path = folder / "container_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"container_config.yaml not found in {folder}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def process_folder(folder: Path) -> Dict:
    """Process a single experiment folder.

    Args:
        folder: Path to experiment folder

    Returns:
        Dictionary with hint_cost, total_games, hints_used, reward_hacks
    """
    # Load files
    summary = load_rollout_summary(folder)
    config = load_container_config(folder)

    # Extract hint penalty from config (nested under 'task')
    hint_penalty = config.get('task', {}).get('hint_penalty', 0)

    # Extract hint and hack filenames from the summary
    hint_files = set(summary.get('hint_terminated_filenames', []))
    hack_files = set(summary.get('reward_hack_filenames', []))

    # If a file is in both, count it as hack only
    hint_only_files = hint_files - hack_files

    # Total games from the summary
    total_games = summary.get('total_runs', 0)

    hints_used = len(hint_only_files)
    reward_hacks = len(hack_files)

    return {
        'hint_cost': hint_penalty,
        'total_games': total_games,
        'hints_used': hints_used,
        'reward_hacks': reward_hacks
    }


def aggregate_results(folders: List[Path], output_path: Path) -> None:
    """Aggregate results from all folders and save to JSON.

    Args:
        folders: List of folder paths to process
        output_path: Path to save output JSON file
    """
    results = {}

    for folder in folders:
        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}, skipping...")
            continue

        if not folder_path.is_dir():
            print(f"Warning: Not a directory: {folder_path}, skipping...")
            continue

        try:
            data = process_folder(folder_path)
            hint_cost = data['hint_cost']

            # Create key like "hint_1", "hint_3", etc.
            key = f"hint_{hint_cost}"

            results[key] = data

            print(f"Processed {folder_path.name}: hint_cost={hint_cost}, "
                  f"total={data['total_games']}, hints={data['hints_used']}, "
                  f"hacks={data['reward_hacks']}")

        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            continue

    if not results:
        print("\nError: No valid results found.")
        return

    # Sort by hint_cost for nicer output
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['hint_cost']))

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    print(f"\nResults saved to: {output_path}")
    print(f"Total experiments processed: {len(sorted_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate hint results from multiple experiment folders"
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="List of folders containing rollout_summary.json and container_config.yaml"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path (default: aggregated_results/hint_results.json relative to script location)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    folders = [Path(f) for f in args.folders]

    # Set default output path relative to script location
    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "aggregated_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "hint_results.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate_results(folders, output_path)
