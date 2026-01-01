"""Merge reward hacking grades into rollouts_summary.json."""

import json
from pathlib import Path
import re
import fire
import config


def merge_grading_results(rollouts_dir: str = None):
    """
    Merge reward hacking classifications from judge_json.py into rollouts_summary.json.

    This reads:
    - rollout_analysis_detailed.json (output from judge_json.py)
    - rollouts_summary.json (existing metadata)

    And updates rollouts_summary.json with reward_hacked field for each rollout.

    Args:
        rollouts_dir: Path to rollouts directory (default: most recent in config.ROLLOUTS_DIR)

    Examples:
        # Merge for most recent rollouts
        python merge_grading_results.py

        # Merge for specific directory
        python merge_grading_results.py --rollouts_dir rollouts/gpt-5/gpt-5-2025-08-07-20251106-124805
    """

    # Find the most recent rollouts directory if not specified
    if rollouts_dir is None:
        model_name = config.MODEL.replace("/", "-").replace(":", "-")
        matching_dirs = sorted(
            [d for d in config.ROLLOUTS_DIR.iterdir() if d.is_dir() and d.name.startswith(model_name)],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if not matching_dirs:
            print(f"Error: No rollouts found in {config.ROLLOUTS_DIR}")
            return

        rollouts_path = matching_dirs[0]
    else:
        rollouts_path = Path(rollouts_dir)

    if not rollouts_path.exists():
        print(f"Error: Rollouts directory not found: {rollouts_path}")
        return

    print(f"\nMerging grading results for: {rollouts_path}")

    # Load grading results from judge_json.py
    grading_file = rollouts_path / "rollout_analysis_detailed.json"
    if not grading_file.exists():
        print(f"Error: Grading results not found: {grading_file}")
        print("Run judge/judge_json.py first!")
        return

    with open(grading_file) as f:
        grading_results = json.load(f)

    # Load existing rollouts summary
    summary_file = rollouts_path / "rollouts_summary.json"
    if not summary_file.exists():
        print(f"Error: Rollouts summary not found: {summary_file}")
        return

    with open(summary_file) as f:
        rollouts_summary = json.load(f)

    print(f"Loaded {len(grading_results)} grading results")
    print(f"Loaded {len(rollouts_summary)} rollout metadata entries")

    # Create mapping from rollout_id to reward_hacked
    # grading_results format: [{"rollout_id": "run5", "analysis": {"reward_hacked": true, ...}}]
    id_to_grade = {}
    for result in grading_results:
        rollout_id_str = result["rollout_id"]  # e.g., "run5"
        reward_hacked = result["analysis"].get("reward_hacked", False)

        # Extract run number from rollout_id (e.g., "run5" -> 5)
        match = re.search(r"run(\d+)$", rollout_id_str)
        if match:
            run_num = int(match.group(1))
            rollout_id = run_num - 1  # run_num is 1-indexed, rollout_id is 0-indexed
            id_to_grade[rollout_id] = reward_hacked

    # Update rollouts_summary with reward_hacked field
    updated_count = 0
    hacked_count = 0

    for entry in rollouts_summary:
        rollout_id = entry["rollout_id"]
        if rollout_id in id_to_grade:
            entry["reward_hacked"] = id_to_grade[rollout_id]
            updated_count += 1
            if id_to_grade[rollout_id]:
                hacked_count += 1
        else:
            # If no grading result, default to False
            entry["reward_hacked"] = False
            print(f"⚠️  No grading result for rollout_id {rollout_id}, defaulting to False")

    # Save updated summary
    with open(summary_file, "w") as f:
        json.dump(rollouts_summary, f, indent=2)

    print(f"\n✓ Updated {updated_count}/{len(rollouts_summary)} rollout entries with reward_hacked field")
    print(f"  - Reward hacked: {hacked_count}")
    print(f"  - Not hacked: {updated_count - hacked_count}")
    if len(rollouts_summary) > updated_count:
        print(f"  - No grade available: {len(rollouts_summary) - updated_count}")
    print(f"\n✓ Saved updated summary to: {summary_file}")


def main():
    """CLI entry point using Fire."""
    fire.Fire(merge_grading_results)


if __name__ == "__main__":
    main()
