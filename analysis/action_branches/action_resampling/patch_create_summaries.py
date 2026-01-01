"""
Patch script to create rollouts_summary.json for existing rollouts.

Run this if you generated rollouts before the summary creation logic was added.
"""

import json
import sys
from pathlib import Path

import fire

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))
import config


def create_rollouts_summary(rollout_dir: Path):
    """
    Create rollouts_summary.json by aggregating run_metadata.json files.

    Args:
        rollout_dir: Directory containing state-runX subdirectories
    """
    # Find all state-runX directories
    state_dirs = sorted(rollout_dir.glob("state-run*"))

    if not state_dirs:
        print(f"  ⚠️  No state-run directories found in {rollout_dir.name}")
        return False

    summary = {}

    for state_dir in state_dirs:
        # Extract run number from directory name (e.g., "state-run1" -> "run1")
        run_id = state_dir.name.replace("state-", "")

        # Read run_metadata.json
        metadata_file = state_dir / "run_metadata.json"
        if not metadata_file.exists():
            print(f"  ⚠️  Missing metadata for {run_id}")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Add to summary
        summary[run_id] = {
            "terminated_on_hint": metadata.get("terminated_on_hint", False),
            "final_step": metadata.get("final_step", 0),
            "termination_reason": metadata.get("termination_reason", "unknown"),
            "game_results": metadata.get("game_results", []),
            "total_hints": metadata.get("total_hints", 0),
        }

    if not summary:
        print("  ❌ No metadata files found")
        return False

    # Save rollouts_summary.json
    summary_file = rollout_dir / "rollouts_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✅ Created {summary_file.name} with {len(summary)} rollouts")
    return True


def create_summaries_for_run(rollouts_base: Path):
    """
    Create rollouts_summary.json for all step directories in a run.

    Args:
        rollouts_base: Base directory containing step_XX subdirectories

    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Find all step_XX directories
    step_dirs = sorted(rollouts_base.glob("step_*"))

    if not step_dirs:
        raise FileNotFoundError(f"No step directories found in {rollouts_base}")

    successful = 0
    failed = 0

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[-1])

        # Find the timestamped rollout directory using model from config
        rollout_dirs = list(step_dir.glob(f"{config.MODEL}-*"))
        if not rollout_dirs:
            print(f"  Step {step_num:2d}: ⚠️  No rollout directory found")
            failed += 1
            continue

        # Use the LATEST (most recent) directory in case of restarts
        rollout_dir = sorted(rollout_dirs)[-1]

        # Check if summary already exists
        summary_file = rollout_dir / "rollouts_summary.json"
        if summary_file.exists():
            print(f"  Step {step_num:2d}: ⏭️  Summary already exists")
            successful += 1
            continue

        # Create summary
        print(f"  Step {step_num:2d}: Creating summary...", end=" ")
        if create_rollouts_summary(rollout_dir):
            successful += 1
        else:
            failed += 1

    print(
        f"\n  Summary: {successful}/{len(step_dirs)} successful, {failed}/{len(step_dirs)} failed"
    )
    return successful, failed


def main(run_name: str = None):
    """
    Create rollouts_summary.json for all existing rollouts.

    Args:
        run_name: Specific run to patch (e.g., "run5"). If None, uses config.RUN_TO_PROCESS
    """
    if run_name:
        rollouts_base = config.BASE_DIR / "rollouts" / run_name
    else:
        rollouts_base = config.ROLLOUTS_DIR
        run_name = config.RUN_TO_PROCESS.replace("state-", "")

    if not rollouts_base.exists():
        print(f"❌ Rollouts directory not found: {rollouts_base}")
        return

    print(f"\n{'=' * 70}")
    print(f"PATCHING ROLLOUTS: {run_name}")
    print(f"{'=' * 70}")
    print(f"Looking for rollout directories in: {rollouts_base}\n")

    # Find all step_XX directories
    step_dirs = sorted(rollouts_base.glob("step_*"))

    if not step_dirs:
        print(f"❌ No step directories found in {rollouts_base}")
        return

    print(f"Found {len(step_dirs)} step directories\n")

    successful = 0
    failed = 0

    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[-1])
        print(f"Step {step_num:2d}:")

        # Find the timestamped rollout directory using model from config
        rollout_dirs = list(step_dir.glob(f"{config.MODEL}-*"))
        if not rollout_dirs:
            print("  ⚠️  No rollout directory found")
            failed += 1
            continue

        # Use the LATEST (most recent) directory in case of restarts
        rollout_dir = sorted(rollout_dirs)[-1]

        # Check if summary already exists
        summary_file = rollout_dir / "rollouts_summary.json"
        if summary_file.exists():
            print("  ⏭️  Summary already exists, skipping")
            successful += 1
            continue

        # Create summary
        if create_rollouts_summary(rollout_dir):
            successful += 1
        else:
            failed += 1

    print(f"\n{'=' * 70}")
    print("PATCH COMPLETE")
    print(f"  Successful: {successful}/{len(step_dirs)}")
    print(f"  Failed: {failed}/{len(step_dirs)}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    fire.Fire(main)
