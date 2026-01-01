"""Regenerate missing rollouts from failed steps."""
import json
from pathlib import Path
import subprocess
import sys
import config
from datetime import datetime

def count_successful_rollouts(step_dir: Path) -> int:
    """Count rollouts with valid run_metadata.json in a step directory."""
    if not step_dir.exists():
        return 0
    
    # Find the timestamped rollout directory
    rollout_dirs = list(step_dir.glob(f"{config.MODEL}-*"))
    if not rollout_dirs:
        return 0
    
    rollout_dir = rollout_dirs[0]
    
    # Count how many run_metadata.json files exist
    # Directories are named either "run*" or "state-run*" depending on the experiment
    metadata_files = list(rollout_dir.glob("*/run_metadata.json"))
    return len(metadata_files)


def get_incomplete_steps(min_threshold: int = None) -> list:
    """
    Find all steps with fewer than min_threshold successful rollouts.
    
    Args:
        min_threshold: Minimum acceptable number of rollouts. 
                       If None, uses ROLLOUTS_PER_STEP (20).
    """
    if min_threshold is None:
        min_threshold = config.ROLLOUTS_PER_STEP
    
    incomplete_steps = []
    
    # Get all state files
    state_files = sorted(config.INPUT_STATE_DIR.glob("ctfish-tictactoe-*.json"))
    
    # Read metadata to get final_step
    metadata_file = config.INPUT_STATE_DIR / "run_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        final_step = metadata.get("final_step")
        if final_step is not None:
            max_step = final_step - 1
            state_files = [f for f in state_files if int(f.stem.split("-")[-1]) <= max_step]
    
    print(f"Checking {len(state_files)} steps for completeness...\n")
    
    for state_file in state_files:
        step_num = int(state_file.stem.split("-")[-1])
        step_dir = config.ROLLOUTS_DIR / f"step_{step_num:02d}"
        
        count = count_successful_rollouts(step_dir)
        expected = config.ROLLOUTS_PER_STEP
        
        if count < min_threshold:
            incomplete_steps.append({
                "step": step_num,
                "count": count,
                "expected": expected,
                "missing": expected - count,
                "state_file": state_file,
                "output_dir": step_dir
            })
            status = "❌" if count == 0 else "⚠️ "
            print(f"{status} Step {step_num:2d}: {count}/{expected} rollouts (below threshold: {min_threshold})")
        else:
            print(f"✅ Step {step_num:2d}: {count}/{expected} rollouts")
    
    return incomplete_steps


def regenerate_step(step_info: dict, retry: bool = False) -> bool:
    """
    Regenerate rollouts for a single incomplete step.
    
    Args:
        step_info: Dictionary with step information
        retry: If True, delete existing rollouts first. If False, append to existing.
    
    Returns:
        True if successful, False otherwise
    """
    step_num = step_info["step"]
    state_file = step_info["state_file"]
    output_dir = step_info["output_dir"]
    
    # If retry, remove the existing timestamped directory
    if retry and output_dir.exists():
        import shutil
        rollout_dirs = list(output_dir.glob(f"{config.MODEL}-*"))
        for rd in rollout_dirs:
            print(f"  🗑️  Removing existing directory: {rd.name}")
            shutil.rmtree(rd)
    
    project_root = config.BASE_DIR.parent
    run_parallel_script = str(project_root / "run-parallel.sh")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert paths to be relative to project root
    state_file_rel = str(state_file.relative_to(project_root))
    output_dir_rel = str(output_dir.relative_to(project_root))
    
    # Generate only the missing count (or all if retry)
    num_to_generate = config.ROLLOUTS_PER_STEP if retry else step_info["missing"]
    
    cmd = [
        "bash",
        run_parallel_script,
        "-n", str(num_to_generate),
        "-m", config.MODEL,
        "-t", config.DOCKER_IMAGE,
        "-s",  # Save state flag
        "-i", state_file_rel,  # Initial state file
        "--base-dir", output_dir_rel,  # Output directory
        "--system-prompt", config.SYSTEM_PROMPT,
        "--user-prompt", config.USER_PROMPT,
        "--win-value", str(config.GAME_CONFIG["win_value"]),
        "--draw-value", str(config.GAME_CONFIG["draw_value"]),
        "--loss-value", str(config.GAME_CONFIG["loss_value"]),
        "--num-practice-games", str(config.GAME_CONFIG["num_practice_games"]),
        "--num-official-games", str(config.GAME_CONFIG["num_official_games"]),
        "--hint-penalty", str(config.GAME_CONFIG["hint_penalty"]),
    ]
    
    print(f"  Generating {num_to_generate} rollouts...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"  STDOUT: {e.stdout[:500]}")
        if e.stderr:
            print(f"  STDERR: {e.stderr[:500]}")
        return False


def main(mode: str = "check", max_parallel: int = 1, min_threshold: int = 10):
    """
    Main function to regenerate missing rollouts.
    
    Args:
        mode: "check" (default) - just show incomplete steps
              "regenerate" - regenerate missing rollouts only
              "retry" - delete and regenerate all rollouts for incomplete steps
        max_parallel: How many steps to process in parallel (default 1 to avoid rate limits)
        min_threshold: Minimum acceptable number of rollouts (default 10)
                       Steps with fewer rollouts will be regenerated
    """
    print(f"\n{'='*70}")
    print(f"CHECKING FOR INCOMPLETE ROLLOUTS: {config.RUN_TO_PROCESS}")
    print(f"Using threshold: {min_threshold} rollouts (will regenerate steps below this)")
    print(f"{'='*70}\n")
    
    incomplete = get_incomplete_steps(min_threshold=min_threshold)
    
    if not incomplete:
        print("\n✅ All steps have complete rollouts!")
        return
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(incomplete)} incomplete steps found")
    print(f"{'='*70}")
    total_missing = sum(s["missing"] for s in incomplete)
    print(f"Total missing rollouts: {total_missing}")
    print(f"\nIncomplete steps: {[s['step'] for s in incomplete]}")
    print(f"{'='*70}\n")
    
    if mode == "check":
        print("Run with mode='regenerate' to generate missing rollouts")
        print("Run with mode='retry' to delete and regenerate incomplete steps")
        return
    
    if mode not in ["regenerate", "retry"]:
        print(f"Invalid mode: {mode}. Use 'check', 'regenerate', or 'retry'")
        return
    
    retry_mode = (mode == "retry")
    
    print(f"Starting regeneration (mode={mode}, max_parallel={max_parallel})...\n")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Process steps with limited parallelism to avoid rate limits
    errors = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_step = {
            executor.submit(regenerate_step, step_info, retry_mode): step_info
            for step_info in incomplete
        }
        
        completed = 0
        total = len(incomplete)
        
        for future in as_completed(future_to_step):
            step_info = future_to_step[future]
            step_num = step_info["step"]
            completed += 1
            
            try:
                success = future.result()
                if success:
                    print(f"[{completed}/{total}] Step {step_num:2d} - ✅ Success")
                else:
                    print(f"[{completed}/{total}] Step {step_num:2d} - ❌ Failed")
                    errors.append(step_num)
            except Exception as e:
                print(f"[{completed}/{total}] Step {step_num:2d} - ❌ Exception: {e}")
                errors.append(step_num)
    
    print(f"\n{'='*70}")
    print(f"REGENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {completed}/{total} steps")
    print(f"Successful: {completed - len(errors)}/{total}")
    print(f"Failed: {len(errors)}/{total}")
    if errors:
        print(f"Failed steps: {errors}")
    print(f"{'='*70}\n")
    
    # Create rollouts_summary.json files
    print(f"Creating rollout summaries...")
    from patch_create_summaries import create_summaries_for_run
    try:
        create_summaries_for_run(config.ROLLOUTS_DIR)
        print(f"✅ Rollout summaries created successfully\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not create rollout summaries: {str(e)}\n")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

