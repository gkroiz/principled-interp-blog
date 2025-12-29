"""Generate N rollouts from each state file in the configured run."""
import fire
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import json
from datetime import datetime

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))
import config


def main(skip_existing: bool = True, max_parallel: int = 5, step: int = None):
    """
    Generate rollouts for all steps in the configured run.
    
    The run to process is specified in config.RUN_TO_PROCESS.
    
    Args:
        skip_existing: Skip steps that already have rollout directories
        max_parallel: Maximum number of steps to process in parallel (default: 5)
        step: If specified, only generate rollouts for this specific step number
    """
    
    if not config.INPUT_STATE_DIR.exists():
        print(f"❌ Error: {config.INPUT_STATE_DIR} not found!")
        print(f"   Make sure you've copied the state folder to input_states/")
        print(f"\n   Example:")
        print(f"   cp -r ../o3_hint_rollouts/.../state-run5 turn_anchors/input_states/")
        return
    
    run_id = config.RUN_TO_PROCESS.replace("state-", "")
    
    print(f"\n{'='*70}")
    print(f"TURN ANCHORS: Processing {run_id}")
    print(f"{'='*70}")
    print(f"Config:")
    print(f"  - Input: {config.INPUT_STATE_DIR}")
    print(f"  - Output: {config.ROLLOUTS_DIR}")
    print(f"  - Rollouts per step: {config.ROLLOUTS_PER_STEP}")
    print(f"  - Hint penalty: {config.GAME_CONFIG['hint_penalty']}")
    print(f"  - Max parallel steps: {max_parallel}")
    print(f"{'='*70}\n")
    
    # Read run metadata to get final_step
    metadata_file = config.INPUT_STATE_DIR / "run_metadata.json"
    if not metadata_file.exists():
        print(f"❌ Error: run_metadata.json not found in {config.INPUT_STATE_DIR}")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    final_step = metadata.get("final_step")
    if final_step is None:
        print(f"❌ Error: 'final_step' not found in run_metadata.json")
        return
    
    print(f"Run terminated at step {final_step}")
    print(f"  Reason: {metadata.get('termination_reason', 'unknown')}")
    print(f"  Terminated on hint: {metadata.get('terminated_on_hint', False)}")
    print(f"  Total hints: {metadata.get('total_hints', 0)}\n")
    
    # Find all state files
    state_files = sorted(config.INPUT_STATE_DIR.glob("ctfish-tictactoe-*.json"))
    
    if not state_files:
        print(f"❌ No state files found in {config.INPUT_STATE_DIR}")
        return
    
    # Filter to only include states BEFORE the terminal step
    # (We don't want to generate rollouts from the state after the terminal action)
    max_step = final_step - 1
    state_files = [f for f in state_files if int(f.stem.split("-")[-1]) <= max_step]
    
    # If specific step requested, filter to just that step
    if step is not None:
        state_files = [f for f in state_files if int(f.stem.split("-")[-1]) == step]
        if not state_files:
            print(f"❌ Error: Step {step} not found or is after terminal action (max step: {max_step})")
            return
        print(f"Processing only step {step}")
    else:
        print(f"Found {len(state_files)} state files (steps 0-{max_step}) to process")
        print(f"  (Excluding steps {max_step + 1}-{final_step} which are after terminal action)\n")
    
    # Create base rollouts directory
    config.ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter out steps to skip
    tasks = []
    for state_file in state_files:
        step_num = int(state_file.stem.split("-")[-1])
        output_dir = config.ROLLOUTS_DIR / f"step_{step_num:02d}"
        
        # If specific step requested, always generate (don't skip existing)
        if skip_existing and output_dir.exists() and step is None:
            print(f"⏭️  Step {step_num:2d} - Already exists, skipping")
            continue
        
        tasks.append((state_file, step_num, output_dir))
    
    if not tasks:
        print("✅ All steps already completed!")
        return
    
    print(f"\nProcessing {len(tasks)} steps with {max_parallel} parallel workers...\n")
    
    # Create error log file
    run_id = config.RUN_TO_PROCESS.replace("state-", "")
    # Extract just the run name (e.g., "run1" from "gpt5/run1")
    run_name = Path(run_id).name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    error_log_path = config.SUMMARY_DIR / f"{run_name}_generation_errors_{timestamp}.log"
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process steps in parallel
    errors = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_step = {
            executor.submit(generate_rollouts_for_step, state_file, step_num, output_dir): (step_num, state_file)
            for state_file, step_num, output_dir in tasks
        }
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_step):
            step_num, state_file = future_to_step[future]
            completed += 1
            
            try:
                success, error_msg = future.result()
                if success:
                    print(f"[{completed}/{total}] Step {step_num:2d} - ✅ Success")
                else:
                    print(f"[{completed}/{total}] Step {step_num:2d} - ❌ Failed")
                    error_info = {
                        "step": step_num,
                        "state_file": str(state_file),
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    }
                    errors.append(error_info)
                    
                    # Check for specific error types
                    if error_msg and "rate" in error_msg.lower():
                        print(f"        ⚠️  Rate limit error detected")
                    elif error_msg and "openai" in error_msg.lower():
                        print(f"        ⚠️  OpenAI API error detected")
                        
            except Exception as e:
                print(f"[{completed}/{total}] Step {step_num:2d} - ❌ Exception: {e}")
                error_info = {
                    "step": step_num,
                    "state_file": str(state_file),
                    "error": str(e),
                    "error_type": "exception",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
    
    # Save error log if there were any errors
    if errors:
        with open(error_log_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n⚠️  {len(errors)} errors occurred. Details saved to:")
        print(f"   {error_log_path}")
        print(f"\nError summary:")
        for err in errors:
            print(f"  - Step {err['step']}: {err['error'][:100]}...")
    
    print(f"\n{'='*70}")
    print(f"✅ Finished processing {run_id}")
    print(f"   Completed: {completed}/{total} steps")
    print(f"   Successful: {completed - len(errors)}/{total}")
    print(f"   Failed: {len(errors)}/{total}")
    print(f"   Rollouts saved to: {config.ROLLOUTS_DIR}")
    print(f"{'='*70}\n")
    
    # Create rollouts_summary.json files for all generated rollouts
    print(f"\n{'='*70}")
    print(f"CREATING ROLLOUT SUMMARIES")
    print(f"{'='*70}\n")
    
    from patch_create_summaries import create_summaries_for_run
    try:
        create_summaries_for_run(config.ROLLOUTS_DIR)
        print(f"\n✅ Rollout summaries created successfully")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not create rollout summaries: {str(e)}")
        print(f"   You can run patch_create_summaries.py manually later")
    
    print(f"\n{'='*70}\n")


def generate_rollouts_for_step(state_file: Path, step_num: int, output_dir: Path) -> Tuple[bool, str]:
    """
    Generate rollouts for a single step.
    
    Args:
        state_file: Path to the state JSON file
        step_num: Step number
        output_dir: Output directory for rollouts
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Prepare to call run-parallel.sh
        project_root = config.BASE_DIR.parent
        run_parallel_script = str(project_root / "run-parallel.sh")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert paths to be relative to project root
        state_file_rel = str(state_file.relative_to(project_root))
        output_dir_rel = str(output_dir.relative_to(project_root))
        
        cmd = [
            "bash",
            run_parallel_script,
            "-n", str(config.ROLLOUTS_PER_STEP),
            "-m", config.MODEL,
            "-t", config.DOCKER_IMAGE,
            "-s",  # Save state
            "-i", state_file_rel,
            "--base-dir", output_dir_rel,
            "--system-prompt", config.SYSTEM_PROMPT,
            "--user-prompt", config.USER_PROMPT,
            "--win-value", str(config.GAME_CONFIG["win_value"]),
            "--draw-value", str(config.GAME_CONFIG["draw_value"]),
            "--loss-value", str(config.GAME_CONFIG["loss_value"]),
            "--num-practice-games", str(config.GAME_CONFIG["num_practice_games"]),
            "--num-official-games", str(config.GAME_CONFIG["num_official_games"]),
            "--hint-penalty", str(config.GAME_CONFIG["hint_penalty"]),
        ]
        
        # Don't capture output so progress is visible
        result = subprocess.run(cmd, cwd=str(project_root))
        
        if result.returncode != 0:
            return (False, f"run-parallel.sh failed with exit code {result.returncode}")
        
        # Rollouts completed successfully
        return (True, "")
        
    except Exception as e:
        return (False, f"Exception during execution: {str(e)}")


if __name__ == "__main__":
    fire.Fire(main)

