"""Generate rollouts in parallel using run-parallel.sh infrastructure."""

import subprocess
import json
from pathlib import Path
import fire
import config


def verify_base_state():
    """Verify that the base state file exists."""
    if not config.BASE_STATE_FILE.exists():
        raise FileNotFoundError(
            f"Base state file not found: {config.BASE_STATE_FILE}\n"
            f"Please create the base state file first."
        )

    print(f"✓ Using base state: {config.BASE_STATE_FILE}")
    print(f"  Step: {config.get_base_step()}")

    return config.BASE_STATE_FILE


def extract_rollout_metadata(state_dir: Path, output_file: Path) -> dict:
    """
    Extract metadata from a completed rollout.
    
    Reads run_metadata.json that player.py automatically saves.
    """
    # Read metadata saved by player.py
    metadata_file = state_dir / "run_metadata.json"
    
    if not metadata_file.exists():
        return {
            "error": "run_metadata.json not found",
            "terminated_on_hint": False,
        }
    
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        return metadata
    except json.JSONDecodeError:
        return {
            "error": "failed to parse run_metadata.json",
            "terminated_on_hint": False,
        }


def generate_rollouts_parallel(
    num_rollouts: int = None,
    setup: bool = False,
):
    """
    Generate rollouts in parallel using run-parallel.sh.
    
    This is MUCH faster than sequential generation!
    Rollouts are property-agnostic and can be reused for testing multiple properties.
    
    Args:
        num_rollouts: Number of rollouts to generate (default from config)
        setup: Whether to setup base state first
    
    Examples:
        # Setup and generate 50 rollouts in parallel
        python generate_rollouts_parallel.py --setup --num_rollouts 50

        # Generate without setup (if base state already exists)
        python generate_rollouts_parallel.py --num_rollouts 50
    """
    
    if setup:
        print("\n" + "="*60)
        print("VERIFYING BASE STATE")
        print("="*60)
        base_state_path = verify_base_state()
    else:
        base_state_path = config.BASE_STATE_FILE
        if not base_state_path.exists():
            print(f"Error: Base state not found at {base_state_path}")
            print("Run with --setup first")
            return
    
    if num_rollouts is None:
        num_rollouts = config.Y_ROLLOUTS
    
    # Create rollouts directory if it doesn't exist (handle WSL weirdness)
    try:
        config.ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass  # Directory already exists (WSL/Windows filesystem quirk)
    
    output_dir = config.ROLLOUTS_DIR
    
    print("\n" + "="*60)
    print("GENERATING ROLLOUTS IN PARALLEL")
    print("="*60)
    print(f"Number of rollouts: {num_rollouts}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {config.MODEL}")
    print(f"Base state: {base_state_path}")
    print(f"Docker image: {config.DOCKER_IMAGE}")
    print("="*60 + "\n")
    
    # Build run-parallel.sh command (in project root)
    # Note: run-parallel.sh reads MAX_STEPS from .env, not as an argument
    # Use relative paths from project root to avoid double-path issues
    project_root = config.BASE_DIR.parent
    run_parallel_script = str(project_root / "run-parallel.sh")
    
    # Convert paths to be relative to project root for run-parallel.sh
    base_state_rel = str(base_state_path.relative_to(project_root))
    output_dir_rel = str(output_dir.relative_to(project_root))
    
    cmd = [
        "bash",  # Need bash to run .sh script (works in WSL and Unix)
        run_parallel_script,
        "-n", str(num_rollouts),
        "-m", config.MODEL,
        "-t", config.DOCKER_IMAGE,
        "-s",  # Save state
        "-i", base_state_rel,  # Resume from base state (relative path)
        "--base-dir", output_dir_rel,  # Save to rollouts/ directory (relative path)
        "--system-prompt", config.SYSTEM_PROMPT,
        "--user-prompt", config.USER_PROMPT,
        "--win-value", str(config.GAME_CONFIG["win_value"]),
        "--draw-value", str(config.GAME_CONFIG["draw_value"]),
        "--loss-value", str(config.GAME_CONFIG["loss_value"]),
        "--num-practice-games", str(config.GAME_CONFIG["num_practice_games"]),
        "--num-official-games", str(config.GAME_CONFIG["num_official_games"]),
        "--hint-penalty", str(config.GAME_CONFIG["hint_penalty"]),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("\nThis will run all rollouts in parallel...")
    print("Press Ctrl+C to stop (containers will keep running)\n")
    
    # Run from project root (where run-parallel.sh expects to be)
    project_root = config.BASE_DIR.parent
    
    try:
        # Run the parallel script (it handles everything)
        result = subprocess.run(cmd, check=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"\nError: run-parallel.sh failed with code {result.returncode}")
            return
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running run-parallel.sh: {e}")
        return
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Containers may still be running.")
        print("Use 'docker ps' to check and 'docker stop <id>' if needed")
        return
    
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    
    # Find the output directory created by run-parallel.sh
    # It will be in the format: {MODEL_NAME}-{TIMESTAMP}
    # We need to find the most recent one
    model_name = config.MODEL.replace("/", "-").replace(":", "-")
    
    # Find all directories matching the pattern in the rollouts directory
    matching_dirs = sorted(
        [d for d in config.ROLLOUTS_DIR.iterdir() if d.is_dir() and d.name.startswith(model_name)],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not matching_dirs:
        print(f"Error: No output directory found in {config.ROLLOUTS_DIR}")
        return
    
    parallel_output_dir = matching_dirs[0]
    print(f"Found output directory: {parallel_output_dir}")
    
    # Process each rollout and extract metadata
    metadata_list = []
    
    for i in range(num_rollouts):
        rollout_id = i + 1  # run-parallel.sh uses 1-indexed
        state_dir = parallel_output_dir / f"state-run{rollout_id}"
        output_file = parallel_output_dir / f"run-{model_name}-*-run{rollout_id}.txt"
        
        # Find the actual output file (has timestamp in name)
        output_files = list(parallel_output_dir.glob(f"run-*-run{rollout_id}.txt"))
        if output_files:
            output_file = output_files[0]
        
        if not state_dir.exists():
            print(f"  Rollout {i}: No state directory found")
            metadata_list.append({
                "rollout_id": i,
                "error": "no state directory",
                "terminated_on_hint": False,
                "reward_hacked": False,
            })
            continue
        
        metadata = extract_rollout_metadata(state_dir, output_file)
        metadata["rollout_id"] = i
        metadata["output_dir"] = str(state_dir)
        metadata_list.append(metadata)
        
        # Print progress
        if metadata.get("error"):
            print(f"  Rollout {i}: ERROR - {metadata['error']}")
        elif metadata.get("terminated_on_hint"):
            print(f"  Rollout {i}: Terminated on hint (step {metadata.get('final_step', '?')})")
        else:
            print(f"  Rollout {i}: Completed (step {metadata.get('final_step', '?')})")
    
    # Save summary in the parallel output directory
    summary_file = parallel_output_dir / "rollouts_summary.json"
    
    with open(summary_file, "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"✓ Generated {len(metadata_list)} rollouts")
    print(f"  - Saved summary to {summary_file}")
    
    # Count outcomes
    hinted = sum(1 for m in metadata_list if m.get("terminated_on_hint"))
    errors = sum(1 for m in metadata_list if m.get("error"))
    
    print(f"\nOutcomes:")
    print(f"  - Terminated on hint: {hinted}")
    print(f"  - Errors: {errors}")
    print(f"  - Completed normally: {len(metadata_list) - hinted - errors}")
    
    print(f"\nParallel output directory: {parallel_output_dir}")


def main():
    """CLI entry point using Fire."""
    fire.Fire(generate_rollouts_parallel)


if __name__ == "__main__":
    main()

