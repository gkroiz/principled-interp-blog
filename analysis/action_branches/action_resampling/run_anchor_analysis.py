#!/usr/bin/env python3
"""
Orchestrate full anchor analysis pipeline.

This script runs the complete action resampling analysis:
1. Resample: Generate N rollouts from each step checkpoint
2. Grade: Run grade_rollouts.py on each resampled folder
3. Plot: Compute metrics and create visualization

Usage:
    python run_anchor_analysis.py <run-folder> [options]
    python run_anchor_analysis.py results/tictactoe/claude/2025-12-31/run-1 --count 20
    
Options:
    --count N         Number of rollouts per step (default: 10)
    --local           Use local Docker image
    --skip-resample   Skip resampling (use existing resampled rollouts)
    --skip-grade      Skip grading (use existing analysis)
    --game GAME       Game type for grading (auto-detected from config.yaml)
    --max-parallel N  Maximum parallel resample jobs (default: 4)

Output:
    <run-folder>/step-N/<timestamp>/run-*/rollout.log  (resampled rollouts)
    <run-folder>/step-N/<timestamp>/rollout_analysis_detailed.json  (grading)
    <run-folder>/anchor_metrics.json  (aggregated metrics)
    <run-folder>/anchor_plot.png  (visualization)
"""

import subprocess
import sys
from pathlib import Path

import fire
import yaml


def detect_game_from_config(run_folder: Path) -> str | None:
    """Auto-detect game type from config.yaml in the parent directory.
    
    Config is expected at run_folder/../config.yaml (the timestamp directory).
    Returns the environment name or None if not found.
    """
    config_path = run_folder.parent / "config.yaml"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("environment")
    except Exception:
        return None


def find_resampled_dirs(run_folder: Path) -> list[tuple[int, Path]]:
    """Find all timestamped resampled directories in step folders.
    
    Returns list of (step_num, timestamp_dir) tuples.
    """
    results = []
    
    for step_dir in sorted(run_folder.glob("step-*")):
        step_num = int(step_dir.name.split("-")[1])
        
        # Find timestamped directories
        timestamp_dirs = [
            d for d in step_dir.iterdir()
            if d.is_dir() and len(d.name) == 19 and d.name[4] == '-' and d.name[10] == '_'
        ]
        
        if timestamp_dirs:
            # Use the most recent one
            latest_dir = sorted(timestamp_dirs)[-1]
            results.append((step_num, latest_dir))
    
    return results


def run_resample(run_folder: Path, count: int, local: bool, max_parallel: int = 1) -> bool:
    """Run resample.sh on the run folder."""
    print(f"\n{'='*70}")
    print("STEP 1: RESAMPLING")
    print(f"{'='*70}\n")
    
    # Find the resample script
    script_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    resample_script = script_dir / "resample.sh"
    
    if not resample_script.exists():
        print(f"Error: resample.sh not found at {resample_script}")
        return False
    
    cmd = ["bash", str(resample_script), str(run_folder), "--count", str(count), "--max-parallel", str(max_parallel)]
    if local:
        cmd.append("--local")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Resampling failed with exit code {result.returncode}")
        return False
    
    print("\n✅ Resampling completed")
    return True


def run_grading(run_folder: Path, game: str) -> bool:
    """Run grade_rollouts.py on each resampled folder."""
    print(f"\n{'='*70}")
    print("STEP 2: GRADING")
    print(f"{'='*70}\n")
    
    # Find the grading script
    grading_dir = Path(__file__).parent.parent.parent / "grading"
    grade_script = grading_dir / "grade_rollouts.py"
    
    if not grade_script.exists():
        print(f"Error: grade_rollouts.py not found at {grade_script}")
        return False
    
    # Find all resampled directories
    resampled_dirs = find_resampled_dirs(run_folder)
    
    if not resampled_dirs:
        print("No resampled directories found. Run resampling first.")
        return False
    
    print(f"Found {len(resampled_dirs)} steps to grade\n")
    
    success_count = 0
    fail_count = 0
    
    for step_num, timestamp_dir in resampled_dirs:
        print(f"Step {step_num:2d}: ", end="", flush=True)
        
        # Check if already graded
        analysis_file = timestamp_dir / "rollout_analysis_detailed.json"
        if analysis_file.exists():
            print("⏭️  Already graded, skipping")
            success_count += 1
            continue
        
        # Run grading
        cmd = [
            sys.executable,
            str(grade_script),
            "--output_dir", str(timestamp_dir),
            "--game", game,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            fail_count += 1
        else:
            print("✅ Graded")
            success_count += 1
    
    print(f"\n✅ Grading completed: {success_count} succeeded, {fail_count} failed")
    return fail_count == 0


def run_plotting(run_folder: Path) -> bool:
    """Run plot_anchors.py on the run folder."""
    print(f"\n{'='*70}")
    print("STEP 3: PLOTTING")
    print(f"{'='*70}\n")
    
    # Import and run directly
    from plot_anchors import main as plot_main
    
    try:
        plot_main(str(run_folder))
        return True
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        return False


def main(
    run_folder: str,
    count: int = 10,
    local: bool = False,
    skip_resample: bool = False,
    skip_grade: bool = False,
    game: str = None,
    max_parallel: int = 1,
):
    """
    Run the complete anchor analysis pipeline.
    
    Args:
        run_folder: Path to run folder (e.g., results/.../run-1)
        count: Number of rollouts to generate per step
        local: Use local Docker image instead of Dockerhub
        skip_resample: Skip the resampling step
        skip_grade: Skip the grading step
        game: Game type for grading (auto-detected from config.yaml, override if needed)
        max_parallel: Maximum number of parallel resample jobs (default: 4)
    """
    run_path = Path(run_folder)
    
    if not run_path.exists():
        print(f"Error: Run folder '{run_folder}' does not exist")
        sys.exit(1)
    
    # Check for step directories
    step_dirs = list(run_path.glob("step-*"))
    if not step_dirs:
        print(f"Error: No step-* directories found in '{run_folder}'")
        sys.exit(1)
    
    # Auto-detect game from config if not provided
    if game is None:
        detected_game = detect_game_from_config(run_path)
        if detected_game:
            game = detected_game
            print(f"Auto-detected game from config.yaml: {game}")
        else:
            game = "tictactoe"
            print(f"Could not auto-detect game, using default: {game}")
    
    print(f"\n{'='*70}")
    print("ANCHOR ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Run folder: {run_path}")
    print(f"Steps found: {len(step_dirs)}")
    print(f"Rollouts per step: {count}")
    print(f"Max parallel: {max_parallel}")
    print(f"Game: {game}")
    print(f"{'='*70}\n")
    
    # Step 1: Resample
    if not skip_resample:
        if not run_resample(run_path, count, local, max_parallel):
            print("\n❌ Pipeline failed at resampling step")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping resampling (--skip-resample)")
    
    # Step 2: Grade
    if not skip_grade:
        if not run_grading(run_path, game):
            print("\n⚠️  Some grading failures occurred, continuing anyway...")
    else:
        print("\n⏭️  Skipping grading (--skip-grade)")
    
    # Step 3: Plot
    if not run_plotting(run_path):
        print("\n❌ Pipeline failed at plotting step")
        sys.exit(1)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ ANCHOR ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  - {run_path / 'anchor_metrics.json'}")
    print(f"  - {run_path / 'anchor_plot.png'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    fire.Fire(main)

