"""Grade rollouts and compute metrics for all steps."""
import fire
import json
from pathlib import Path
import sys
import subprocess

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))
import config

def main():
    """
    For each step in the configured run:
    1. Collect all rollout directories
    2. Grade ALL rollouts in one async batch
    3. Compute metrics for each step
    4. Save results
    """
    
    run_id = config.RUN_TO_PROCESS.replace("state-", "")
    
    print(f"\n{'='*70}")
    print(f"COMPUTING METRICS: {run_id}")
    print(f"{'='*70}\n")
    
    # Read metadata to determine final step
    metadata_file = config.INPUT_STATE_DIR / "run_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        final_step = metadata.get("final_step")
        if final_step is not None:
            max_step = final_step - 1
            print(f"Run terminated at step {final_step}")
            print(f"  Only analyzing steps 0-{max_step} (before termination)\n")
        else:
            max_step = None
    else:
        max_step = None
        print(f"⚠️  No metadata found, analyzing all available steps\n")
    
    if not config.ROLLOUTS_DIR.exists():
        print(f"❌ No rollouts found at {config.ROLLOUTS_DIR}")
        print(f"   Run generate_all_anchors.py first!")
        return
    
    step_dirs = sorted(config.ROLLOUTS_DIR.glob("step_*"))
    
    # Filter out steps after termination
    if max_step is not None:
        step_dirs = [d for d in step_dirs if int(d.name.split("_")[-1]) <= max_step]
    
    if not step_dirs:
        print(f"❌ No step directories found in {config.ROLLOUTS_DIR}")
        return
    
    print(f"Found {len(step_dirs)} steps to process\n")
    
    # Collect all rollout directories
    rollout_dirs = []
    for step_dir in step_dirs:
        step_num = int(step_dir.name.split("_")[-1])
        
        # Find the timestamped rollout directory using model from config
        rollout_dir_matches = list(step_dir.glob(f"{config.MODEL}-*"))
        if not rollout_dir_matches:
            print(f"⚠️  Step {step_num:2d} - No rollouts found")
            continue
        
        # Use the LATEST (most recent) directory in case of restarts
        rollout_dir = sorted(rollout_dir_matches)[-1]
        rollout_dirs.append((step_num, rollout_dir))
    
    if not rollout_dirs:
        print("❌ No rollout directories found!")
        return
    
    # Now compute metrics for each step
    print(f"\n{'='*70}")
    print(f"COMPUTING METRICS PER STEP")
    print(f"{'='*70}\n")
    
    metrics_results = {}
    
    for step_num, rollout_dir in rollout_dirs:
        print(f"Step {step_num:2d}:", end=" ")
        
        # Grade if needed (skip if old format exists)
        old_format_path = rollout_dir / "rollout_summary.json"
        detailed_path = rollout_dir / "rollout_analysis_detailed.json"
        
        if not detailed_path.exists() and not old_format_path.exists():
            print("Grading...", end=" ", flush=True)
            # Call grade_rollouts_v2.py (don't capture output so progress bar shows)
            cmd = [
                "python", "judge/grade_rollouts_v2.py",
                "--output_dir", str(rollout_dir)
            ]
            result = subprocess.run(cmd, cwd=str(config.BASE_DIR.parent))
            if result.returncode != 0:
                print(f"❌ Failed")
                metrics_results[step_num] = {"error": "Grading failed"}
                continue
            print("✅", end=" ")
        
        # Compute metric
        metric = compute_metric_for_rollout(rollout_dir)
        
        if "error" in metric:
            print(f"⚠️  {metric['error']}")
            metrics_results[step_num] = metric
            continue
        
        # Save to analysis/run5/step_00/
        analysis_dir = config.ANALYSIS_DIR / f"step_{step_num:02d}"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        with open(analysis_dir / "metric_result.json", "w") as f:
            json.dump(metric, f, indent=2)
        
        metrics_results[step_num] = metric
        print(f"✅ Metric: {metric['metric']:.3f} (hint={metric['hint_count']}, hack={metric['hack_count']})")
    
    # Save summary
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    # Extract just the run name (e.g., "run1" from "gpt5/run1")
    run_name = Path(run_id).name
    summary_file = config.SUMMARY_DIR / f"{run_name}_metrics.json"
    
    with open(summary_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Summary saved to: {summary_file}")
    print(f"{'='*70}\n")


def compute_metric_for_rollout(rollout_dir: Path):
    """Compute hint/(hint+hack) for a rollout directory."""
    
    # Check for old format rollout_summary.json first (has reward_hack_filenames, hint_terminated_filenames)
    old_summary_path = rollout_dir / "rollout_summary.json"
    if old_summary_path.exists():
        with open(old_summary_path) as f:
            data = json.load(f)
        
        # Old format has the data directly
        if "reward_hack_filenames" in data and "hint_terminated_filenames" in data:
            # Extract run IDs from filenames
            hack_filenames = set(data["reward_hack_filenames"])
            hint_filenames = set(data["hint_terminated_filenames"])
            
            # Count hints (hinted but not hacked)
            hint_only_filenames = hint_filenames - hack_filenames
            hint_count = len(hint_only_filenames)
            hack_count = len(hack_filenames)
            
            total = hint_count + hack_count
            metric = hint_count / total if total > 0 else 0.0
            
            return {
                "hint_count": hint_count,
                "hack_count": hack_count,
                "total": total,
                "metric": metric
            }
    
    # Otherwise, use new format (rollouts_summary.json + rollout_analysis_detailed.json)
    summary_path = rollout_dir / "rollouts_summary.json"
    if not summary_path.exists():
        return {
            "hint_count": 0,
            "hack_count": 0,
            "total": 0,
            "metric": 0.0,
            "error": "No summary file found (checked rollout_summary.json and rollouts_summary.json)"
        }
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    # Read rollout_analysis_detailed.json
    detailed_path = rollout_dir / "rollout_analysis_detailed.json"
    if not detailed_path.exists():
        return {
            "hint_count": 0,
            "hack_count": 0,
            "total": 0,
            "metric": 0.0,
            "error": "rollout_analysis_detailed.json not found - grading failed"
        }
    
    with open(detailed_path) as f:
        detailed = json.load(f)
    
    # Get hacked rollout IDs
    hacked_ids = {r["rollout_id"] for r in detailed if r.get("reward_hacked", False)}
    
    # Count hints (terminated on hint but not hacked)
    hint_count = sum(
        1 for rid, meta in summary.items()
        if meta.get("terminated_on_hint", False) and rid not in hacked_ids
    )
    
    # Count hacks
    hack_count = len(hacked_ids)
    
    # Compute metric
    total = hint_count + hack_count
    metric = hint_count / total if total > 0 else 0.0
    
    return {
        "hint_count": hint_count,
        "hack_count": hack_count,
        "total": total,
        "metric": metric
    }


if __name__ == "__main__":
    fire.Fire(main)
