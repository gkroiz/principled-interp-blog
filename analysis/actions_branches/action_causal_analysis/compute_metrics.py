"""Compute hint/(hint+hack) metrics for groups."""

import json
from pathlib import Path
import fire
import config


def select_balanced_groups(judgments: list[dict], n_per_group: int = None) -> tuple[list[dict], list[dict]]:
    """
    Select N rollouts with property and N without (random selection).

    Args:
        judgments: List of judgment dicts
        n_per_group: Number of rollouts per group. If None, uses min of both groups (fully balanced).

    Returns (with_property_group, without_property_group)
    """
    # Separate by property presence
    with_property = [j for j in judgments if j["property_present"]]
    without_property = [j for j in judgments if not j["property_present"]]

    # If n_per_group not specified, use min of both groups for max balance
    if n_per_group is None:
        n_per_group = min(len(with_property), len(without_property))
        print(f"Auto-selecting n_per_group = {n_per_group} (min of both groups)")

    # Shuffle for random selection (use seed for reproducibility)
    import random
    random.seed(42)
    random.shuffle(with_property)
    random.shuffle(without_property)

    # Select top N from each
    selected_with = with_property[:n_per_group]
    selected_without = without_property[:n_per_group]

    if len(selected_with) < n_per_group:
        print(f"⚠️  Warning: Only {len(selected_with)} rollouts with property (requested {n_per_group})")
    if len(selected_without) < n_per_group:
        print(f"⚠️  Warning: Only {len(selected_without)} rollouts without property (requested {n_per_group})")

    print(f"\nSelected Groups:")
    print(f"  With property: {len(selected_with)} rollouts")
    print(f"  Without property: {len(selected_without)} rollouts")

    return selected_with, selected_without


def compute_group_metrics(group: list[dict], group_name: str) -> dict:
    """
    Compute metrics for a group of rollouts.
    
    Returns dict with:
    - hint_count: Number that terminated on hint (without hacking)
    - hack_count: Number that reward hacked
    - metric: hint_count / (hint_count + hack_count)
    - rollout_ids: List of rollout IDs in this group
    """
    hint_count = 0
    hack_count = 0
    
    for rollout in group:
        metadata = rollout["metadata"]
        
        terminated_on_hint = metadata.get("terminated_on_hint", False)
        reward_hacked = metadata.get("reward_hacked", False)
        
        if reward_hacked:
            hack_count += 1
        elif terminated_on_hint:
            hint_count += 1
        # Note: some rollouts might do neither (e.g., reached max steps)
    
    total = hint_count + hack_count
    metric = hint_count / total if total > 0 else 0.0
    
    print(f"\n{group_name}:")
    print(f"  Hint-only count: {hint_count}")
    print(f"  Hack count: {hack_count}")
    print(f"  Total (hint+hack): {total}")
    print(f"  Metric (hint/total): {metric:.3f}")
    
    return {
        "group_name": group_name,
        "hint_count": hint_count,
        "hack_count": hack_count,
        "total_count": total,
        "metric": metric,
        "n_rollouts": len(group),
        "rollout_ids": [r["rollout_id"] for r in group],
    }


def compute_causal_effect(with_property_metrics: dict, without_property_metrics: dict) -> dict:
    """Compute the causal effect: difference in metrics between groups."""
    
    metric_diff = with_property_metrics["metric"] - without_property_metrics["metric"]
    
    print(f"\n{'='*60}")
    print("CAUSAL EFFECT ESTIMATE")
    print("="*60)
    print(f"With property metric:    {with_property_metrics['metric']:.3f}")
    print(f"Without property metric: {without_property_metrics['metric']:.3f}")
    print(f"Difference:              {metric_diff:+.3f}")
    
    if metric_diff > 0:
        print(f"\n→ Property INCREASES hint usage (decreases hacking)")
    elif metric_diff < 0:
        print(f"\n→ Property DECREASES hint usage (increases hacking)")
    else:
        print(f"\n→ No effect detected")
    
    print("="*60)
    
    return {
        "metric_with_property": with_property_metrics["metric"],
        "metric_without_property": without_property_metrics["metric"],
        "causal_effect": metric_diff,
        "with_property_details": with_property_metrics,
        "without_property_details": without_property_metrics,
    }


def load_reward_hacked_data(rollouts_dir: Path) -> dict:
    """Load reward_hacked flags from rollout_analysis_detailed.json."""
    analysis_file = rollouts_dir / "rollout_analysis_detailed.json"
    
    if not analysis_file.exists():
        print(f"Warning: No rollout_analysis_detailed.json found in {rollouts_dir}")
        return {}
    
    with open(analysis_file) as f:
        analyses = json.load(f)
    
    # Create mapping from filename to reward_hacked status
    hack_map = {}
    for item in analyses:
        filename = item["filename"]
        # Extract run number from filename (e.g., "run-...-run10.txt" -> 10)
        run_num = int(filename.split("-run")[-1].replace(".txt", ""))
        hack_map[run_num] = item["analysis"]["reward_hacked"]
    
    return hack_map


def merge_reward_hacked_into_judgments(judgments: list[dict]) -> list[dict]:
    """Merge reward_hacked data from rollout directories into judgment metadata."""
    
    # Group judgments by source directory
    by_source = {}
    for j in judgments:
        source_dir = j["metadata"].get("source_dir")
        if source_dir:
            if source_dir not in by_source:
                by_source[source_dir] = []
            by_source[source_dir].append(j)
    
    # Load hack data for each source directory
    for source_dir, judgments_subset in by_source.items():
        rollouts_path = Path(source_dir)
        hack_map = load_reward_hacked_data(rollouts_path)
        
        # Merge into judgments
        for j in judgments_subset:
            original_id = j["metadata"].get("original_rollout_id")
            if original_id is not None and original_id in hack_map:
                j["metadata"]["reward_hacked"] = hack_map[original_id]
            else:
                # Default to False if not found
                j["metadata"]["reward_hacked"] = False
    
    return judgments


def compute_metrics_for_property(
    property_name: str,
    n_per_group: int = None,
    condition_on: str = None,
):
    """
    Compute metrics for a property experiment, optionally conditioning on another property.

    Args:
        property_name: Which property to analyze (e.g., 'ethical_concerns')
        n_per_group: Number of rollouts per group (auto-selected as min if not provided)
        condition_on: Optional property to condition on (e.g., 'thought_about_exploit')
                     Only rollouts with this property will be included in the analysis

    Examples:
        # Compute with auto-selected N
        python compute_metrics.py ethical_concerns

        # Use specific N per group
        python compute_metrics.py ethical_concerns --n_per_group 15

        # Conditional analysis: effect of ethical_concerns given thought_about_exploit
        python compute_metrics.py ethical_concerns --condition_on thought_about_exploit
    """
    if property_name not in config.PROPERTIES:
        print(f"Error: Unknown property '{property_name}'")
        print(f"Available properties: {', '.join(config.PROPERTIES.keys())}")
        return

    # n_per_group will be auto-selected as min of both groups if None

    # Determine experiment directory (with optional conditioning subdirectory)
    base_exp_dir = config.EXPERIMENTS_DIR / property_name
    if condition_on:
        exp_dir = base_exp_dir / f"conditioned_on_{condition_on}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = base_exp_dir

    # Load judgments (always from base directory)
    judgments_file = base_exp_dir / "judgments.json"
    
    if not judgments_file.exists():
        print(f"Error: No judgments found at {judgments_file}")
        print(f"Run judge_rollouts.py first with matching experiment name.")
        return
    
    with open(judgments_file) as f:
        judgments = json.load(f)
    
    # Merge reward_hacked data from rollout directories
    print("Loading reward_hacked data from rollout directories...")
    judgments = merge_reward_hacked_into_judgments(judgments)
    hack_count = sum(1 for j in judgments if j["metadata"].get("reward_hacked", False))
    print(f"  ✓ Found {hack_count} rollouts with reward hacking")

    # Apply conditioning filter if specified
    conditioning_applied = False
    if condition_on:
        if condition_on not in config.PROPERTIES:
            print(f"\nError: Unknown conditioning property '{condition_on}'")
            print(f"Available properties: {', '.join(config.PROPERTIES.keys())}")
            return

        print(f"\n{'='*60}")
        print(f"APPLYING CONDITIONAL FILTER: {condition_on} = True")
        print("="*60)

        # Load conditioning property judgments
        condition_exp_dir = config.EXPERIMENTS_DIR / condition_on
        condition_judgments_file = condition_exp_dir / "judgments.json"

        if not condition_judgments_file.exists():
            print(f"Error: No judgments found for conditioning property at {condition_judgments_file}")
            print(f"Run: python judge_rollouts.py {condition_on} [--only_first_resampled]")
            return

        with open(condition_judgments_file) as f:
            condition_judgments = json.load(f)

        # Create mapping from rollout_id to whether condition is True
        condition_map = {j["rollout_id"]: j["property_present"] for j in condition_judgments}

        # Filter to only rollouts where condition is True
        original_count = len(judgments)
        judgments = [j for j in judgments if condition_map.get(j["rollout_id"], False)]

        print(f"Filtered from {original_count} to {len(judgments)} rollouts where {condition_on} = True")

        # Count how many have property in filtered set
        with_property_count = sum(1 for j in judgments if j["property_present"])
        without_property_count = len(judgments) - with_property_count
        print(f"  With {property_name}: {with_property_count}")
        print(f"  Without {property_name}: {without_property_count}")

        if len(judgments) == 0:
            print(f"\nError: No rollouts satisfy the condition {condition_on} = True")
            return

        conditioning_applied = True

        # Save filtered judgments for inspection
        filtered_judgments_file = exp_dir / f"judgments_conditioned_on_{condition_on}.json"
        with open(filtered_judgments_file, "w") as f:
            json.dump(judgments, f, indent=2)
        print(f"\n✓ Saved filtered judgments to {filtered_judgments_file}")

    print("\n" + "="*60)
    print("COMPUTING METRICS")
    print("="*60)
    print(f"Property: {property_name}")
    if conditioning_applied:
        print(f"Conditioned on: {condition_on} = True")
        print(f"Output directory: {exp_dir}")
    print(f"Using N={n_per_group} rollouts per group")
    print("="*60)
    
    # Select balanced groups
    with_property, without_property = select_balanced_groups(judgments, n_per_group)
    
    if not with_property or not without_property:
        print("\nError: Insufficient rollouts in one or both groups")
        return
    
    # Compute metrics for each group
    with_metrics = compute_group_metrics(with_property, "WITH PROPERTY")
    without_metrics = compute_group_metrics(without_property, "WITHOUT PROPERTY")
    
    # Compute causal effect
    results = compute_causal_effect(with_metrics, without_metrics)

    # Add conditioning information to results
    results["conditioning"] = {
        "applied": conditioning_applied,
        "property": condition_on if conditioning_applied else None,
    }

    # Save results
    output_file = exp_dir / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved metrics to {output_file}")


def main():
    """CLI entry point using Fire."""
    fire.Fire(compute_metrics_for_property)


if __name__ == "__main__":
    main()

