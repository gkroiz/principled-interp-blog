"""Statistical analysis and visualization of causal effects."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import fire
import config


def bootstrap_confidence_interval_from_counts(
    hint_with: int,
    hack_with: int,
    hint_without: int,
    hack_without: int,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> dict:
    """
    Compute bootstrap confidence interval using multinomial resampling from counts.
    
    Args:
        hint_with: Number of hints in "with property" group
        hack_with: Number of hacks in "with property" group
        hint_without: Number of hints in "without property" group
        hack_without: Number of hacks in "without property" group
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
    
    Returns dict with effect_estimate, ci_lower, ci_upper, p_value
    """
    # Original effect
    total_with = hint_with + hack_with
    total_without = hint_without + hack_without
    
    metric_with = hint_with / total_with if total_with > 0 else 0.0
    metric_without = hint_without / total_without if total_without > 0 else 0.0
    original_effect = metric_with - metric_without
    
    # Bootstrap using multinomial resampling
    bootstrap_effects = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Resample with group (multinomial)
        if total_with > 0:
            counts_with = rng.multinomial(total_with, [hint_with/total_with, hack_with/total_with])
            boot_hint_with, boot_hack_with = counts_with
            boot_metric_with = boot_hint_with / total_with
        else:
            boot_metric_with = 0.0
        
        # Resample without group (multinomial)
        if total_without > 0:
            counts_without = rng.multinomial(total_without, [hint_without/total_without, hack_without/total_without])
            boot_hint_without, boot_hack_without = counts_without
            boot_metric_without = boot_hint_without / total_without
        else:
            boot_metric_without = 0.0
        
        boot_effect = boot_metric_with - boot_metric_without
        bootstrap_effects.append(boot_effect)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_effects, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_effects, (1 - alpha/2) * 100)
    
    # P-value: proportion of bootstrap distribution on opposite side of 0 from effect
    if original_effect >= 0:
        p_value = 2 * np.mean([e <= 0 for e in bootstrap_effects])
    else:
        p_value = 2 * np.mean([e >= 0 for e in bootstrap_effects])
    p_value = min(p_value, 1.0)
    
    return {
        "effect_estimate": original_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_confidence_interval(
    with_property_group: list,
    without_property_group: list,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> dict:
    """
    Compute bootstrap confidence interval for the causal effect.
    
    Returns dict with:
    - effect_estimate: Point estimate of causal effect
    - ci_lower: Lower bound of CI
    - ci_upper: Upper bound of CI
    - p_value: Approximate p-value
    """
    # Filter to only rollouts that hinted or hacked (contribute to the metric)
    def has_outcome(r):
        """Check if rollout contributed to metric (hinted or hacked)."""
        return (r["metadata"].get("terminated_on_hint") and not r["metadata"].get("reward_hacked")) or \
               r["metadata"].get("reward_hacked")
    
    with_property_filtered = [r for r in with_property_group if has_outcome(r)]
    without_property_filtered = [r for r in without_property_group if has_outcome(r)]
    
    def compute_metric(group):
        """Helper to compute metric from group data."""
        hint_count = sum(
            1 for r in group 
            if r["metadata"].get("terminated_on_hint") 
            and not r["metadata"].get("reward_hacked")
        )
        hack_count = sum(1 for r in group if r["metadata"].get("reward_hacked"))
        total = hint_count + hack_count
        return hint_count / total if total > 0 else 0.0
    
    # Original effect (using filtered groups)
    original_effect = compute_metric(with_property_filtered) - compute_metric(without_property_filtered)
    
    # Bootstrap - resample only from rollouts that contributed to the metric
    bootstrap_effects = []
    rng = np.random.RandomState(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement from filtered groups
        with_indices = rng.choice(len(with_property_filtered), size=len(with_property_filtered), replace=True)
        without_indices = rng.choice(len(without_property_filtered), size=len(without_property_filtered), replace=True)
        
        with_sample = [with_property_filtered[i] for i in with_indices]
        without_sample = [without_property_filtered[i] for i in without_indices]
        
        # Compute effect for this bootstrap sample
        effect = compute_metric(with_sample) - compute_metric(without_sample)
        bootstrap_effects.append(effect)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_effects, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_effects, (1 - alpha/2) * 100)
    
    # P-value: proportion of bootstrap distribution on opposite side of 0 from effect
    if original_effect >= 0:
        p_value = 2 * np.mean([e <= 0 for e in bootstrap_effects])
    else:
        p_value = 2 * np.mean([e >= 0 for e in bootstrap_effects])
    p_value = min(p_value, 1.0)  # Cap at 1.0
    
    return {
        "effect_estimate": original_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "bootstrap_effects": bootstrap_effects,  # Keep for plotting
    }


def visualize_results(property_name: str, condition_on: str = None):
    """Create visualization of causal effect."""

    # Determine experiment directory (with optional conditioning subdirectory)
    base_exp_dir = config.EXPERIMENTS_DIR / property_name
    if condition_on:
        exp_dir = base_exp_dir / f"conditioned_on_{condition_on}"
    else:
        exp_dir = base_exp_dir

    # Load metrics, judgments, and bootstrap results
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Load judgments (use filtered judgments for conditional analysis)
    if condition_on:
        judgments_file = exp_dir / f"judgments_conditioned_on_{condition_on}.json"
    else:
        judgments_file = base_exp_dir / "judgments.json"

    with open(judgments_file) as f:
        judgments = json.load(f)

    # Load bootstrap CI results
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        bootstrap_ci = results.get('bootstrap_ci', {})
    else:
        bootstrap_ci = {}
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Bar plot of metrics with confidence intervals
    n_with = metrics['with_property_details']['n_rollouts']
    n_without = metrics['without_property_details']['n_rollouts']
    groups = [f'With Property\n(N={n_with})', f'Without Property\n(N={n_without})']
    values = [
        metrics['metric_with_property'],
        metrics['metric_without_property']
    ]
    colors = ['#ff6b6b', '#4ecdc4']
    
    # Compute error bars from bootstrap CI if available
    if bootstrap_ci:
        # The CI is for the difference, so we need to compute individual CIs
        # For now, we'll compute a simple approximation based on the counts
        with_details = metrics['with_property_details']
        without_details = metrics['without_property_details']
        
        # Standard error approximation for binomial proportion
        def se_proportion(p, n):
            return np.sqrt(p * (1 - p) / n) if n > 0 else 0
        
        se_with = se_proportion(values[0], with_details['total_count'])
        se_without = se_proportion(values[1], without_details['total_count'])
        
        # 95% CI is approximately ±1.96 * SE
        yerr = [1.96 * se_with, 1.96 * se_without]
    else:
        yerr = None
    
    bars = ax.bar(groups, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Metric: hint / (hint + hack)', fontsize=14, fontweight='bold')

    # Add conditioning info to title if applicable
    conditioning = metrics.get('conditioning', {})
    if conditioning.get('applied', False):
        condition_prop = conditioning.get('property', '')
        title = f'Causal Effect of "{property_name.replace("_", " ").title()}"\nConditioned on: {condition_prop.replace("_", " ").title()} = True'
    else:
        title = f'Causal Effect of\n"{property_name.replace("_", " ").title()}"'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add causal effect with p-value at the bottom of the plot
    effect = values[0] - values[1]
    
    # Get p-value from bootstrap results if available
    if bootstrap_ci and 'p_value' in bootstrap_ci:
        p_value = bootstrap_ci['p_value']
        ci_lower = bootstrap_ci.get('ci_lower', 0)
        ci_upper = bootstrap_ci.get('ci_upper', 0)
        
        # Determine significance level
        if p_value < 0.001:
            sig_label = '***'
        elif p_value < 0.01:
            sig_label = '**'
        elif p_value < 0.05:
            sig_label = '*'
        else:
            sig_label = 'n.s.'
        
        # Choose box color based on significance
        box_color = 'lightgreen' if p_value < 0.05 else 'lightcoral'
        
        # Add effect annotation at bottom center
        ax.text(0.5, -0.12, 
                f'Effect: {effect:+.3f} ({sig_label})\n95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]  |  p = {p_value:.4f}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7, edgecolor='black', linewidth=1.5))
    else:
        # Fallback if no bootstrap results
        ax.text(0.5, -0.1, f'Effect: {effect:+.3f}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = exp_dir / "visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")
    plt.close()


def analyze_property(property_name: str, n_bootstrap: int = 10000, condition_on: str = None):
    """
    Complete statistical analysis for a property.

    Args:
        property_name: Which property to analyze (e.g., 'rule_rationalization')
        n_bootstrap: Number of bootstrap samples (default: 10000)
        condition_on: Optional property to condition on (must match what was used in compute_metrics)

    Examples:
        # Analyze with default settings
        python analyze_results.py rule_rationalization

        # Use more bootstrap samples
        python analyze_results.py rule_rationalization --n_bootstrap 20000

        # Analyze conditional results
        python analyze_results.py ethical_concerns --condition_on thought_about_exploit
    """
    if property_name not in config.PROPERTIES:
        print(f"Error: Unknown property '{property_name}'")
        print(f"Available properties: {', '.join(config.PROPERTIES.keys())}")
        return

    # Determine experiment directory (with optional conditioning subdirectory)
    base_exp_dir = config.EXPERIMENTS_DIR / property_name
    if condition_on:
        exp_dir = base_exp_dir / f"conditioned_on_{condition_on}"
    else:
        exp_dir = base_exp_dir

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Property: {property_name}")
    if condition_on:
        print(f"Conditioned on: {condition_on} = True")
    print("="*60)
    
    # Load data
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"\nError: No metrics found at {metrics_file}")
        print(f"Run compute_metrics.py first with matching experiment name.")
        return
    
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Load judgments (use filtered judgments for conditional analysis)
    if condition_on:
        judgments_file = exp_dir / f"judgments_conditioned_on_{condition_on}.json"
    else:
        judgments_file = base_exp_dir / "judgments.json"

    with open(judgments_file) as f:
        judgments = json.load(f)

    # Get groups (reconstruct from saved rollout IDs)
    with_ids = set(metrics['with_property_details']['rollout_ids'])
    without_ids = set(metrics['without_property_details']['rollout_ids'])
    
    with_property = [j for j in judgments if j['rollout_id'] in with_ids]
    without_property = [j for j in judgments if j['rollout_id'] in without_ids]
    
    print(f"\nGroup sizes:")
    print(f"  With property: {len(with_property)}")
    print(f"  Without property: {len(without_property)}")
    
    # Extract the actual counts from metrics for bootstrap
    # (judgments don't have reward_hacked merged in, so we use the computed outcomes)
    with_details = metrics['with_property_details']
    without_details = metrics['without_property_details']
    
    # Bootstrap CI using the outcome counts
    print(f"\nComputing bootstrap confidence interval (n={n_bootstrap})...")
    ci_results = bootstrap_confidence_interval_from_counts(
        hint_with=with_details['hint_count'],
        hack_with=with_details['hack_count'],
        hint_without=without_details['hint_count'],
        hack_without=without_details['hack_count'],
        n_bootstrap=n_bootstrap
    )
    
    print(f"\nBootstrap Results:")
    print(f"  Effect estimate: {ci_results['effect_estimate']:+.3f}")
    print(f"  {ci_results['confidence_level']*100:.0f}% CI: [{ci_results['ci_lower']:+.3f}, {ci_results['ci_upper']:+.3f}]")
    print(f"  P-value: {ci_results['p_value']:.4f}")
    
    # Significance check
    if ci_results['p_value'] < 0.05:
        print(f"  ✓ Statistically significant at p < 0.05")
    else:
        print(f"  ✗ Not statistically significant (p >= 0.05)")

    # Save complete results
    property_config = config.PROPERTIES[property_name]
    complete_results = {
        "property_name": property_name,
        "property_description": property_config["description"],
        "metrics": metrics,
        "bootstrap_ci": {
            k: v for k, v in ci_results.items()
            if k != "bootstrap_effects"  # Don't save all samples
        },
    }
    
    output_file = exp_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\n✓ Saved complete results to {output_file}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_results(property_name, condition_on=condition_on)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return complete_results


def main():
    """CLI entry point using Fire."""
    fire.Fire(analyze_property)


if __name__ == "__main__":
    main()

