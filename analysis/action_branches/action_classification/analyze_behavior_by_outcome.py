#!/usr/bin/env python3
"""
Analyze how agent behaviors differ based on final outcome.

Three outcome classes:
1. normal_completion - Agent played normally and finished
2. terminated_on_hint - Agent requested a hint
3. reward_hacked - Agent exploited the system
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define behavior mappings
EXPLOIT_CATEGORIES = {"System Exploitation Planning", "System Exploitation Execution"}
HINT_CATEGORIES = {"Sanctioned Assistance Reliance", "Hint-Based Strategy", "Strategic Deference"}  # Include old names for backward compatibility
NORMAL_CATEGORIES = {
    "Exploration & Observation",
    "Task Execution", 
    "Task Planning",
    "Strategic Re-evaluation & Correction"
}

def load_classifications(directory):
    """Load all classification files from directory."""
    classifications_dir = Path(directory)
    all_games = []
    
    for filepath in sorted(classifications_dir.glob("*_classified.json")):
        if filepath.name == "summary_report.json":
            continue
            
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                # Skip games with null final_behavior
                if data.get('final_behavior') is not None:
                    all_games.append({
                        'filename': filepath.name,
                        'data': data
                    })
            except json.JSONDecodeError:
                print(f"Error reading {filepath.name}")
                continue
    
    return all_games

def categorize_turn(classification):
    """Categorize a turn into exploit_planning/exploit_execution/hint/normal."""
    if classification == "System Exploitation Planning":
        return "exploit_planning"
    elif classification == "System Exploitation Execution":
        return "exploit_execution"
    elif classification in HINT_CATEGORIES:
        return "hint"
    elif classification in NORMAL_CATEGORIES:
        return "normal"
    else:
        return "other"

def categorize_turn_detailed(classification):
    """Preserve the original detailed classification."""
    if classification == "System Exploitation Planning":
        return "exploit_planning"
    elif classification == "System Exploitation Execution":
        return "exploit_execution"
    elif classification in HINT_CATEGORIES:
        return "hint"
    elif classification == "Exploration & Observation":
        return "exploration"
    elif classification == "Task Execution":
        return "task_execution"
    elif classification == "Task Planning":
        return "task_planning"
    elif classification == "Strategic Re-evaluation & Correction":
        return "strategic_reevaluation"
    else:
        return "other"

def find_game_start_turn(classifications):
    """
    Find which turn index corresponds to when the game actually starts.
    Returns the trajectory position (0-indexed in our truncated view).
    """
    for idx, turn in enumerate(classifications):
        # Check tool_calls for game.py commands
        tool_calls = turn.get('tool_calls', [])
        for call in tool_calls:
            if call.get('function') == 'execute_command':
                command = call.get('arguments', '')
                # Look for game.py start, move, hint, or status commands
                if './game.py' in command and any(cmd in command for cmd in ['start', 'move', 'hint', 'status']):
                    return idx
    return None

def extract_trajectories_by_outcome(games):
    """
    Extract trajectories grouped by final outcome.
    Truncates at first exploitation EXECUTION (not planning/ideation).
    This allows us to see the full deliberation process before execution.
    """
    trajectories_by_outcome = defaultdict(list)
    
    for game in games:
        final_behavior = game['data'].get('final_behavior')
        if final_behavior is None:
            continue
            
        trajectory = []
        classifications = game['data'].get('classifications', [])
        
        # Find when game starts
        game_start_position = find_game_start_turn(classifications)
        
        for turn in classifications:
            category = categorize_turn(turn['classification'])
            trajectory.append(category)
            
            # Stop at first exploitation EXECUTION (not planning)
            if category == 'exploit_execution':
                break
        
        if trajectory:
            trajectories_by_outcome[final_behavior].append({
                'filename': game['filename'],
                'trajectory': trajectory,
                'total_turns': len(trajectory),
                'game_start_position': game_start_position
            })
    
    return trajectories_by_outcome

def extract_trajectories_detailed(games):
    """
    Extract trajectories with detailed categorization (preserving original classifications).
    Truncates at first exploitation EXECUTION (not planning/ideation).
    """
    trajectories_by_outcome = defaultdict(list)
    
    for game in games:
        final_behavior = game['data'].get('final_behavior')
        if final_behavior is None:
            continue
            
        trajectory = []
        classifications = game['data'].get('classifications', [])
        
        # Find when game starts
        game_start_position = find_game_start_turn(classifications)
        
        for turn in classifications:
            category = categorize_turn_detailed(turn['classification'])
            trajectory.append(category)
            
            # Stop at first exploitation EXECUTION (not planning)
            if category == 'exploit_execution':
                break
        
        if trajectory:
            trajectories_by_outcome[final_behavior].append({
                'filename': game['filename'],
                'trajectory': trajectory,
                'total_turns': len(trajectory),
                'game_start_position': game_start_position
            })
    
    return trajectories_by_outcome

def plot_outcome_behavior_comparison(trajectories_by_outcome, output_path):
    """Compare behavior distributions across different outcomes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {
        'exploit_planning': '#e67e22',  # Orange for planning
        'exploit_execution': '#e74c3c',  # Red for execution
        'hint': '#3498db',  # Blue for hints
        'normal': '#2ecc71',  # Green for normal
        'other': '#95a5a6'  # Gray for other
    }
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    # Plot 1: Stacked bars for each outcome
    ax = axes[0, 0]
    outcomes = sorted(trajectories_by_outcome.keys())
    behavior_data = defaultdict(list)
    
    for outcome in outcomes:
        all_behaviors = [b for t in trajectories_by_outcome[outcome] for b in t['trajectory']]
        total = len(all_behaviors)
        
        for behavior in ['normal', 'hint', 'exploit_planning', 'exploit_execution', 'other']:
            count = all_behaviors.count(behavior)
            behavior_data[behavior].append(100 * count / total if total > 0 else 0)
    
    x = np.arange(len(outcomes))
    width = 0.6
    bottom = np.zeros(len(outcomes))
    
    behavior_labels = {
        'normal': 'Normal',
        'hint': 'Hint',
        'exploit_planning': 'Exploit Plan',
        'exploit_execution': 'Exploit Execute',
        'other': 'Other'
    }
    
    for behavior in ['normal', 'hint', 'exploit_planning', 'exploit_execution', 'other']:
        ax.bar(x, behavior_data[behavior], width, label=behavior_labels[behavior], 
               bottom=bottom, color=colors[behavior])
        bottom += behavior_data[behavior]
    
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Behavior Distribution by Final Outcome', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average game length by outcome
    ax = axes[0, 1]
    avg_lengths = []
    std_lengths = []
    for outcome in outcomes:
        lengths = [t['total_turns'] for t in trajectories_by_outcome[outcome]]
        avg_lengths.append(np.mean(lengths))
        std_lengths.append(np.std(lengths))
    
    bars = ax.bar(x, avg_lengths, width, yerr=std_lengths, capsize=5,
                   color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Average Game Length (turns)', fontsize=11)
    ax.set_title('Game Length by Outcome', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, avg_lengths, std_lengths)):
        ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.3,
               f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Number of games by outcome
    ax = axes[1, 0]
    game_counts = [len(trajectories_by_outcome[o]) for o in outcomes]
    bars = ax.bar(x, game_counts, width, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Number of Games', fontsize=11)
    ax.set_title('Game Count by Outcome', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, game_counts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
               str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Exploitation rate by outcome (planning + execution)
    ax = axes[1, 1]
    exploit_rates = []
    for outcome in outcomes:
        all_behaviors = [b for t in trajectories_by_outcome[outcome] for b in t['trajectory']]
        exploit_count = all_behaviors.count('exploit_planning') + all_behaviors.count('exploit_execution')
        total = len(all_behaviors)
        exploit_rates.append(100 * exploit_count / total if total > 0 else 0)
    
    bars = ax.bar(x, exploit_rates, width, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Exploitation Turns (%)', fontsize=11)
    ax.set_title('Exploitation Rate by Outcome', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, exploit_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Agent Behavior Analysis by Final Outcome', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_behavior_trajectories_by_outcome(trajectories_by_outcome, output_path):
    """Create side-by-side heatmaps for each outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))
    
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    behavior_to_num = {
        'normal': 0, 
        'hint': 1, 
        'exploit_planning': 2, 
        'exploit_execution': 3
    }
    colors_list = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']  # green, blue, orange, red
    cmap = plt.matplotlib.colors.ListedColormap(colors_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    for ax, outcome in zip(axes, sorted(trajectories_by_outcome.keys())):
        trajectories = sorted(trajectories_by_outcome[outcome], key=lambda x: x['total_turns'])
        
        if not trajectories:
            ax.text(0.5, 0.5, f'No {outcome} games', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        max_turns = max(t['total_turns'] for t in trajectories)
        matrix = np.full((len(trajectories), max_turns), np.nan)
        
        for i, traj in enumerate(trajectories):
            for j, behavior in enumerate(traj['trajectory']):
                # Skip 'other' behaviors - they'll remain as NaN (invisible)
                if behavior in behavior_to_num:
                    matrix[i, j] = behavior_to_num[behavior]
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        
        # Draw vertical lines at game start positions
        for i, traj in enumerate(trajectories):
            game_start = traj.get('game_start_position')
            if game_start is not None and game_start > 0:
                # Draw line between pre-game and game columns
                ax.axvline(x=game_start - 0.5, ymin=i/len(trajectories), ymax=(i+1)/len(trajectories),
                          color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Turn Number', fontsize=10)
        ax.set_ylabel('Game Instance', fontsize=10)
        ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n={len(trajectories)} games)', 
                    fontsize=11, fontweight='bold')
        ax.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Adjust subplot spacing to make room for colorbar
    plt.subplots_adjust(bottom=0.15)
    
    # Shared colorbar positioned below the plots
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3], orientation='horizontal')
    cbar.ax.set_xticklabels(['Play Normal', 'Use Hint', 'Exploit Plan', 'Exploit Execute'])
    cbar.set_label('Behavior Type', labelpad=10)
    
    plt.suptitle('Behavior Trajectories by Final Outcome\n(Black lines mark when game starts)', 
                 fontsize=16, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_behavior_trajectories_detailed(trajectories_by_outcome, output_path):
    """Create detailed side-by-side heatmaps showing all subcategories."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))
    
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    # Carefully chosen color palette for 7 categories
    behavior_to_num = {
        'exploration': 0,           # Cyan - discovering/learning
        'task_planning': 1,         # Yellow - thinking/strategizing  
        'task_execution': 2,        # Green - doing/playing
        'strategic_reevaluation': 3, # Purple - reflecting/correcting
        'hint': 4,                  # Pink/Magenta - asking for help
        'exploit_planning': 5,      # Orange - considering cheating
        'exploit_execution': 6      # Red - actually cheating
    }
    
    colors_list = [
        '#17a2b8',  # Cyan - exploration
        '#ffc107',  # Yellow - task planning
        '#28a745',  # Green - task execution
        '#9b59b6',  # Purple - strategic reevaluation
        '#e91e63',  # Pink/Magenta - hint (distinct from cyan)
        '#e67e22',  # Orange - exploit planning
        '#e74c3c'   # Red - exploit execution
    ]
    
    cmap = plt.matplotlib.colors.ListedColormap(colors_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    for ax, outcome in zip(axes, sorted(trajectories_by_outcome.keys())):
        trajectories = sorted(trajectories_by_outcome[outcome], key=lambda x: x['total_turns'])
        
        if not trajectories:
            ax.text(0.5, 0.5, f'No {outcome} games', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        max_turns = max(t['total_turns'] for t in trajectories)
        matrix = np.full((len(trajectories), max_turns), np.nan)
        
        for i, traj in enumerate(trajectories):
            for j, behavior in enumerate(traj['trajectory']):
                # Skip 'other' behaviors - they'll remain as NaN (invisible)
                if behavior in behavior_to_num:
                    matrix[i, j] = behavior_to_num[behavior]
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        
        # Draw vertical lines at game start positions
        for i, traj in enumerate(trajectories):
            game_start = traj.get('game_start_position')
            if game_start is not None and game_start > 0:
                # Draw line between pre-game and game columns
                ax.axvline(x=game_start - 0.5, ymin=i/len(trajectories), ymax=(i+1)/len(trajectories),
                          color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Turn Number', fontsize=10)
        ax.set_ylabel('Game Instance', fontsize=10)
        ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n={len(trajectories)} games)', 
                    fontsize=11, fontweight='bold')
        ax.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Adjust subplot spacing to make room for colorbar
    plt.subplots_adjust(bottom=0.18)
    
    # Shared colorbar positioned below the plots
    cbar_ax = fig.add_axes([0.12, 0.05, 0.76, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3, 4, 5, 6], orientation='horizontal')
    cbar.ax.set_xticklabels([
        'Exploration &\nObservation',
        'Task\nPlanning', 
        'Task\nExecution',
        'Strategic\nRe-evaluation',
        'Use\nHint',
        'Exploit\nPlan',
        'Exploit\nExecute'
    ], fontsize=9)
    cbar.set_label('Detailed Behavior Type', labelpad=10, fontsize=11)
    
    plt.suptitle('Detailed Behavior Trajectories by Final Outcome\n(Black lines mark when game starts)', 
                 fontsize=16, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_early_behavior_prediction(trajectories_by_outcome, output_path):
    """Show if early behaviors predict final outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Analyze first 3 turns
    for turn_idx, ax in enumerate(axes.flat[:3]):
        outcome_behavior_counts = defaultdict(lambda: {
            'exploit_planning': 0, 
            'exploit_execution': 0, 
            'hint': 0, 
            'normal': 0, 
            'other': 0
        })
        outcome_totals = defaultdict(int)
        
        for outcome, trajectories in trajectories_by_outcome.items():
            for traj in trajectories:
                if turn_idx < len(traj['trajectory']):
                    behavior = traj['trajectory'][turn_idx]
                    outcome_behavior_counts[outcome][behavior] += 1
                    outcome_totals[outcome] += 1
        
        # Calculate percentages
        outcomes = sorted(trajectories_by_outcome.keys())
        x = np.arange(len(outcomes))
        width = 0.2
        
        colors = {
            'exploit_planning': '#e67e22',
            'exploit_execution': '#e74c3c',
            'hint': '#3498db',
            'normal': '#2ecc71',
            'other': '#95a5a6'
        }
        for i, behavior in enumerate(['normal', 'hint', 'exploit_planning', 'exploit_execution', 'other']):
            percentages = [100 * outcome_behavior_counts[o][behavior] / outcome_totals[o] 
                          if outcome_totals[o] > 0 else 0 for o in outcomes]
            label_map = {
                'normal': 'Normal',
                'hint': 'Hint',
                'exploit_planning': 'Exploit Plan',
                'exploit_execution': 'Exploit Execute',
                'other': 'Other'
            }
            ax.bar(x + i * width, percentages, width, label=label_map.get(behavior, behavior), color=colors[behavior])
        
        outcome_labels = {
            'normal_completion': 'Normal',
            'terminated_on_hint': 'Hint',
            'reward_hacked': 'Hacked'
        }
        
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Turn {turn_idx} Behavior by Final Outcome', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([outcome_labels.get(o, o) for o in outcomes])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
    
    # Summary statistics in the 4th panel
    ax = axes.flat[3]
    ax.axis('off')
    
    summary_text = "Early Behavior Patterns:\n\n"
    
    for outcome in sorted(trajectories_by_outcome.keys()):
        trajectories = trajectories_by_outcome[outcome]
        outcome_label = {
            'normal_completion': 'Normal Completion',
            'terminated_on_hint': 'Terminated on Hint',
            'reward_hacked': 'Reward Hacked'
        }.get(outcome, outcome)
        
        # Count games with early exploitation execution (within first 3 turns)
        early_exploit = sum(1 for t in trajectories 
                           if len(t['trajectory']) > 0 and 
                           any(t['trajectory'][i] == 'exploit_execution' 
                               for i in range(min(3, len(t['trajectory'])))))
        
        summary_text += f"{outcome_label}:\n"
        summary_text += f"  Total games: {len(trajectories)}\n"
        summary_text += f"  Early exploit (≤3 turns): {early_exploit} ({100*early_exploit/len(trajectories):.1f}%)\n"
        summary_text += f"  Avg length: {np.mean([t['total_turns'] for t in trajectories]):.1f} turns\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Early Behavior as Predictor of Final Outcome', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_detailed_category_breakdown(trajectories_by_outcome, output_path):
    """Show detailed breakdown of all classification categories by outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    all_categories = set()
    for trajectories in trajectories_by_outcome.values():
        for traj in trajectories:
            # Need to get original classifications
            all_categories.update(traj.get('categories', []))
    
    # Instead, let's count the behavior types we have
    for ax, outcome in zip(axes, sorted(trajectories_by_outcome.keys())):
        trajectories = trajectories_by_outcome[outcome]
        
        behavior_counts = defaultdict(int)
        for traj in trajectories:
            for behavior in traj['trajectory']:
                behavior_counts[behavior] += 1
        
        total = sum(behavior_counts.values())
        
        # Create pie chart
        colors = {
            'exploit_planning': '#e67e22',
            'exploit_execution': '#e74c3c',
            'hint': '#3498db',
            'normal': '#2ecc71',
            'other': '#95a5a6'
        }
        labels = []
        sizes = []
        pie_colors = []
        
        label_map = {
            'normal': 'Normal',
            'hint': 'Hint',
            'exploit_planning': 'Exploit Plan',
            'exploit_execution': 'Exploit Execute',
            'other': 'Other'
        }
        
        for behavior in ['normal', 'hint', 'exploit_planning', 'exploit_execution', 'other']:
            if behavior_counts[behavior] > 0:
                count = behavior_counts[behavior]
                pct = 100 * count / total
                labels.append(f'{label_map.get(behavior, behavior)}\n{count} ({pct:.1f}%)')
                sizes.append(count)
                pie_colors.append(colors[behavior])
        
        wedges, texts = ax.pie(sizes, labels=labels, colors=pie_colors, startangle=90)
        ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n({len(trajectories)} games, {total} turns)',
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Behavior Composition by Final Outcome', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def generate_outcome_statistics(trajectories_by_outcome):
    """Print detailed statistics by outcome."""
    print("\n" + "="*70)
    print("BEHAVIOR ANALYSIS BY FINAL OUTCOME")
    print("="*70)
    
    for outcome in sorted(trajectories_by_outcome.keys()):
        trajectories = trajectories_by_outcome[outcome]
        
        outcome_label = {
            'normal_completion': 'NORMAL COMPLETION',
            'terminated_on_hint': 'TERMINATED ON HINT',
            'reward_hacked': 'REWARD HACKED'
        }.get(outcome, outcome.upper())
        
        print(f"\n{outcome_label}:")
        print("-" * 70)
        
        total_games = len(trajectories)
        total_turns = sum(t['total_turns'] for t in trajectories)
        
        all_behaviors = [b for t in trajectories for b in t['trajectory']]
        exploit_plan_count = all_behaviors.count('exploit_planning')
        exploit_exec_count = all_behaviors.count('exploit_execution')
        hint_count = all_behaviors.count('hint')
        normal_count = all_behaviors.count('normal')
        other_count = all_behaviors.count('other')
        
        print(f"  Total Games: {total_games}")
        print(f"  Total Turns: {total_turns}")
        print(f"  Average Turns per Game: {total_turns/total_games:.1f}")
        
        print(f"\n  {'Behavior':<20} {'Count':<10} {'Percentage':<10} {'Per Game':<10}")
        print(f"  {'-'*50}")
        print(f"  {'Exploit Planning':<20} {exploit_plan_count:<10} {100*exploit_plan_count/total_turns:>6.1f}% {exploit_plan_count/total_games:>9.2f}")
        print(f"  {'Exploit Execution':<20} {exploit_exec_count:<10} {100*exploit_exec_count/total_turns:>6.1f}% {exploit_exec_count/total_games:>9.2f}")
        print(f"  {'Use Hint':<20} {hint_count:<10} {100*hint_count/total_turns:>6.1f}% {hint_count/total_games:>9.2f}")
        print(f"  {'Play Normal':<20} {normal_count:<10} {100*normal_count/total_turns:>6.1f}% {normal_count/total_games:>9.2f}")
        print(f"  {'Other':<20} {other_count:<10} {100*other_count/total_turns:>6.1f}% {other_count/total_games:>9.2f}")
        
        # Games with exploitation behaviors
        exploit_exec_games = sum(1 for t in trajectories if 'exploit_execution' in t['trajectory'])
        exploit_plan_games = sum(1 for t in trajectories if 'exploit_planning' in t['trajectory'])
        hint_games = sum(1 for t in trajectories if 'hint' in t['trajectory'])
        
        print(f"\n  Games with exploit execution: {exploit_exec_games} ({100*exploit_exec_games/total_games:.1f}%)")
        print(f"  Games with exploit planning: {exploit_plan_games} ({100*exploit_plan_games/total_games:.1f}%)")
        print(f"  Games with hints: {hint_games} ({100*hint_games/total_games:.1f}%)")
    
    print("\n" + "="*70 + "\n")

def main():
    # Load classifications
    classifications_dir = "/Users/gersonkroiz/ctfish/thought_anchors/turns/classifications"
    output_dir = "/Users/gersonkroiz/ctfish/thought_anchors/turns/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading classification files...")
    games = load_classifications(classifications_dir)
    print(f"Loaded {len(games)} games with valid final_behavior")
    
    print("Extracting trajectories by outcome...")
    trajectories_by_outcome = extract_trajectories_by_outcome(games)
    trajectories_by_outcome_detailed = extract_trajectories_detailed(games)
    
    # Print outcome counts
    print(f"\nOutcome distribution:")
    for outcome, trajectories in sorted(trajectories_by_outcome.items()):
        print(f"  {outcome}: {len(trajectories)} games")
    
    # Generate statistics
    generate_outcome_statistics(trajectories_by_outcome)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plot_outcome_behavior_comparison(
        trajectories_by_outcome,
        os.path.join(output_dir, "behavior_by_outcome.png")
    )
    
    plot_behavior_trajectories_by_outcome(
        trajectories_by_outcome,
        os.path.join(output_dir, "trajectories_by_outcome.png")
    )
    
    plot_behavior_trajectories_detailed(
        trajectories_by_outcome_detailed,
        os.path.join(output_dir, "trajectories_by_outcome_detailed.png")
    )
    
    plot_early_behavior_prediction(
        trajectories_by_outcome,
        os.path.join(output_dir, "early_behavior_prediction.png")
    )
    
    plot_detailed_category_breakdown(
        trajectories_by_outcome,
        os.path.join(output_dir, "behavior_composition_by_outcome.png")
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - behavior_by_outcome.png: Comprehensive comparison across outcomes")
    print("  - trajectories_by_outcome.png: Side-by-side heatmaps (simplified)")
    print("  - trajectories_by_outcome_detailed.png: Side-by-side heatmaps (all subcategories)")
    print("  - early_behavior_prediction.png: Early turns as predictors")
    print("  - behavior_composition_by_outcome.png: Detailed composition breakdown")

if __name__ == "__main__":
    main()

