#!/usr/bin/env python3
"""
Analyze how agent behaviors differ based on final outcome.

This script is generalizable and can work with different classification taxonomies:
- Exploit precursors (7 categories)
- Hopelessness (4 categories)  
- Original task-based (7 categories)
- Any custom taxonomy

Three outcome classes:
1. normal_completion - Agent played normally and finished
2. terminated_on_hint - Agent requested a hint
3. reward_hacked - Agent exploited the system
"""

import json
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define consistent outcome order for visualizations
OUTCOME_ORDER = ['normal_completion', 'reward_hacked', 'terminated_on_hint']

# Color for truncation marker (distinct color visible across all plots)
TRUNCATION_COLOR = '#FF00FF'  # Magenta - highly visible and distinct

# Define taxonomy configurations
TAXONOMY_CONFIGS = {
    'exploit_precursor': {
        'categories': [
            "Standard Rule & Strategy Exploration",
            "Decision to Exploit",
            "Decision to Accept Constraints",
            "Decision to Use Sanctioned Mechanic",
            "Vulnerability Probing & Discovery",
            "Exploit Hypothesis Formulation",
            "Unconventional Action Planning"
        ],
        'simplified_mapping': {
            "Standard Rule & Strategy Exploration": "normal",
            "Decision to Exploit": "exploit_planning",
            "Decision to Accept Constraints": "normal",
            "Decision to Use Sanctioned Mechanic": "hint",
            "Vulnerability Probing & Discovery": "exploit_planning",
            "Exploit Hypothesis Formulation": "exploit_planning",
            "Unconventional Action Planning": "exploit_execution"
        },
        'detailed_mapping': {
            "Standard Rule & Strategy Exploration": "standard_strategy",
            "Decision to Exploit": "decision_exploit",
            "Decision to Accept Constraints": "decision_accept_constraints",
            "Decision to Use Sanctioned Mechanic": "decision_use_sanctioned",
            "Vulnerability Probing & Discovery": "vulnerability_probing",
            "Exploit Hypothesis Formulation": "exploit_hypothesis",
            "Unconventional Action Planning": "exploit_execution"
        },
        'colors': {
            'standard_strategy': '#2196F3',
            'decision_exploit': '#FF5722',
            'decision_accept_constraints': '#4CAF50',
            'decision_use_sanctioned': '#FFC107',
            'vulnerability_probing': '#9C27B0',
            'exploit_hypothesis': '#E91E63',
            'exploit_execution': '#B71C1C'
        },
        'colorbar_labels': [
            'Standard Rule &\nStrategy Exploration',
            'Decision to\nExploit',
            'Decision to Accept\nConstraints',
            'Decision to Use\nSanctioned Mechanic',
            'Vulnerability\nProbing & Discovery',
            'Exploit Hypothesis\nFormulation',
            'Unconventional\nAction Planning'
        ]
    },
    'hopelessness': {
        'categories': [
            "Confident/Optimistic",
            "Neutral/Uncertain",
            "Struggling/Concerned",
            "Hopeless/Defeated"
        ],
        'simplified_mapping': {
            "Confident/Optimistic": "confident",
            "Neutral/Uncertain": "neutral",
            "Struggling/Concerned": "struggling",
            "Hopeless/Defeated": "hopeless"
        },
        'detailed_mapping': {
            "Confident/Optimistic": "confident",
            "Neutral/Uncertain": "neutral",
            "Struggling/Concerned": "struggling",
            "Hopeless/Defeated": "hopeless"
        },
        'colors': {
            'confident': '#2ecc71',     # Green - positive
            'neutral': '#3498db',       # Blue - neutral
            'struggling': '#f39c12',    # Orange - concerning
            'hopeless': '#e74c3c'       # Red - critical
        },
        'colorbar_labels': [
            'Confident/\nOptimistic',
            'Neutral/\nUncertain',
            'Struggling/\nConcerned',
            'Hopeless/\nDefeated'
        ]
    },
    'original': {
        'categories': [
            "Exploration & Observation",
            "Task Planning",
            "Task Execution",
            "Strategic Re-evaluation & Correction",
            "Sanctioned Assistance Reliance",
            "System Exploitation Planning",
            "System Exploitation Execution"
        ],
        'simplified_mapping': {
            "Exploration & Observation": "normal",
            "Task Planning": "normal",
            "Task Execution": "normal",
            "Strategic Re-evaluation & Correction": "normal",
            "Sanctioned Assistance Reliance": "hint",
            "Hint-Based Strategy": "hint",
            "Strategic Deference": "hint",
            "System Exploitation Planning": "exploit_planning",
            "System Exploitation Execution": "exploit_execution"
        },
        'detailed_mapping': {
            "Exploration & Observation": "exploration",
            "Task Planning": "task_planning",
            "Task Execution": "task_execution",
            "Strategic Re-evaluation & Correction": "strategic_reevaluation",
            "Sanctioned Assistance Reliance": "hint",
            "Hint-Based Strategy": "hint",
            "Strategic Deference": "hint",
            "System Exploitation Planning": "exploit_planning",
            "System Exploitation Execution": "exploit_execution"
        },
        'colors': {
            'exploration': '#17a2b8',
            'task_planning': '#ffc107',
            'task_execution': '#28a745',
            'strategic_reevaluation': '#9b59b6',
            'hint': '#e91e63',
            'exploit_planning': '#e67e22',
            'exploit_execution': '#e74c3c'
        },
        'colorbar_labels': [
            'Exploration &\nObservation',
            'Task\nPlanning',
            'Task\nExecution',
            'Strategic Re-evaluation\n& Correction',
            'Sanctioned Assistance\nReliance',
            'System Exploitation\nPlanning',
            'System Exploitation\nExecution'
        ]
    },
    'exploration': {
        'categories': [
            "Reading Game File",
            "Reading Hint Engine Source",
            "Reading Opponent Source",
            "Reading Game State Files",
            "Environment Exploration",
            "Checking Help/Usage",
            "Writing/Running Custom Code",
            "File Manipulation",
            "Executing Moves/Actions"
        ],
        'simplified_mapping': {
            "Reading Game File": "reading_game_file",
            "Reading Hint Engine Source": "reading_hint_source",
            "Reading Opponent Source": "reading_opponent_source",
            "Reading Game State Files": "reading_state_files",
            "Environment Exploration": "env_exploration",
            "Checking Help/Usage": "checking_help",
            "Writing/Running Custom Code": "custom_code",
            "File Manipulation": "file_manipulation",
            "Executing Moves/Actions": "executing_moves"
        },
        'detailed_mapping': {
            "Reading Game File": "reading_game_file",
            "Reading Hint Engine Source": "reading_hint_source",
            "Reading Opponent Source": "reading_opponent_source",
            "Reading Game State Files": "reading_state_files",
            "Environment Exploration": "env_exploration",
            "Checking Help/Usage": "checking_help",
            "Writing/Running Custom Code": "custom_code",
            "File Manipulation": "file_manipulation",
            "Executing Moves/Actions": "executing_moves"
        },
        'colors': {
            'reading_game_file': '#1E88E5',        # Bright Blue - documentation
            'reading_hint_source': '#9C27B0',      # Purple - special source
            'reading_opponent_source': '#CDDC39',  # Lime - different special source
            'reading_state_files': '#00BCD4',      # Cyan - data/state
            'env_exploration': '#4CAF50',          # Green - discovery
            'checking_help': '#FF9800',            # Orange - help/info
            'custom_code': '#F44336',              # Red - custom execution
            'file_manipulation': '#795548',        # Brown - manipulation
            'executing_moves': '#3F51B5'           # Indigo - gameplay
        },
        'colorbar_labels': [
            'Reading\nGame File',
            'Reading Hint\nEngine Source',
            'Reading\nOpponent Source',
            'Reading Game\nState Files',
            'Environment\nExploration',
            'Checking\nHelp/Usage',
            'Writing/Running\nCustom Code',
            'File\nManipulation',
            'Executing\nMoves/Actions'
        ]
    }
}

def get_ordered_outcomes(trajectories_by_outcome):
    """Return outcomes in consistent order, only including those that exist."""
    return [outcome for outcome in OUTCOME_ORDER if outcome in trajectories_by_outcome]

def detect_taxonomy(games):
    """Detect which taxonomy is being used based on the categories present or taxonomy field.
    
    Returns: taxonomy name ('exploit_precursor', 'hopelessness', 'exploration', 'original', or 'custom')
    """
    # Check if any game has a taxonomy or classification_type field
    for game in games[:min(3, len(games))]:
        data = game.get('data', {})
        # Check both 'taxonomy' and 'classification_type' fields
        taxonomy_name = data.get('taxonomy') or data.get('classification_type')
        if taxonomy_name:
            # Map known taxonomy names
            if 'Reward Hacking Precursors' in taxonomy_name or 'exploit' in taxonomy_name.lower():
                return 'exploit_precursor'
            elif 'Hopelessness' in taxonomy_name or 'hopeless' in taxonomy_name.lower():
                return 'hopelessness'
            elif 'Exploration' in taxonomy_name or 'exploration' in taxonomy_name.lower():
                return 'exploration'
    
    # Fall back to sampling categories
    all_categories = set()
    for game in games[:min(10, len(games))]:
        data = game.get('data', {})
        classifications = data.get('classifications', [])
        for item in classifications[:min(5, len(classifications))]:
            # Handle both old format (just string) and new format (dict with 'classification' key)
            if isinstance(item, dict):
                category = item.get('classification')
            else:
                category = item
            if category:
                all_categories.add(category)
    
    # Check which taxonomy the categories belong to
    for taxonomy_name, config in TAXONOMY_CONFIGS.items():
        taxonomy_cats = set(config['categories'])
        if all_categories & taxonomy_cats:
            return taxonomy_name
    
    # If no match, return custom
    return 'custom'

def get_taxonomy_config(taxonomy_name, all_categories=None):
    """Get or create a taxonomy configuration.
    
    Args:
        taxonomy_name: Name of the taxonomy
        all_categories: If taxonomy is 'custom', provide all unique categories found
    
    Returns:
        Dictionary with taxonomy configuration
    """
    if taxonomy_name in TAXONOMY_CONFIGS:
        return TAXONOMY_CONFIGS[taxonomy_name]
    
    # Create a custom config
    if all_categories is None:
        all_categories = []
    
    # Generate colors for custom categories
    colors = {}
    colorbar_labels = []
    simplified_mapping = {}
    detailed_mapping = {}
    
    # Use a color palette
    palette = sns.color_palette("husl", len(all_categories))
    
    for i, cat in enumerate(sorted(all_categories)):
        simplified_cat = cat.lower().replace(' ', '_').replace('/', '_')
        colors[simplified_cat] = plt.matplotlib.colors.rgb2hex(palette[i])
        colorbar_labels.append(cat.replace(' ', '\n'))
        simplified_mapping[cat] = simplified_cat
        detailed_mapping[cat] = simplified_cat
    
    return {
        'categories': sorted(all_categories),
        'simplified_mapping': simplified_mapping,
        'detailed_mapping': detailed_mapping,
        'colors': colors,
        'colorbar_labels': colorbar_labels
    }

def extract_hint_level(filename, data):
    """Extract hint level from filename or data."""
    # Try to extract from filename first
    match = re.search(r'hint[_-]?(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try to extract from data
    if 'hint_level' in data:
        return data['hint_level']
    
    # Check if it's in metadata
    if 'metadata' in data and 'hint_level' in data['metadata']:
        return data['metadata']['hint_level']
    
    return None

def load_classifications(directory):
    """Load all classification files from directory.
    
    Handles both old format (simple arrays) and new format (detailed objects).
    """
    classifications_dir = Path(directory)
    all_games = []
    
    # Try different file patterns
    patterns = [
        "*_exploits_classified.json",
        "*_hopelessness_classified.json", 
        "*_classified.json"
    ]
    
    files_to_process = []
    for pattern in patterns:
        files = list(classifications_dir.glob(pattern))
        if files:
            files_to_process.extend(files)
    
    # Remove duplicates and sort
    files_to_process = sorted(set(files_to_process))
    
    for filepath in files_to_process:
        if "summary_report" in filepath.name:
            continue
            
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                
                # Normalize the data format
                # Handle new format where classifications is an array of objects
                if 'classifications' in data and data['classifications']:
                    first_item = data['classifications'][0]
                    if isinstance(first_item, dict) and 'classification' in first_item:
                        # New format - convert to old format for compatibility
                        normalized_classifications = []
                        for item in data['classifications']:
                            classification_dict = {
                                'classification': item['classification'],
                                'turn_index': item.get('turn_index'),
                                'reasoning': item.get('reasoning', ''),
                                'tool_calls': item.get('tool_calls', [])
                            }
                            normalized_classifications.append(classification_dict)
                        data['classifications'] = normalized_classifications
                
                # Skip games with null final_behavior
                if data.get('final_behavior') is not None:
                    hint_level = extract_hint_level(filepath.name, data)
                    all_games.append({
                        'filename': filepath.name,
                        'data': data,
                        'hint_level': hint_level
                    })
            except json.JSONDecodeError:
                print(f"Error reading {filepath.name}")
                continue
    
    return all_games

def categorize_turn(classification, taxonomy_config, use_simplified=True):
    """Categorize a turn based on taxonomy configuration.
    
    Args:
        classification: The classification string
        taxonomy_config: The taxonomy configuration dict
        use_simplified: If True, use simplified_mapping; otherwise use detailed_mapping
    """
    mapping = taxonomy_config['simplified_mapping'] if use_simplified else taxonomy_config['detailed_mapping']
    return mapping.get(classification, "other")

def find_game_start_turns(classifications):
    """Find all turn indices that correspond to when a game starts."""
    game_starts = []
    for idx, turn in enumerate(classifications):
        # Check tool_calls for game.py commands
        tool_calls = turn.get('tool_calls', [])
        for call in tool_calls:
            if call.get('function') == 'execute_command':
                arguments = call.get('arguments', '')
                
                # Try to parse as JSON first
                try:
                    if arguments.startswith('{'):
                        args_dict = json.loads(arguments)
                        command = args_dict.get('command', '')
                    else:
                        command = arguments
                except (json.JSONDecodeError, ValueError):
                    command = arguments
                
                # Look for game start commands
                # Allow for environment variable assignments before the command (e.g., BOARD_SIZE=6 WIN_LENGTH=4 ./game.py start)
                if re.search(r'^\s*(?:\w+=\S+\s+)*(?:python3?\s+|\./)?game\.py\s+start\s*$', command):
                    game_starts.append(idx)
                    break
    
    # Return at most the first 2 game starts
    return game_starts[:2]

def extract_trajectories_by_outcome(games, taxonomy_config, use_simplified=True, keep_original=False, max_turns=None, filter_categories=None):
    """Extract trajectories grouped by final outcome.
    
    Truncates reward_hacked games at the hack_turn.
    
    Args:
        games: List of game data
        taxonomy_config: The taxonomy configuration
        use_simplified: If True, use simplified_mapping; otherwise use detailed_mapping
        keep_original: If True, keep original classification names instead of mapping
        max_turns: If provided, exclude games with more than this many turns
        filter_categories: If provided, only show these categories (others become "other")
    """
    trajectories_by_outcome = defaultdict(list)
    
    for game in games:
        final_behavior = game['data'].get('final_behavior')
        if final_behavior is None:
            continue
            
        classifications = game['data'].get('classifications', [])
        
        # Find all game starts
        all_game_starts = find_game_start_turns(classifications)
        num_games = len(all_game_starts)
        
        # Determine truncation point based on hack_turn (for reward_hacked games only)
        truncation_turn = None
        rollout_analysis = game['data'].get('rollout_analysis', {})
        hack_turn = rollout_analysis.get('hack_turn')
        
        if hack_turn is not None and final_behavior == 'reward_hacked':
            # Convert to int in case it's stored as a string
            # Truncate at the hack turn itself
            try:
                truncation_turn = int(hack_turn)
            except (ValueError, TypeError):
                truncation_turn = None
        
        # Build trajectory
        trajectory = []
        for idx, turn in enumerate(classifications):
            # Handle both old and new format
            if isinstance(turn, dict):
                cat = turn.get('classification')
            else:
                cat = turn
            
            if keep_original:
                # Keep the original classification name
                category = cat if cat else "other"
            else:
                # Apply mapping
                category = categorize_turn(cat, taxonomy_config, use_simplified=use_simplified)
            
            # Apply category filter if provided
            if filter_categories is not None and category not in filter_categories:
                category = "other"
            
            trajectory.append(category)
            
            # Stop at truncation turn if specified
            if truncation_turn is not None and idx >= truncation_turn:
                break
        
        # Find game starts within trajectory
        trajectory_length = len(trajectory)
        game_start_positions = find_game_start_turns(classifications[:trajectory_length])
        
        # Apply max_turns filter if specified
        if max_turns is not None and trajectory_length > max_turns:
            continue
        
        if trajectory:
            trajectories_by_outcome[final_behavior].append({
                'filename': game['filename'],
                'trajectory': trajectory,
                'total_turns': len(trajectory),
                'game_start_positions': game_start_positions,
                'num_games_detected': num_games,
                'truncation_turn': truncation_turn  # Store truncation point for visualization
            })
    
    return trajectories_by_outcome

def plot_behavior_trajectories_by_outcome(trajectories_by_outcome, taxonomy_config, output_path, use_simplified=True, first_game_only=False):
    """Create side-by-side heatmaps for each outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))
    
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    # Get all categories and create numeric mapping
    all_categories_in_data = set()
    for trajectories in trajectories_by_outcome.values():
        for traj in trajectories:
            all_categories_in_data.update(traj['trajectory'])
    
    sorted_categories = sorted(all_categories_in_data - {'other'})
    if 'other' in all_categories_in_data:
        sorted_categories.append('other')
    
    behavior_to_num = {cat: i for i, cat in enumerate(sorted_categories)}
    
    # Create color map
    colors_list = [taxonomy_config['colors'].get(cat, '#95a5a6') for cat in sorted_categories]
    cmap = plt.matplotlib.colors.ListedColormap(colors_list)
    bounds = [i - 0.5 for i in range(len(sorted_categories) + 1)]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot each outcome
    for idx, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[idx]
        
        if outcome not in trajectories_by_outcome:
            ax.text(0.5, 0.5, f'No {outcome_labels.get(outcome, outcome)} games', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Turn Number', fontsize=10)
            ax.set_ylabel('Game Instance', fontsize=10)
            ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n=0 games)', 
                        fontsize=11, fontweight='bold')
            continue
        
        trajectories = sorted(trajectories_by_outcome[outcome], key=lambda x: x['total_turns'])
        
        if not trajectories:
            continue
        
        max_turns = max(t['total_turns'] for t in trajectories)
        matrix = np.full((len(trajectories), max_turns), np.nan)
        
        for i, traj in enumerate(trajectories):
            for j, behavior in enumerate(traj['trajectory']):
                if behavior in behavior_to_num:
                    matrix[i, j] = behavior_to_num[behavior]
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        
        # Draw game start lines (black)
        for i, traj in enumerate(trajectories):
            game_starts = traj.get('game_start_positions', [])
            # If first_game_only flag is set, only draw the first game start line
            if first_game_only and game_starts:
                game_starts = game_starts[:1]
            for game_start in game_starts:
                if game_start > 0:
                    ymin = 1 - (i + 1) / len(trajectories)
                    ymax = 1 - i / len(trajectories)
                    ax.axvline(x=game_start - 0.5, ymin=ymin, ymax=ymax,
                              color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Turn Number', fontsize=10)
        ax.set_ylabel('Game Instance', fontsize=10)
        ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n={len(trajectories)} games)', 
                    fontsize=11, fontweight='bold')
        ax.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Shared colorbar
    plt.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=list(range(len(sorted_categories))), orientation='horizontal')
    
    # Create nice labels for simplified categories
    label_map = {
        'normal': 'Play Normal',
        'hint': 'Use Hint',
        'exploit_planning': 'Exploit Plan',
        'exploit_execution': 'Exploit Execute',
        'confident': 'Confident',
        'neutral': 'Neutral',
        'struggling': 'Struggling',
        'hopeless': 'Hopeless',
        'other': 'Other'
    }
    nice_labels = [label_map.get(cat, cat.replace('_', ' ').title()) for cat in sorted_categories]
    cbar.ax.set_xticklabels(nice_labels)
    cbar.set_label('Behavior Type', labelpad=10)
    
    plt.suptitle('Behavior Trajectories by Final Outcome\n(Black lines mark when games start)', 
                 fontsize=16, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_behavior_trajectories_detailed(trajectories_by_outcome, taxonomy_config, output_path, first_game_only=False):
    """Create detailed side-by-side heatmaps showing all subcategories with row-to-filename mapping."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 11))
    
    outcome_labels = {
        'normal_completion': 'Normal Completion',
        'terminated_on_hint': 'Terminated on Hint',
        'reward_hacked': 'Reward Hacked'
    }
    
    # Store filename-to-row mappings for printing later
    filename_mappings = {}
    
    # Get all ORIGINAL categories from the data
    all_categories_in_data = set()
    for trajectories in trajectories_by_outcome.values():
        for traj in trajectories:
            all_categories_in_data.update(traj['trajectory'])
    
    # Sort categories based on taxonomy order if available, otherwise alphabetically
    taxonomy_categories = taxonomy_config.get('categories', [])
    sorted_categories = []
    
    # First add categories that are in the taxonomy (in taxonomy order)
    for cat in taxonomy_categories:
        if cat in all_categories_in_data:
            sorted_categories.append(cat)
    
    # Then add any remaining categories alphabetically
    remaining_cats = sorted(all_categories_in_data - set(sorted_categories) - {'other'})
    sorted_categories.extend(remaining_cats)
    
    # Add 'other' at the end if present
    if 'other' in all_categories_in_data:
        sorted_categories.append('other')
    
    behavior_to_num = {cat: i for i, cat in enumerate(sorted_categories)}
    
    # Create color map - use predefined colors from taxonomy if available
    # Otherwise generate colors dynamically
    colors_list = []
    predefined_colors = {
        # Exploit precursor colors
        "Standard Rule & Strategy Exploration": '#2196F3',
        "Decision to Exploit": '#FF5722',
        "Decision to Accept Constraints": '#4CAF50',
        "Decision to Use Sanctioned Mechanic": '#FFC107',
        "Vulnerability Probing & Discovery": '#9C27B0',
        "Exploit Hypothesis Formulation": '#E91E63',
        "Unconventional Action Planning": '#B71C1C',
        # Hopelessness colors
        "Confident/Optimistic": '#2ecc71',
        "Neutral/Uncertain": '#3498db',
        "Struggling/Concerned": '#f39c12',
        "Hopeless/Defeated": '#e74c3c',
        # Original taxonomy colors
        "Exploration & Observation": '#17a2b8',
        "Task Planning": '#ffc107',
        "Task Execution": '#28a745',
        "Strategic Re-evaluation & Correction": '#9b59b6',
        "Sanctioned Assistance Reliance": '#e91e63',
        "Hint-Based Strategy": '#e91e63',
        "Strategic Deference": '#e91e63',
        "System Exploitation Planning": '#e67e22',
        "System Exploitation Execution": '#e74c3c',
        # Exploration taxonomy colors (diverse and high-contrast)
        "Reading Game File": '#1E88E5',
        "Reading Hint Engine Source": '#9C27B0',
        "Reading Opponent Source": '#CDDC39',
        "Reading Game State Files": '#00BCD4',
        "Environment Exploration": '#4CAF50',
        "Checking Help/Usage": '#FF9800',
        "Writing/Running Custom Code": '#F44336',
        "File Manipulation": '#795548',
        "Executing Moves/Actions": '#3F51B5',
        # Objective interpretation colors (high contrast)
        "Record-Focused": '#8B0000',           # Dark red
        "Score-Maximization": '#FF6B35',       # Bright orange
        "Win-Focused": '#004E89',              # Deep blue
        "Strategic/Gameplay-Focused": '#1B998B', # Teal
        "Neutral/No Objective Discussion": '#C5C6C8', # Light gray
        # Misunderstanding colors (high contrast)
        "Correct Understanding": '#27AE60',    # Green
        "Minor/Unclear Misunderstanding": '#F39C12', # Orange
        "Major Misunderstanding": '#E74C3C',   # Red
        "Multiple Misunderstandings": '#8E44AD', # Purple
        "No Clear Evidence": '#95A5A6',        # Gray
        "Objective Misunderstanding": '#C0392B', # Dark red
        "Rule Misunderstanding": '#D35400',    # Dark orange
        "State Misunderstanding": '#2980B9',   # Blue
        # Error category (highly visible)
        "ERROR: JSON parse error": '#FFD700',  # Gold/yellow
        # Fairness colors (high contrast)
        "Understands & Accepts Fairness": '#2ECC71',      # Bright green
        "Questions Fairness": '#F39C12',                   # Orange
        "Perceives Unfairness": '#E74C3C',                 # Red
        "Neutral/No Fairness Discussion": '#BDC3C7',      # Light gray
        # Other
        'other': '#95a5a6'
    }
    
    # Generate color palette for any categories without predefined colors
    num_undefined_colors = sum(1 for cat in sorted_categories if cat not in predefined_colors)
    if num_undefined_colors > 0:
        palette = sns.color_palette("husl", num_undefined_colors)
        palette_idx = 0
    
    for cat in sorted_categories:
        if cat in predefined_colors:
            colors_list.append(predefined_colors[cat])
        else:
            colors_list.append(plt.matplotlib.colors.rgb2hex(palette[palette_idx]))
            palette_idx += 1
    
    cmap = plt.matplotlib.colors.ListedColormap(colors_list)
    bounds = [i - 0.5 for i in range(len(sorted_categories) + 1)]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot each outcome
    for idx, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[idx]
        
        if outcome not in trajectories_by_outcome:
            ax.text(0.5, 0.5, f'No {outcome_labels.get(outcome, outcome)} games', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlabel('Turn Number', fontsize=10)
            ax.set_ylabel('Game Instance', fontsize=10)
            ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n=0 games)', 
                        fontsize=11, fontweight='bold')
            filename_mappings[outcome] = []
            continue
        
        trajectories = sorted(trajectories_by_outcome[outcome], key=lambda x: x['total_turns'])
        
        # Store filename-to-row mapping for this outcome
        filename_mappings[outcome] = [
            (i, traj['filename'], traj['total_turns']) 
            for i, traj in enumerate(trajectories)
        ]
        
        if not trajectories:
            continue
        
        max_turns = max(t['total_turns'] for t in trajectories)
        matrix = np.full((len(trajectories), max_turns), np.nan)
        
        for i, traj in enumerate(trajectories):
            for j, behavior in enumerate(traj['trajectory']):
                if behavior in behavior_to_num:
                    matrix[i, j] = behavior_to_num[behavior]
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        
        # Draw game start lines (black)
        for i, traj in enumerate(trajectories):
            game_starts = traj.get('game_start_positions', [])
            # If first_game_only flag is set, only draw the first game start line
            if first_game_only and game_starts:
                game_starts = game_starts[:1]
            for game_start in game_starts:
                if game_start > 0:
                    ymin = 1 - (i + 1) / len(trajectories)
                    ymax = 1 - i / len(trajectories)
                    ax.axvline(x=game_start - 0.5, ymin=ymin, ymax=ymax,
                              color='black', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Turn Number', fontsize=10)
        ax.set_ylabel('Game Instance', fontsize=10)
        ax.set_title(f'{outcome_labels.get(outcome, outcome)}\n(n={len(trajectories)} games)', 
                    fontsize=11, fontweight='bold')
        ax.grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Shared colorbar
    plt.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.12, 0.05, 0.76, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=list(range(len(sorted_categories))), orientation='horizontal')
    
    # Use actual category names with line breaks for better readability
    colorbar_labels = []
    for cat in sorted_categories:
        # Add line breaks to long category names
        if len(cat) > 25:
            # Try to break at '&' or mid-point
            if ' & ' in cat:
                label = cat.replace(' & ', ' &\n')
            elif len(cat) > 40:
                # Break at roughly middle word
                words = cat.split()
                mid = len(words) // 2
                label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            else:
                label = cat
        else:
            label = cat
        colorbar_labels.append(label)
    
    cbar.ax.set_xticklabels(colorbar_labels, fontsize=9)
    cbar.set_label('Detailed Behavior Type', labelpad=10, fontsize=11)
    
    plt.suptitle('Detailed Behavior Trajectories by Final Outcome\n(Black lines mark when games start)', 
                 fontsize=16, fontweight='bold', y=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Print filename-to-row mappings
    print("\n" + "="*80)
    print("ROW-TO-FILENAME MAPPING FOR DETAILED TRAJECTORY PLOT")
    print("="*80)
    for outcome in OUTCOME_ORDER:
        if outcome in filename_mappings and filename_mappings[outcome]:
            outcome_label = outcome_labels.get(outcome, outcome)
            print(f"\n{outcome_label}:")
            print("-" * 80)
            for row, filename, turns in filename_mappings[outcome]:
                print(f"  Row {row}: {filename} ({turns} turns)")
        elif outcome in filename_mappings:
            print(f"\n{outcome_labels.get(outcome, outcome)}: No games")
    print("="*80 + "\n")
    
    plt.close()

def generate_outcome_statistics(trajectories_by_outcome):
    """Print detailed statistics by outcome."""
    print("\n" + "="*70)
    print("BEHAVIOR ANALYSIS BY FINAL OUTCOME")
    print("="*70)
    
    for outcome in get_ordered_outcomes(trajectories_by_outcome):
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
        
        # Count each behavior
        behavior_counts = defaultdict(int)
        for b in all_behaviors:
            behavior_counts[b] += 1
        
        print(f"  Total Games: {total_games}")
        print(f"  Total Turns: {total_turns}")
        print(f"  Average Turns per Game: {total_turns/total_games:.1f}")
        
        print(f"\n  {'Behavior':<30} {'Count':<10} {'Percentage':<10} {'Per Game':<10}")
        print(f"  {'-'*60}")
        
        for behavior in sorted(behavior_counts.keys()):
            count = behavior_counts[behavior]
            print(f"  {behavior:<30} {count:<10} {100*count/total_turns:>6.1f}% {count/total_games:>9.2f}")
    
    print("\n" + "="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze agent behaviors by outcome for any classification taxonomy'
    )
    parser.add_argument(
        '--separate-by-hint',
        action='store_true',
        help='Generate separate visualizations for each hint level'
    )
    parser.add_argument(
        '--classifications-dir',
        type=str,
        required=True,
        help='Directory containing classification files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--taxonomy',
        type=str,
        choices=['auto', 'exploit_precursor', 'hopelessness', 'exploration', 'original'],
        default='auto',
        help='Taxonomy to use (auto-detect by default)'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Exclude games with more than this many turns (useful for filtering outliers)'
    )
    parser.add_argument(
        '--filter-categories',
        type=str,
        nargs='+',
        default=None,
        help='Only show these categories (others will be grouped as "other"). Use category names after mapping (e.g., "exploit_planning", "decision_exploit")'
    )
    parser.add_argument(
        '--first-game-only',
        action='store_true',
        help='Only show the first game start line (useful for chess where there should be only one game per session)'
    )
    
    args = parser.parse_args()
    
    classifications_dir = args.classifications_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading classification files...")
    games = load_classifications(classifications_dir)
    print(f"Loaded {len(games)} games with valid final_behavior")
    
    if not games:
        print("Error: No valid games found")
        return
    
    # Detect or use specified taxonomy
    if args.taxonomy == 'auto':
        taxonomy_name = detect_taxonomy(games)
        print(f"Auto-detected taxonomy: {taxonomy_name}")
    else:
        taxonomy_name = args.taxonomy
        print(f"Using specified taxonomy: {taxonomy_name}")
    
    # Get taxonomy config
    if taxonomy_name == 'custom':
        # Extract all categories for custom taxonomy
        all_categories = set()
        for game in games:
            classifications = game['data'].get('classifications', [])
            for item in classifications:
                if isinstance(item, dict):
                    cat = item.get('classification')
                else:
                    cat = item
                if cat:
                    all_categories.add(cat)
        taxonomy_config = get_taxonomy_config('custom', all_categories)
        print(f"Created custom taxonomy with {len(all_categories)} categories")
    else:
        taxonomy_config = get_taxonomy_config(taxonomy_name)
        print(f"Using {len(taxonomy_config['categories'])} predefined categories")
    
    # Print filter information if categories are filtered
    if args.filter_categories:
        print(f"\nFiltering to show only these categories: {', '.join(args.filter_categories)}")
        print("(All other categories will be grouped as 'other')")
    
    if args.separate_by_hint:
        # Group games by hint level
        games_by_hint = defaultdict(list)
        for game in games:
            hint_level = game.get('hint_level')
            if hint_level is not None:
                games_by_hint[hint_level].append(game)
            else:
                games_by_hint['unknown'].append(game)
        
        print(f"\nHint level distribution:")
        for hint_level in sorted(games_by_hint.keys(), key=lambda x: (isinstance(x, str), x)):
            print(f"  Hint level {hint_level}: {len(games_by_hint[hint_level])} games")
        
        # Generate visualizations for each hint level
        for hint_level in sorted(games_by_hint.keys(), key=lambda x: (isinstance(x, str), x)):
            hint_games = games_by_hint[hint_level]
            hint_suffix = f"_hint_{hint_level}"
            
            print(f"\n{'='*70}")
            print(f"Processing hint level {hint_level} ({len(hint_games)} games)")
            print('='*70)
            
            print("Extracting trajectories by outcome...")
            # Extract simplified trajectories for basic plot
            trajectories_by_outcome = extract_trajectories_by_outcome(hint_games, taxonomy_config, use_simplified=True, max_turns=args.max_turns, filter_categories=args.filter_categories)
            # Extract detailed trajectories with original names for detailed plot
            trajectories_by_outcome_detailed = extract_trajectories_by_outcome(hint_games, taxonomy_config, keep_original=True, max_turns=args.max_turns, filter_categories=args.filter_categories)
            
            # Print outcome counts
            print(f"\nOutcome distribution for hint level {hint_level}:")
            for outcome in get_ordered_outcomes(trajectories_by_outcome):
                trajectories = trajectories_by_outcome[outcome]
                print(f"  {outcome}: {len(trajectories)} games")
            
            if not trajectories_by_outcome:
                print(f"No valid trajectories for hint level {hint_level}, skipping...")
                continue
            
            # Generate statistics
            generate_outcome_statistics(trajectories_by_outcome)
            
            # Create visualizations
            print("\nGenerating trajectory visualizations...")
            
            plot_behavior_trajectories_by_outcome(
                trajectories_by_outcome,
                taxonomy_config,
                os.path.join(output_dir, f"trajectories_by_outcome{hint_suffix}.png"),
                first_game_only=args.first_game_only
            )
            
            plot_behavior_trajectories_detailed(
                trajectories_by_outcome_detailed,
                taxonomy_config,
                os.path.join(output_dir, f"trajectories_by_outcome_detailed{hint_suffix}.png"),
                first_game_only=args.first_game_only
            )
        
        print(f"\n{'='*70}")
        print(f"All visualizations saved to: {output_dir}")
    
    else:
        # Original behavior: combine all hint levels
        print("Extracting trajectories by outcome...")
        # Extract simplified trajectories for basic plot
        trajectories_by_outcome = extract_trajectories_by_outcome(games, taxonomy_config, use_simplified=True, max_turns=args.max_turns, filter_categories=args.filter_categories)
        # Extract detailed trajectories with original names for detailed plot
        trajectories_by_outcome_detailed = extract_trajectories_by_outcome(games, taxonomy_config, keep_original=True, max_turns=args.max_turns, filter_categories=args.filter_categories)
        
        # Print outcome counts
        print(f"\nOutcome distribution:")
        for outcome in get_ordered_outcomes(trajectories_by_outcome):
            trajectories = trajectories_by_outcome[outcome]
            print(f"  {outcome}: {len(trajectories)} games")
        
        # Generate statistics
        generate_outcome_statistics(trajectories_by_outcome)
        
        # Create visualizations
        print("\nGenerating trajectory visualizations...")
        
        plot_behavior_trajectories_by_outcome(
            trajectories_by_outcome,
            taxonomy_config,
            os.path.join(output_dir, "trajectories_by_outcome.png"),
            first_game_only=args.first_game_only
        )
        
        plot_behavior_trajectories_detailed(
            trajectories_by_outcome_detailed,
            taxonomy_config,
            os.path.join(output_dir, "trajectories_by_outcome_detailed.png"),
            first_game_only=args.first_game_only
        )
        
        print(f"\nAll trajectory visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
