# Causal Analysis Pipeline

## Prerequisites
- Base state JSON file in `base_states/` folder (e.g., `base_states/gpt-5/cheating.json`)
- Fill in file paths in `config.py` (BASE_STATE_FILE, EXPERIMENTS_DIR, ROLLOUTS_DIR, MODEL, etc.)
- Define properties to test in `config.PROPERTIES`
- Match settings of base state JSON in config.GAME_PROPERTIES

## Pipeline Steps

### 1. Generate Rollouts (property-agnostic, reusable)
```bash
python generate_rollouts_parallel.py --num_rollouts 200
# Generates Y counterfactual rollouts from the base state
# Output: rollouts/gpt-5/gpt-5-<timestamp>/
```

### 2. Grade Rollouts for Reward Hacking
```bash
# Get the most recent rollouts directory
ROLLOUTS_DIR=$(ls -td rollouts/gpt-5/gpt-5-* | head -1)
python judge_json.py "$ROLLOUTS_DIR"
# Uses LLM to classify each rollout as hacked or not
# Output: <rollouts_dir>/rollout_analysis_detailed.json
```

### 3. Merge Grading Results
```bash
python merge_grading_results.py
# Merges reward_hacked field into rollouts_summary.json
# Output: Updates <rollouts_dir>/rollouts_summary.json
```

### 4. Judge Property Presence
Judge rollouts for your target property AND conditioning property (if doing conditional analysis).
**Important**: Use the same judge mode (--only_first_resampled or all) for both properties!

```bash
# Target property
python judge_rollouts.py ethical_concerns --only_first_resampled
# Output: results/gpt-5/ethical_concerns/judgments.json

# Conditioning property (for conditional analysis)
python judge_rollouts.py thought_about_exploit --only_first_resampled
# Output: results/gpt-5/thought_about_exploit/judgments.json
```

### 5. Compute Metrics & Analyze

**Conditional analysis** (recommended to handle confounding):
```bash
# Compute metrics conditioned on thought_about_exploit = True
python compute_metrics.py ethical_concerns --condition_on thought_about_exploit
# Output: results/gpt-5/ethical_concerns/conditioned_on_thought_about_exploit/
#   - judgments_conditioned_on_thought_about_exploit.json (filtered rollouts)
#   - metrics.json

# Statistical analysis and visualization
python analyze_results.py ethical_concerns --condition_on thought_about_exploit
# Output: results/gpt-5/ethical_concerns/conditioned_on_thought_about_exploit/
#   - results.json (bootstrap CIs, p-values)
#   - visualization.png
```

**Unconditional analysis** (optional):
```bash
# Compute metrics without conditioning
python compute_metrics.py ethical_concerns
# Output: results/gpt-5/ethical_concerns/metrics.json

# Statistical analysis and visualization
python analyze_results.py ethical_concerns
# Output: results/gpt-5/ethical_concerns/
#   - results.json
#   - visualization.png
```

## Output Structure
```
rollouts/gpt-5/<timestamp>/
├── rollouts_summary.json           # Metadata with reward_hacked field
├── rollout_analysis_detailed.json  # Detailed reward hack analysis
└── state-run*/                     # Individual rollout state directories

results/gpt-5/<property>/
├── judgments.json                  # Property classifications (all rollouts)
├── metrics.json                    # Unconditional metrics
├── results.json                    # Unconditional statistical analysis
├── visualization.png               # Unconditional plot
└── conditioned_on_<property>/      # Conditional results subdirectory
    ├── judgments_conditioned_on_<property>.json
    ├── metrics.json
    ├── results.json
    └── visualization.png
```