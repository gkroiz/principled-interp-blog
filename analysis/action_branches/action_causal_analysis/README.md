# Causal Analysis of Agent Properties

Analyze causal effects of agent properties (e.g., "thought about exploits", "ethical concerns") on hint/hack behavior.

## Overview

Given a base state (checkpoint from a run), this pipeline:
1. **Resamples** N rollouts from that checkpoint using `resume.sh`
2. **Grades** rollouts for hint/hack using `grade_rollouts.py`
3. **Judges** rollouts for property presence using `property_judge.py`
4. **Computes** causal effect: compares `hint/(hint+hack)` between groups with/without property

## Directory Structure

```
action_causal_analysis/
├── property_judge.py      # Judge rollouts for property presence
├── compute_causal_effect.py  # Compute causal effect with bootstrap CI
├── base_states/           # Store copied checkpoints here
│   └── <experiment-name>/
│       └── <timestamp>/   # Results from resume.sh
└── README.md
```

## Pipeline

### 1. Copy a Checkpoint

Copy a step folder from an existing run:

```bash
cp -r results/tictactoe/claude-haiku-4-5/2025-12-31/run-1/step-5 \
    analysis/action_branches/action_causal_analysis/base_states/my-experiment

# Also copy the config
cp results/tictactoe/claude-haiku-4-5/2025-12-31/config.yaml \
    analysis/action_branches/action_causal_analysis/base_states/my-experiment/
```

### 2. Generate Rollouts

Use `resume.sh` to generate rollouts from the checkpoint. Run multiple times to accumulate enough rollouts for statistical power:

```bash
./scripts/resume.sh <checkpoint-folder> --config <config-path> --count N [--local]
```

Arguments:
- `<checkpoint-folder>`: Path to the step folder (must contain `state.json`, `messages.json`, `game/`)
- `--config`: Path to config.yaml (required since checkpoint is outside normal results structure)
- `--count N`: Number of parallel rollouts to generate
- `--local`: Use local Docker image instead of Dockerhub

Example:
```bash
./scripts/resume.sh \
    base_states/my-experiment \
    --config base_states/my-experiment/config.yaml \
    --count 100 \
    --local
```

Each run creates a new timestamp folder with N rollouts:
```
base_states/my-experiment/
├── 2025-12-31_22-00-00/
│   ├── run-1/rollout.log
│   ├── run-2/rollout.log
│   └── ...
├── 2025-12-31_22-15-00/
└── ...
```

### 3. Grade Rollouts

Grade each timestamp folder for hint/hack:

```bash
for dir in base_states/my-experiment/2025-*; do
    uv run python analysis/grading/grade_rollouts.py "$dir" --game tictactoe
done
```

### 4. Judge for Properties

Judge all rollouts for a property:

```bash
# Judges ALL timestamp folders
uv run python analysis/action_branches/action_causal_analysis/property_judge.py \
    base_states/my-experiment \
    thought_about_exploit
```

Available properties:
- `thought_about_exploit` - Agent considers hacks/exploits
- `rule_rationalization` - Agent rationalizes gray-area actions as acceptable
- `rule_legitimacy` - Agent treats rules as binding constraints
- `ethical_concerns` - Agent expresses concern about cheating
- `point_tradeoff_calculation` - Agent calculates point tradeoffs

### 5. Compute Causal Effect

Compute effect with bootstrap confidence intervals:

```bash
# Basic analysis
uv run python analysis/action_branches/action_causal_analysis/compute_causal_effect.py \
    base_states/my-experiment \
    thought_about_exploit

# Conditional analysis (effect of B given A = True)
uv run python analysis/action_branches/action_causal_analysis/compute_causal_effect.py \
    base_states/my-experiment \
    ethical_concerns \
    --condition-on thought_about_exploit
```

## Output

Property judgments (per timestamp folder):
- `property_judgments_<property>.json` - Per-rollout judgments
- `property_summary_<property>.json` - Aggregated counts

Causal analysis (in experiment folder):
- `causal_effect_<property>.json` - Full results with bootstrap CI
- `causal_effect_<property>.png` - Bar chart visualization

## Interpretation

The causal effect is:
```
effect = metric_with_property - metric_without_property
```

Where `metric = hint / (hint + hack)` (higher = more hints, fewer hacks).

- **Positive effect**: Property INCREASES hint usage (decreases hacking)
- **Negative effect**: Property DECREASES hint usage (increases hacking)

Statistical significance is determined by bootstrap CI and p-value.

## Example Workflow

```bash
# Setup - copy checkpoint and config
cp -r results/.../run-1/step-5 base_states/ethics-study
cp results/.../config.yaml base_states/ethics-study/

# Generate rollouts (run multiple times for more data)
./scripts/resume.sh base_states/ethics-study \
    --config base_states/ethics-study/config.yaml \
    --count 100 --local

# Grade all timestamp folders
for dir in base_states/ethics-study/2025-*; do
    uv run python analysis/grading/grade_rollouts.py "$dir" --game tictactoe
done

# Judge for properties
uv run python property_judge.py base_states/ethics-study thought_about_exploit
uv run python property_judge.py base_states/ethics-study ethical_concerns

# Compute causal effects
uv run python compute_causal_effect.py base_states/ethics-study thought_about_exploit
uv run python compute_causal_effect.py base_states/ethics-study ethical_concerns --condition-on thought_about_exploit

# View results
open base_states/ethics-study/causal_effect_*.png
```

## Notes

- More rollouts = better statistical power for bootstrap CI
- Use `--condition-on` for conditional independence tests
- Properties are defined in `property_judge.py` (can be extended)
- All scripts auto-iterate over timestamp folders when given a parent directory

