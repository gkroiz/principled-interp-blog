# Action Resampling Analysis

Computes causal attribution of agent actions by resampling from each checkpoint in a run and measuring the `hint/(hint+hack)` metric.

## Overview

Given a completed run with step checkpoints, this pipeline:
1. **Resamples** N rollouts from each step checkpoint using `resume.sh`
2. **Grades** each resampled rollout using `grade_rollouts.py`
3. **Plots** the metric across steps to visualize causal effects

## Usage

### Quick Start (Single Command)

```bash
# Run full pipeline on a completed run
uv run python analysis/action_branches/action_resampling/run_anchor_analysis.py \
    results/tictactoe/claude-haiku-4-5/2025-12-31_18-28-38/run-1 \
    --count 20

# With options
uv run python analysis/action_branches/action_resampling/run_anchor_analysis.py \
    results/.../run-1 \
    --count 20 \
    --local \
    --max-parallel 4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--count N` | Number of rollouts per step | 10 |
| `--local` | Use local Docker image | False |
| `--max-parallel N` | Parallel resample jobs | 1 |
| `--skip-resample` | Skip resampling step | False |
| `--skip-grade` | Skip grading step | False |
| `--game GAME` | Game type (auto-detected) | from config.yaml |

### Running Steps Individually

```bash
# 1. Resample only
./scripts/resample.sh results/.../run-1 --count 20 --max-parallel 4

# 2. Grade a specific resampled folder
uv run python analysis/grading/grade_rollouts.py \
    results/.../run-1/step-0/2025-12-31_22-00-00 \
    --game tictactoe

# 3. Plot only (after resampling and grading)
uv run python plots/action_resampling/plot_anchors.py \
    results/.../run-1
```

## Input Structure

Expects a run folder with step checkpoints from `run.sh` or `run_experiments.py`:

```
results/<env>/<model>/<timestamp>/
├── config.yaml                    # Experiment config (game type auto-detected)
├── rollout_analysis_detailed.json # Grading of original runs (baseline)
├── run-1/
│   ├── rollout.log
│   ├── step-0/
│   │   ├── state.json
│   │   ├── messages.json
│   │   └── game/
│   ├── step-1/
│   │   └── ...
│   └── step-N/  (terminal state, excluded from resampling)
└── run-2/
    └── ...
```

## Output Structure

After running the pipeline:

```
run-1/
├── step-0/
│   └── 2025-12-31_22-00-00/       # Resampled rollouts
│       ├── run-1/rollout.log
│       ├── run-2/rollout.log
│       └── rollout_analysis_detailed.json
├── step-1/
│   └── 2025-12-31_22-01-00/
│       └── ...
├── anchor_metrics.json            # Aggregated metrics
└── anchor_plot.png                # Visualization
```

## Metric Computation

The metric is `hint / (hint + hack)` where:
- **Hack takes precedence**: if a rollout both hacks AND hints, it counts as hack only
- **Baseline**: Original runs from parent folder serve as "resampling from before action 0"

### Plot Interpretation

**Top panel (line plot):**
- X-axis: "Resampling from before action N"
  - x=0: baseline (original runs)
  - x=1: resampling from step-0 (action 0 is fixed)
  - x=N: resampling from step N-1 (actions 0 to N-1 are fixed)
- Y-axis: hint/(hint+hack) metric (0 = all hacks, 1 = all hints)

**Bottom panel (bar chart):**
- Causal effect of each action
- Bar at action N = metric[N] - metric[N-1]
- Green = action increased hint usage
- Red = action increased hack usage

## Files

| File | Purpose |
|------|---------|
| `run_anchor_analysis.py` | Orchestrates the full pipeline |
| `plots/action_resampling/plot_anchors.py` | Computes metrics and creates visualization |

## Example Workflow

```bash
# 1. Run initial experiment
uv run ./scripts/run.sh dev_config.yaml --count 5

# 2. Grade original runs
uv run python analysis/grading/grade_rollouts.py \
    results/tictactoe/claude-haiku-4-5/2025-12-31_18-28-38 \
    --game tictactoe

# 3. Run anchor analysis on one run
uv run python analysis/action_branches/action_resampling/run_anchor_analysis.py \
    results/tictactoe/claude-haiku-4-5/2025-12-31_18-28-38/run-1 \
    --count 20 --local

# 4. View results
open results/.../run-1/anchor_plot.png
cat results/.../run-1/anchor_metrics.json
```

## Notes

- The **last step** (terminal state) is automatically excluded from resampling since the game is already over
- Game type is auto-detected from `config.yaml` in the parent directory
- Use `--skip-resample` to re-grade and re-plot without regenerating rollouts
- Use `--skip-grade` to just re-plot with existing grading results
