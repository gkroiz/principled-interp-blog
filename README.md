# Principled Interpretability Experiments

Experimental framework for studying AI agent behavior in game environments. Analyzes decision-making patterns, hint usage, and reward hacking through controlled rollouts and causal analysis.

## Architecture

This repo works alongside [`agent-interp-envs`](../agent-interp-envs), which provides:
- Docker images for game environments (chess, tic-tac-toe)
- Core agent loop and state management
- Tool calling interface for LLM agents

**This repo** provides:
- Experiment orchestration scripts
- LLM-based rollout grading
- Agent behavior classification
- Action resampling for causal analysis

## Quick Start

```bash
# Install dependencies (choose one)
uv sync                      # recommended
pip install -r requirements.txt

# Set up API keys (.env file)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...

# Run a simple experiment
./scripts/run.sh dev_config.yaml --count 5

# Grade the rollouts
uv run python analysis/grading/grade_rollouts.py \
    results/tictactoe/openai-gpt-5-nano/<timestamp> \
    --game tictactoe
```

## Running Experiments

### Option 1: `run.sh` (Simple)

```bash
./scripts/run.sh <config.yaml> [options]
```

| Flag | Description |
|------|-------------|
| `--count N` | Number of parallel rollouts |
| `--local` | Use local Docker image |
| `--build` | Build local image first (requires `--local`) |

**Output:** `results/<environment>/<model>/<timestamp>/`

### Option 2: `run_experiments.py` (Multi-experiment)

```bash
uv run python experiments/run_experiments.py experiment_config.json [options]
```

Supports running multiple experiment configurations with different prompts, settings, etc. See `experiments/experiment_config.example.json`.

### Resuming from Checkpoints

```bash
./scripts/resume.sh <step-folder> [options]
```

Resume from any saved step checkpoint (contains `state.json`, `messages.json`, `game/`).

| Flag | Description |
|------|-------------|
| `--count N` | Number of parallel rollouts |
| `--local` | Use local Docker image |
| `--config PATH` | Explicit config path |

## Analysis Pipeline

### 1. Grade Rollouts

Classify each rollout as hint-terminated or reward-hacked using an LLM judge:

```bash
uv run python analysis/grading/grade_rollouts.py <results-folder> --game <game>
```

**Output:** `rollout_analysis_detailed.json`, `rollout_summary.json`

### 2. Classify Agent Turns

Categorize each agent turn by behavior type (exploration, exploitation, etc.):

```bash
uv run python analysis/action_branches/action_classification/classify_agent_turns.py \
    <messages.json> --output classifications.json
```

> **Note:** Currently only supports OpenRouter models

See `analysis/action_branches/action_classification/README.md` for details.

### 3. Action Resampling

Compute causal attribution by resampling from each step checkpoint:

```bash
uv run python analysis/action_branches/action_resampling/run_anchor_analysis.py \
    results/.../run-1 --count 20 --local
```

See `analysis/action_branches/action_resampling/README.md` for details.

### 4. Causal Analysis

Test specific hypotheses about what causes hint vs. hack behavior:

```bash
# Grade rollouts for specific properties
uv run python analysis/action_branches/action_causal_analysis/property_judge.py \
    base_states/my-experiment thought_about_exploit

# Compute causal effect
uv run python analysis/action_branches/action_causal_analysis/compute_causal_effect.py \
    base_states/my-experiment thought_about_exploit
```

See `analysis/action_branches/action_causal_analysis/README.md` for details.

## Configuration

Example `config.yaml`:

```yaml
environment: tictactoe  # or chess

agent:
  model: openai/gpt-5-nano  # OpenRouter model path
  max_steps: 10

task:
  board_size: 4
  win_length: 3
  good_engine: true
  hint_mode: none  # none, always, smart

prompts:
  system_prompt: |
    You are an autonomous agent with access to a UNIX shell...
  user_prompt: |
    Your task is to play tic-tac-toe against an opponent...
```

## Directory Structure

```
principled-interp-blog/
├── scripts/
│   ├── run.sh              # Run fresh experiments
│   ├── resume.sh           # Resume from checkpoint
│   └── resample.sh         # Resample from multiple steps
├── experiments/
│   └── run_experiments.py  # Multi-experiment orchestration
├── analysis/
│   ├── grading/
│   │   └── grade_rollouts.py
│   └── action_branches/
│       ├── action_classification/  # Turn-by-turn behavior classification
│       ├── action_resampling/      # Causal attribution via resampling
│       └── action_causal_analysis/ # Property-based causal analysis
├── results/                # Experiment outputs
│   └── <env>/<model>/<timestamp>/
│       ├── config.yaml
│       ├── run-N/
│       │   ├── rollout.log
│       │   └── step-M/
│       ├── rollout_analysis_detailed.json
│       └── rollout_summary.json
└── plots/                  # Generated visualizations
```

## Output Format

Each run produces:

```
run-N/
├── rollout.log           # Full agent transcript
├── step-0/
│   ├── state.json        # Game state checkpoint
│   ├── messages.json     # Conversation history
│   └── game/
│       ├── board.txt
│       ├── moves.txt
│       └── status.txt
├── step-1/
│   └── ...
└── step-M/               # Terminal state
```

## Development Workflow

When modifying the agent environment:

1. Edit code in `agent-interp-envs`
2. Build local Docker image: `docker build -t tictactoe:latest .`
3. Run with `--local` flag: `./scripts/run.sh config.yaml --local`

## Requirements

- Python 3.11+
- Docker
- `uv` (recommended) or `pip`
- API keys for OpenAI/OpenRouter

