# Turn Anchors Analysis

Automated turn-by-turn analysis system that resamples from each state in a run and computes the `hint/(hint+hack)` metric.

## Directory Structure

```
turn_anchors/
├── config.py                    # Configuration (edit this!)
├── generate_all_anchors.py      # Generate rollouts from each turn
├── compute_anchor_metrics.py    # Grade rollouts and compute metrics
├── visualize_anchors.py         # Create visualization
│
├── input_states/                # Put state-run folders here
│   └── state-run5/
│       ├── ctfish-tictactoe-0000.json
│       ├── ctfish-tictactoe-0001.json
│       └── ...
│
├── rollouts/                    # Generated rollouts (auto-created)
│   └── run5/
│       ├── step_00/
│       ├── step_01/
│       └── ...
│
├── analysis/                    # Grading results (auto-created)
│   └── run5/
│       ├── step_00/
│       │   └── metric_result.json
│       └── ...
│
└── summary/                     # Final results (auto-created)
    ├── run5_metrics.json
    └── run5_plot.png
```

## Usage

### 1. Setup

Copy a state-run folder from your experiments:

```bash
# Copy state folder
cp -r ../o3_hint_rollouts/o3-2025-04-16-20251030-105921/state-run5 turn_anchors/input_states/
```

### 2. Configure

Edit `config.py` and update these settings:

```python
# Which run to process
RUN_TO_PROCESS = "state-run5"

# CRITICAL: Match the original experiment's hint penalty!
GAME_CONFIG = {
    # ...
    "hint_penalty": 1,  # Use the value from the original run
}

# Also update USER_PROMPT to match (e.g., "costs 1 points" or "costs 7 points")
```

### 3. Run the Pipeline

```bash
cd turn_anchors

# Step 1: Generate N rollouts from each turn (with parallelization)
python3 generate_all_anchors.py

# Step 2: Grade rollouts and compute metrics
python3 compute_anchor_metrics.py

# Step 3: Visualize results
python3 visualize_anchors.py
```

### Parallelization

By default, `generate_all_anchors.py` processes **5 steps in parallel**. Each step internally runs N rollouts in parallel (via `run-parallel.sh`), so this gives you excellent throughput.

To adjust parallelization, you can edit `config.py`:

```python
# In generate_all_anchors.py call
python3 generate_all_anchors.py --max-parallel 10  # Process 10 steps at once
```

**Example with your setup (n=20 per step, 5 parallel steps):**
- 5 steps running simultaneously
- Each step runs 20 rollouts in parallel
- Total: up to 100 Docker containers running at once
- Much faster than sequential processing!

## Output

- **`rollouts/run5/step_XX/`**: Contains N rollouts generated from turn XX
- **`analysis/run5/step_XX/metric_result.json`**: Contains hint count, hack count, and metric for turn XX
- **`summary/run5_metrics.json`**: Aggregated metrics for all turns
- **`summary/run5_plot.png`**: Visualization of metric vs turn number

## Configuration Details

### Critical Settings

These MUST match the original experiment that generated the states:

- `hint_penalty`: The actual hint cost in the game (1, 7, etc.)
- `USER_PROMPT`: Should mention the correct hint cost
- `SYSTEM_PROMPT`: Should match the original
- `GAME_CONFIG`: All game parameters (win/draw/loss values, practice games, etc.)

### Rollout Settings

- `ROLLOUTS_PER_STEP`: How many rollouts to generate per turn (default: 50)
- `MODEL`: Which model to use (default: "o3-2025-04-16")
- `MAX_STEPS`: Maximum steps per rollout (default: 40)

## Tips

- **Resume**: Run `generate_all_anchors.py` again to continue if interrupted (uses `skip_existing=True`)
- **Multiple runs**: Change `RUN_TO_PROCESS` in `config.py` and rerun
- **Debug**: Check `rollouts/runX/step_XX/o3-2025-04-16-TIMESTAMP/` for individual rollout logs
- **Verify config**: Always verify `hint_penalty` matches the original run before generating rollouts!
- **Error logs**: If errors occur during generation, check `summary/runX_generation_errors_TIMESTAMP.log` for details
- **Patch old rollouts**: If you have rollouts from before summary creation was added, run `python3 patch_create_summaries.py`

### Patching Old Rollouts

If you generated rollouts before the `rollouts_summary.json` creation logic was added, you can retroactively create the summary files:

```bash
# Patch rollouts for the configured run
python3 patch_create_summaries.py

# Or patch a specific run
python3 patch_create_summaries.py --run-name run5
```

This will:
- Find all `step_XX/` directories
- Aggregate `run_metadata.json` files in each
- Create `rollouts_summary.json` for each step
- Skip steps that already have summaries

### Error Handling

The system automatically logs errors during rollout generation:

- **Rate limit errors**: Detected and flagged with `[RATE_LIMIT]` or `[RATE_LIMIT_429]`
- **OpenAI API errors**: Detected and flagged with `[OPENAI_API_ERROR]`
- **Server errors**: Detected and flagged with `[OPENAI_SERVER_ERROR]` (500, 502, 503)
- **Error log location**: `summary/run5_generation_errors_TIMESTAMP.log`

Example error log entry:
```json
{
  "step": 5,
  "state_file": "turn_anchors/input_states/state-run5/ctfish-tictactoe-0005.json",
  "error": "[RATE_LIMIT] Rate limit exceeded. Please try again later.",
  "timestamp": "2025-11-02T14:23:45.123456"
}
```

If you encounter rate limits, you can:
1. Wait a few minutes and rerun (it will skip completed steps)
2. Reduce `MAX_PARALLEL_STEPS` to lower concurrency
3. Reduce `ROLLOUTS_PER_STEP` to generate fewer rollouts per step

## Example Workflow

```bash
# Analyze run5 (hint penalty = 1)
cp -r ../o3_hint_rollouts/.../state-run5 turn_anchors/input_states/
# Edit config.py: RUN_TO_PROCESS = "state-run5", hint_penalty = 1
python3 generate_all_anchors.py
python3 compute_anchor_metrics.py
python3 visualize_anchors.py

# Analyze run9 (hint penalty = 7)
cp -r ../o3_hint_rollouts/.../state-run9 turn_anchors/input_states/
# Edit config.py: RUN_TO_PROCESS = "state-run9", hint_penalty = 7, update USER_PROMPT
python3 generate_all_anchors.py
python3 compute_anchor_metrics.py
python3 visualize_anchors.py
```

Now you have `summary/run5_plot.png` and `summary/run9_plot.png` for comparison!

