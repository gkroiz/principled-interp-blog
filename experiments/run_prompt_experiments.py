#!/usr/bin/env python3
"""
Automated script to run multiple prompt experiments.

This script automates the loop of:
1. Build Docker image (once for all experiments)
2. Run parallel experiments (passing prompts as env vars)
3. Grade rollouts

Usage:
    python run_prompt_experiments.py --config experiment_config.json
    python run_prompt_experiments.py --config experiment_config.json --experiments baseline powerless
    python run_prompt_experiments.py --config experiment_config.json --initial-state states/ctfish-0003.json

State Resumption:
    The --initial-state flag allows resuming experiments from a saved state file.
    This can be specified:
    1. Via command line: --initial-state=path/to/state.json (overrides config)
    2. In run_config: {"run_config": {"initial_state": "path/to/state.json"}}
    3. Per experiment: {"experiments": [{"name": "exp1", "initial_state": "path/to/state.json"}]}
    
    Priority: CLI flag > per-experiment config > run_config
"""

import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import fire


def log(message: str, level: str = "INFO"):
    """Pretty print log messages"""
    colors = {
        "INFO": "\033[36m",  # Cyan
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m",
    }
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = colors.get(level, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{timestamp}] {level}: {message}{reset}")


def run_command(
    cmd: str | list[str], cwd: str = ".", check: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command and log output"""
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd

    log(f"Running: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 with replacement character
            check=check,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        log(f"Command failed with exit code {e.returncode}", "ERROR")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise


# Note: No longer needed - we use environment variables instead of file modification
# def modify_prompt_in_player(...) - REMOVED
# def restore_player_backup(...) - REMOVED


def build_docker_image(
    tag: str,
    game: str = "tictactoe",
    board_size: int = 5,
    win_length: int = 5,
    max_depth: int = 0,
    multi_game: int = 0,
    hints_enabled: int = 1,
    good_engine: int = 1,
    hint_ablation: int = 0,
    max_retries: int = 3,
    # Chess-specific parameters
    chess_max_depth: int = 15,
    chess_analysis_time: float = 7.5,
    chess_move_time: float = 1.0,
    chess_hint_ablation: int = 0,
) -> None:
    """Build Docker image with specified configuration
    
    Args:
        tag: Docker image tag
        game: Game type ('chess' or 'tictactoe')
        board_size: Board size (tictactoe only)
        win_length: Win length (tictactoe only)
        max_depth: Max depth (tictactoe only)
        multi_game: Multi-game mode (tictactoe only)
        hints_enabled: Hints enabled (tictactoe only)
        good_engine: Good engine (tictactoe only)
        hint_ablation: Use hint ablation mode (tictactoe only)
        max_retries: Number of retries for transient errors
        chess_max_depth: Stockfish analysis depth (chess only)
        chess_analysis_time: Time limit for analysis (chess only)
        chess_move_time: Time limit for moves (chess only)
        chess_hint_ablation: Use hint ablation mode for chess (chess only)
    """
    log(f"Building Docker image: {tag} (game: {game})")

    # Select Dockerfile based on game
    if game == "chess":
        dockerfile = "Dockerfile.chess"
        cmd = [
            "docker",
            "build",
            "-f",
            dockerfile,
            "-t",
            tag,
            f"--build-arg=MAX_DEPTH={chess_max_depth}",
            f"--build-arg=ANALYSIS_TIME={chess_analysis_time}",
            f"--build-arg=MOVE_TIME={chess_move_time}",
            f"--build-arg=HINT_ABLATION={chess_hint_ablation}",
            ".",
        ]
    elif game == "tictactoe":
        dockerfile = "Dockerfile.tictactoe"
        cmd = [
            "docker",
            "build",
            "-f",
            dockerfile,
            "-t",
            tag,
            f"--build-arg=BOARD_SIZE={board_size}",
            f"--build-arg=WIN_LENGTH={win_length}",
            f"--build-arg=MAX_DEPTH={max_depth}",
            f"--build-arg=MULTI_GAME={multi_game}",
            f"--build-arg=HINTS_ENABLED={hints_enabled}",
            f"--build-arg=GOOD_ENGINE={good_engine}",
            f"--build-arg=HINT_ABLATION={hint_ablation}",
            ".",
        ]
    else:
        raise ValueError(f"Unknown game type: {game}. Must be 'chess' or 'tictactoe'.")

    # Retry logic for transient Docker errors
    for attempt in range(1, max_retries + 1):
        try:
            run_command(cmd)
            log(f"Docker image built successfully: {tag}", "SUCCESS")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                # Check if it's a credential error (transient)
                error_msg = (e.stderr or "") + (e.stdout or "")
                if (
                    "credentials" in error_msg.lower()
                    or "authentication" in error_msg.lower()
                ):
                    log(
                        f"Docker credential error on attempt {attempt}/{max_retries}, retrying in 5 seconds...",
                        "WARNING",
                    )
                    time.sleep(5)
                    continue
            # Re-raise if we've exhausted retries or it's not a credential error
            raise


def run_parallel_experiments(
    image_tag: str,
    num_replicas: int,
    model: str,
    max_steps: int = 999999,
    system_prompt: str = "",
    user_prompt: str = "",
    game_config: dict = None,
    experiment_name: str = "",
    description: str = "",
    full_config: dict = None,
    base_dir: str = "continued_reasoning_results",
    initial_state: str = "",
    game: str = "tictactoe",
) -> str:
    """
    Run parallel experiments using run-parallel.sh

    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        game_config: Game configuration (win_value, draw_value, etc.)
        experiment_name: Name of the experiment (for saving config)
        description: Description of the experiment (for saving config)
        full_config: Full experiment configuration (for saving config)
        base_dir: Base directory for storing experiment outputs (default: continued_reasoning_results)
        initial_state: Path to state file to resume from (optional)
        game: Game type ('chess' or 'tictactoe', default: tictactoe)

    Returns:
        Output directory path
    """
    log(f"Running {num_replicas} replicas with {model} (game: {game})")

    if game_config is None:
        game_config = {}

    # Update .env with MODEL and MAX_STEPS
    env_path = Path(".env")
    if env_path.exists():
        env_content = env_path.read_text()
        env_lines = env_content.split("\n")

        # Update MODEL and MAX_STEPS
        updated = set()
        for i, line in enumerate(env_lines):
            if line.startswith("MODEL="):
                env_lines[i] = f"MODEL={model}"
                updated.add("MODEL")
            elif line.startswith("MAX_STEPS="):
                env_lines[i] = f"MAX_STEPS={max_steps}"
                updated.add("MAX_STEPS")

        # Add if not present
        if "MODEL" not in updated:
            env_lines.append(f"MODEL={model}")
        if "MAX_STEPS" not in updated:
            env_lines.append(f"MAX_STEPS={max_steps}")

        env_path.write_text("\n".join(env_lines))

    # Record output directories before running
    # output_base = Path("rh_gpt5_tests")
    output_base = Path(base_dir)
    existing_dirs = set(output_base.iterdir()) if output_base.exists() else set()

    cmd = [
        "bash",
        "run-parallel.sh",
        "-n",
        str(num_replicas),
        "-m",
        model,
        "-t",
        image_tag,
        "--system-prompt",
        system_prompt,
        "--user-prompt",
        user_prompt,
        "-s",  # Save state
        "--base-dir",
        base_dir,
        "--game",
        game,
    ]

    # Add initial state file if provided
    if initial_state:
        # Validate that the file exists
        initial_state_path = Path(initial_state)
        if not initial_state_path.exists():
            raise ValueError(f"Initial state file not found: {initial_state}")
        cmd.extend(["-i", initial_state])
        log(f"Resuming from state file: {initial_state}", "INFO")

    # Add game config parameters
    if "win_value" in game_config:
        cmd.extend(["--win-value", str(game_config["win_value"])])
    if "draw_value" in game_config:
        cmd.extend(["--draw-value", str(game_config["draw_value"])])
    if "loss_value" in game_config:
        cmd.extend(["--loss-value", str(game_config["loss_value"])])
    if "num_practice_games" in game_config:
        cmd.extend(["--num-practice-games", str(game_config["num_practice_games"])])
    if "num_official_games" in game_config:
        cmd.extend(["--num-official-games", str(game_config["num_official_games"])])
    if "hint_penalty" in game_config:
        cmd.extend(["--hint-penalty", str(game_config["hint_penalty"])])

    # Start the command in the background
    log(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Wait for output directory to be created
    output_dir = None
    max_wait = 30  # Wait up to 30 seconds for directory to appear
    for _ in range(max_wait):
        time.sleep(1)
        if output_base.exists():
            new_dirs = set(output_base.iterdir()) - existing_dirs
            if new_dirs:
                output_dir = max(new_dirs, key=lambda x: x.stat().st_mtime)
                log(f"Found output directory: {output_dir}", "SUCCESS")

                # Save experiment config immediately
                if experiment_name and full_config is not None:
                    log(
                        "Saving experiment config (while runs are in progress)...",
                        "INFO",
                    )
                    save_experiment_config(
                        str(output_dir),
                        experiment_name,
                        system_prompt,
                        user_prompt,
                        description,
                        full_config,
                        initial_state,
                    )
                break

    if output_dir is None:
        # Kill the process if we couldn't find the directory
        process.kill()
        raise ValueError("Could not find output directory after 30 seconds")

    # Now wait for the process to complete
    stdout, stderr = process.communicate()

    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    return str(output_dir)


def grade_rollouts(output_dir: str, model: str = "google/gemini-2.5-flash", game: str = "tictactoe") -> dict:
    """
    DISABLED: Grading is currently disabled due to WSL path issues.

    To grade manually, run:
        python judge/grade_rollouts_v2.py --output_dir=<output_dir> --game=<game>

    Returns:
        Empty summary dictionary
    """
    log(f"Grading disabled for {output_dir}", "WARNING")
    log(
        f"To grade manually, run: python judge/grade_rollouts_v2.py --output_dir={output_dir} --game={game}",
        "INFO",
    )
    return {}


def substitute_template_variables(text: str, config: dict) -> str:
    """
    Substitute template variables in text with values from config.

    Supports variables from docker_config and game_config:
    - {board_size}, {win_length}, {max_depth}, {multi_game}, {hints_enabled}, {good_engine}
    - {win_value}, {draw_value}, {loss_value}
    - {num_practice_games}, {num_official_games}, {hint_penalty}
    - {practice_games_lost}, {practice_game_win_rate}
    - {win_value_doubled} (calculated as win_value * 2)
    """
    docker_config = config.get("docker_config", {})
    game_config = config.get("game_config", {})

    # Build substitution dictionary
    substitutions = {
        # Docker config values
        "board_size": docker_config.get("board_size", 5),
        "win_length": docker_config.get("win_length", 5),
        "max_depth": docker_config.get("max_depth", 0),
        "multi_game": docker_config.get("multi_game", 0),
        "hints_enabled": docker_config.get("hints_enabled", 1),
        "good_engine": docker_config.get("good_engine", 1),
        # Game config values
        "win_value": game_config.get("win_value", 10),
        "draw_value": game_config.get("draw_value", 0),
        "loss_value": game_config.get("loss_value", 0),
        "num_practice_games": game_config.get("num_practice_games", 1),
        "num_official_games": game_config.get("num_official_games", 1),
        "hint_penalty": game_config.get("hint_penalty", 10),
        "practice_games_lost": game_config.get("practice_games_lost", 0),
        "practice_game_win_rate": game_config.get("practice_game_win_rate", 0),
    }

    # Add calculated values
    substitutions["win_value_doubled"] = substitutions["win_value"] * 2

    # Perform substitution
    result = text
    for key, value in substitutions.items():
        result = result.replace(f"{{{key}}}", str(value))

    return result


def save_experiment_config(
    output_dir: str,
    experiment_name: str,
    system_prompt: str,
    user_prompt: str,
    description: str,
    config: dict,
    initial_state: str = "",
) -> None:
    """Save experiment configuration to output directory"""
    exp_config = {
        "experiment": {
            "name": experiment_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "description": description,
            "timestamp": datetime.now().isoformat(),
        },
        "docker_config": config.get("docker_config", {}),
        "run_config": config.get("run_config", {}),
        "game_config": config.get("game_config", {}),
    }
    
    # Add initial_state if provided
    if initial_state:
        exp_config["experiment"]["initial_state_file"] = initial_state

    config_file = Path(output_dir) / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(exp_config, f, indent=2)

    log(f"Saved experiment config to: {config_file}", "SUCCESS")


def run_single_experiment(
    experiment: dict, config: dict, docker_tag: str = "ctfish-experiments:latest", base_dir: str = "continued_reasoning_results", initial_state_override: str = "", game: str = "tictactoe"
) -> dict:
    """Run a single experiment using environment variables (no rebuild needed)"""
    exp_name = experiment["name"]
    system_prompt = experiment.get("system_prompt", "")
    user_prompt = experiment.get("user_prompt", "")
    description = experiment.get("description", "")
    
    # Get initial_state from experiment config or override
    initial_state = initial_state_override or experiment.get("initial_state", "") or config.get("run_config", {}).get("initial_state", "")

    # Merge experiment-specific game_config for template substitution
    merged_config = config.copy()
    if "game_config" in experiment:
        merged_config["game_config"] = {**config.get("game_config", {}), **experiment["game_config"]}

    # Substitute template variables from merged config
    system_prompt = substitute_template_variables(system_prompt, merged_config)
    user_prompt = substitute_template_variables(user_prompt, merged_config)

    log("=" * 80, "INFO")
    log(f"STARTING EXPERIMENT: {exp_name}", "SUCCESS")
    log(f"Description: {description}", "INFO")
    log(
        f"User Prompt: {user_prompt[:100]}..."
        if len(user_prompt) > 100
        else f"User Prompt: {user_prompt}",
        "INFO",
    )
    if initial_state:
        log(f"Initial State: {initial_state}", "INFO")
    log("=" * 80, "INFO")

    start_time = time.time()

    try:
        # Run parallel experiments with prompts passed as environment variables
        # Config is saved automatically as soon as output directory is created
        log("Step 1: Running parallel experiments", "INFO")
        run_config = config.get("run_config", {})
        game_config = config.get("game_config", {})
        
        # Merge experiment-specific game_config (overrides global game_config)
        if "game_config" in experiment:
            game_config = {**game_config, **experiment["game_config"]}
        
        # Log hint_penalty if set
        if "hint_penalty" in game_config:
            log(f"Hint penalty: {game_config['hint_penalty']} points", "INFO")
        
        output_dir = run_parallel_experiments(
            image_tag=docker_tag,
            num_replicas=run_config.get("num_replicas", 10),
            model=run_config.get("model", "openai/gpt-oss-20b"),
            max_steps=run_config.get("max_steps", 999999),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            game_config=game_config,
            experiment_name=exp_name,
            description=description,
            full_config=config,
            base_dir=base_dir,
            initial_state=initial_state,
            game=game,
        )

        # Grading is disabled - will print commands at the end
        summary = {}

        elapsed = time.time() - start_time

        result = {
            "name": exp_name,
            "status": "success",
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
            "summary": summary,
            "game": game,
        }

        log("=" * 80, "INFO")
        log(f"EXPERIMENT COMPLETE: {exp_name}", "SUCCESS")
        log(f"Output directory: {output_dir}", "INFO")
        log(f"Elapsed time: {elapsed / 60:.1f} minutes", "INFO")
        log("=" * 80, "INFO")

        return result

    except Exception as e:
        log(f"EXPERIMENT FAILED: {exp_name}", "ERROR")
        log(f"Error: {e}", "ERROR")

        # Print full traceback for debugging
        import traceback

        traceback.print_exc()

        return {
            "name": exp_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - start_time,
        }


def run_experiments(
    config_file: str = "experiment_config.json",
    experiments: list[str] | None = None,
    skip_build: bool = False,
    base_dir: str = "continued_reasoning_results",
    initial_state: str = "",
    game: str = "tictactoe",
):
    """
    Run multiple prompt experiments

    Args:
        config_file: Path to experiment configuration JSON
        experiments: List of experiment names to run (runs all if None)
        skip_build: Skip Docker build step (useful for testing)
        base_dir: Base directory for storing experiment outputs (default: continued_reasoning_results)
        initial_state: Path to state file to resume from (optional, overrides config)
        game: Game type ('chess' or 'tictactoe', default: tictactoe)
    """
    log("=" * 80, "SUCCESS")
    log("STARTING PROMPT EXPERIMENTS", "SUCCESS")
    log("=" * 80, "SUCCESS")

    # Load config
    config_path = Path(config_file)
    if not config_path.exists():
        log(f"Config file not found: {config_file}", "ERROR")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    all_experiments = config.get("experiments", [])

    # Filter experiments if specified
    if experiments:
        all_experiments = [exp for exp in all_experiments if exp["name"] in experiments]
        if not all_experiments:
            log(f"No matching experiments found for: {experiments}", "ERROR")
            sys.exit(1)

    log(f"Running {len(all_experiments)} experiments", "INFO")
    for exp in all_experiments:
        log(f"  - {exp['name']}: {exp.get('description', '')}", "INFO")

    # Build Docker image ONCE for all experiments
    docker_tag = "ctfish-experiments:latest"
    docker_config = config.get("docker_config", {})

    if not skip_build:
        log("\n" + "=" * 80, "INFO")
        log(f"BUILDING DOCKER IMAGE (once for all experiments, game: {game})", "INFO")
        log("=" * 80, "INFO")
        try:
            build_docker_image(
                tag=docker_tag,
                game=game,
                board_size=docker_config.get("board_size", 5),
                win_length=docker_config.get("win_length", 5),
                max_depth=docker_config.get("max_depth", 0),
                multi_game=docker_config.get("multi_game", 0),
                hints_enabled=docker_config.get("hints_enabled", 1),
                good_engine=docker_config.get("good_engine", 1),
                hint_ablation=docker_config.get("hint_ablation", 0),
                chess_max_depth=docker_config.get("chess_max_depth", 15),
                chess_analysis_time=docker_config.get("chess_analysis_time", 7.5),
                chess_move_time=docker_config.get("chess_move_time", 1.0),
                chess_hint_ablation=docker_config.get("chess_hint_ablation", 0),
            )
            log("Docker image ready for all experiments!", "SUCCESS")
        except Exception as e:
            log(f"Docker build failed: {e}", "ERROR")
            log("Cannot continue without Docker image", "ERROR")
            sys.exit(1)
    else:
        log(f"\nSkipping Docker build, using existing image: {docker_tag}", "WARNING")

    # Run all experiments in parallel using threading
    results_queue = queue.Queue()
    threads = []

    log("\n" + "=" * 80, "INFO")
    log(f"LAUNCHING {len(all_experiments)} EXPERIMENTS IN PARALLEL", "SUCCESS")
    log("=" * 80, "INFO")

    def run_experiment_thread(experiment, config, docker_tag, exp_num, base_dir, initial_state_override, game):
        """Run a single experiment in a thread"""
        log(f"[Experiment {exp_num}] Starting: {experiment['name']}", "INFO")
        result = run_single_experiment(experiment, config, docker_tag=docker_tag, base_dir=base_dir, initial_state_override=initial_state_override, game=game)
        results_queue.put((exp_num, result))
        log(f"[Experiment {exp_num}] Completed: {experiment['name']}", "SUCCESS")

    # Launch all experiment threads
    for i, experiment in enumerate(all_experiments, 1):
        thread = threading.Thread(
            target=run_experiment_thread, args=(experiment, config, docker_tag, i, base_dir, initial_state, game)
        )
        thread.start()
        threads.append(thread)
        # Small delay between thread launches to avoid race conditions
        time.sleep(1)

    log(f"All {len(threads)} experiments launched! Waiting for completion...", "INFO")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results in order
    results_dict = {}
    while not results_queue.empty():
        exp_num, result = results_queue.get()
        results_dict[exp_num] = result

    results = [results_dict[i] for i in sorted(results_dict.keys())]

    # Final summary
    log("\n" + "=" * 80, "SUCCESS")
    log("ALL EXPERIMENTS COMPLETE", "SUCCESS")
    log("=" * 80, "SUCCESS")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    log(
        f"Successful: {successful}/{len(results)}", "SUCCESS" if failed == 0 else "INFO"
    )
    if failed > 0:
        log(f"Failed: {failed}/{len(results)}", "ERROR")

    log("\nResults summary:", "INFO")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        log(f"{status_icon} {result['name']}", "INFO")
        if result["status"] == "success":
            log(f"    Output: {result.get('output_dir', 'N/A')}", "INFO")
        else:
            log(f"    Error: {result.get('error', 'Unknown error')}", "ERROR")

    # Print grading commands for successful experiments as one command
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        log("\n" + "=" * 80, "INFO")
        log("TO GRADE EXPERIMENTS, COPY AND RUN THIS COMMAND:", "SUCCESS")
        log("=" * 80, "INFO")

        # Build single command with && \
        commands = []
        for result in successful_results:
            game_param = result.get("game", "tictactoe")
            commands.append(
                f"python judge/grade_rollouts_v2.py --output_dir={result['output_dir']} --game={game_param}"
            )

        combined_command = " && \\\n".join(commands)
        print(combined_command)

        log("=" * 80 + "\n", "INFO")


def main(
    config: str = "experiment_config.json",
    experiments: str | None = None,
    skip_build: bool = False,
    base_dir: str = "continued_reasoning_results",
    initial_state: str = "",
    game: str = "tictactoe",
):
    """
    Run automated prompt experiments

    Args:
        config: Path to experiment configuration JSON
        experiments: Comma-separated list of experiment names to run (runs all if not specified)
        skip_build: Skip Docker build step (useful for testing)
        base_dir: Base directory for storing experiment outputs (default: continued_reasoning_results)
        initial_state: Path to state file to resume from (optional, overrides config)
        game: Game type ('chess' or 'tictactoe', default: tictactoe)

    Examples:
        # Run all experiments in config
        python run_prompt_experiments.py
        
        # Run with chess
        python run_prompt_experiments.py --game=chess

        # Run specific experiments
        python run_prompt_experiments.py --experiments=baseline,powerless

        # Skip Docker build (reuse existing images)
        python run_prompt_experiments.py --skip-build
        
        # Use custom base directory
        python run_prompt_experiments.py --base-dir=my_experiments
        
        # Resume from a state file
        python run_prompt_experiments.py --initial-state=states/ctfish-0003.json
    """
    print(f"Running experiments: {experiments}")
    if not isinstance(experiments, tuple):
        exp_list = experiments.split(",") if experiments else None
    else:
        exp_list = experiments
    run_experiments(config, exp_list, skip_build, base_dir, initial_state, game)


if __name__ == "__main__":
    fire.Fire(main)
