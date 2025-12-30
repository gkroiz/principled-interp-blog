#!/usr/bin/env python3
"""
Automated script to run multiple prompt experiments.

This script automates the loop of:
1. Prepare Docker image (pull from Docker Hub or use local)
2. Run parallel experiments (generates YAML config and launches containers)
3. Monitor execution and save results

Usage:
    python run_experiments.py --config experiment_config.json
    python run_experiments.py --config experiment_config.json --experiments experiment1 experiment2
    python run_experiments.py --config experiment_config.json --initial-state states/ctfish-0003.json

Container Sources:
    Configure in docker_config section of your config file:

    1. Pull from Docker Hub:
        "docker_config": {
            "container_source": "dockerhub",
            "container_image": "gkroiz/agent-interp-envs:tictactoe-latest"
        }

    2. Use existing local image (default):
        "docker_config": {
            "container_source": "local",
            "container_image": "my-image:latest"
        }

State Resumption:
    The --initial-state flag allows resuming experiments from a saved state file.
    This can be specified:
    1. Via command line: --initial-state=path/to/state.json (overrides config)
    2. In default_config: {"default_config": {"initial_state": "path/to/state.json"}}
    3. Per experiment: {"experiments": [{"name": "exp1", "initial_state": "path/to/state.json"}]}

    Priority: CLI flag > per-experiment config > default_config
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
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking priority.

    Args:
        base: Base dictionary (e.g., default_config)
        override: Override dictionary (e.g., experiment config)

    Returns:
        Merged dictionary where override values take priority

    Example:
        base = {"agent": {"model": "gpt-4", "max_steps": 30}, "foo": "bar"}
        override = {"agent": {"model": "claude"}, "baz": "qux"}
        result = {"agent": {"model": "claude", "max_steps": 30}, "foo": "bar", "baz": "qux"}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override takes priority for non-dict values or new keys
            result[key] = value

    return result


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



def pull_docker_image(image: str, max_retries: int = 3) -> None:
    """Pull Docker image from registry

    Args:
        image: Docker image to pull (e.g., 'gkroiz/agent-interp-envs:latest')
        max_retries: Number of retries for transient errors
    """
    log(f"Pulling Docker image: {image}")

    cmd = ["docker", "pull", image]

    # Retry logic for transient Docker errors
    for attempt in range(1, max_retries + 1):
        try:
            run_command(cmd)
            log(f"Docker image pulled successfully: {image}", "SUCCESS")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                log(
                    f"Pull failed on attempt {attempt}/{max_retries}, retrying in 5 seconds...",
                    "WARNING",
                )
                time.sleep(5)
                continue
            # Re-raise if we've exhausted retries
            raise


def verify_local_docker_image(image: str) -> bool:
    """Verify that a Docker image exists locally

    Args:
        image: Docker image name/tag to verify

    Returns:
        True if image exists locally, False otherwise
    """
    log(f"Verifying local Docker image: {image}")

    try:
        result = subprocess.run(
            ["docker", "images", "-q", image],
            capture_output=True,
            text=True,
            check=True,
        )
        exists = bool(result.stdout.strip())
        if exists:
            log(f"Docker image found locally: {image}", "SUCCESS")
        else:
            log(f"Docker image not found locally: {image}", "WARNING")
        return exists
    except subprocess.CalledProcessError:
        return False


def generate_yaml_config(
    experiment: dict,
    default_config: dict,
    game_config: dict,
    output_path: Path,
) -> None:
    """
    Generate YAML config file for agent-interp-envs containers

    Args:
        experiment: Experiment configuration (will override defaults)
        default_config: Default configuration (base values)
        game_config: Legacy game configuration (fallback for task)
        output_path: Path to write YAML config
    """
    # Merge default_config with experiment config (experiment takes priority)
    merged = deep_merge(default_config, experiment)

    # Get task configuration (from merged config or fall back to game_config)
    task_config = merged.get("task", game_config.copy())

    # Build config structure matching agent-interp-envs format
    agent_config = merged.get("agent", {})
    config = {
        "environment": merged.get("environment", "tictactoe"),
        "agent": {
            "model": agent_config.get("model", "openai/gpt-4"),
            "max_steps": agent_config.get("max_steps", 30),
        },
        "task": task_config,
        "prompts": merged.get("prompts", {
            "system_prompt": "",
            "user_prompt": "",
        }),
    }

    # Write YAML config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def launch_container(
    image_tag: str,
    output_file: Path,
    env_vars: dict,
    config_file: Path = None,
    state_dir: Path = None,
    initial_state: str = "",
) -> subprocess.Popen:
    """
    Launch a single Docker container

    Args:
        image_tag: Docker image to run
        output_file: Path to write stdout/stderr
        env_vars: Dictionary of environment variables to pass to container
        config_file: Path to YAML config file to mount at /opt/config.yaml (optional)
        state_dir: Directory to mount for state saving (optional)
        initial_state: Path to initial state file for resumption (optional)

    Returns:
        Popen process object
    """
    # Build docker command
    docker_args = [
        "docker", "run",
        "--rm",
        # Note: Don't use --user flag for agent-interp-envs containers
        # They need root access to install packages in entry_point.py
    ]

    # Read .env file and pass as environment variables
    env_path = Path(".env")
    if env_path.exists():
        # Mount the file for backward compatibility
        docker_args.extend(["-v", f"{env_path.absolute()}:/app/.env"])

        # Also read and pass as env vars (required for agent-interp-envs)
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#') and '=' in line:
                    # Split on first = only
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value:  # Only add if value is not empty
                        docker_args.extend(["-e", f"{key}={value}"])

    # Mount config file if provided (for agent-interp-envs containers)
    if config_file and config_file.exists():
        docker_args.extend(["-v", f"{config_file.absolute()}:/opt/config.yaml:ro"])

    # Add all environment variables
    for key, value in env_vars.items():
        if value is not None:
            docker_args.extend(["-e", f"{key}={value}"])

    # Handle state directory and initial state
    if state_dir:
        state_dir.mkdir(parents=True, exist_ok=True)
        docker_args.extend([
            "-v", f"{state_dir.absolute()}:/tmp/output"
        ])

        if initial_state:
            # Mount initial state file
            initial_state_path = Path(initial_state).absolute()
            initial_state_dir = initial_state_path.parent
            initial_state_basename = initial_state_path.name

            docker_args.extend([
                "-v", f"{initial_state_dir}:/tmp/initial-state:ro",
                "-e", f"RESUME_FROM=/tmp/initial-state/{initial_state_basename}"
            ])

    # Add image tag
    docker_args.append(image_tag)

    # Open output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    outfile = open(output_file, "w")

    # Launch container
    process = subprocess.Popen(
        docker_args,
        stdout=outfile,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Store output file handle for cleanup
    process._output_file = outfile

    return process


def run_experiment_replicas(
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
    base_dir: str = "results",
    initial_state: str = "",
    game: str = "tictactoe",
    experiment: dict = None,
    default_config: dict = None,
) -> str:
    """
    Run multiple replicas of a single experiment in parallel

    Args:
        image_tag: Docker image tag to run
        num_replicas: Number of parallel runs
        model: Model name
        max_steps: Maximum steps per run
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        game_config: Game configuration (win_value, draw_value, etc.)
        experiment_name: Name of the experiment (for saving config)
        description: Description of the experiment (for saving config)
        full_config: Full experiment configuration (for saving config)
        base_dir: Base directory for storing experiment outputs (default: results)
        initial_state: Path to state file to resume from (optional)
        game: Game type ('chess' or 'tictactoe', default: tictactoe)
        experiment: Full experiment dict for YAML config generation (optional)
        default_config: Default configuration for YAML config generation (optional)

    Returns:
        Output directory path
    """
    log(f"Running {num_replicas} replicas with {model} (game: {game})")

    if game_config is None:
        game_config = {}

    # Validate initial state file if provided
    if initial_state:
        initial_state_path = Path(initial_state)
        if not initial_state_path.exists():
            raise ValueError(f"Initial state file not found: {initial_state}")
        log(f"Resuming from state file: {initial_state}", "INFO")

    # Set up output directory and config files
    output_dir, config_file, model_name, timestamp = setup_experiment_output(
        model=model,
        base_dir=base_dir,
        experiment_name=experiment_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        description=description,
        full_config=full_config,
        initial_state=initial_state,
        experiment=experiment,
        default_config=default_config,
        game_config=game_config,
    )

    # Launch containers
    start_time = time.time()
    processes = []

    log(f"\nLaunching {num_replicas} containers...", "INFO")
    for i in range(1, num_replicas + 1):
        run_id = f"{model_name}-{timestamp}-run{i}"
        output_file = output_dir / f"run-{run_id}.txt"
        state_dir = output_dir / f"state-run{i}"

        log(f"[{i}/{num_replicas}] Launching container {i}...", "INFO")
        log(f"    Output: {output_file}", "INFO")
        if initial_state or True:  # Always enable state saving
            log(f"    State: {state_dir}", "INFO")

        process = launch_container(
            image_tag=image_tag,
            output_file=output_file,
            env_vars={},  # API keys come from .env, config comes from YAML
            config_file=config_file,  # Mount YAML config for agent-interp-envs
            state_dir=state_dir,
            initial_state=initial_state,
        )
        processes.append(process)

        # Small delay between launches to avoid race conditions
        time.sleep(0.5)

    # Monitor container execution
    elapsed = monitor_containers(processes, start_time)

    # Create experiment summary
    create_experiment_summary(
        output_dir=output_dir,
        model=model,
        model_name=model_name,
        timestamp=timestamp,
        num_replicas=num_replicas,
        max_steps=max_steps,
        image_tag=image_tag,
        initial_state=initial_state,
        game=game,
        start_time=start_time,
        elapsed=elapsed,
    )

    return str(output_dir)


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
        "default_config": config.get("default_config", {}),
        "game_config": config.get("game_config", {}),
    }

    # Add initial_state if provided
    if initial_state:
        exp_config["experiment"]["initial_state_file"] = initial_state

    config_file = Path(output_dir) / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(exp_config, f, indent=2)

    log(f"Saved experiment config to: {config_file}", "SUCCESS")


def setup_experiment_output(
    model: str,
    base_dir: str,
    experiment_name: str,
    system_prompt: str,
    user_prompt: str,
    description: str,
    full_config: dict,
    initial_state: str,
    experiment: dict,
    default_config: dict,
    game_config: dict,
) -> tuple[Path, Path | None, str, str]:
    """
    Set up output directory and configuration files for an experiment

    Args:
        model: Model name
        base_dir: Base directory for storing experiment outputs
        experiment_name: Name of the experiment
        system_prompt: System prompt
        user_prompt: User prompt
        description: Experiment description
        full_config: Full experiment configuration
        initial_state: Path to state file to resume from (optional)
        experiment: Full experiment dict for YAML config generation
        default_config: Default configuration for YAML config generation
        game_config: Game configuration

    Returns:
        Tuple of (output_dir, config_file, model_name, timestamp)
    """
    # Create output directory with timestamp
    model_name = model.replace("/", "-").replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(base_dir) / f"{model_name}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Output directory: {output_dir}", "SUCCESS")

    # Save experiment config immediately
    if experiment_name and full_config is not None:
        log("Saving experiment config...", "INFO")
        save_experiment_config(
            str(output_dir),
            experiment_name,
            system_prompt,
            user_prompt,
            description,
            full_config,
            initial_state,
        )

    # Generate YAML config file for agent-interp-envs containers
    config_file = None
    if experiment is not None and default_config is not None:
        config_file = output_dir / "container_config.yaml"
        log(f"Generating YAML config: {config_file}", "INFO")

        # Prepare experiment dict with prompts
        experiment_with_prompts = experiment.copy()
        if "prompts" not in experiment_with_prompts:
            experiment_with_prompts["prompts"] = {}

        # Set prompts (template variables will be substituted by agent-interp-envs)
        experiment_with_prompts["prompts"]["system_prompt"] = system_prompt
        experiment_with_prompts["prompts"]["user_prompt"] = user_prompt

        generate_yaml_config(
            experiment=experiment_with_prompts,
            default_config=default_config,
            game_config=game_config,
            output_path=config_file,
        )

    return output_dir, config_file, model_name, timestamp


def monitor_containers(processes: list[subprocess.Popen], start_time: float) -> int:
    """
    Monitor running containers and print status updates

    Args:
        processes: List of container processes to monitor
        start_time: Experiment start time

    Returns:
        Total elapsed time in seconds
    """
    log(f"\n✅ All {len(processes)} containers started!", "SUCCESS")
    log("Monitoring progress...\n", "INFO")

    last_status_time = time.time()
    status_interval = 10  # Print status every 10 seconds

    while True:
        # Check if all processes are done
        running = sum(1 for p in processes if p.poll() is None)
        completed = len(processes) - running

        # Print status update
        current_time = time.time()
        if current_time - last_status_time >= status_interval:
            elapsed = int(current_time - start_time)
            elapsed_fmt = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
            progress = int(completed * 100 / len(processes))

            log(f"Status: {running} running, {completed}/{len(processes)} completed ({progress}%) - Elapsed: {elapsed_fmt}", "INFO")
            last_status_time = current_time

        if running == 0:
            break

        time.sleep(2)

    # Clean up output file handles
    for process in processes:
        if hasattr(process, '_output_file'):
            process._output_file.close()

    elapsed = int(time.time() - start_time)
    elapsed_fmt = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"

    log(f"\n✨ All runs completed!", "SUCCESS")
    log(f"Total time: {elapsed_fmt}", "INFO")

    return elapsed


def create_experiment_summary(
    output_dir: Path,
    model: str,
    model_name: str,
    timestamp: str,
    num_replicas: int,
    max_steps: int,
    image_tag: str,
    initial_state: str,
    game: str,
    start_time: float,
    elapsed: int,
) -> None:
    """
    Create experiment summary JSON file

    Args:
        output_dir: Output directory path
        model: Model name
        model_name: Sanitized model name
        timestamp: Experiment timestamp
        num_replicas: Number of replicas run
        max_steps: Maximum steps per run
        image_tag: Docker image tag used
        initial_state: Path to initial state file (optional)
        game: Game type
        start_time: Experiment start time
        elapsed: Total elapsed time in seconds
    """
    elapsed_fmt = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"

    log(f"Output files: {output_dir}/run-*.txt", "INFO")
    log(f"State directories: {output_dir}/state-run*/", "INFO")

    # Create summary JSON
    summary_file = output_dir / "experiment_summary.json"
    summary = {
        "experiment": {
            "model": model,
            "model_name": model_name,
            "num_replicas": num_replicas,
            "max_steps": max_steps,
            "docker_image_tag": image_tag,
            "save_state_enabled": True,
            "initial_state_file": initial_state if initial_state else None,
            "game": game,
        },
        "execution": {
            "timestamp": timestamp,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_formatted": elapsed_fmt,
        },
        "results": {
            "completed": num_replicas,
            "running": 0,
        },
        "outputs": {
            "output_directory": str(output_dir),
            "log_files_pattern": f"run-{model_name}-{timestamp}-run*.txt",
            "state_directories_pattern": "state-run*/",
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Experiment summary: {summary_file}", "INFO")


def execute_single_experiment(
    experiment: dict,
    exp_num: int,
    docker_tag: str,
    config: dict,
    initial_state: str,
    base_dir: str,
    game: str,
    results_queue: queue.Queue,
) -> None:
    """
    Execute a single experiment (runs in a thread)

    Args:
        experiment: Experiment configuration
        exp_num: Experiment number (for logging)
        docker_tag: Docker image tag to use
        config: Full configuration dict
        initial_state: Path to state file to resume from (optional)
        base_dir: Base directory for storing experiment outputs
        game: Game type ('chess' or 'tictactoe')
        results_queue: Queue to put results in
    """
    exp_name = experiment["name"]
    log(f"[Experiment {exp_num}] Starting: {exp_name}", "INFO")

    start_time = time.time()

    try:
        # Merge default_config with experiment config (experiment takes priority)
        default_config = config.get("default_config", {})
        merged_config = deep_merge(default_config, experiment)

        # Get prompts from merged config (supports both experiment.prompts and top-level for backward compatibility)
        prompts = merged_config.get("prompts", {})
        system_prompt = prompts.get("system_prompt", merged_config.get("system_prompt", ""))
        user_prompt = prompts.get("user_prompt", merged_config.get("user_prompt", ""))
        description = merged_config.get("description", "")

        # Get initial_state (CLI override > experiment config > default_config)
        exp_initial_state = initial_state or merged_config.get("initial_state", "")

        log("=" * 80, "INFO")
        log(f"EXPERIMENT: {exp_name}", "SUCCESS")
        log(f"Description: {description}", "INFO")
        log(
            f"User Prompt: {user_prompt[:100]}..."
            if len(user_prompt) > 100
            else f"User Prompt: {user_prompt}",
            "INFO",
        )
        if exp_initial_state:
            log(f"Initial State: {exp_initial_state}", "INFO")
        log("=" * 80, "INFO")

        # Get merged configs (experiment values override defaults)
        game_config = merged_config.get("game_config", config.get("game_config", {}))
        agent_config = merged_config.get("agent", {})

        # Log hint_penalty if set
        task_config = merged_config.get("task", {})
        if "hint_penalty" in task_config:
            log(f"Hint penalty: {task_config['hint_penalty']} points", "INFO")
        elif "hint_penalty" in game_config:
            log(f"Hint penalty: {game_config['hint_penalty']} points", "INFO")

        # Run experiment replicas
        output_dir = run_experiment_replicas(
            image_tag=docker_tag,
            num_replicas=merged_config.get("num_replicas", 10),
            model=agent_config.get("model", "openai/gpt-4"),
            max_steps=agent_config.get("max_steps", 30),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            game_config=game_config,
            experiment_name=exp_name,
            description=description,
            full_config=config,
            base_dir=base_dir,
            initial_state=exp_initial_state,
            game=game,
            experiment=experiment,
            default_config=default_config,
        )

        elapsed = time.time() - start_time

        result = {
            "name": exp_name,
            "status": "success",
            "output_dir": output_dir,
            "elapsed_seconds": elapsed,
            "summary": {},
            "game": game,
        }

        log("=" * 80, "INFO")
        log(f"EXPERIMENT COMPLETE: {exp_name}", "SUCCESS")
        log(f"Output directory: {output_dir}", "INFO")
        log(f"Elapsed time: {elapsed / 60:.1f} minutes", "INFO")
        log("=" * 80, "INFO")

    except Exception as e:
        log(f"EXPERIMENT FAILED: {exp_name}", "ERROR")
        log(f"Error: {e}", "ERROR")

        import traceback
        traceback.print_exc()

        result = {
            "name": exp_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - start_time,
        }

    results_queue.put((exp_num, result))
    log(f"[Experiment {exp_num}] Completed: {exp_name}", "SUCCESS")


def run_experiments(
    config_file: str = "experiment_config.json",
    experiments: list[str] | None = None,
    skip_build: bool = False,
    base_dir: str = "results",
    initial_state: str = "",
    game: str = "tictactoe",
):
    """
    Run multiple prompt experiments

    Args:
        config_file: Path to experiment configuration JSON
        experiments: List of experiment names to run (runs all if None)
        skip_build: Skip Docker build step (useful for testing)
        base_dir: Base directory for storing experiment outputs (default: results)
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

    # Determine container source and image to use
    docker_config = config.get("docker_config", {})
    container_source = docker_config.get("container_source", "local")  # "dockerhub" or "local" (default)

    # Determine docker_tag based on container source
    if container_source == "dockerhub":
        # Use image from Docker Hub
        docker_tag = docker_config.get("container_image", "gkroiz/agent-interp-envs:tictactoe-latest")
    elif container_source == "local":
        # Use user-specified local image (default)
        docker_tag = docker_config.get("container_image")
        if not docker_tag:
            log("Error: container_image must be specified when using container_source='local'", "ERROR")
            sys.exit(1)
    else:
        log(f"Error: Unknown container_source '{container_source}'. Must be 'dockerhub' or 'local'", "ERROR")
        sys.exit(1)

    if not skip_build:
        log("\n" + "=" * 80, "INFO")
        if container_source == "dockerhub":
            log(f"PULLING DOCKER IMAGE FROM DOCKER HUB", "INFO")
            log(f"Image: {docker_tag}", "INFO")
            log("=" * 80, "INFO")
            try:
                pull_docker_image(docker_tag)
                log("Docker image ready for all experiments!", "SUCCESS")
            except Exception as e:
                log(f"Docker pull failed: {e}", "ERROR")
                log("Cannot continue without Docker image", "ERROR")
                sys.exit(1)
        elif container_source == "local":
            log(f"USING LOCAL DOCKER IMAGE", "INFO")
            log(f"Image: {docker_tag}", "INFO")
            log("=" * 80, "INFO")
            try:
                if not verify_local_docker_image(docker_tag):
                    raise ValueError(f"Local Docker image not found: {docker_tag}")
                log("Docker image ready for all experiments!", "SUCCESS")
            except Exception as e:
                log(f"Docker image verification failed: {e}", "ERROR")
                log("Cannot continue without Docker image", "ERROR")
                sys.exit(1)
    else:
        log(f"\nSkipping container setup, using existing image: {docker_tag}", "WARNING")

    # Run all experiments in parallel using threading
    results_queue = queue.Queue()
    threads = []

    log("\n" + "=" * 80, "INFO")
    log(f"LAUNCHING {len(all_experiments)} EXPERIMENTS IN PARALLEL", "SUCCESS")
    log("=" * 80, "INFO")

    # Launch all experiment threads
    for i, experiment in enumerate(all_experiments, 1):
        thread = threading.Thread(
            target=execute_single_experiment,
            args=(experiment, i, docker_tag, config, initial_state, base_dir, game, results_queue)
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
                f"python judge/grade_rollouts.py --output_dir={result['output_dir']} --game={game_param}"
            )

        combined_command = " && \\\n".join(commands)
        print(combined_command)

        log("=" * 80 + "\n", "INFO")


def main(
    config: str = "experiment_config.json",
    experiments: str | None = None,
    skip_build: bool = False,
    base_dir: str = "results",
    initial_state: str = "",
    game: str = "tictactoe",
):
    """
    Run automated prompt experiments

    Args:
        config: Path to experiment configuration JSON
        experiments: Comma-separated list of experiment names to run (runs all if not specified)
        skip_build: Skip Docker build step (useful for testing)
        base_dir: Base directory for storing experiment outputs (default: results)
        initial_state: Path to state file to resume from (optional, overrides config)
        game: Game type ('chess' or 'tictactoe', default: tictactoe)
    """
    print(f"Running experiments: {experiments}")
    if not isinstance(experiments, tuple):
        exp_list = experiments.split(",") if experiments else None
    else:
        exp_list = experiments
    run_experiments(config, exp_list, skip_build, base_dir, initial_state, game)


if __name__ == "__main__":
    fire.Fire(main)
