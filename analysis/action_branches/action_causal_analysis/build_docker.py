"""Build Docker image with the correct configuration for causal analysis."""

import subprocess
import fire
import config


def build_docker_image(force_rebuild: bool = False):
    """
    Build Docker image matching the original experiment configuration.
    
    Args:
        force_rebuild: Force rebuild even if image exists (uses --no-cache)
    
    This builds a Docker image with:
    - Board size: 6x6
    - Win condition: 4-in-a-row
    - Multi-game mode (1 practice + 1 official game)
    - Expert opponent (good_engine=1)
    """
    print("=" * 60)
    print("BUILDING DOCKER IMAGE FOR CAUSAL ANALYSIS")
    print("=" * 60)
    print(f"Image tag: {config.DOCKER_IMAGE}")
    print(f"Board size: {config.DOCKER_BUILD_CONFIG['board_size']}x{config.DOCKER_BUILD_CONFIG['board_size']}")
    print(f"Win length: {config.DOCKER_BUILD_CONFIG['win_length']}")
    print(f"Multi-game mode: {config.DOCKER_BUILD_CONFIG['multi_game']}")
    print(f"Expert opponent: {config.DOCKER_BUILD_CONFIG['good_engine']}")
    print("=" * 60 + "\n")
    
    cmd = [
        "docker", "build",
        "-f", "Dockerfile.tictactoe",
        "-t", config.DOCKER_IMAGE,
        f"--build-arg=BOARD_SIZE={config.DOCKER_BUILD_CONFIG['board_size']}",
        f"--build-arg=WIN_LENGTH={config.DOCKER_BUILD_CONFIG['win_length']}",
        f"--build-arg=MAX_DEPTH={config.DOCKER_BUILD_CONFIG['max_depth']}",
        f"--build-arg=MULTI_GAME={config.DOCKER_BUILD_CONFIG['multi_game']}",
        f"--build-arg=GOOD_ENGINE={config.DOCKER_BUILD_CONFIG['good_engine']}",
    ]
    
    if force_rebuild:
        cmd.append("--no-cache")
    
    cmd.append(".")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Run from project root (parent directory)
    project_root = config.BASE_DIR.parent
    
    try:
        result = subprocess.run(cmd, cwd=str(project_root), check=True)
        print("\n" + "=" * 60)
        print("✓ Docker image built successfully!")
        print("=" * 60)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ Docker build failed!")
        print("=" * 60)
        raise


def main(force_rebuild: bool = False):
    """CLI entry point using Fire."""
    build_docker_image(force_rebuild=force_rebuild)


if __name__ == "__main__":
    fire.Fire(main)

