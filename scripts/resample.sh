#!/bin/bash
#
# Resample from each step in a run folder.
#
# For each step-N/ folder in the run, calls resume.sh to generate
# multiple continuations from that checkpoint.
#
# Usage:
#   ./scripts/resample.sh <run-folder> [options]
#
# Options:
#   --count N     Run N parallel rollouts per step (default: 10)
#   --local       Use local Docker image instead of Dockerhub
#   --step N      Only resample from step N (default: all steps)
#   --max-parallel M  Max steps to process in parallel (default: 1)
#
# Output structure:
#   <run-folder>/step-0/<timestamp>/run-1/rollout.log
#   <run-folder>/step-0/<timestamp>/run-2/rollout.log
#   <run-folder>/step-1/<timestamp>/run-1/rollout.log
#   ...

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_SCRIPT="$SCRIPT_DIR/resume.sh"

print_header() {
  echo -e "${BOLD}╔════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}║${NC}  ${BLUE}Action Resampling${NC}"
  echo -e "${BOLD}╠════════════════════════════════════════════╣${NC}"
  echo -e "${BOLD}║${NC}  Run folder:    ${YELLOW}$RUN_FOLDER${NC}"
  echo -e "${BOLD}║${NC}  Steps found:   ${YELLOW}$NUM_STEPS${NC}"
  echo -e "${BOLD}║${NC}  Count/step:    ${YELLOW}$COUNT${NC}"
  echo -e "${BOLD}║${NC}  Max parallel:  ${YELLOW}$MAX_PARALLEL${NC}"
  echo -e "${BOLD}╚════════════════════════════════════════════╝${NC}"
}

# Parse arguments
RUN_FOLDER=""
COUNT=10
MAX_PARALLEL=1
SINGLE_STEP=""
LOCAL_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --count)
      COUNT=$2
      shift 2
      ;;
    --local)
      LOCAL_FLAG="--local"
      shift
      ;;
    --step)
      SINGLE_STEP=$2
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL=$2
      shift 2
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      if [[ -z "$RUN_FOLDER" ]]; then
        RUN_FOLDER="$1"
      else
        echo "Unexpected argument: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$RUN_FOLDER" ]]; then
  echo "Usage: $0 <run-folder> [--count N] [--local] [--step N] [--max-parallel M]"
  exit 1
fi

RUN_FOLDER=$(realpath "$RUN_FOLDER")

if [[ ! -d "$RUN_FOLDER" ]]; then
  echo -e "${RED}Error: Run folder '$RUN_FOLDER' does not exist.${NC}"
  exit 1
fi

# Find all step directories
if [[ -n "$SINGLE_STEP" ]]; then
  STEP_DIRS=("$RUN_FOLDER/step-$SINGLE_STEP")
  if [[ ! -d "${STEP_DIRS[0]}" ]]; then
    echo -e "${RED}Error: Step directory '${STEP_DIRS[0]}' does not exist.${NC}"
    exit 1
  fi
else
  STEP_DIRS=()
  while IFS= read -r dir; do
    STEP_DIRS+=("$dir")
  done < <(find "$RUN_FOLDER" -maxdepth 1 -type d -name "step-*" | sort -V)
  
  # Exclude the last step (terminal state after game ended)
  if [[ ${#STEP_DIRS[@]} -gt 1 ]]; then
    LAST_IDX=$((${#STEP_DIRS[@]} - 1))
    LAST_STEP="${STEP_DIRS[$LAST_IDX]}"
    STEP_DIRS=("${STEP_DIRS[@]:0:$LAST_IDX}")
    echo -e "${YELLOW}Excluding final step $(basename "$LAST_STEP") (terminal state)${NC}"
  fi
fi

NUM_STEPS=${#STEP_DIRS[@]}

if [[ $NUM_STEPS -eq 0 ]]; then
  echo -e "${RED}Error: No step-* directories found in '$RUN_FOLDER'.${NC}"
  exit 1
fi

print_header
echo ""

# Process steps
COMPLETED=0
FAILED=0

process_step() {
  local step_dir="$1"
  local step_name=$(basename "$step_dir")
  
  echo -e "${BLUE}[$step_name]${NC} Starting resample with $COUNT rollouts..."
  
  if "$RESUME_SCRIPT" "$step_dir" --count "$COUNT" $LOCAL_FLAG; then
    echo -e "${GREEN}[$step_name]${NC} ✓ Completed"
    return 0
  else
    echo -e "${RED}[$step_name]${NC} ✗ Failed"
    return 1
  fi
}

# Create temp dir for tracking results
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Process steps with parallelization
if [[ $MAX_PARALLEL -eq 1 ]]; then
  # Sequential processing
  for step_dir in "${STEP_DIRS[@]}"; do
    if process_step "$step_dir"; then
      ((COMPLETED++))
    else
      ((FAILED++))
    fi
  done
else
  # Parallel processing - launch jobs, track via temp files and PIDs
  PIDS=()
  
  for step_dir in "${STEP_DIRS[@]}"; do
    step_name=$(basename "$step_dir")
    echo -e "${BLUE}[$step_name]${NC} Starting resample with $COUNT rollouts..."
    
    (
      if "$RESUME_SCRIPT" "$step_dir" --count "$COUNT" $LOCAL_FLAG > /dev/null 2>&1; then
        echo -e "${GREEN}[$step_name]${NC} ✓ Completed"
        touch "$TEMP_DIR/$step_name.done"
      else
        echo -e "${RED}[$step_name]${NC} ✗ Failed"
        touch "$TEMP_DIR/$step_name.failed"
      fi
    ) &
    PIDS+=($!)
    
    # If we've hit max parallel, wait for one to finish before continuing
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
      # Check each PID and remove finished ones
      NEW_PIDS=()
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          NEW_PIDS+=("$pid")
        fi
      done
      PIDS=("${NEW_PIDS[@]}")
      
      # If still at max, sleep and check again
      if [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; then
        sleep 0.5
      fi
    done
  done
  
  # Wait for all remaining jobs
  wait
  
  # Count results from temp files
  COMPLETED=$(ls "$TEMP_DIR"/*.done 2>/dev/null | wc -l | tr -d ' ')
  FAILED=$(ls "$TEMP_DIR"/*.failed 2>/dev/null | wc -l | tr -d ' ')
fi

# Print summary
echo ""
echo -e "${BOLD}════════════════════════════════════════════${NC}"
echo -e "${BOLD}Summary:${NC}"
echo -e "  ${GREEN}✓ Completed:${NC} $COMPLETED"
echo -e "  ${RED}✗ Failed:${NC} $FAILED"
echo -e "  Total steps: $NUM_STEPS"
echo -e "${BOLD}════════════════════════════════════════════${NC}"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi

