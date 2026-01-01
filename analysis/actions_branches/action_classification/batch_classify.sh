#!/bin/bash
# Batch classify conversation state files (PARALLELIZED)
# Only processes the final step directory's messages.json from each run directory

set -e

# Default values
INPUT_DIR=""
OUTPUT_DIR="./analysis/action_branches/action_classifcation/classifications"
MODEL="google/gemini-2.5-pro"
PARALLEL_JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CLASSIFIER_TYPE="standard"

# Print usage
usage() {
    echo "Usage: $0 --directory <input_dir> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  -d, --directory DIR        Input directory containing state-run* subdirectories"
    echo ""
    echo "Optional arguments:"
    echo "  -o, --output-dir DIR       Output directory for classifications (default: ./thought_anchors/classifications)"
    echo "  -m, --model MODEL          Model to use for classification (default: google/gemini-2.5-pro)"
    echo "  -j, --jobs N               Number of parallel jobs (default: auto-detect CPU count)"
    echo "  -t, --type TYPE            Classifier type: 'standard', 'exploits', 'exploration', 'hopelessness', 'fairness', 'misunderstandings', or 'objective' (default: standard)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Classifier types:"
    echo "  standard         : Uses classify_agent_task_approach.py (general task approach taxonomy)"
    echo "  exploits         : Uses classify_agent_exploit_precursors.py (reward hacking precursors)"
    echo "  exploration      : Uses classify_agent_exploration_types.py (exploration action types)"
    echo "  hopelessness     : Uses classify_agent_hopelessness.py (perceived hopelessness)"
    echo "  fairness         : Uses classify_agent_fairness.py (fair vs unfair play perception)"
    echo "  misunderstandings: Uses classify_agent_misunderstandings.py (understanding vs misunderstanding)"
    echo "  objective        : Uses classify_agent_objective_interpretation.py (how agent interprets objective)"
    echo ""
    echo "Examples:"
    echo "  $0 --directory results/"
    echo "  $0 -d results/ -o ./classifications/"
    echo "  $0 -d results/ -m google/gemini-2.5-pro -j 8"
    echo "  $0 -d results/ -t exploits -j 8"
    echo "  $0 -d results/ -t exploration -j 8"
    echo "  $0 -d results/ -t hopelessness -j 8"
    echo "  $0 -d results/ -t fairness -j 8"
    echo "  $0 -d results/ -t misunderstandings -j 8"
    echo "  $0 -d results/ -t objective -j 8"
    echo ""
    echo "Note: For each state-runXX directory, only the final step directory is processed"
    echo "      (the highest numbered step-* directory's messages.json file)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -t|--type)
            CLASSIFIER_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            usage
            ;;
    esac
done

# Check if required directory argument provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --directory argument is required"
    echo ""
    usage
fi

# Validate and set classifier script
if [ "$CLASSIFIER_TYPE" = "exploits" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_exploit_precursors.py"
    CLASSIFIER_NAME="Exploit Precursors"
    OUTPUT_SUFFIX="exploits_classified"
elif [ "$CLASSIFIER_TYPE" = "exploration" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_exploration_types.py"
    CLASSIFIER_NAME="Exploration Types"
    OUTPUT_SUFFIX="exploration_classified"
elif [ "$CLASSIFIER_TYPE" = "hopelessness" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_hopelessness.py"
    CLASSIFIER_NAME="Hopelessness"
    OUTPUT_SUFFIX="hopelessness_classified"
elif [ "$CLASSIFIER_TYPE" = "fairness" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_fairness.py"
    CLASSIFIER_NAME="Fairness"
    OUTPUT_SUFFIX="fairness_classified"
elif [ "$CLASSIFIER_TYPE" = "misunderstandings" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_misunderstandings.py"
    CLASSIFIER_NAME="Misunderstandings"
    OUTPUT_SUFFIX="misunderstandings_classified"
elif [ "$CLASSIFIER_TYPE" = "objective" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_objective_interpretation.py"
    CLASSIFIER_NAME="Objective Interpretation"
    OUTPUT_SUFFIX="objective_classified"
elif [ "$CLASSIFIER_TYPE" = "standard" ]; then
    CLASSIFIER_SCRIPT="analysis/actions_branches/action_classification/classification_scripts/classify_agent_task_approach.py"
    CLASSIFIER_NAME="Task Approach"
    OUTPUT_SUFFIX="classified"
else
    echo "Error: Invalid classifier_type '$CLASSIFIER_TYPE'. Must be 'standard', 'exploits', 'exploration', 'hopelessness', 'fairness', 'misunderstandings', or 'objective'."
    exit 1
fi

# Check if classifier script exists
if [ ! -f "$CLASSIFIER_SCRIPT" ]; then
    echo "Error: Classifier script not found: $CLASSIFIER_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize arrays for tracking processes
PIDS=()
PROGRESS_PID=""

# Signal handler for Ctrl+C
cleanup_on_interrupt() {
    echo ""
    echo "Interrupted! Cleaning up background jobs..."
    
    # Kill progress monitor
    if [ -n "$PROGRESS_PID" ]; then
        kill $PROGRESS_PID 2>/dev/null || true
        wait $PROGRESS_PID 2>/dev/null || true
    fi
    
    # Kill all worker processes
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    
    # Wait a bit for graceful shutdown
    sleep 0.5
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        kill -9 $pid 2>/dev/null || true
    done
    
    echo "Cleanup complete. Exiting."
    exit 130
}

trap cleanup_on_interrupt SIGINT

# Find all state-run* directories
echo "Searching for state-run directories in $INPUT_DIR..."
STATE_DIRS=$(find "$INPUT_DIR" -type d -name "state-run*" | sort)

if [ -z "$STATE_DIRS" ]; then
    echo "No state-run* directories found in $INPUT_DIR"
    exit 1
fi

# For each state-run directory, find the highest-numbered step directory and its messages.json
FILES=""
DIR_COUNT=0
for state_dir in $STATE_DIRS; do
    DIR_COUNT=$((DIR_COUNT + 1))
    # Find the highest numbered step-* directory
    LAST_STEP_DIR=$(find "$state_dir" -maxdepth 1 -type d -name "step-*" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_STEP_DIR" ]; then
        MESSAGES_FILE="$LAST_STEP_DIR/messages.json"
        if [ -f "$MESSAGES_FILE" ]; then
            FILES="$FILES$MESSAGES_FILE"$'\n'
        fi
    fi
done

# Remove trailing newline
FILES=$(echo "$FILES" | sed '/^$/d')

if [ -z "$FILES" ]; then
    echo "No messages.json files found in step-* directories"
    exit 1
fi

FILE_COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
echo "Found $DIR_COUNT state-run directories"
echo "Processing $FILE_COUNT final state files (one per run)"
echo "Classifier: $CLASSIFIER_NAME ($CLASSIFIER_SCRIPT)"
echo "Using model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Create temp directory for logs and progress tracking
TEMP_LOG_DIR=$(mktemp -d)
PROGRESS_FILE="$TEMP_LOG_DIR/progress.txt"
echo "0" > "$PROGRESS_FILE"
trap "rm -rf $TEMP_LOG_DIR" EXIT

# Function to process a single file
process_file() {
    local file="$1"
    local current="$2"
    local total="$3"
    local output_dir="$4"
    local model="$5"
    local log_file="$6"
    local progress_file="$7"
    local classifier_script="$8"
    local output_suffix="$9"
    
    # Extract step directory name, state-run directory name, and experiment directory name
    local STEP_DIR=$(basename $(dirname "$file"))
    local STATE_DIR=$(basename $(dirname $(dirname "$file")))
    local EXPERIMENT_DIR=$(basename $(dirname $(dirname $(dirname "$file"))))

    # Generate output filename with experiment directory to avoid collisions
    local OUTPUT_FILE="$output_dir/${EXPERIMENT_DIR}_${STATE_DIR}_${STEP_DIR}_${output_suffix}.json"

    echo "[$current/$total] Processing: $EXPERIMENT_DIR/$STATE_DIR/$STEP_DIR" >> "$log_file"
    
    # Skip if output already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  -> Skipping (output already exists)" >> "$log_file"
        # Increment progress
        local count=$(cat "$progress_file")
        echo $((count + 1)) > "$progress_file"
        return 0
    fi
    
    # Run classification
    if python "$classifier_script" "$file" --output "$OUTPUT_FILE" --model "$model" 2>&1 | sed 's/^/  /' >> "$log_file"; then
        echo "  -> ✓ Saved: $(basename "$OUTPUT_FILE")" >> "$log_file"
        # Increment progress
        local count=$(cat "$progress_file")
        echo $((count + 1)) > "$progress_file"
        return 0
    else
        echo "  -> ✗ Failed to process" >> "$log_file"
        # Increment progress even on failure
        local count=$(cat "$progress_file")
        echo $((count + 1)) > "$progress_file"
        return 1
    fi
}

export -f process_file

# Start progress monitor in background
(
    while true; do
        if [ -f "$PROGRESS_FILE" ]; then
            COMPLETED=$(cat "$PROGRESS_FILE")
            printf "\rProgress: %d/%d completed (%d%%)" "$COMPLETED" "$FILE_COUNT" $((COMPLETED * 100 / FILE_COUNT))
        fi
        sleep 1
    done
) &
PROGRESS_PID=$!

# Process files in parallel
CURRENT=0
LOG_FILES=()

for file in $FILES; do
    CURRENT=$((CURRENT + 1))
    
    # Create log file for this job
    LOG_FILE="$TEMP_LOG_DIR/job_$CURRENT.log"
    touch "$LOG_FILE"  # Pre-create the log file
    LOG_FILES+=("$LOG_FILE")
    
    # Launch job in background
    process_file "$file" "$CURRENT" "$FILE_COUNT" "$OUTPUT_DIR" "$MODEL" "$LOG_FILE" "$PROGRESS_FILE" "$CLASSIFIER_SCRIPT" "$OUTPUT_SUFFIX" &
    PIDS+=($!)
    
    # If we've reached the parallel limit, wait for one job to complete
    # Use a portable approach compatible with bash 3.2 (macOS default)
    if [ ${#PIDS[@]} -ge $PARALLEL_JOBS ]; then
        # Wait for any job to complete
        while true; do
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                fi
            done
            # If at least one job completed, break
            if [ ${#NEW_PIDS[@]} -lt ${#PIDS[@]} ]; then
                PIDS=("${NEW_PIDS[@]}")
                break
            fi
            sleep 0.1
        done
    fi
done

# Wait for all remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Stop progress monitor
kill $PROGRESS_PID 2>/dev/null || true
wait $PROGRESS_PID 2>/dev/null || true
printf "\rProgress: %d/%d completed (100%%)\n" "$FILE_COUNT" "$FILE_COUNT"

# Print all logs in order
echo ""
echo "Job Results:"
echo "============"
for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$log_file" ]; then
        cat "$log_file"
        echo ""
    fi
done

echo "Processing complete!"
echo "Results saved in: $OUTPUT_DIR"

# Generate summary report
echo ""
echo "Generating summary report..."
python - <<EOF
import json
import os
from pathlib import Path
from collections import defaultdict

output_dir = Path("$OUTPUT_DIR")
output_suffix = "$OUTPUT_SUFFIX"
classifier_name = "$CLASSIFIER_NAME"

# Match files from this classification run
results_files = list(output_dir.glob(f"*_{output_suffix}.json"))

if not results_files:
    print("No classification results found")
    exit(0)

print(f"Classifier: {classifier_name}")
print(f"Found {len(results_files)} classification results")
print("="*70)

# Aggregate statistics
total_turns = 0
category_totals = defaultdict(int)
file_summaries = []

for result_file in results_files:
    with open(result_file) as f:
        data = json.load(f)
    
    turns = data.get("total_assistant_turns", 0)
    total_turns += turns
    
    for cat, count in data.get("category_counts", {}).items():
        category_totals[cat] += count
    
    file_summaries.append({
        "file": result_file.name,
        "turns": turns,
        "categories": data.get("category_counts", {})
    })

# Print overall summary
print(f"\nOVERALL SUMMARY - {classifier_name} Classification")
print("="*70)
print(f"Total files processed: {len(results_files)}")
print(f"Total assistant turns: {total_turns}")
print(f"\nAggregate category distribution:")

for cat, count in sorted(category_totals.items(), key=lambda x: -x[1]):
    pct = (count / total_turns * 100) if total_turns > 0 else 0
    print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")

# Save summary
summary_file = output_dir / f"summary_report_{output_suffix}.json"
with open(summary_file, 'w') as f:
    json.dump({
        "classifier_name": classifier_name,
        "classifier_type": output_suffix.replace("_classified", ""),
        "total_files": len(results_files),
        "total_turns": total_turns,
        "category_totals": dict(category_totals),
        "files": file_summaries
    }, f, indent=2)

print(f"\nDetailed summary saved to: {summary_file}")
EOF

echo ""
echo "Done!"

