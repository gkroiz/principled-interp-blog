#!/bin/bash
# git  classify conversation state files (PARALLELIZED)
# Only processes the final (highest-numbered) state file from each run directory

set -e

# Check if directory argument provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory> [output_dir] [model] [parallel_jobs]"
    echo ""
    echo "Examples:"
    echo "  $0 continued_reasoning_results/2_games/"
    echo "  $0 continued_reasoning_results/2_games/ ./classifications/"
    echo "  $0 continued_reasoning_results/gpt-5-2025-08-07-20251030-014424/ ./classifications/"
    echo "  $0 continued_reasoning_results/2_games/ ./classifications/ google/gemini-2.5-pro 8"
    echo ""
    echo "Note: For each state-runXX directory, only the final state file is processed"
    echo "      (since each state file contains cumulative history)"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-./thought_anchors/classifications}"
MODEL="${3:-google/gemini-2.5-pro}"
PARALLEL_JOBS="${4:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

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

# For each state-run directory, find the highest-numbered state file
FILES=""
DIR_COUNT=0
for state_dir in $STATE_DIRS; do
    DIR_COUNT=$((DIR_COUNT + 1))
    # Find the highest numbered ctfish-tictactoe-*.json file
    LAST_FILE=$(ls "$state_dir"/ctfish-tictactoe-*.json 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_FILE" ]; then
        FILES="$FILES$LAST_FILE"$'\n'
    fi
done

# Remove trailing newline
FILES=$(echo "$FILES" | sed '/^$/d')

if [ -z "$FILES" ]; then
    echo "No ctfish-tictactoe-*.json files found in state-run directories"
    exit 1
fi

FILE_COUNT=$(echo "$FILES" | wc -l | tr -d ' ')
echo "Found $DIR_COUNT state-run directories"
echo "Processing $FILE_COUNT final state files (one per run)"
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
    
    # Extract state-run directory name, experiment directory name, and file basename
    local STATE_DIR=$(basename $(dirname "$file"))
    local EXPERIMENT_DIR=$(basename $(dirname $(dirname "$file")))
    local FILE_BASENAME=$(basename "$file" .json)
    
    # Generate output filename with experiment directory to avoid collisions
    local OUTPUT_FILE="$output_dir/${EXPERIMENT_DIR}_${STATE_DIR}_${FILE_BASENAME}_classified.json"
    
    echo "[$current/$total] Processing: $EXPERIMENT_DIR/$STATE_DIR/$FILE_BASENAME" >> "$log_file"
    
    # Skip if output already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  -> Skipping (output already exists)" >> "$log_file"
        # Increment progress
        local count=$(cat "$progress_file")
        echo $((count + 1)) > "$progress_file"
        return 0
    fi
    
    # Run classification
    if python thought_anchors/classify_agent_turns.py "$file" --output "$OUTPUT_FILE" --model "$model" 2>&1 | sed 's/^/  /' >> "$log_file"; then
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
    process_file "$file" "$CURRENT" "$FILE_COUNT" "$OUTPUT_DIR" "$MODEL" "$LOG_FILE" "$PROGRESS_FILE" &
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
results_files = list(output_dir.glob("*_classified.json"))

if not results_files:
    print("No classification results found")
    exit(0)

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
print(f"\nOVERALL SUMMARY")
print("="*70)
print(f"Total files processed: {len(results_files)}")
print(f"Total assistant turns: {total_turns}")
print(f"\nAggregate category distribution:")

for cat, count in sorted(category_totals.items(), key=lambda x: -x[1]):
    pct = (count / total_turns * 100) if total_turns > 0 else 0
    print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")

# Save summary
summary_file = output_dir / "summary_report.json"
with open(summary_file, 'w') as f:
    json.dump({
        "total_files": len(results_files),
        "total_turns": total_turns,
        "category_totals": dict(category_totals),
        "files": file_summaries
    }, f, indent=2)

print(f"\nDetailed summary saved to: {summary_file}")
EOF

echo ""
echo "Done!"

