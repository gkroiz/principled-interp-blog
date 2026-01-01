#!/usr/bin/env python3
"""
Script to add rollout analysis data to existing classification files.

This script reads existing classification JSON files and adds the rollout_analysis
and final_behavior fields by looking up the corresponding rollout_analysis_detailed.json
files. This allows you to enrich existing classifications without re-running the
expensive LLM classification process.

Usage:
    python add_rollout_analysis.py <classifications_dir> [--dry-run] [--verbose]

Examples:
    # Add rollout analysis to all files in classifications directory
    python add_rollout_analysis.py thought_anchors/classifications/
    
    # Dry run to see what would be updated
    python add_rollout_analysis.py thought_anchors/classifications/ --dry-run
    
    # Verbose output showing details
    python add_rollout_analysis.py thought_anchors/classifications/ --verbose
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Import the load_rollout_analysis function from classify_agent_turns.py
# We'll use the same logic to maintain consistency
sys.path.insert(0, str(Path(__file__).parent))
from classify_agent_turns import load_rollout_analysis


def update_classification_with_rollout_analysis(
    classification_file: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> bool:
    """
    Update a single classification file with rollout analysis data.
    
    Returns True if the file was updated, False otherwise.
    """
    try:
        # Read existing classification
        with open(classification_file, 'r') as f:
            classification_data = json.load(f)
        
        # Check if rollout analysis already exists
        if "rollout_analysis" in classification_data and classification_data["rollout_analysis"] is not None:
            if verbose:
                print(f"  ⏭️  Skipping {classification_file.name}: already has rollout analysis")
            return False
        
        # Get input file path
        input_file = classification_data.get("input_file")
        if not input_file:
            print(f"  ⚠️  Warning: {classification_file.name} has no input_file field, skipping")
            return False
        
        # Load rollout analysis
        rollout_info = load_rollout_analysis(input_file)
        
        if rollout_info is None:
            if verbose:
                print(f"  ℹ️  No rollout analysis found for {classification_file.name}")
            # Add None values so we know we checked
            classification_data["rollout_analysis"] = None
            classification_data["final_behavior"] = None
        else:
            # Determine final behavior
            if rollout_info["reward_hacked"]:
                final_behavior = "reward_hacked"
            elif rollout_info["terminated_on_hint"]:
                final_behavior = "terminated_on_hint"
            else:
                final_behavior = "normal_completion"
            
            # Add rollout analysis fields
            classification_data["rollout_analysis"] = rollout_info
            classification_data["final_behavior"] = final_behavior
            
            if verbose:
                print(f"  ✅ Found rollout analysis for {classification_file.name}:")
                print(f"     - Final behavior: {final_behavior}")
                if rollout_info["reward_hacked"]:
                    print(f"     - Hack turn: {rollout_info['hack_turn']}")
        
        # Save updated classification
        if not dry_run:
            with open(classification_file, 'w') as f:
                json.dump(classification_data, f, indent=2)
            if not verbose:
                print(f"  ✅ Updated {classification_file.name}")
        else:
            if not verbose:
                print(f"  🔍 Would update {classification_file.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {classification_file.name}: {e}")
        return False


def process_directory(
    classifications_dir: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Process all classification files in a directory.
    
    Returns a dict with statistics about the updates.
    """
    # Find all JSON files
    json_files = list(classifications_dir.glob("*_classified.json"))
    
    if not json_files:
        print(f"No *_classified.json files found in {classifications_dir}")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 0}
    
    print(f"\nFound {len(json_files)} classification files")
    print(f"{'DRY RUN - ' if dry_run else ''}Processing...\n")
    
    stats = {
        "total": len(json_files),
        "updated": 0,
        "skipped": 0,
        "errors": 0
    }
    
    for json_file in sorted(json_files):
        try:
            was_updated = update_classification_with_rollout_analysis(
                json_file,
                dry_run=dry_run,
                verbose=verbose
            )
            if was_updated:
                stats["updated"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"  ❌ Error: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add rollout analysis data to existing classification files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add rollout analysis to all files
  python add_rollout_analysis.py thought_anchors/classifications/
  
  # Dry run to see what would be updated
  python add_rollout_analysis.py thought_anchors/classifications/ --dry-run
  
  # Verbose output showing details
  python add_rollout_analysis.py thought_anchors/classifications/ --verbose
        """
    )
    parser.add_argument(
        "classifications_dir",
        type=Path,
        help="Directory containing classification JSON files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually modifying files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each file"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not args.classifications_dir.exists():
        print(f"Error: Directory {args.classifications_dir} does not exist")
        exit(1)
    
    if not args.classifications_dir.is_dir():
        print(f"Error: {args.classifications_dir} is not a directory")
        exit(1)
    
    # Process all files
    stats = process_directory(
        args.classifications_dir,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files: {stats['total']}")
    print(f"Updated: {stats['updated']}")
    print(f"Skipped (already had rollout analysis): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    if args.dry_run:
        print("\n⚠️  This was a DRY RUN - no files were modified")
        print("Run without --dry-run to actually update the files")


if __name__ == "__main__":
    main()

