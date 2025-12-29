#!/usr/bin/env python3
"""
Analyze classification results and generate reports.

Usage:
    python analyze_classifications.py <classifications_dir>
    python analyze_classifications.py classification_results.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any


def load_classification_file(file_path: Path) -> Dict[str, Any]:
    """Load a single classification result file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_single_file(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single classification result."""
    classifications = data.get("classifications", [])
    
    # Count transitions between categories
    transitions = []
    for i in range(len(classifications) - 1):
        from_cat = classifications[i]["classification"]
        to_cat = classifications[i + 1]["classification"]
        transitions.append((from_cat, to_cat))
    
    # Identify sequences of same category
    sequences = []
    if classifications:
        current_cat = classifications[0]["classification"]
        current_seq_start = 0
        
        for i in range(1, len(classifications)):
            if classifications[i]["classification"] != current_cat:
                sequences.append({
                    "category": current_cat,
                    "start": current_seq_start,
                    "end": i - 1,
                    "length": i - current_seq_start
                })
                current_cat = classifications[i]["classification"]
                current_seq_start = i
        
        # Add final sequence
        sequences.append({
            "category": current_cat,
            "start": current_seq_start,
            "end": len(classifications) - 1,
            "length": len(classifications) - current_seq_start
        })
    
    return {
        "total_turns": len(classifications),
        "category_counts": data.get("category_counts", {}),
        "transitions": transitions,
        "sequences": sequences
    }


def analyze_directory(directory: Path) -> Dict[str, Any]:
    """Analyze all classification files in a directory."""
    classification_files = list(directory.glob("*_classified.json"))
    
    if not classification_files:
        print(f"No *_classified.json files found in {directory}")
        return {}
    
    all_analyses = []
    aggregate_transitions = Counter()
    aggregate_categories = defaultdict(int)
    
    for file_path in classification_files:
        data = load_classification_file(file_path)
        analysis = analyze_single_file(data)
        
        all_analyses.append({
            "file": file_path.name,
            "analysis": analysis
        })
        
        # Aggregate
        for cat, count in analysis["category_counts"].items():
            aggregate_categories[cat] += count
        
        for transition in analysis["transitions"]:
            aggregate_transitions[transition] += 1
    
    return {
        "num_files": len(classification_files),
        "files": all_analyses,
        "aggregate_categories": dict(aggregate_categories),
        "aggregate_transitions": {f"{f} -> {t}": c for (f, t), c in aggregate_transitions.most_common()}
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results."""
    print("="*70)
    print("CLASSIFICATION ANALYSIS")
    print("="*70)
    
    if "num_files" in analysis:
        # Directory analysis
        print(f"\nFiles analyzed: {analysis['num_files']}")
        
        print("\nAggregate Category Distribution:")
        total = sum(analysis["aggregate_categories"].values())
        for cat, count in sorted(analysis["aggregate_categories"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nTop Category Transitions:")
        for transition, count in list(analysis["aggregate_transitions"].items())[:15]:
            print(f"  {transition:50s}: {count:4d}")
        
        print("\nPer-File Analysis:")
        for file_data in analysis["files"][:5]:  # Show first 5
            print(f"\n  {file_data['file']}:")
            file_analysis = file_data["analysis"]
            print(f"    Total turns: {file_analysis['total_turns']}")
            print(f"    Sequences: {len(file_analysis['sequences'])}")
            
            # Show longest sequences
            longest = sorted(file_analysis['sequences'], key=lambda x: -x['length'])[:3]
            print(f"    Longest category sequences:")
            for seq in longest:
                print(f"      - {seq['category']} ({seq['length']} turns)")
    
    else:
        # Single file analysis
        print(f"\nTotal turns: {analysis['total_turns']}")
        
        print("\nCategory Distribution:")
        total = analysis['total_turns']
        for cat, count in sorted(analysis["category_counts"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nCategory Sequences ({len(analysis['sequences'])} total):")
        for i, seq in enumerate(analysis['sequences'], 1):
            print(f"  {i}. {seq['category']} (turns {seq['start']}-{seq['end']}, length {seq['length']})")
        
        print("\nCategory Transitions:")
        transition_counts = Counter(analysis['transitions'])
        for (from_cat, to_cat), count in transition_counts.most_common(15):
            print(f"  {from_cat:30s} -> {to_cat:30s}: {count:2d}")


def generate_markdown_report(analysis: Dict[str, Any], output_file: Path):
    """Generate a markdown report of the analysis."""
    with open(output_file, 'w') as f:
        f.write("# Classification Analysis Report\n\n")
        
        if "num_files" in analysis:
            f.write(f"**Files Analyzed:** {analysis['num_files']}\n\n")
            
            f.write("## Aggregate Category Distribution\n\n")
            total = sum(analysis["aggregate_categories"].values())
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            for cat, count in sorted(analysis["aggregate_categories"].items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                f.write(f"| {cat} | {count} | {pct:.1f}% |\n")
            
            f.write("\n## Top Category Transitions\n\n")
            f.write("| Transition | Count |\n")
            f.write("|------------|-------|\n")
            for transition, count in list(analysis["aggregate_transitions"].items())[:20]:
                f.write(f"| {transition} | {count} |\n")
        
        else:
            f.write(f"**Total Turns:** {analysis['total_turns']}\n\n")
            
            f.write("## Category Distribution\n\n")
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|------------|\n")
            total = analysis['total_turns']
            for cat, count in sorted(analysis["category_counts"].items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                f.write(f"| {cat} | {count} | {pct:.1f}% |\n")
            
            f.write("\n## Category Sequences\n\n")
            for i, seq in enumerate(analysis['sequences'], 1):
                f.write(f"{i}. **{seq['category']}** (turns {seq['start']}-{seq['end']}, length {seq['length']})\n")
    
    print(f"\nMarkdown report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze classification results"
    )
    parser.add_argument(
        "input",
        help="Path to classification file or directory containing classification files"
    )
    parser.add_argument(
        "--report", "-r",
        help="Generate markdown report to specified file",
        default=None
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return 1
    
    if input_path.is_dir():
        print(f"Analyzing directory: {input_path}")
        analysis = analyze_directory(input_path)
    else:
        print(f"Analyzing file: {input_path}")
        data = load_classification_file(input_path)
        analysis = analyze_single_file(data)
    
    if not analysis:
        print("No data to analyze")
        return 1
    
    print_analysis(analysis)
    
    if args.report:
        report_path = Path(args.report)
        generate_markdown_report(analysis, report_path)
    
    return 0


if __name__ == "__main__":
    exit(main())

