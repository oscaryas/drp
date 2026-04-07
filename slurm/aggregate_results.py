#!/usr/bin/env python
"""
Aggregate cross-token influence results from multiple SLURM array jobs.

This script combines the individual .pt files produced by each array task
into a single consolidated output file.
"""

import argparse
import os
import torch
from pathlib import Path
from collections import defaultdict
import re


def find_result_files(results_dir, model_name=None, pattern=None):
    """
    Find all result files in the results directory.

    Args:
        results_dir: Directory containing result files
        model_name: Optional model name to filter by
        pattern: Optional regex pattern to match filenames

    Returns:
        List of (task_id, filepath) tuples, sorted by task_id
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Default pattern matches files like: model_cross_token_influence_task_0000.pt
    if pattern is None:
        if model_name:
            pattern = f"{re.escape(model_name)}_cross_token_influence_task_(\\d+)\\.pt"
        else:
            pattern = r".*_cross_token_influence_task_(\d+)\.pt"

    pattern_re = re.compile(pattern)

    result_files = []
    for file_path in results_path.glob("*.pt"):
        match = pattern_re.match(file_path.name)
        if match:
            task_id = int(match.group(1))
            result_files.append((task_id, file_path))

    # Sort by task_id
    result_files.sort(key=lambda x: x[0])

    return result_files


def load_and_merge_results(result_files, verbose=False):
    """
    Load all result files and merge them into a single dictionary.

    Args:
        result_files: List of (task_id, filepath) tuples
        verbose: Print progress messages

    Returns:
        Merged results dictionary
    """
    merged_results = {}
    prompt_index = 0

    for task_id, file_path in result_files:
        if verbose:
            print(f"Loading task {task_id}: {file_path.name}")

        try:
            results = torch.load(file_path, map_location='cpu')

            # Each file contains a dict with keys 0, 1, 2, ... for each prompt
            # We need to renumber these to be sequential across all files
            for local_idx in sorted(results.keys()):
                merged_results[prompt_index] = results[local_idx]
                if verbose:
                    prompt = merged_results[prompt_index]['prompt']
                    print(f"  Prompt {prompt_index}: {prompt[:60]}...")
                prompt_index += 1

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

    return merged_results


def print_summary(results):
    """Print a summary of the merged results."""
    print("\n" + "="*80)
    print("Aggregation Summary")
    print("="*80)
    print(f"Total prompts: {len(results)}")

    if results:
        # Get shape information from first result
        first_key = list(results.keys())[0]
        first_result = results[first_key]

        print(f"\nMetrics per prompt:")
        for metric_name in ['frobenius_norm', 'spectral_norm',
                           'participation_ratio', 'entropy_effective_rank']:
            if metric_name in first_result:
                shape = first_result[metric_name].shape
                print(f"  {metric_name}: {shape}")

        # List all prompts
        print(f"\nPrompts:")
        for i in sorted(results.keys()):
            prompt = results[i]['prompt']
            print(f"  {i:3d}: {prompt[:70]}...")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cross-token influence results from SLURM array jobs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing individual result files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save aggregated results"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name to filter result files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional regex pattern to match result filenames"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    print("="*80)
    print("Cross-Token Influence - Results Aggregation")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output file: {args.output_file}")
    if args.model_name:
        print(f"Model name filter: {args.model_name}")
    if args.pattern:
        print(f"Filename pattern: {args.pattern}")
    print("="*80)

    # Find result files
    print("\nSearching for result files...")
    result_files = find_result_files(
        args.results_dir,
        model_name=args.model_name,
        pattern=args.pattern
    )

    if not result_files:
        print("Error: No result files found!")
        return 1

    print(f"Found {len(result_files)} result files")
    print(f"Task IDs: {[task_id for task_id, _ in result_files]}")

    # Load and merge results
    print("\nLoading and merging results...")
    merged_results = load_and_merge_results(result_files, verbose=args.verbose)

    if not merged_results:
        print("Error: No results loaded!")
        return 1

    # Print summary
    print_summary(merged_results)

    # Save aggregated results
    print(f"\nSaving aggregated results to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(merged_results, args.output_file)
    print("Aggregation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
