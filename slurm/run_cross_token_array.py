#!/usr/bin/env python
"""
SLURM array job runner for cross-token influence computation.

This script is designed to be called from a SLURM array job, where each
array task processes one or more prompts from a prompts file.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path to import coupling
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from coupling import run_cross_token_influence_hf


def load_prompts(prompts_file):
    """Load prompts from file (one prompt per line)."""
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def get_prompts_for_task(prompts, task_id, prompts_per_task=1):
    """
    Get the prompt(s) for a specific task ID.

    Args:
        prompts: List of all prompts
        task_id: SLURM array task ID
        prompts_per_task: Number of prompts to process per task

    Returns:
        List of prompts for this task
    """
    start_idx = task_id * prompts_per_task
    end_idx = start_idx + prompts_per_task

    task_prompts = prompts[start_idx:end_idx]

    if not task_prompts:
        raise ValueError(
            f"Task ID {task_id} is out of range. "
            f"Total prompts: {len(prompts)}, prompts per task: {prompts_per_task}"
        )

    return task_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-token influence computation for SLURM array job"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="SLURM array task ID"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to file containing prompts (one per line)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model path or name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--prompts-per-task",
        type=int,
        default=1,
        help="Number of prompts to process per task (default: 1)"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="use_4bit",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    # Print configuration
    print("="*80)
    print("Cross-Token Influence - SLURM Array Job")
    print("="*80)
    print(f"Task ID: {args.task_id}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Prompts per task: {args.prompts_per_task}")
    print(f"Use 4-bit quantization: {args.use_4bit}")
    print(f"Device: {args.device}")
    print("="*80)

    # Load prompts
    print("\nLoading prompts...")
    all_prompts = load_prompts(args.prompts_file)
    print(f"Total prompts in file: {len(all_prompts)}")

    # Get prompts for this task
    task_prompts = get_prompts_for_task(
        all_prompts, args.task_id, args.prompts_per_task
    )
    print(f"Prompts for this task: {len(task_prompts)}")

    for i, prompt in enumerate(task_prompts):
        print(f"  {i}: {prompt[:80]}...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model_name = os.path.normpath(os.path.basename(args.model_path))

    # Configure quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        print("Not using quantization")
        bnb_config = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
        )
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1

    # Run cross-token influence computation
    print("\n" + "="*80)
    print("Running cross-token influence computation...")
    print("="*80 + "\n")

    try:
        results = run_cross_token_influence_hf(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            prompts=task_prompts,
            save=False,  # We'll save manually with task-specific naming
            device=args.device,
            verbose=args.verbose
        )
        print("\nComputation completed successfully!")
    except Exception as e:
        print(f"\nError during computation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results with task-specific filename
    output_file = os.path.join(
        args.output_dir,
        f"{model_name}_cross_token_influence_task_{args.task_id:04d}.pt"
    )

    print(f"\nSaving results to: {output_file}")
    torch.save(results, output_file)
    print("Results saved successfully!")

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Task ID: {args.task_id}")
    print(f"Prompts processed: {len(task_prompts)}")
    print(f"Output file: {output_file}")

    # Print shapes for first result
    if results:
        first_key = list(results.keys())[0]
        first_result = results[first_key]
        print(f"\nResult shapes (prompt 0):")
        for metric_name in ['frobenius_norm', 'spectral_norm',
                           'participation_ratio', 'entropy_effective_rank']:
            if metric_name in first_result:
                shape = first_result[metric_name].shape
                print(f"  {metric_name}: {shape}")

    print("="*80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
