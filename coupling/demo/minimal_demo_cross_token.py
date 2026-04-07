import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from coupling import run_cross_token_influence_hf


def run(model_path="meta-llama/Meta-Llama-3-8B"):
    """
    Minimal demo for computing cross-token influence metrics.

    For each layer l and input token j, computes multiple metrics from the residual Jacobian
    J_residual = ∂f^l(X^{l-1})_T / ∂x_j^{l-1}:

    Metrics:
    - Frobenius norm: sqrt(sum(S²))
    - Spectral norm: max(S)
    - Participation ratio: sum(S)² / sum(S²)
    - Entropy effective rank: exp(-Σ p_i log p_i)

    Returns a dict with each metric as a tensor of shape [num_layers, T+1].
    """
    model_name = os.path.normpath(os.path.basename(model_path))

    # Load model in 4-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    prompts = ["What is the capital of France? The capital is"]

    # Run cross-token influence computation
    out = run_cross_token_influence_hf(
        model, tokenizer, model_name, prompts,
        save=True, verbose=True
    )

    # Print results
    print("\n" + "="*80)
    print("Cross-Token Influence Results")
    print("="*80)
    for i, result in out.items():
        print(f"\nPrompt {i}: {result['prompt']}")

        # Get shapes (all metrics have the same shape)
        shape = result['frobenius_norm'].shape
        print(f"\nMetrics shape: {shape}")
        print(f"  [num_layers={shape[0]}, num_input_tokens={shape[1]}]")

        # Display Frobenius norm
        print(f"\n{'─'*80}")
        print("FROBENIUS NORM (||J_residual||_F)")
        print(f"{'─'*80}")
        print(f"First layer (all input tokens → last token):")
        print(f"  {result['frobenius_norm'][0]}")
        print(f"Last layer (all input tokens → last token):")
        print(f"  {result['frobenius_norm'][-1]}")

        # Display Spectral norm
        print(f"\n{'─'*80}")
        print("SPECTRAL NORM (largest singular value)")
        print(f"{'─'*80}")
        print(f"First layer:")
        print(f"  {result['spectral_norm'][0]}")
        print(f"Last layer:")
        print(f"  {result['spectral_norm'][-1]}")

        # Display Participation ratio
        print(f"\n{'─'*80}")
        print("PARTICIPATION RATIO (sum(S)² / sum(S²))")
        print(f"{'─'*80}")
        print(f"First layer:")
        print(f"  {result['participation_ratio'][0]}")
        print(f"Last layer:")
        print(f"  {result['participation_ratio'][-1]}")

        # Display Entropy effective rank
        print(f"\n{'─'*80}")
        print("ENTROPY EFFECTIVE RANK (exp(-Σ p_i log p_i))")
        print(f"{'─'*80}")
        print(f"First layer:")
        print(f"  {result['entropy_effective_rank'][0]}")
        print(f"Last layer:")
        print(f"  {result['entropy_effective_rank'][-1]}")

        print(f"\n{'='*80}\n")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cross-token influence metrics for transformer models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model path or name (default: meta-llama/Meta-Llama-3-8B)"
    )
    args = parser.parse_args()
    run(model_path=args.model_path)
