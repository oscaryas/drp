import os
from collections import defaultdict

import torch

from .jacobian import jacobian
from .utils import timestamp


def cross_token_influence_from_hooks(hooks, index=-1, chunks=4, verbose=False, device="cuda"):
    """
    Computes cross-token influence metrics for each layer.

    For each layer l and input token j, computes multiple metrics from the residual Jacobian
    J_residual = ∂f^l(X^{l-1})_T / ∂x_j^{l-1}, where f^l excludes the skip connection.

    Metrics computed:
    - Frobenius norm: sqrt(sum(S^2))
    - Spectral norm: max(S) (largest singular value)
    - Participation ratio: sum(S)^2 / sum(S^2)
    - Entropy effective rank: exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s_i)

    Args:
        hooks: dict of representations before and after skip connection
               hooks[layer] = {0: x_in, 1: x_out}
        index: Output token index (T). Use -1 for last token.
        chunks: Number of chunks for Jacobian computation
        verbose: Print timestamped progress messages
        device: Device for computation

    Returns:
        metrics_dict: dict with keys:
            'frobenius_norm': tensor of shape [num_layers, T+1]
            'spectral_norm': tensor of shape [num_layers, T+1]
            'participation_ratio': tensor of shape [num_layers, T+1]
            'entropy_effective_rank': tensor of shape [num_layers, T+1]
    """
    # Get sequence length from first hook
    first_layer = list(hooks.keys())[0]
    x_in_sample = hooks[first_layer][0]
    seq_len = x_in_sample.shape[1]

    # Convert negative index to positive
    if index < 0:
        T = seq_len + index
    else:
        T = index

    num_layers = len(hooks)
    num_input_tokens = T + 1  # All tokens j from 0 to T (includes self-influence)

    timestamp(f"Computing cross-token influence for {num_layers} layers, {num_input_tokens} input tokens") if verbose else None

    # Initialize output tensors
    frobenius_norms = torch.zeros(num_layers, num_input_tokens, device=device)
    spectral_norms = torch.zeros(num_layers, num_input_tokens, device=device)
    participation_ratios = torch.zeros(num_layers, num_input_tokens, device=device)
    entropy_ranks = torch.zeros(num_layers, num_input_tokens, device=device)

    # Iterate over layers
    for layer_idx, layer_key in enumerate(hooks):
        timestamp(f"Processing layer {layer_idx + 1}/{num_layers}: {layer_key}") if verbose else None

        x_in = hooks[layer_key][0]
        x_out = hooks[layer_key][1]
        dim = x_out.shape[-1]

        # Compute influence from each input token j to output token T
        for j in range(num_input_tokens):
            timestamp(f"  Computing Jacobian from token {j} to token {T}") if verbose else None

            # Compute Jacobian: ∂(x_out)_T / ∂(x_in)_j
            J = jacobian(x_out, x_in, index=index, index_in=j, chunks=chunks, device=device).detach()

            # Subtract identity to get residual Jacobian
            J_residual = J - torch.eye(dim, device=device)

            # Compute SVD to get singular values
            _, S, _ = torch.linalg.svd(J_residual, full_matrices=False)

            # Compute metrics from singular values
            # 1. Frobenius norm: sqrt(sum(S^2))
            frobenius_norm = torch.sqrt(torch.sum(S ** 2))

            # 2. Spectral norm: largest singular value
            spectral_norm = S[0]  # Singular values are sorted in descending order

            # 3. Participation ratio: sum(S)^2 / sum(S^2)
            sum_S = torch.sum(S)
            sum_S_squared = torch.sum(S ** 2)
            participation_ratio = (sum_S ** 2) / (sum_S_squared + 1e-10)  # Add epsilon for numerical stability

            # 4. Entropy effective rank: exp(-sum(p_i * log(p_i)))
            p = S / (sum_S + 1e-10)  # Normalize to get probabilities
            # Filter out very small values to avoid log(0)
            p = p[p > 1e-10]
            entropy = -torch.sum(p * torch.log(p))
            entropy_rank = torch.exp(entropy)

            # Store results
            frobenius_norms[layer_idx, j] = frobenius_norm
            spectral_norms[layer_idx, j] = spectral_norm
            participation_ratios[layer_idx, j] = participation_ratio
            entropy_ranks[layer_idx, j] = entropy_rank

            if verbose:
                timestamp(f"  Token {j} metrics - Frob: {frobenius_norm:.4f}, "
                         f"Spec: {spectral_norm:.4f}, PR: {participation_ratio:.4f}, "
                         f"ER: {entropy_rank:.4f}")

    timestamp("Cross-token influence computation complete") if verbose else None

    return {
        'frobenius_norm': frobenius_norms,
        'spectral_norm': spectral_norms,
        'participation_ratio': participation_ratios,
        'entropy_effective_rank': entropy_ranks
    }


def run_cross_token_influence_hf(model, tokenizer, model_name, prompts, start=None, end=None,
                                   save=False, out_path=None, device="cuda", verbose=False):
    """
    Runs cross-token influence experiment for HuggingFace model and tokenizer.

    Args:
        model: HuggingFace model (PreTrainedModel)
        tokenizer: HuggingFace tokenizer (PreTrainedTokenizer)
        model_name: Model name for saving (e.g., "Meta-Llama-3-8B")
        prompts: List of string prompts
        start: Start index for prompts (default: 0)
        end: End index for prompts (default: len(prompts))
        save: Whether to save output to file
        out_path: Directory to save output (default: current working directory)
        device: Device for computation (default: "cuda")
        verbose: Print timestamped progress messages

    Returns:
        out: dict where out[i] = {
            "prompt": str,
            "frobenius_norm": tensor of shape [num_layers, T+1],
            "spectral_norm": tensor of shape [num_layers, T+1],
            "participation_ratio": tensor of shape [num_layers, T+1],
            "entropy_effective_rank": tensor of shape [num_layers, T+1]
        }
    """
    out = defaultdict(dict)
    start = start if start is not None else 0
    end = end if end is not None else len(prompts)

    for i, prompt in zip(range(start, end), prompts):
        timestamp(f"Running prompt {i + 1} of {end}")

        out[i] = {"prompt": prompt}
        print(prompt) if verbose else None

        # Tokenize prompt
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens.input_ids
        num_tokens = input_ids.shape[1]
        print(f"Number of tokens: {num_tokens}") if verbose else None

        # Compute chunks based on token count (same formula as run_coupling_hf)
        chunks = 2 * (num_tokens // 20) + 5 + i
        print(f"Number of chunks: {chunks}") if verbose else None

        # Run forward pass
        input_ids = input_ids.to(device)
        outputs = model(input_ids, output_hidden_states=True)
        L = len(outputs.hidden_states) - 1  # Number of layers

        # Format as hooks
        outputs_zip = {}
        for j in range(L):
            outputs_zip[f"block_{j}"] = {0: outputs.hidden_states[j], 1: outputs.hidden_states[j+1]}

        # Compute cross-token influence metrics
        metrics_dict = cross_token_influence_from_hooks(
            outputs_zip, index=-1, chunks=chunks, verbose=verbose, device=device
        )

        # Store all metrics
        out[i].update(metrics_dict)
        timestamp(f"Ended prompt {i + 1}") if verbose else None

    # Save if requested
    if save:
        if out_path is None:
            out_path = os.getcwd()
            print(f"Saving enabled but out_path not specified. Saving in `{out_path}`")
        out_file = os.path.join(out_path, "_".join((model_name, "cross_token_influence.pt")))
        torch.save(out, out_file)
        print(f"Saved results to: {out_file}")

    return out
