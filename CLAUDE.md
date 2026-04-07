# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research package for computing transformer block coupling metrics from PyTorch hooks. This implements the coupling metric m(J₁, J₂) described in "Transformer Block Coupling and its Correlation with Generalization in LLMs" (ICLR 2025).

The package computes Jacobians of residual connections across transformer layers and measures coupling between singular vectors, normalized by p-norms.

## Installation

Install in development mode:
```bash
pip install -e .
```

Or directly from GitHub:
```bash
pip install git+https://github.com/sugolov/coupling.git
```

## Running Demos

**Block coupling demo** (original metric):
```bash
python coupling/demo/minimal_demo_hf.py --model-path meta-llama/Meta-Llama-3-8B
```

**Cross-token influence demo** (new metrics):
```bash
python coupling/demo/minimal_demo_cross_token.py --model-path meta-llama/Meta-Llama-3-8B
```

Both demos can run without arguments (defaults to Meta-Llama-3-8B) or as modules:
```bash
python -m coupling.demo.minimal_demo_hf --model-path <model-path>
python -m coupling.demo.minimal_demo_cross_token --model-path <model-path>
```

**Note**: Demo files import from the package (e.g., `from coupling import run_coupling_hf`). For programmatic use, always import from the top-level package.

Google Colab demo: https://colab.research.google.com/drive/1ronRmxr0yJO8Re0iJeqp055IoiU7oLOI?usp=sharing

## Running on SLURM Clusters

The `slurm/` directory contains scripts for running cross-token influence experiments on HPC clusters using SLURM job arrays. This setup enables parallel processing of multiple prompts, with each array task processing one or more prompts independently on a single GPU.

**Quick Start:**

1. Edit prompts file: `slurm/prompts.txt` (one prompt per line)
2. Configure SLURM script: `slurm/run_cross_token_influence.sbatch`
   - Set `COUPLING_DIR` path
   - Configure Python environment (conda/venv/modules)
   - Adjust `--array=0-N` for N+1 prompts
   - Set partition and GPU requirements
3. Submit job: `sbatch slurm/run_cross_token_influence.sbatch`
4. Monitor: `squeue -u $USER` and check `logs/cross_token_*.out`
5. Aggregate results: `python slurm/aggregate_results.py --results-dir results --output-file results/output.pt`

**Key Files:**
- `slurm/run_cross_token_influence.sbatch` - SLURM batch script with job configuration
- `slurm/run_cross_token_array.py` - Python runner that maps task IDs to prompts
- `slurm/prompts.txt` - Input prompts (one per line)
- `slurm/aggregate_results.py` - Combines results from all array tasks
- `slurm/README.md` - Detailed documentation and troubleshooting

Each array task saves results as `{model_name}_cross_token_influence_task_{id:04d}.pt`. See `slurm/README.md` for advanced usage, troubleshooting, and cluster-specific configurations.

## Architecture

### Core Module Structure

The package follows a pipeline architecture with two main computational paths:

**1. Block Coupling (original paper metric)**

1. **main.py** - Entry point with two main functions:
   - `coupling_from_hooks()`: Computes coupling from pre-collected hooks (generic)
   - `run_coupling_hf()`: End-to-end runner for HuggingFace models

2. **metrics.py** - Coupling metric computation:
   - Main metric: `diag_sv_trace_similarity()` computes ||U₂ᵀJ₁V₂ - S₁||_F / ||s₁||_p
   - Returns both trace-normalized and p-norm-normalized variants
   - Generates coupling matrices for all layer pairs (i,j)
   - Returns two dictionaries: `coupling_ujv` and `coupling_vju` (transposed roles of U/V)
   - Also contains legacy functions: `diag_sv_similarity()` and `diag_sv_similarity_k()`

**2. Cross-Token Influence (new addition)**

3. **influence.py** - Cross-token influence metrics:
   - `cross_token_influence_from_hooks()`: Computes influence from each input token j to output token T
   - For each layer, computes metrics from residual Jacobian J_residual = ∂f^l(X^{l-1})_T / ∂x_j^{l-1}
   - Four metrics computed: Frobenius norm, spectral norm, participation ratio, entropy effective rank
   - `run_cross_token_influence_hf()`: End-to-end runner for HuggingFace models

**Shared Components**

4. **jacobian.py** - Jacobian computation via PyTorch autograd:
   - `jacobian()`: Uses `functorch.experimental.chunk_vmap` for vectorized gradient computation across output dimensions
   - Creates identity matrix and vmaps over it to compute full Jacobian efficiently
   - Supports both same-token (index=index_in) and cross-token (index≠index_in) Jacobians
   - `svd()`: Wrapper for computing SVD via either "torch" (standard) or "random" (randomized) methods
   - `randomsvd()` and `randomevd()`: Implement randomized SVD for large matrices using power iterations
   - Key: Computes J = ∂f^l(X^(l-1))/∂x for residual function f

5. **utils.py** - Minimal utilities (timestamp function for logging)

### Data Flow

```
HF Model → Hidden States → Hooks Dict → Jacobians → SVD → Coupling Metrics
```

Hooks format: `hooks[layer] = {0: x_in, 1: x_out}` where x_out = x_in + f(x_in)

The Jacobian computation subtracts identity: `J - torch.eye(dim)` to get the residual Jacobian.

### Key Parameters

- `chunks`: Number of chunks for vectorized Jacobian computation using `chunk_vmap`. Higher values reduce memory usage but increase computation time. In `run_coupling_hf()`, auto-computed as `2 * (num_tokens // 20) + 5 + i` where `i` is the prompt index. For manual use via `coupling_from_hooks()`, default is 4.
- `num_sing_vecs`: Tuple of K values for top-K singular vectors (default: (10,30,50)). Determines how many singular vectors to use when computing coupling metrics. Results are computed for each K value.
- `p`: Order of p-norm for coupling normalization (default: 2). Used in denominator of metric: ||s₁||_p
- `index`: Token index for Jacobian computation (default: -1, last token). Specifies which output token position to compute Jacobian for.
- `index_in`: Input token index for Jacobian (defaults to `index` if not specified). Allows computing cross-position Jacobians.
- `svd_method`: Either "torch" (standard SVD via torch.linalg.svd) or "random" (randomized SVD for large matrices, uses K, L, E, ITS parameters)

## Dependencies

- PyTorch 2.6.0
- Transformers 4.50.3 (for HuggingFace model integration)
- bitsandbytes 0.45.4 (for quantization support)
- datasets 3.5.0
- functorch (for `chunk_vmap`)

## Output Format

**Block coupling** (`run_coupling_hf()`) returns a nested dictionary:
```python
out[prompt_index] = {
    "prompt": str,
    "coupling_ujv": {K: {"trace": tensor, "norm": tensor}},  # for each K in num_sing_vecs
    "coupling_vju": {K: {"trace": tensor, "norm": tensor}}
}
```

Both `coupling_ujv` and `coupling_vju` contain L×L matrices (where L is number of layers) with coupling values between all layer pairs. The difference is which singular vectors (U or V) are used in the metric computation.

**Cross-token influence** (`run_cross_token_influence_hf()`) returns:
```python
out[prompt_index] = {
    "prompt": str,
    "frobenius_norm": tensor[num_layers, T+1],
    "spectral_norm": tensor[num_layers, T+1],
    "participation_ratio": tensor[num_layers, T+1],
    "entropy_effective_rank": tensor[num_layers, T+1]
}
```

Each metric tensor has shape [num_layers, T+1] where T+1 is the number of input tokens (0 to T inclusive). For each layer and input token j, the value represents the influence from token j to the output token T.

## Mathematical Notation

**Block Coupling Metric**

Measures how aligned the Jacobian J₁ is with the singular vector structure of J₂:

- **m_K(J₁, J₂)** = ||U₂ᵀJ₁V₂ - S₁||_F / ||s₁||_p
  - U₂, V₂: Top-K left/right singular vectors of J₂
  - S₁: Top-K singular values of J₁ (as diagonal matrix)
  - ||·||_F: Frobenius norm
  - ||s₁||_p: p-norm of singular values vector

The code computes both `ujv` (using U and V as shown above) and `vju` (swapping roles of U and V) variants.

**Cross-Token Influence Metrics**

For each layer l and input token j, computes metrics from the residual Jacobian J_residual = ∂f^l(X^{l-1})_T / ∂x_j^{l-1}, where T is the output token (typically last token):

1. **Frobenius norm**: ||J_residual||_F = √(Σ S²)
2. **Spectral norm**: max(S) - largest singular value
3. **Participation ratio**: (Σ S)² / (Σ S²) - measures how evenly spread the singular values are
4. **Entropy effective rank**: exp(-Σ p_i log p_i) where p_i = s_i / Σ s_i - another measure of singular value distribution

## Additional Notes

- The `extra/` directory contains supplementary code for alignment, Vision Transformers (coupling_vit.py), and plotting utilities
- Output is saved as `{model_name}_coupling.pt` or `{model_name}_cross_token_influence.pt` when `save=True`
- Supports quantized models via BitsAndBytesConfig (4-bit loading by default in demos)
- No test suite is currently included in the repository
- The package exports main functions via `coupling/__init__.py`: `coupling_from_hooks`, `run_coupling_hf`, `cross_token_influence_from_hooks`, `run_cross_token_influence_hf`, `jacobian`, `metrics`, `svd`
- **Critical**: The Jacobian computation subtracts the identity matrix to isolate the residual function contribution: `J - torch.eye(dim)`. This is done in both block coupling and cross-token influence computations.
