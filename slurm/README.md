# SLURM Setup for Cross-Token Influence Computation

This directory contains scripts for running cross-token influence experiments on HPC clusters using SLURM job arrays.

## Overview

The setup allows you to:
- Process multiple prompts in parallel using SLURM array jobs
- Each array task processes one or more prompts independently
- Results are saved separately and can be aggregated later
- Efficient use of cluster resources with single GPU per job

## Files

- `run_cross_token_influence.sbatch` - SLURM batch script
- `run_cross_token_array.py` - Python runner for array tasks
- `prompts.txt` - Sample prompts file (one prompt per line)
- `aggregate_results.py` - Script to combine results from all tasks
- `README.md` - This file

## Quick Start

### 1. Prepare Your Prompts

Edit `prompts.txt` to include your prompts (one per line):

```bash
nano prompts.txt
```

Or create from your own file:

```bash
cat > prompts.txt << EOF
Your first prompt here
Your second prompt here
Your third prompt here
EOF
```

### 2. Configure the SLURM Script

Edit `run_cross_token_influence.sbatch` to match your cluster:

```bash
nano run_cross_token_influence.sbatch
```

**Important settings to modify:**

```bash
# Set the path to your coupling repository
COUPLING_DIR="/path/to/coupling"

# Set the model you want to use
MODEL_PATH="meta-llama/Meta-Llama-3-8B"

# Configure your Python environment (choose one):
# conda activate your_env_name
# OR
# source /path/to/venv/bin/activate
# OR
# module load python/3.10 cuda/12.1
```

**Adjust SLURM parameters:**

```bash
#SBATCH --array=0-9          # Adjust for number of prompts (0-N where N = num_prompts-1)
#SBATCH --time=02:00:00      # Adjust based on model size and prompt length
#SBATCH --mem=32G            # Adjust based on model requirements
#SBATCH --partition=gpu      # Set to your cluster's GPU partition name
```

### 3. Create Required Directories

```bash
mkdir -p logs results
```

### 4. Submit the Job

```bash
sbatch run_cross_token_influence.sbatch
```

### 5. Monitor Your Jobs

Check job status:
```bash
squeue -u $USER
```

Check output logs:
```bash
tail -f logs/cross_token_*.out
```

Check error logs:
```bash
tail -f logs/cross_token_*.err
```

### 6. Aggregate Results

Once all jobs complete, combine the results:

```bash
python aggregate_results.py \
    --results-dir results \
    --output-file results/Meta-Llama-3-8B_cross_token_influence_all.pt \
    --model-name Meta-Llama-3-8B \
    --verbose
```

## Advanced Usage

### Processing Multiple Prompts Per Task

If you have many prompts and want to reduce the number of jobs, you can process multiple prompts per task:

1. Edit the SLURM script to adjust the array size:
   ```bash
   # For 100 prompts with 10 prompts per task = 10 jobs
   #SBATCH --array=0-9
   ```

2. Add the `--prompts-per-task` argument:
   ```bash
   python slurm/run_cross_token_array.py \
       --task-id ${SLURM_ARRAY_TASK_ID} \
       --prompts-file "${PROMPTS_FILE}" \
       --model-path "${MODEL_PATH}" \
       --output-dir "${OUTPUT_DIR}" \
       --prompts-per-task 10 \
       --verbose
   ```

### Using Different Models

To test multiple models in parallel, create separate prompts files and submit multiple jobs:

```bash
# Submit job for LLaMA-3-8B
sbatch --export=MODEL_PATH=meta-llama/Meta-Llama-3-8B run_cross_token_influence.sbatch

# Submit job for LLaMA-2-7B
sbatch --export=MODEL_PATH=meta-llama/Llama-2-7b-hf run_cross_token_influence.sbatch
```

### Disabling 4-bit Quantization

If you have enough GPU memory and want to run without quantization:

Edit the Python call in the SLURM script:
```bash
python slurm/run_cross_token_array.py \
    ... \
    --no-4bit \
    --verbose
```

### Custom GPU Requirements

For larger models, request more resources:

```bash
#SBATCH --gres=gpu:a100:1    # Request A100 GPU
#SBATCH --mem=64G            # Request more memory
#SBATCH --time=04:00:00      # Request more time
```

### Partial Job Resubmission

If some tasks fail, you can resubmit only those tasks:

```bash
# Resubmit only tasks 5, 6, and 7
sbatch --array=5-7 run_cross_token_influence.sbatch
```

## Common SLURM Commands

```bash
# Submit job
sbatch run_cross_token_influence.sbatch

# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job JOBID

# Cancel a job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Cancel specific array tasks
scancel JOBID_5  # Cancel only task 5 of job JOBID

# Check job accounting (after completion)
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Check available partitions
sinfo

# Check your job history
sacct -u $USER --starttime=2024-01-01
```

## Troubleshooting

### Job fails immediately

Check the error log:
```bash
cat logs/cross_token_JOBID_TASKID.err
```

Common issues:
- Python environment not activated correctly
- CUDA/GPU not available
- Model not found or access denied
- Prompts file path incorrect

### Out of memory errors

Try one or more of:
1. Increase `--mem` in SLURM script
2. Ensure 4-bit quantization is enabled (`--use-4bit`)
3. Use smaller model
4. Reduce `--cpus-per-task`

### CUDA out of memory

- Ensure only one process per GPU (default)
- Enable 4-bit quantization
- Request GPU with more memory (e.g., A100 instead of V100)

### Jobs pending for long time

- Check queue: `squeue -u $USER`
- Your partition may be busy
- Try different partition: `#SBATCH --partition=other_gpu`
- Check cluster status: `sinfo`

### Module/conda errors

Make sure the Python environment section in the SLURM script matches your cluster setup:

```bash
# For conda
conda activate your_env_name

# For modules
module purge
module load python/3.10 cuda/12.1

# For virtualenv
source /path/to/venv/bin/activate
```

## Output Format

Individual task results are saved as:
```
results/ModelName_cross_token_influence_task_0000.pt
results/ModelName_cross_token_influence_task_0001.pt
...
```

Each file contains a dictionary:
```python
{
    0: {
        "prompt": "...",
        "frobenius_norm": tensor([num_layers, T+1]),
        "spectral_norm": tensor([num_layers, T+1]),
        "participation_ratio": tensor([num_layers, T+1]),
        "entropy_effective_rank": tensor([num_layers, T+1])
    },
    1: {...},  # If prompts_per_task > 1
    ...
}
```

Aggregated results have the same format but with all prompts from all tasks combined.

## Example Workflow

Complete example for processing 50 prompts:

```bash
# 1. Navigate to coupling directory
cd /path/to/coupling/slurm

# 2. Create your prompts file with 50 prompts
cat your_prompts.txt > prompts.txt

# 3. Edit SLURM script
nano run_cross_token_influence.sbatch
# Set: #SBATCH --array=0-49
# Configure paths and environment

# 4. Create directories
mkdir -p logs results

# 5. Submit job
sbatch run_cross_token_influence.sbatch
# Note the job ID (e.g., 12345)

# 6. Monitor progress
watch -n 30 'squeue -u $USER'
tail -f logs/cross_token_12345_0.out

# 7. Check for failures (after jobs complete)
grep -l "Exit code: 0" logs/cross_token_12345_*.out | wc -l
# Should equal 50 if all succeeded

# 8. Aggregate results
python aggregate_results.py \
    --results-dir results \
    --output-file results/Meta-Llama-3-8B_all_prompts.pt \
    --verbose

# 9. Load and analyze in Python
python
>>> import torch
>>> results = torch.load('results/Meta-Llama-3-8B_all_prompts.pt')
>>> len(results)  # Should be 50
>>> results[0]['frobenius_norm'].shape  # e.g., torch.Size([32, 11])
```

## Cluster-Specific Notes

Different clusters may have different configurations. Here are some common variations:

### Slurm Partitions
- Some clusters use `--partition=gpu`
- Others use `--partition=gpu-shared` or `--partition=interactive`
- Check with `sinfo` or your cluster documentation

### GPU Specification
```bash
# Generic
#SBATCH --gres=gpu:1

# Specific GPU type
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100

# Multiple GPUs (for multi-GPU models)
#SBATCH --gres=gpu:2
```

### Module Systems
Your cluster might require specific modules:
```bash
module load gcc/11.2.0
module load cuda/12.1.0
module load cudnn/8.9.0
module load python/3.10.8
```

Check your cluster's documentation or ask your system administrator.

## Support

For issues specific to:
- SLURM configuration: Contact your cluster administrator
- Coupling code: See main repository README
- These scripts: Check logs and error messages first

## License

These scripts are provided as part of the coupling package. See main LICENSE file.
