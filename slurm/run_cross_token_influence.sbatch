#!/bin/bash
#SBATCH --job-name=cross_token_influence
#SBATCH --output=logs/cross_token_%A_%a.out
#SBATCH --error=logs/cross_token_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Optional: Specify partition/queue (uncomment and modify as needed)
# #SBATCH --partition=gpu

# Optional: Specify GPU type (uncomment and modify as needed)
# #SBATCH --constraint=a100
# #SBATCH --gres=gpu:a100:1

# Optional: Email notifications (uncomment and modify as needed)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@domain.com

#########################################################
# Configuration - Modify these for your setup
#########################################################

# Path to your coupling repository
COUPLING_DIR="/path/to/coupling"

# Path to prompts file (one prompt per line)
PROMPTS_FILE="${COUPLING_DIR}/slurm/prompts.txt"

# Output directory for results
OUTPUT_DIR="${COUPLING_DIR}/slurm/results"

# Model to use (HuggingFace model path)
MODEL_PATH="meta-llama/Meta-Llama-3-8B"

# Python environment setup - choose ONE of the following:

# Option 1: Conda environment
# conda activate your_env_name

# Option 2: Virtual environment
# source /path/to/venv/bin/activate

# Option 3: Load modules (common on HPC clusters)
# module purge
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

#########################################################
# Job execution
#########################################################

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Navigate to the coupling directory
cd "${COUPLING_DIR}"

# Print GPU information
echo "GPU information:"
nvidia-smi

# Print Python environment
echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo ""
echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

# Run the cross-token influence computation
echo ""
echo "=========================================="
echo "Starting cross-token influence computation..."
echo "=========================================="

python slurm/run_cross_token_array.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --prompts-file "${PROMPTS_FILE}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --verbose

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
