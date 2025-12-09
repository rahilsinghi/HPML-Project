#!/bin/bash
#SBATCH --account=pr_289_general
#SBATCH --job-name=firstsight_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

# FirstSight: Baseline Evaluation (Consolidated)
# Profiles Qwen2-VL-2B model performance

echo "============================================"
echo "FirstSight - Baseline Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"
echo ""

# Load modules
module purge
module load python/intel/3.8.6
module load cuda/11.3.1
module load gcc/10.2.0

# Configure scratch space
export SCRATCH_DIR=/scratch/${USER}
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export TMPDIR=${SCRATCH_DIR}/tmp
export HF_HOME=${SCRATCH_DIR}/huggingface
export TRANSFORMERS_CACHE=${SCRATCH_DIR}/transformers
export TORCH_HOME=${SCRATCH_DIR}/torch

echo "Environment configured:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# Activate environment
VENV_PATH=${SCRATCH_DIR}/envs/firstsight
if [ -d "$VENV_PATH" ]; then
    echo "Activating venv: $VENV_PATH"
    source $VENV_PATH/bin/activate
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run: bash scripts/hpc_setup.sh --mode venv"
    exit 1
fi

# Navigate to working directory
cd $SLURM_SUBMIT_DIR

# Add src to PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# Print GPU info
echo "============================================"
echo "GPU Information:"
echo "============================================"
nvidia-smi
echo ""

# Print Python environment
echo "============================================"
echo "Python Environment:"
echo "============================================"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo ""

# Create output directory
mkdir -p experiments/baseline

# Run baseline evaluation
echo "============================================"
echo "Running Baseline Evaluation"
echo "============================================"
echo ""

python src/baseline/baseline_eval.py

# Check if results were created
if [ -f "experiments/baseline/baseline_"*".json" ]; then
    echo ""
    echo "✓ Baseline results saved successfully"
    ls -lh experiments/baseline/baseline_*.json
else
    echo ""
    echo "⚠ Warning: No baseline results found"
fi

echo ""
echo "============================================"
echo "Job Complete!"
echo "End Time: $(date)"
echo "============================================"

