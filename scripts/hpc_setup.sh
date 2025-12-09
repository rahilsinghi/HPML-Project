#!/bin/bash
# FirstSight: Consolidated HPC Environment Setup
# Supports both conda and venv modes
# Usage: bash hpc_setup.sh --mode [conda|venv]

set -e

echo "=========================================="
echo "FirstSight - HPC Environment Setup"
echo "=========================================="
echo ""

# Parse arguments
MODE="venv"  # Default to venv (better for large models)
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash hpc_setup.sh --mode [conda|venv]"
      exit 1
      ;;
  esac
done

echo "Setup mode: $MODE"
echo ""

# Load modules
echo "Loading modules..."
module purge

if [ "$MODE" == "conda" ]; then
    module load anaconda3/2020.07
    module load cuda/11.8.0
    module load gcc/10.2.0
else
    module load python/intel/3.8.6
    module load cuda/11.3.1
    module load gcc/10.2.0
fi

# Configure scratch space (important for large VLMs)
echo "Configuring scratch space..."
export SCRATCH_DIR=/scratch/${USER}
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1
export TMPDIR=${SCRATCH_DIR}/tmp
export HF_HOME=${SCRATCH_DIR}/huggingface
export TRANSFORMERS_CACHE=${SCRATCH_DIR}/transformers
export TORCH_HOME=${SCRATCH_DIR}/torch

# Create directories
mkdir -p $TMPDIR
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $TORCH_HOME

echo "Scratch directories configured:"
echo "  TMPDIR: $TMPDIR"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# Setup environment based on mode
if [ "$MODE" == "conda" ]; then
    echo "Creating conda environment 'firstsight'..."
    conda create -n firstsight python=3.10 -y
    
    echo "Activating conda environment..."
    source activate firstsight
    
    # Install PyTorch with CUDA 11.8
    echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    
else
    # venv mode (recommended for scratch space)
    VENV_PATH=${SCRATCH_DIR}/envs/firstsight
    echo "Creating venv at $VENV_PATH..."
    python3 -m venv $VENV_PATH
    
    echo "Activating venv..."
    source $VENV_PATH/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    python -m pip install --upgrade pip --no-cache-dir
    
    # Install PyTorch with CUDA 11.3
    echo "Installing PyTorch 1.12.1 with CUDA 11.3..."
    pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
fi

# Install common dependencies
echo ""
echo "Installing Transformers and dependencies..."
pip install --no-cache-dir transformers==4.36.0
pip install --no-cache-dir accelerate==0.25.0
pip install --no-cache-dir bitsandbytes==0.41.3
pip install --no-cache-dir peft==0.7.1
pip install --no-cache-dir datasets==2.15.0
pip install --no-cache-dir pillow
pip install --no-cache-dir av
pip install --no-cache-dir tqdm
pip install --no-cache-dir pandas
pip install --no-cache-dir matplotlib
pip install --no-cache-dir seaborn
pip install --no-cache-dir scipy
pip install --no-cache-dir scikit-learn

# Install profiling tools
echo "Installing profiling tools..."
pip install --no-cache-dir py3nvml
pip install --no-cache-dir gpustat

# Install YAML support
pip install --no-cache-dir pyyaml

# Optional: Install logging tools
echo "Installing optional logging tools..."
pip install --no-cache-dir tensorboard || echo "tensorboard install failed (optional)"
pip install --no-cache-dir wandb || echo "wandb install failed (optional)"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

if [ "$MODE" == "conda" ]; then
    echo "Environment: firstsight (conda)"
    echo "To activate: source activate firstsight"
else
    echo "Environment: $VENV_PATH (venv)"
    echo "To activate: source $VENV_PATH/bin/activate"
fi

echo ""
echo "Cache locations:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""
echo "Next steps:"
echo "  1. Run baseline: sbatch scripts/slurm/run_baseline.slurm"
echo "  2. Run quantization: sbatch scripts/slurm/run_quantization.slurm"
echo "  3. Run distillation: sbatch scripts/slurm/run_distillation.slurm"
echo ""

