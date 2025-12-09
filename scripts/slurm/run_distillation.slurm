#!/bin/bash
#SBATCH --account=pr_289_general
#SBATCH --job-name=firstsight_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --output=distillation_%j.out
#SBATCH --error=distillation_%j.err

# FirstSight: Knowledge Distillation
# Distills EgoGPT-7b (teacher) → Qwen2-VL-2B (student)

echo "============================================"
echo "FirstSight - VLM Knowledge Distillation"
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
echo "  TMPDIR: $TMPDIR"
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
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""

# Verify configuration file exists
CONFIG_FILE="configs/distillation_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Configuration file: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p experiments/distillation

# Run distillation training
echo "============================================"
echo "Starting Knowledge Distillation"
echo "============================================"
echo ""
echo "Teacher: EgoGPT-7b-EgoIT (9B params)"
echo "Student: Qwen2-VL-2B-Instruct (2B params)"
echo ""

python -m src.distillation.distill_vlm $CONFIG_FILE

DISTILL_EXIT_CODE=$?

if [ $DISTILL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================"
    echo "⚠ Distillation failed with exit code: $DISTILL_EXIT_CODE"
    echo "============================================"
    exit $DISTILL_EXIT_CODE
fi

echo ""
echo "✓ Distillation training complete!"
echo ""

# Check if model was saved
BEST_MODEL_DIR="experiments/distillation/best_student_model"
if [ -d "$BEST_MODEL_DIR" ]; then
    echo "✓ Best model saved: $BEST_MODEL_DIR"
    ls -lh $BEST_MODEL_DIR
else
    echo "⚠ Warning: Best model directory not found"
fi

echo ""

# Run evaluation
echo "============================================"
echo "Running Evaluation"
echo "============================================"
echo ""

if [ -d "$BEST_MODEL_DIR" ]; then
    python -m src.distillation.evaluate $BEST_MODEL_DIR
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Evaluation complete!"
        
        # Display results summary if it exists
        SUMMARY_FILE="experiments/distillation/evaluation_summary.txt"
        if [ -f "$SUMMARY_FILE" ]; then
            echo ""
            echo "============================================"
            echo "Evaluation Summary:"
            echo "============================================"
            cat $SUMMARY_FILE
        fi
    else
        echo ""
        echo "⚠ Evaluation failed with exit code: $EVAL_EXIT_CODE"
    fi
else
    echo "⚠ Skipping evaluation (no model found)"
fi

echo ""
echo "============================================"
echo "Job Complete!"
echo "End Time: $(date)"
echo "============================================"
echo ""

# Print final summary
echo "Output files:"
ls -lh experiments/distillation/

# Memory usage summary
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used --format=csv

