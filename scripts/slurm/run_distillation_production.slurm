#!/bin/bash
#SBATCH --account=pr_289_general
#SBATCH --job-name=firstsight_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --output=distillation_production_%j.out
#SBATCH --error=distillation_production_%j.err

echo "============================================"
echo "FirstSight - VLM Knowledge Distillation"
echo "PRODUCTION RUN: 5000 samples, 10 epochs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

# Load modules
module purge
module load anaconda3/2020.07
module load cuda/11.3.1
module load gcc/10.2.0

# Set cache directories to scratch space
export TRANSFORMERS_CACHE=/scratch/${USER}/transformers
export HF_HOME=/scratch/${USER}/huggingface
export TORCH_HOME=/scratch/${USER}/torch
export PYTHONNOUSERSITE=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/rs9174/envs/firstsight

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "============================================"
echo "GPU Information:"
nvidia-smi
echo "============================================"

# Create output directory
mkdir -p experiments/distillation_production

# Run distillation with production config
echo "Starting PRODUCTION distillation:"
echo "  Teacher: Qwen/Qwen2-VL-7B-Instruct"
echo "  Student: Qwen/Qwen2-VL-2B-Instruct"
echo "  Dataset: 5000 train / 1000 val samples"
echo "  Epochs: 10"
echo "  Expected time: 2-3 hours"
echo ""

python -m src.distillation.distill_vlm configs/distillation_config_production.yaml

# Run evaluation if model was saved
if [ -d "experiments/distillation_production/best_student_model" ]; then
    echo ""
    echo "============================================"
    echo "Running evaluation on production model..."
    echo "============================================"
    python -m src.distillation.evaluate experiments/distillation_production/best_student_model
fi

echo "============================================"
echo "Job Complete: $(date)"
echo "============================================"

