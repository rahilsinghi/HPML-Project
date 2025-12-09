#!/bin/bash
#SBATCH --account=pr_289_general
#SBATCH --job-name=firstsight_1k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --output=distillation_1k_%j.out
#SBATCH --error=distillation_1k_%j.err

echo "============================================"
echo "FirstSight - VLM Knowledge Distillation"
echo "1K SAMPLES: 1000 train, 200 val, 10 epochs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

# Load modules
module purge
module load anaconda3/2020.07
module load cuda/11.3.1
module load gcc/10.2.0

# Set cache directories
export TRANSFORMERS_CACHE=/scratch/${USER}/transformers
export HF_HOME=/scratch/${USER}/huggingface
export TORCH_HOME=/scratch/${USER}/torch
export PYTHONNOUSERSITE=1

# Activate conda
eval "$(conda shell.bash hook)"
conda activate /scratch/rs9174/envs/firstsight

echo "Python: $(which python)"
echo "Version: $(python --version)"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "============================================"
nvidia-smi
echo "============================================"

mkdir -p experiments/distillation_1k

echo "Starting 1K distillation run..."
python -m src.distillation.distill_vlm configs/distillation_config_1k.yaml

if [ -d "experiments/distillation_1k/best_student_model" ]; then
    echo "Running evaluation..."
    python -m src.distillation.evaluate experiments/distillation_1k/best_student_model
fi

echo "============================================"
echo "Job Complete: $(date)"
echo "============================================"

