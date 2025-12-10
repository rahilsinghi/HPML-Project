#!/bin/bash
#SBATCH --account=class
#SBATCH --job-name=firstsight_lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --mem=80GB
#SBATCH --partition=a100_2
#SBATCH --output=training_lora_%j.out
#SBATCH --error=training_lora_%j.err

echo "============================================"
echo "FirstSight - LoRA Fine-tuning (No Quantization)"
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
conda activate /scratch/${USER}/envs/firstsight

echo "Python: $(which python)"
echo "Version: $(python --version)"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "============================================"
nvidia-smi
echo "============================================"

mkdir -p results/training/lora
mkdir -p results/profiling

echo "Starting LoRA fine-tuning (no quantization)..."
echo "Using DeepSpeed ZeRO-3 (NOT DistributedDataParallel)"
deepspeed --num_gpus=4 \
    src/training/train.py \
    --config configs/training_config_lora.yaml \
    --mode lora \
    --deepspeed configs/deepspeed_zero3.json

if [ -d "results/training/lora" ]; then
    echo "Results saved in results/training/lora"
    ls -lh results/training/lora/
fi

echo "============================================"
echo "Job Complete: $(date)"
echo "============================================"

