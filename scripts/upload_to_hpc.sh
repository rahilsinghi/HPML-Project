#!/bin/bash
# FirstSight: Consolidated HPC Upload Script
# Usage: bash upload_to_hpc.sh [--all|--baseline|--distillation]

set -e

echo "=========================================="
echo "FirstSight: Upload to HPC"
echo "=========================================="
echo ""

# Configuration
HPC_USER="rs9174"
HPC_HOST="greene.hpc.nyu.edu"
HPC_DIR="/scratch/${HPC_USER}/firstsight"

# Parse arguments
UPLOAD_MODE="all"
if [ $# -gt 0 ]; then
    case $1 in
        --all)
            UPLOAD_MODE="all"
            ;;
        --baseline)
            UPLOAD_MODE="baseline"
            ;;
        --distillation)
            UPLOAD_MODE="distillation"
            ;;
        *)
            echo "Usage: bash upload_to_hpc.sh [--all|--baseline|--distillation]"
            exit 1
            ;;
    esac
fi

echo "Upload mode: $UPLOAD_MODE"
echo "Target: ${HPC_USER}@${HPC_HOST}:${HPC_DIR}"
echo ""

# Create remote directory structure
echo "Creating remote directories..."
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_DIR}/{src/{baseline,distillation,utils},configs,scripts/slurm,experiments/{baseline,quantization,distillation}}"

# Upload setup script (always needed)
echo "Uploading setup script..."
scp scripts/hpc_setup.sh ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/scripts/

# Upload based on mode
if [ "$UPLOAD_MODE" == "all" ] || [ "$UPLOAD_MODE" == "baseline" ]; then
    echo "Uploading baseline scripts..."
    scp src/baseline/*.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/baseline/
    scp src/utils/*.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/utils/
    scp scripts/slurm/run_baseline.slurm ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/scripts/slurm/
    scp scripts/slurm/run_quantization.slurm ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/scripts/slurm/
fi

if [ "$UPLOAD_MODE" == "all" ] || [ "$UPLOAD_MODE" == "distillation" ]; then
    echo "Uploading distillation scripts..."
    scp src/distillation/*.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/distillation/ 2>/dev/null || echo "Note: distillation scripts not yet created"
    scp src/__init__.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/
    scp src/baseline/__init__.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/baseline/
    scp src/distillation/__init__.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/distillation/ 2>/dev/null || echo "Note: __init__.py will be uploaded when created"
    scp src/utils/__init__.py ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/src/utils/
    scp configs/distillation_config.yaml ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/configs/ 2>/dev/null || echo "Note: config will be uploaded when created"
    scp scripts/slurm/run_distillation.slurm ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/scripts/slurm/ 2>/dev/null || echo "Note: distillation SLURM job will be uploaded when created"
fi

# Upload requirements
if [ -f "requirements.txt" ]; then
    echo "Uploading requirements.txt..."
    scp requirements.txt ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/
fi

echo ""
echo "=========================================="
echo "✓ Upload complete!"
echo "=========================================="
echo ""
echo "Next steps on HPC:"
echo ""
echo "1. SSH to HPC:"
echo "   ssh ${HPC_USER}@${HPC_HOST}"
echo ""
echo "2. Navigate to project:"
echo "   cd ${HPC_DIR}"
echo ""
echo "3. Setup environment (first time only):"
echo "   bash scripts/hpc_setup.sh --mode venv"
echo ""
echo "4. Submit jobs:"
if [ "$UPLOAD_MODE" == "baseline" ]; then
    echo "   sbatch scripts/slurm/run_baseline.slurm"
    echo "   sbatch scripts/slurm/run_quantization.slurm"
elif [ "$UPLOAD_MODE" == "distillation" ]; then
    echo "   sbatch scripts/slurm/run_distillation.slurm"
else
    echo "   sbatch scripts/slurm/run_baseline.slurm"
    echo "   sbatch scripts/slurm/run_quantization.slurm"
    echo "   sbatch scripts/slurm/run_distillation.slurm"
fi
echo ""
echo "5. Monitor jobs:"
echo "   squeue -u ${HPC_USER}"
echo "   tail -f *.out"
echo ""
echo "=========================================="

