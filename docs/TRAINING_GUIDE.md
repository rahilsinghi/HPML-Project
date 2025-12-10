# Training Guide: LLaVA-OneVision Fine-tuning on Ego4D

High-performance fine-tuning pipeline for LLaVA-OneVision on Ego4D dataset.

## Overview

Two training modes:
- **Baseline (M0)**: Evaluation only, no fine-tuning
- **QLoRA (M1)**: 4-bit NF4 quantization + LoRA fine-tuning for 10 epochs

## Quick Start

### 1. Configure Dataset

Edit `configs/training_config.yaml`:
```yaml
data:
  data_path: "data/FirstSight-Dataset/datasets/Ego4D/Ego4D.json"
  video_folder: "data/FirstSight-Dataset/Ego4d"
  video_ids_file: null  # Optional: path to .txt with video IDs (one per line)
```

### 2. Test on CPU (Optional - for local testing)

```bash
# Run 1 epoch on CPU with real data
python scripts/run_training_cpu.py

# Or directly:
python -m src.training.train \
    --config configs/training_config.yaml \
    --mode qlora \
    --use_cpu
```

**Note:** The training pipeline uses REAL Ego4D data by default. No dummy data generation needed.

### 3. Run Baseline Evaluation

**Local:**
```bash
python -m src.training.train \
    --config configs/training_config.yaml \
    --mode baseline \
    --deepspeed configs/deepspeed_zero3.json
```

**HPC:**
```bash
sbatch scripts/slurm/run_training_baseline.slurm
```

### 4. Run QLoRA Fine-tuning

**Local:**
```bash
python -m src.training.train \
    --config configs/training_config.yaml \
    --mode qlora \
    --deepspeed configs/deepspeed_zero3.json
```

**HPC:**
```bash
sbatch scripts/slurm/run_training_qlora.slurm
```

## Dataset Format

Ego4D JSON format:
```json
[
  {
    "id": "video_id",
    "video": "path/to/video.mp4",
    "audio": "path/to/audio.mp3",  // Optional
    "conversations": [
      {"from": "human", "value": "<speech>\n<image>\nQuestion"},
      {"from": "gpt", "value": "Answer"}
    ]
  }
]
```

### Filtering by Video IDs

Create a `.txt` file with video IDs (one per line):
```
52fb1a73-2ba7-413d-8596-33615daecc73-1
dda77c62-2b60-4a66-baef-ebc25ff7bb79
```

Set in config:
```yaml
data:
  video_ids_file: "data/video_ids.txt"
```

## Outputs

### Baseline Mode
- `results/training/baseline/baseline_results.json` - Evaluation metrics
- `results/profiling/baseline_profiling.json` - Profiling metrics

### LoRA Mode
- `results/training/lora/final_model/` - Fine-tuned model
- `results/training/lora/training_results.json` - Training metrics
- `results/profiling/lora_profiling.json` - Profiling metrics

### QLoRA Mode
- `results/training/qlora/final_model/` - Fine-tuned model
- `results/training/qlora/training_results.json` - Training metrics
- `results/profiling/qlora_profiling.json` - Profiling metrics

## Profiling Metrics

The profiler tracks:
- Wall clock time (total, epoch, step)
- Forward/backward/optimizer step times (CUDA events)
- Tokens/samples per second
- Peak GPU memory usage
- Step time statistics (mean, median, p95)

## Configuration

All settings in `configs/training_config.yaml`:
- Model: `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- LoRA hyperparameters
- Training hyperparameters
- Dataset paths
- FlashAttention flag

## Requirements

- PyTorch 2.1+
- Transformers 4.36+
- PEFT 0.7+
- bitsandbytes 0.41+
- DeepSpeed
- decord (for video loading)
- librosa (for audio loading)

## Monitoring Jobs on HPC

```bash
# Check status
squeue -u $USER

# View output
tail -f training_baseline_*.out
tail -f training_qlora_*.out

# View errors
tail -f training_*.err
```

