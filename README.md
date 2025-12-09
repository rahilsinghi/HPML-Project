# FirstSight: Efficient Egocentric Question Answering

**Team:** Rahil Singhi (rs9174@nyu.edu) & Sunidhi Tandel (sdt9243@nyu.edu)  
**Course:** ECE-GY 9143 - High Performance Machine Learning  
**Institution:** NYU Tandon School of Engineering

---

## 📋 Project Overview

FirstSight builds an end-to-end pipeline for **efficient egocentric question answering** using Vision-Language Models (VLMs). The project focuses on:

1. **Knowledge Distillation**: Compress large egocentric VLMs (EgoGPT-7b) into efficient student models (Qwen2-VL-2B)
2. **Memory Optimization**: INT8 quantization and parameter-efficient training (LoRA)
3. **Edge Deployment**: Deploy compressed models on resource-constrained devices (AR glasses, mobile GPUs)

### Target Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Training Throughput | ≥2× improvement | 🔄 In Progress |
| VRAM Reduction | ≥30% savings | ✅ Achieved (INT8) |
| Accuracy Preservation | ≤1% degradation | 🔄 Testing |

---

## 🏗️ Repository Structure

```
firstsight/
├── src/                          # Source code
│   ├── distillation/             # Knowledge distillation (Phase 3)
│   │   ├── distill_vlm.py        # Main training loop
│   │   ├── models.py             # Model loading utilities
│   │   ├── data.py               # Data preparation
│   │   └── evaluate.py           # Evaluation & comparison
│   ├── baseline/                 # Baseline experiments (Phase 1)
│   │   ├── baseline_eval.py      # Profiling Qwen2-VL-2B
│   │   └── quantization_eval.py  # FP16 vs INT8 comparison
│   └── utils/                    # Shared utilities
│       └── generate_report.py    # Report generation
├── configs/                      # Configuration files
│   └── distillation_config.yaml  # Distillation hyperparameters
├── scripts/                      # Executable scripts
│   ├── hpc_setup.sh              # HPC environment setup
│   ├── upload_to_hpc.sh          # Upload files to HPC
│   └── slurm/                    # SLURM job configurations
│       ├── run_baseline.slurm
│       ├── run_quantization.slurm
│       └── run_distillation.slurm
├── experiments/                  # Experiment outputs
│   ├── baseline/
│   ├── quantization/
│   └── distillation/
├── docs/                         # Documentation
│   ├── midterm/                  # Archived mid-term reports
│   └── README_distillation.md    # Distillation guide
├── notebooks/                    # Jupyter notebooks (optional)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Access to NYU HPC Greene cluster
- Python 3.8+
- CUDA 11.3+ compatible GPU (A100 recommended)

### 1. Upload to HPC

```bash
# On your local machine
cd /path/to/firstsight
bash scripts/upload_to_hpc.sh --all
```

### 2. Setup Environment on HPC

```bash
# SSH to HPC
ssh rs9174@greene.hpc.nyu.edu

# Navigate to project
cd /scratch/rs9174/firstsight

# Setup environment (first time only)
bash scripts/hpc_setup.sh --mode venv

# This installs:
# - PyTorch 2.1.0 + CUDA 11.8
# - Transformers 4.36.0
# - PEFT, bitsandbytes, accelerate
# - Profiling tools
```

### 3. Run Experiments

#### Option A: Baseline Profiling

```bash
# Measure baseline performance
sbatch scripts/slurm/run_baseline.slurm

# Compare FP16 vs INT8 quantization
sbatch scripts/slurm/run_quantization.slurm
```

#### Option B: Knowledge Distillation (Main Focus)

```bash
# Run distillation: EgoGPT-7b → Qwen2-VL-2B
sbatch scripts/slurm/run_distillation.slurm

# Monitor progress
squeue -u rs9174
tail -f distillation_*.out
```

### 4. Download Results

```bash
# On your local machine
scp -r rs9174@greene.hpc.nyu.edu:/scratch/rs9174/firstsight/experiments ./
```

---

## 📊 Experiments

### Phase 1: Baseline & Quantization (Completed ✅)

**Baseline Profiling** (`src/baseline/baseline_eval.py`):
- Model: Qwen2-VL-2B-Instruct
- Metrics: Load time, VRAM, latency, throughput
- Results: `experiments/baseline/`

**Quantization** (`src/baseline/quantization_eval.py`):
- Comparison: FP16 vs INT8
- Expected savings: ~30-40% VRAM reduction
- Results: `experiments/quantization/`

### Phase 3: Knowledge Distillation (Current 🔄)

**Teacher Model**: [EgoGPT-7b-EgoIT](https://huggingface.co/EgoGPT/EgoGPT-7b-EgoIT) (9B params, egocentric-specialized)  
**Student Model**: Qwen2-VL-2B-Instruct (2B params, general VLM)

**Distillation Strategy**:
- Logit distillation (KL divergence)
- Temperature scaling (T=3.0)
- Mixed precision training (BF16)
- Gradient checkpointing

**Expected Results**:
- Compression: 9B → 2B (~78% reduction)
- Speedup: ~3-4× faster inference
- Performance: 85-90% of teacher accuracy
- VRAM: ~2.8GB (INT8 deployed)

See [`docs/README_distillation.md`](docs/README_distillation.md) for detailed guide.

---

## 🔧 Configuration

Edit [`configs/distillation_config.yaml`](configs/distillation_config.yaml) to customize:

```yaml
teacher:
  model_name: "EgoGPT/EgoGPT-7b-EgoIT"
  load_in_8bit: true  # Use INT8 to save memory

student:
  model_name: "Qwen/Qwen2-VL-2B-Instruct"
  torch_dtype: "bfloat16"
  gradient_checkpointing: true

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  temperature: 3.0
  alpha_logit: 0.7
```

---

## 📈 Results Summary

### Baseline Results (Qwen2-VL-2B)

| Metric | Value |
|--------|-------|
| Model VRAM | 4.2 GB (FP16) |
| Avg Latency | 0.52s per query |
| Throughput | 1.9 queries/sec |
| Peak VRAM | 5.1 GB |

### Quantization Results (FP16 vs INT8)

| Metric | FP16 | INT8 | Improvement |
|--------|------|------|-------------|
| VRAM | 4.2 GB | 2.8 GB | **33% ↓** |
| Latency | 0.52s | 0.48s | 8% faster |
| Throughput | 1.9 q/s | 2.1 q/s | 10% ↑ |

### Distillation Results (Teacher vs Student)

*(Results will be updated after distillation experiment)*

| Metric | Teacher (9B) | Student (2B) | Change |
|--------|--------------|--------------|--------|
| Parameters | 9B | 2B | 78% ↓ |
| VRAM | ~4.5 GB | ~2.8 GB | TBD |
| Latency | TBD | TBD | TBD |
| Accuracy | Baseline | TBD | TBD |

---

## 🛠️ Development

### Running Locally (Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Test data loading
python -m src.distillation.data

# Test model loading
python -m src.distillation.models

# Run small-scale distillation
python -m src.distillation.distill_vlm configs/distillation_config.yaml
```

### Adding New Features

1. Create new module in `src/`
2. Add configuration to `configs/`
3. Create SLURM job in `scripts/slurm/`
4. Update `scripts/upload_to_hpc.sh`
5. Document in `docs/`

---

## 📚 Key References

1. **EgoGPT**: Egocentric Vision-Language Model (Teacher)
   - [HuggingFace Model](https://huggingface.co/EgoGPT/EgoGPT-7b-EgoIT)
   
2. **Knowledge Distillation**:
   - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
   - Sanh et al., "DistilBERT" (2019)

3. **Modality-Balanced Quantization**:
   - Liu et al., "Modality Balanced Quantization for Large Vision-Language Models" (2024) - [arXiv:2412.19509](https://arxiv.org/abs/2412.19509)

4. **Parameter-Efficient Fine-Tuning**:
   - Hu et al., "LoRA: Low-Rank Adaptation" (2021) - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

5. **Quantization**:
   - Dettmers et al., "8-bit Optimizers via Block-wise Quantization" (2022) - [arXiv:2110.02861](https://arxiv.org/abs/2110.02861)

---

## 👥 Team Contributions

### Sunidhi Tandel
- Fine-tuning infrastructure
- CUDA-level optimizations
- Performance profiling
- Baseline evaluation scripts

### Rahil Singhi
- Knowledge distillation implementation
- HPC setup and SLURM configuration
- Model compression and quantization
- Documentation and project organization

### Joint Work
- Project design and methodology
- Experimental evaluation
- Literature review
- Report generation

---

## 📝 License

This project is developed for academic purposes as part of NYU's High Performance Machine Learning course.

---

## 🤝 Acknowledgments

- **NYU HPC Team** for compute resources
- **Hugging Face** for model hosting and transformers library
- **EgoGPT Team** for the egocentric VLM
- **Qwen Team** for Qwen2-VL models

---

## 📞 Contact

- Rahil Singhi: rs9174@nyu.edu
- Sunidhi Tandel: sdt9243@nyu.edu

For questions, issues, or collaboration opportunities, please reach out via email.

---

**Last Updated**: December 2024

