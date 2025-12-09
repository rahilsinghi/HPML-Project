# Knowledge Distillation Guide for FirstSight

Complete guide for implementing VLM knowledge distillation from EgoGPT-7b to Qwen2-VL-2B.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Running Distillation](#running-distillation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Evaluation](#evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Expected Results](#expected-results)

---

## Overview

### Goal

Compress a large egocentric VLM (EgoGPT-7b, 9B parameters) into a smaller, efficient student model (Qwen2-VL-2B, 2B parameters) while preserving egocentric QA performance.

### Key Benefits

- **Compression**: 78% parameter reduction (9B → 2B)
- **Speed**: 3-4× faster inference
- **Memory**: ~40% VRAM reduction
- **Deployment**: Enables edge device deployment

### Timeline

- **Setup**: 30 minutes
- **Training**: 8-10 hours (3 epochs)
- **Evaluation**: 1 hour
- **Total**: ~12 hours (single A100 GPU)

---

## Theoretical Background

### Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model:

1. **Soft Targets**: Teacher's output probabilities (with temperature scaling)
2. **Hard Targets**: Ground truth labels (if available)
3. **Combined Loss**: Weighted sum of soft and hard losses

### Loss Function

```
L_total = α · L_distill + (1 - α) · L_hard

Where:
- L_distill = KL(P_teacher || P_student) · T²
- L_hard = CrossEntropy(P_student, labels)
- α = distillation weight (0.7 recommended)
- T = temperature (3.0 recommended)
```

### Temperature Scaling

Temperature T "softens" the probability distribution:

```python
P_soft = softmax(logits / T)
```

- **T = 1**: Standard softmax (hard predictions)
- **T > 1**: Softer probabilities (reveals "dark knowledge")
- **T = 3-4**: Good for VLM distillation

### Why It Works for VLMs

1. **Teacher's soft probabilities** reveal similarity between tokens
2. **Visual-linguistic alignment** is partially preserved
3. **Egocentric patterns** learned by teacher transfer to student
4. **Student learns "how to think"** not just "what to answer"

---

## Architecture

### Teacher Model: EgoGPT-7b-EgoIT

- **Parameters**: 9B (estimated from architecture)
- **Specialization**: Fine-tuned on egocentric data (Ego4D)
- **Strengths**: Strong egocentric reasoning, spatial understanding
- **Format**: Causal LM with vision encoder

**Loading Strategy**:
- INT8 quantization (via bitsandbytes)
- Frozen parameters (no gradient computation)
- Auto device mapping

### Student Model: Qwen2-VL-2B-Instruct

- **Parameters**: 2B
- **Base**: General-purpose VLM
- **Trainable**: All parameters (or LoRA if memory-constrained)
- **Format**: Causal LM with vision encoder

**Training Strategy**:
- BF16 mixed precision
- Gradient checkpointing
- Small batch size + gradient accumulation

---

## Implementation Details

### File Structure

```
src/distillation/
├── distill_vlm.py    # Main training loop
├── models.py         # Model loading utilities
├── data.py           # Data preparation
└── evaluate.py       # Evaluation & comparison
```

### Key Functions

#### 1. Model Loading (`models.py`)

```python
teacher, teacher_processor = load_teacher_model(
    model_name="EgoGPT/EgoGPT-7b-EgoIT",
    load_in_8bit=True  # INT8 quantization
)

student, student_processor = load_student_model(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    gradient_checkpointing=True
)
```

#### 2. Data Preparation (`data.py`)

```python
train_loader, val_loader = create_train_val_split(
    train_samples=400,  # Manageable for 1-day training
    val_samples=100,
    batch_size=2,
    processor=student_processor
)
```

#### 3. Distillation Forward Pass (`distill_vlm.py`)

```python
outputs = distill_forward(
    teacher_model=teacher,
    student_model=student,
    inputs=batch,
    temperature=3.0,
    alpha=0.7
)

# Returns:
# - loss: Combined distillation + hard label loss
# - distill_loss: KL divergence
# - hard_loss: Cross-entropy (if labels available)
```

#### 4. Training Epoch

```python
train_stats = train_epoch(
    teacher_model=teacher,
    student_model=student,
    dataloader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,  # For mixed precision
    config=config,
    epoch=epoch
)
```

### Memory Optimization

To fit both models on a single A100 (80GB):

1. **Teacher in INT8**: ~4.5GB VRAM
2. **Student in BF16**: ~4GB VRAM
3. **Gradient checkpointing**: Saves ~30% memory
4. **Batch size = 2**: Reduces activation memory
5. **Gradient accumulation = 8**: Effective batch size = 16

**Expected total VRAM**: 25-35GB

---

## Running Distillation

### Step 1: Configuration

Edit `configs/distillation_config.yaml`:

```yaml
training:
  epochs: 3              # 3 epochs is sufficient
  batch_size: 2          # Small for memory efficiency
  learning_rate: 1.0e-4  # Conservative LR
  temperature: 3.0       # Standard for VLM distillation
  alpha_logit: 0.7       # Favor distillation over hard labels
```

### Step 2: Upload to HPC

```bash
# On local machine
bash scripts/upload_to_hpc.sh --distillation
```

### Step 3: Submit Job

```bash
# On HPC
cd /scratch/rs9174/firstsight
sbatch scripts/slurm/run_distillation.slurm
```

### Step 4: Monitor

```bash
# Check queue
squeue -u rs9174

# View output
tail -f distillation_*.out

# Check errors
tail -f distillation_*.err

# GPU usage
nvidia-smi
```

### Step 5: Check Results

After completion, check:

```bash
ls experiments/distillation/

# Expected files:
# - best_student_model/       (best checkpoint)
# - final_student_model/      (final checkpoint)
# - training_history_*.json   (loss curves)
# - evaluation_*.json         (comparison results)
# - evaluation_summary.txt    (human-readable summary)
```

---

## Hyperparameter Tuning

### Temperature (T)

**Effect**: Controls how much "dark knowledge" transfers

- **T = 1**: No softening (equivalent to standard training)
- **T = 2-3**: Mild softening (good for similar models)
- **T = 3-4**: Strong softening (good for large teacher-student gap)
- **T > 5**: Too soft (may hurt performance)

**Recommendation**: Start with T=3.0

### Distillation Weight (α)

**Effect**: Balances distillation vs. hard label loss

- **α = 1.0**: Pure distillation (no hard labels)
- **α = 0.7**: Favor distillation, some hard supervision
- **α = 0.5**: Equal weight
- **α = 0.3**: Favor hard labels

**Recommendation**: α=0.7 for VLM distillation

### Learning Rate

**Effect**: Training speed and stability

- **Too high** (>1e-3): Unstable, may diverge
- **Optimal** (1e-4 to 5e-4): Stable convergence
- **Too low** (<1e-5): Slow convergence

**Recommendation**: 1e-4 with cosine decay

### Batch Size & Gradient Accumulation

**Trade-off**: Memory vs. training time

- **Large batch** (32+): Better gradients, needs more VRAM
- **Small batch** (2-4): Fits in memory, noisier gradients
- **Solution**: Gradient accumulation (effective batch = batch_size × accumulation_steps)

**Recommendation**: batch_size=2, accumulation_steps=8 (effective=16)

---

## Evaluation

### Automatic Evaluation

The SLURM job automatically runs evaluation after training:

```bash
# Loads best_student_model
# Compares teacher vs student on test set
# Generates comparison report
```

### Manual Evaluation

```bash
# Activate environment
source /scratch/${USER}/envs/firstsight/bin/activate

# Run evaluation
python -m src.distillation.evaluate experiments/distillation/best_student_model
```

### Metrics

**Model Size**:
- Parameter count
- Compression ratio
- Memory footprint

**Performance**:
- Inference latency (seconds per query)
- Throughput (queries per second)
- VRAM usage

**Accuracy** (if ground truth available):
- Exact Match (EM)
- F1 Score
- BLEU/ROUGE

### Expected Output

```
==========================================
COMPARISON: Teacher vs Student
==========================================

MODEL SIZE & COMPRESSION:
  Teacher parameters: 9.00B
  Student parameters: 2.00B
  Compression ratio: 4.50×
  Size reduction: 77.8%

MEMORY USAGE:
  Teacher VRAM: 4.50 GB
  Student VRAM: 2.80 GB
  VRAM savings: 1.70 GB (37.8%)

PERFORMANCE:
  Teacher latency: 0.620s
  Student latency: 0.180s
  Speedup: 3.44×
  Throughput improvement: 3.44×
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size: `batch_size: 1`
2. Enable gradient checkpointing (already on)
3. Use INT8 for both models
4. Request A100 80GB (not 40GB)

**Edit config**:
```yaml
training:
  batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 16  # Increase to maintain effective batch
```

### Issue: Job Fails Immediately

**Symptoms**:
```
Error: Virtual environment not found
```

**Solution**:
```bash
# Setup environment first
bash scripts/hpc_setup.sh --mode venv
```

### Issue: NaN Loss

**Symptoms**:
```
Epoch 1, Batch 10 | Loss: nan
```

**Causes & Solutions**:
1. **Learning rate too high**: Reduce to 5e-5
2. **Gradient explosion**: Check gradient clipping (max_norm=1.0)
3. **Mixed precision instability**: Switch to FP32 (slower but stable)

**Edit config**:
```yaml
training:
  learning_rate: 5.0e-5  # Reduce from 1e-4
```

### Issue: Slow Training

**Symptoms**:
```
Time per epoch: >6 hours
```

**Optimization**:
1. Increase batch size if memory allows
2. Reduce num_workers in dataloader
3. Use fewer training samples for debugging

**Quick test config**:
```yaml
data:
  train_samples: 100  # Reduce from 400
  val_samples: 20     # Reduce from 100
```

### Issue: Model Not Saved

**Symptoms**:
```
⚠ Warning: Best model directory not found
```

**Check**:
```bash
# Verify output directory was created
ls -lh experiments/distillation/

# Check disk space
df -h /scratch/${USER}

# Check permissions
ls -ld experiments/distillation/
```

---

## Expected Results

### Timeline (Single A100 80GB)

| Phase | Duration | Notes |
|-------|----------|-------|
| Model download | 5-10 min | First time only (cached after) |
| Epoch 1 | 2-3 hours | Includes validation |
| Epoch 2 | 2-3 hours | |
| Epoch 3 | 2-3 hours | |
| Evaluation | 30-60 min | |
| **Total** | **~10 hours** | Within 12-hour SLURM limit |

### Performance Targets

| Metric | Teacher | Student | Target |
|--------|---------|---------|--------|
| Parameters | 9B | 2B | 4.5× compression |
| VRAM | 4.5 GB | 2.8 GB | 38% reduction |
| Latency | 0.6s | 0.18s | 3-4× faster |
| Accuracy | 100% | 85-90% | ≤10% drop |

### Success Criteria

✅ **Minimum Viable**:
- Training completes without errors
- Student model saved successfully
- Compression ratio > 3×
- Speedup > 2×

✅ **Target**:
- Compression ratio ≈ 4.5×
- VRAM reduction ≥ 30%
- Speedup ≥ 3×
- Accuracy retention ≥ 85%

✅ **Stretch Goal**:
- Compression ratio ≈ 4.5×
- VRAM reduction ≥ 40%
- Speedup ≥ 4×
- Accuracy retention ≥ 90%

---

## Next Steps

### After Successful Distillation

1. **Deploy Student Model**:
   - Export to ONNX/TensorRT
   - Test on edge GPU (RTX 4090)
   - Measure real-world latency

2. **Further Compression**:
   - Apply INT8 quantization to student
   - Test W4A8 (4-bit weights, 8-bit activations)
   - Combine with pruning

3. **Fine-tuning** (Optional):
   - Fine-tune student on task-specific data
   - Apply LoRA for parameter efficiency
   - Evaluate on full EgoSchema benchmark

4. **Integration**:
   - Combine with teammate's fine-tuning work
   - Create unified pipeline
   - Prepare final demo

---

## References

### Papers

1. **Knowledge Distillation**:
   - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
   - [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

2. **DistilBERT** (NLP distillation):
   - Sanh et al., "DistilBERT, a distilled version of BERT" (2019)
   - [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

3. **VLM Compression**:
   - Liu et al., "Modality Balanced Quantization for Large Vision-Language Models" (2024)
   - [arXiv:2412.19509](https://arxiv.org/abs/2412.19509)

### Tools & Frameworks

- **Hugging Face Transformers**: Model loading and training
- **bitsandbytes**: INT8 quantization
- **PyTorch**: Deep learning framework
- **PEFT**: Parameter-efficient fine-tuning (future work)

---

## Contact

For questions or issues:
- Rahil Singhi: rs9174@nyu.edu
- Sunidhi Tandel: sdt9243@nyu.edu

---

**Last Updated**: December 2024

