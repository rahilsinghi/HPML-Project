"""
FirstSight: Knowledge Distillation for Vision-Language Models
Main training loop for distilling EgoGPT-7b → Qwen2-VL-2B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast
from transformers import get_cosine_schedule_with_warmup
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging
import yaml

from .models import load_teacher_model, load_student_model, print_model_summary
from .data import create_train_val_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def distill_forward(
    teacher_model,
    student_model,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 3.0,
    alpha: float = 0.7,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Forward pass for knowledge distillation.
    
    Computes:
    - Logit distillation loss (KL divergence)
    - Optional hard label loss
    
    Args:
        teacher_model: Teacher model (frozen)
        student_model: Student model (trainable)
        inputs: Batch of inputs
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss (1-alpha for hard loss)
        device: Device to use
    
    Returns:
        Dictionary with losses and logits
    """
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Teacher forward pass (no gradients)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        teacher_logits = teacher_outputs.logits
    
    # Student forward pass (with gradients)
    student_outputs = student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    student_logits = student_outputs.logits
    
    # Handle vocabulary size mismatch between teacher and student
    # Align vocabularies by taking the minimum size
    vocab_size_teacher = teacher_logits.size(-1)
    vocab_size_student = student_logits.size(-1)
    min_vocab_size = min(vocab_size_teacher, vocab_size_student)
    
    if vocab_size_teacher != vocab_size_student:
        logger.debug(f"Vocab size mismatch: teacher={vocab_size_teacher}, student={vocab_size_student}")
        logger.debug(f"Aligning to min vocab size: {min_vocab_size}")
        teacher_logits = teacher_logits[..., :min_vocab_size]
        student_logits = student_logits[..., :min_vocab_size]
    
    # Compute distillation loss (KL divergence with temperature scaling)
    # Use log_softmax for numerical stability
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss
    distill_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Compute hard label loss if labels are provided
    hard_loss = torch.tensor(0.0, device=device)
    if "labels" in inputs and inputs["labels"] is not None:
        labels = inputs["labels"].to(device)
        # Ensure labels don't exceed aligned vocabulary size
        labels_masked = torch.where(labels >= min_vocab_size, -100, labels)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels_masked.view(-1),
            ignore_index=-100
        )
    
    # Combined loss
    total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
    
    return {
        "loss": total_loss,
        "distill_loss": distill_loss,
        "hard_loss": hard_loss,
        "teacher_logits": teacher_logits,
        "student_logits": student_logits
    }


def train_epoch(
    teacher_model,
    student_model,
    dataloader,
    optimizer,
    scheduler,
    config: Dict,
    epoch: int,
    device: str = "cuda"
) -> Dict:
    """
    Train for one epoch.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dictionary
        epoch: Current epoch number
        device: Device to use
    
    Returns:
        Dictionary with epoch statistics
    """
    student_model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    total_distill_loss = 0.0
    total_hard_loss = 0.0
    num_batches = 0
    
    temperature = config["training"]["temperature"]
    alpha = config["training"]["alpha_logit"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch}")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass with mixed precision (bfloat16)
        with autocast(dtype=torch.bfloat16):
            outputs = distill_forward(
                teacher_model=teacher_model,
                student_model=student_model,
                inputs=batch,
                temperature=temperature,
                alpha=alpha,
                device=device
            )
            
            loss = outputs["loss"] / gradient_accumulation_steps
        
        # Backward pass (no scaler needed for bfloat16)
        loss.backward()
        
        # Accumulate losses
        total_loss += outputs["loss"].item()
        total_distill_loss += outputs["distill_loss"].item()
        total_hard_loss += outputs["hard_loss"].item()
        num_batches += 1
        
        # Update weights after accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log progress
            if (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Distill: {total_distill_loss/num_batches:.4f} | "
                    f"LR: {lr:.2e}"
                )
        
        # Memory cleanup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_distill_loss = total_distill_loss / num_batches
    avg_hard_loss = total_hard_loss / num_batches
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch} Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Time: {epoch_time:.2f}s")
    logger.info(f"Avg Loss: {avg_loss:.4f}")
    logger.info(f"Avg Distill Loss: {avg_distill_loss:.4f}")
    logger.info(f"Avg Hard Loss: {avg_hard_loss:.4f}")
    logger.info(f"{'='*70}\n")
    
    return {
        "epoch": epoch,
        "loss": avg_loss,
        "distill_loss": avg_distill_loss,
        "hard_loss": avg_hard_loss,
        "time": epoch_time
    }


def validate(
    teacher_model,
    student_model,
    dataloader,
    config: Dict,
    device: str = "cuda"
) -> Dict:
    """
    Validate the student model.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataloader: Validation dataloader
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Dictionary with validation statistics
    """
    student_model.eval()
    teacher_model.eval()
    
    total_loss = 0.0
    total_distill_loss = 0.0
    num_batches = 0
    
    temperature = config["training"]["temperature"]
    alpha = config["training"]["alpha_logit"]
    
    logger.info("Running validation...")
    
    with torch.no_grad():
        for batch in dataloader:
            with autocast():
                outputs = distill_forward(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    inputs=batch,
                    temperature=temperature,
                    alpha=alpha,
                    device=device
                )
            
            total_loss += outputs["loss"].item()
            total_distill_loss += outputs["distill_loss"].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_distill_loss = total_distill_loss / num_batches
    
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation Distill Loss: {avg_distill_loss:.4f}\n")
    
    return {
        "val_loss": avg_loss,
        "val_distill_loss": avg_distill_loss
    }


def train_distillation(config_path: str = "configs/distillation_config.yaml"):
    """
    Main training function for VLM distillation.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("="*70)
    logger.info("FirstSight: VLM Knowledge Distillation")
    logger.info("="*70)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"\nConfiguration loaded from: {config_path}")
    logger.info(f"Teacher: {config['teacher']['model_name']}")
    logger.info(f"Student: {config['student']['model_name']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load models
    logger.info("\n" + "="*70)
    logger.info("Loading Models")
    logger.info("="*70)
    
    teacher_model, teacher_processor = load_teacher_model(
        model_name=config["teacher"]["model_name"],
        load_in_8bit=config["teacher"]["load_in_8bit"]
    )
    
    student_model, student_processor = load_student_model(
        model_name=config["student"]["model_name"],
        torch_dtype=getattr(torch, config["student"]["torch_dtype"])
    )
    
    print_model_summary(teacher_model, student_model)
    
    # Create dataloaders
    logger.info("\n" + "="*70)
    logger.info("Preparing Data")
    logger.info("="*70)
    
    train_loader, val_loader = create_train_val_split(
        data_path=config["data"].get("data_path"),
        train_data_path=config["data"].get("train_data_path"),
        val_data_path=config["data"].get("val_data_path"),
        train_samples=config["data"]["train_samples"],
        val_samples=config["data"]["val_samples"],
        batch_size=config["training"]["batch_size"],
        processor=student_processor  # Use student processor
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * config["training"]["epochs"] // config["training"]["gradient_accumulation_steps"]
    warmup_steps = config["training"]["warmup_steps"]
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info("Using bfloat16 mixed precision (no GradScaler needed)")
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting Training")
    logger.info("="*70)
    
    training_history = []
    best_val_loss = float('inf')
    
    # Get output directory from config
    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_stats = train_epoch(
            teacher_model=teacher_model,
            student_model=student_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            device=device
        )
        
        # Validate
        val_stats = validate(
            teacher_model=teacher_model,
            student_model=student_model,
            dataloader=val_loader,
            config=config,
            device=device
        )
        
        # Combine stats
        epoch_stats = {**train_stats, **val_stats}
        training_history.append(epoch_stats)
        
        # Save checkpoint if best
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            checkpoint_path = output_dir / config["output"]["best_model_name"]
            student_model.save_pretrained(checkpoint_path)
            student_processor.save_pretrained(checkpoint_path)
            logger.info(f"✓ Saved best model checkpoint: {checkpoint_path}")
        
        # Save training history
        history_path = output_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    # Final save
    final_model_path = output_dir / config["output"]["final_model_name"]
    student_model.save_pretrained(final_model_path)
    student_processor.save_pretrained(final_model_path)
    logger.info(f"\n✓ Training complete! Final model saved: {final_model_path}")
    
    logger.info("\n" + "="*70)
    logger.info("Training Summary")
    logger.info("="*70)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total epochs: {config['training']['epochs']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/distillation_config.yaml"
    train_distillation(config_path)

