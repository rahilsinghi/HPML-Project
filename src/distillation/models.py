"""
FirstSight: Model Loading Utilities for Knowledge Distillation
Teacher: Qwen2-VL-7B-Instruct (8.3B params)
Student: Qwen2-VL-2B-Instruct (2.2B params)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_teacher_model(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    load_in_8bit: bool = True,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = torch.bfloat16
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """
    Load Qwen2-VL-7B teacher model with INT8 quantization for memory efficiency.
    
    Args:
        model_name: HuggingFace model identifier
        load_in_8bit: Enable INT8 quantization (recommended for teacher)
        device_map: Device mapping strategy
        torch_dtype: Data type for non-quantized tensors
    
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading teacher model: {model_name}")
    logger.info(f"INT8 quantization: {load_in_8bit}")
    
    try:
        if load_in_8bit:
            # Use bitsandbytes for INT8 quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set teacher model to eval mode (no gradient computation)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("✓ Teacher model loaded successfully")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        logger.info(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  VRAM: {memory_allocated:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        raise


def load_student_model(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """
    Load Qwen2-VL-2B student model for distillation training.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy
        torch_dtype: Data type (BF16 recommended for training)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading student model: {model_name}")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        
        # Set model to training mode
        model.train()
        
        logger.info("✓ Student model loaded successfully")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        logger.info(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  VRAM: {memory_allocated:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load student model: {e}")
        raise


def get_model_info(model) -> dict:
    """
    Get model information for logging and debugging.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "size_gb": total_params * 4 / 1e9,  # Assuming FP32
        "is_training": model.training
    }
    
    if torch.cuda.is_available():
        info["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    
    return info


def print_model_summary(teacher_model, student_model):
    """
    Print comprehensive summary of teacher and student models.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
    """
    logger.info("=" * 70)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 70)
    
    teacher_info = get_model_info(teacher_model)
    student_info = get_model_info(student_model)
    
    logger.info("\nTeacher Model (EgoGPT-7b):")
    logger.info(f"  Total parameters: {teacher_info['total_params'] / 1e9:.2f}B")
    logger.info(f"  Trainable parameters: {teacher_info['trainable_params'] / 1e6:.2f}M")
    logger.info(f"  Training mode: {teacher_info['is_training']}")
    if "vram_allocated_gb" in teacher_info:
        logger.info(f"  VRAM allocated: {teacher_info['vram_allocated_gb']:.2f} GB")
    
    logger.info("\nStudent Model (Qwen2-VL-2B):")
    logger.info(f"  Total parameters: {student_info['total_params'] / 1e9:.2f}B")
    logger.info(f"  Trainable parameters: {student_info['trainable_params'] / 1e6:.2f}M")
    logger.info(f"  Training mode: {student_info['is_training']}")
    if "vram_allocated_gb" in student_info:
        logger.info(f"  VRAM allocated: {student_info['vram_allocated_gb']:.2f} GB")
    
    compression_ratio = teacher_info['total_params'] / student_info['total_params']
    logger.info(f"\nCompression ratio: {compression_ratio:.2f}×")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Test model loading
    logger.info("Testing model loading...")
    
    teacher, teacher_processor = load_teacher_model()
    student, student_processor = load_student_model()
    
    print_model_summary(teacher, student)
    
    logger.info("\n✓ All models loaded successfully!")

