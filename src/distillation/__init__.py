"""
Knowledge distillation module for VLMs
Teacher: EgoGPT-7b-EgoIT
Student: Qwen2-VL-2B-Instruct
"""

from .distill_vlm import distill_forward, train_epoch
from .models import load_teacher_model, load_student_model
from .data import create_egocentric_dataloader

__all__ = [
    'distill_forward',
    'train_epoch',
    'load_teacher_model',
    'load_student_model',
    'create_egocentric_dataloader'
]

