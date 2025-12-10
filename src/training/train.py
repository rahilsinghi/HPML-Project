"""
Main training script for LLaVA-OneVision fine-tuning.
Supports Baseline (evaluation) and QLoRA fine-tuning modes.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. QLoRA mode will not work.")

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. QLoRA quantization will not work.")

from .dataset import Ego4DDataset
from .profiling import TrainingProfiler


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_processor(
    model_name: str,
    use_qlora: bool = False,
    use_flash_attn: bool = False,
    torch_dtype: str = "bfloat16",
    use_cpu: bool = False,
):
    """
    Load model and processor (GPU only).
    
    Args:
        model_name: HuggingFace model name
        use_qlora: Whether to use QLoRA (4-bit quantization)
        use_flash_attn: Whether to use FlashAttention
        torch_dtype: Torch dtype (bfloat16/float16)
        use_cpu: Not used (kept for compatibility, always uses GPU)
    """
    # Set dtype for GPU
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Configure quantization for QLoRA
    if use_qlora and BITSANDBYTES_AVAILABLE:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
        if use_qlora and not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes")
    
    # Load model on GPU
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Enable FlashAttention if requested
    if use_flash_attn:
        try:
            if hasattr(model.config, "use_flash_attention_2"):
                model.config.use_flash_attention_2 = True
            print("FlashAttention enabled")
        except Exception as e:
            print(f"Warning: Could not enable FlashAttention: {e}")
    
    print("Model loaded on GPU")
    
    return model, processor


def setup_lora(model, config: Dict):
    """Setup LoRA adapters."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules", None),  # Auto-detect if None
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple exact match
    exact_match = sum(1 for p, l in zip(pred_str, label_str) if p.strip() == l.strip())
    accuracy = exact_match / len(pred_str) if pred_str else 0.0
    
    return {"accuracy": accuracy, "exact_match": exact_match}


def main():
    parser = argparse.ArgumentParser(description="Train LLaVA-OneVision on Ego4D")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["baseline", "lora", "qlora"], default="qlora")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]
    
    # Determine training mode
    use_qlora = args.mode == "qlora"
    use_lora = args.mode == "lora"
    use_flash_attn = config.get("use_flash_attn", False)
    
    print(f"Mode: {args.mode}")
    print(f"LoRA: {use_lora}")
    print(f"QLoRA: {use_qlora}")
    print(f"FlashAttention: {use_flash_attn}")
    
    # Load model and processor (GPU only)
    model, processor = load_model_and_processor(
        model_name=model_config["model_name"],
        use_qlora=use_qlora,  # QLoRA uses quantization
        use_flash_attn=use_flash_attn,
        torch_dtype=model_config.get("torch_dtype", "bfloat16"),
        use_cpu=False,  # GPU only
    )
    
    # Setup LoRA if LoRA or QLoRA mode
    if use_lora or use_qlora:
        model = setup_lora(model, config.get("lora", {}))
    
    # Load dataset (from HuggingFace or local file)
    dataset = Ego4DDataset(
        data_path=data_config.get("data_path"),
        processor=processor,
        video_ids_file=data_config.get("video_ids_file"),
        max_frames=data_config.get("max_frames", 8),
        fps=data_config.get("fps", 1),
        video_folder=data_config.get("video_folder"),
        audio_folder=data_config.get("audio_folder"),
        hf_dataset_name=data_config.get("hf_dataset_name"),
        hf_dataset_config=data_config.get("hf_dataset_config"),
    )
    
    # Initialize profiler
    profiling_dir = output_config.get("profiling_dir", "results/profiling")
    profiler = TrainingProfiler(output_dir=profiling_dir)
    
    # Training arguments - results saved to results folder
    output_dir = Path(output_config.get("output_dir", "results/training")) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    print(f"Profiling metrics will be saved to: {profiling_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("epochs", 10) if (use_lora or use_qlora) else 1,
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 2e-4),
        warmup_steps=training_config.get("warmup_steps", 100),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("log_interval", 10),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps", 500),
        fp16=False,
        bf16=True,  # bf16 for GPU training
        deepspeed=args.deepspeed,
        report_to="none",
        # Loss and optimizer settings
        optim=training_config.get("optimizer", "adamw_torch"),  # AdamW optimizer
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),  # Learning rate scheduler
        save_total_limit=training_config.get("save_total_limit", 3),  # Keep only last N checkpoints
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "loss"),
        greater_is_better=False,  # Lower loss is better
        save_safetensors=True,  # Save as safetensors format (not pickle)
    )
    
    # Custom trainer class to integrate profiling
    class ProfiledTrainer(Trainer):
        def __init__(self, profiler, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.profiler = profiler
        
        def training_step(self, model, inputs):
            # Profile forward
            self.profiler.start_forward()
            loss = super().training_step(model, inputs)
            self.profiler.end_forward()
            
            # Get batch info for profiling
            batch_size = inputs.get("input_ids", torch.tensor([])).shape[0]
            num_tokens = inputs.get("input_ids", torch.tensor([])).numel()
            
            # Record step (backward/optimizer timing handled by Trainer internally)
            self.profiler.record_step(batch_size, num_tokens)
            
            return loss
    
    # Create trainer
    trainer = ProfiledTrainer(
        profiler=profiler,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
        tokenizer=processor.tokenizer,
    )
    
    # Baseline: evaluation only
    if args.mode == "baseline":
        print("Running baseline evaluation...")
        profiler.start_epoch()
        
        eval_results = trainer.evaluate()
        print(f"Baseline evaluation results: {eval_results}")
        
        profiler.end_epoch()
        
        # Save results
        results_path = output_dir / "baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save profiling metrics (JSON and CSV)
        profiler.save("baseline_profiling.json")
        profiler.save_csv("baseline_profiling.csv")
    
    # LoRA/QLoRA: fine-tuning
    else:
        mode_name = "QLoRA" if use_qlora else "LoRA"
        print(f"Starting {mode_name} fine-tuning...")
        profiler.start_epoch()
        
        # Train (Trainer handles epochs internally)
        trainer.train()
        
        profiler.end_epoch()
        
        # Save final model at last epoch as checkpoint and safetensors
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final model (last epoch) to: {final_model_dir}")
        print("Format: safetensors + checkpoint files")
        
        # Save model in safetensors format
        model.save_pretrained(
            final_model_dir,
            safe_serialization=True  # Use safetensors format (.safetensors files)
        )
        processor.save_pretrained(final_model_dir)
        
        # Also save as checkpoint format (includes optimizer state if available)
        checkpoint_dir = output_dir / "checkpoint-final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(checkpoint_dir)  # Saves as safetensors (due to save_safetensors=True)
        
        print(f"✓ Final model saved to: {final_model_dir}")
        print(f"  - Model weights: model.safetensors (safetensors format)")
        print(f"  - Config: config.json")
        print(f"  - Processor: tokenizer files")
        print(f"✓ Checkpoint saved to: {checkpoint_dir}")
        print(f"  - Includes optimizer state and training state")
        
        # Final evaluation
        eval_results = trainer.evaluate()
        print(f"\nFinal evaluation results: {eval_results}")
        
        # Save results
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save profiling metrics (JSON and CSV)
        profiler.save(f"{args.mode}_profiling.json")
        profiler.save_csv(f"{args.mode}_profiling.csv")
        
        print(f"\nModel checkpoints saved:")
        print(f"  - Final model: {final_model_dir}")
        print(f"  - Epoch checkpoints: {output_dir}/checkpoint-epoch-*/")
        print(f"  - All models saved in safetensors format")
    
    # Save command used for this run
    command_log_path = output_dir / "command_log.txt"
    command_used = " ".join(sys.argv)
    with open(command_log_path, 'w') as f:
        f.write(f"Command executed: {command_used}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Config file: {args.config}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"DeepSpeed config: {args.deepspeed}\n")
    
    print(f"\nCommand log saved to: {command_log_path}")
    print(f"Command used: {command_used}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

