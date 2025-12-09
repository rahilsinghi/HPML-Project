"""
FirstSight: Evaluation Script for Knowledge Distillation
Compares teacher and student model performance on egocentric QA
"""

import torch
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging
from tqdm import tqdm
import gc

from .models import load_teacher_model, load_student_model, get_model_info
from .data import create_egocentric_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    processor,
    dataloader,
    model_name: str,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate a single model on the dataloader.
    
    Args:
        model: Model to evaluate
        processor: Model processor
        dataloader: Evaluation dataloader
        model_name: Name for logging
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*70}")
    
    model.eval()
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "samples": []
    }
    
    total_latency = 0.0
    total_tokens = 0
    num_samples = 0
    
    # Get model info
    model_info = get_model_info(model)
    results["model_info"] = {
        "total_params": model_info["total_params"],
        "size_gb": model_info["size_gb"],
        "vram_gb": model_info.get("vram_allocated_gb", 0)
    }
    
    logger.info(f"Model parameters: {model_info['total_params'] / 1e9:.2f}B")
    logger.info(f"VRAM: {model_info.get('vram_allocated_gb', 0):.2f} GB")
    
    # Evaluate on all samples
    logger.info("\nRunning inference...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            questions = batch["questions"]
            
            for i, question in enumerate(questions):
                # Prepare single input
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"Question: {question}\nAnswer concisely:"}]
                }]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
                
                # Measure inference time
                torch.cuda.synchronize() if device == "cuda" else None
                start_time = time.time()
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=1.0
                )
                
                torch.cuda.synchronize() if device == "cuda" else None
                latency = time.time() - start_time
                
                # Decode answer
                answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Measure memory
                peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                
                # Store results
                results["samples"].append({
                    "question": question,
                    "answer": answer,
                    "latency_s": latency,
                    "peak_vram_gb": peak_vram,
                    "num_tokens": len(outputs[0])
                })
                
                total_latency += latency
                total_tokens += len(outputs[0])
                num_samples += 1
                
                # Cleanup
                del inputs, outputs
                if num_samples % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
    
    # Compute aggregate metrics
    results["summary"] = {
        "num_samples": num_samples,
        "avg_latency_s": total_latency / num_samples,
        "total_time_s": total_latency,
        "avg_tokens": total_tokens / num_samples,
        "throughput_samples_per_sec": num_samples / total_latency,
        "throughput_tokens_per_sec": total_tokens / total_latency,
        "avg_vram_gb": sum(s["peak_vram_gb"] for s in results["samples"]) / num_samples
    }
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"{model_name} - Evaluation Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Samples evaluated: {num_samples}")
    logger.info(f"Avg latency: {results['summary']['avg_latency_s']:.3f}s")
    logger.info(f"Throughput: {results['summary']['throughput_samples_per_sec']:.2f} samples/s")
    logger.info(f"Avg tokens: {results['summary']['avg_tokens']:.1f}")
    logger.info(f"Avg VRAM: {results['summary']['avg_vram_gb']:.2f} GB")
    logger.info(f"{'='*70}\n")
    
    return results


def compare_models(
    teacher_results: Dict,
    student_results: Dict
) -> Dict:
    """
    Compare teacher and student model results.
    
    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
    
    Returns:
        Comparison dictionary
    """
    logger.info("\n" + "="*70)
    logger.info("COMPARISON: Teacher vs Student")
    logger.info("="*70)
    
    teacher_summary = teacher_results["summary"]
    student_summary = student_results["summary"]
    teacher_info = teacher_results["model_info"]
    student_info = student_results["model_info"]
    
    comparison = {
        "model_size": {
            "teacher_params_b": teacher_info["total_params"] / 1e9,
            "student_params_b": student_info["total_params"] / 1e9,
            "compression_ratio": teacher_info["total_params"] / student_info["total_params"],
            "size_reduction_pct": (1 - student_info["total_params"] / teacher_info["total_params"]) * 100
        },
        "memory": {
            "teacher_vram_gb": teacher_info["vram_gb"],
            "student_vram_gb": student_info["vram_gb"],
            "vram_reduction_gb": teacher_info["vram_gb"] - student_info["vram_gb"],
            "vram_reduction_pct": (1 - student_info["vram_gb"] / teacher_info["vram_gb"]) * 100 if teacher_info["vram_gb"] > 0 else 0
        },
        "performance": {
            "teacher_latency_s": teacher_summary["avg_latency_s"],
            "student_latency_s": student_summary["avg_latency_s"],
            "speedup": teacher_summary["avg_latency_s"] / student_summary["avg_latency_s"],
            "teacher_throughput": teacher_summary["throughput_samples_per_sec"],
            "student_throughput": student_summary["throughput_samples_per_sec"],
            "throughput_improvement": student_summary["throughput_samples_per_sec"] / teacher_summary["throughput_samples_per_sec"]
        }
    }
    
    # Print comparison table
    logger.info("\n MODEL SIZE & COMPRESSION:")
    logger.info(f"  Teacher parameters: {comparison['model_size']['teacher_params_b']:.2f}B")
    logger.info(f"  Student parameters: {comparison['model_size']['student_params_b']:.2f}B")
    logger.info(f"  Compression ratio: {comparison['model_size']['compression_ratio']:.2f}×")
    logger.info(f"  Size reduction: {comparison['model_size']['size_reduction_pct']:.1f}%")
    
    logger.info("\n MEMORY USAGE:")
    logger.info(f"  Teacher VRAM: {comparison['memory']['teacher_vram_gb']:.2f} GB")
    logger.info(f"  Student VRAM: {comparison['memory']['student_vram_gb']:.2f} GB")
    logger.info(f"  VRAM savings: {comparison['memory']['vram_reduction_gb']:.2f} GB ({comparison['memory']['vram_reduction_pct']:.1f}%)")
    
    logger.info("\n PERFORMANCE:")
    logger.info(f"  Teacher latency: {comparison['performance']['teacher_latency_s']:.3f}s")
    logger.info(f"  Student latency: {comparison['performance']['student_latency_s']:.3f}s")
    logger.info(f"  Speedup: {comparison['performance']['speedup']:.2f}×")
    logger.info(f"  Teacher throughput: {comparison['performance']['teacher_throughput']:.2f} samples/s")
    logger.info(f"  Student throughput: {comparison['performance']['student_throughput']:.2f} samples/s")
    logger.info(f"  Throughput improvement: {comparison['performance']['throughput_improvement']:.2f}×")
    
    logger.info("="*70 + "\n")
    
    return comparison


def evaluate_distillation(
    student_model_path: str = "experiments/distillation/best_student_model",
    num_eval_samples: int = 50,
    batch_size: int = 4
):
    """
    Main evaluation function: compares teacher and distilled student.
    
    Args:
        student_model_path: Path to distilled student model
        num_eval_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
    """
    logger.info("="*70)
    logger.info("FirstSight: Distillation Evaluation")
    logger.info("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load teacher model
    logger.info("\nLoading teacher model...")
    teacher_model, teacher_processor = load_teacher_model(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        load_in_8bit=True
    )
    
    # Load distilled student model
    logger.info("\nLoading distilled student model...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    student_model = Qwen2VLForConditionalGeneration.from_pretrained(
        student_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    student_processor = AutoProcessor.from_pretrained(
        student_model_path,
        trust_remote_code=True
    )
    
    logger.info("✓ Models loaded successfully")
    
    # Create evaluation dataloader
    logger.info("\nPreparing evaluation data...")
    eval_dataloader = create_egocentric_dataloader(
        batch_size=batch_size,
        num_samples=num_eval_samples,
        shuffle=False,
        processor=teacher_processor  # Use teacher processor for consistency
    )
    
    # Evaluate teacher
    teacher_results = evaluate_model(
        model=teacher_model,
        processor=teacher_processor,
        dataloader=eval_dataloader,
        model_name="EgoGPT-7b (Teacher)",
        device=device
    )
    
    # Clear memory
    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluate student
    student_results = evaluate_model(
        model=student_model,
        processor=student_processor,
        dataloader=eval_dataloader,
        model_name="Qwen2-VL-2B (Distilled Student)",
        device=device
    )
    
    # Compare results
    comparison = compare_models(teacher_results, student_results)
    
    # Save results
    output_dir = Path("experiments/distillation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    full_results = {
        "teacher": teacher_results,
        "student": student_results,
        "comparison": comparison
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"✓ Evaluation results saved to: {results_file}")
    
    # Generate summary report
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("FirstSight: Distillation Evaluation Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Teacher: EgoGPT-7b ({comparison['model_size']['teacher_params_b']:.2f}B params)\n")
        f.write(f"Student: Qwen2-VL-2B ({comparison['model_size']['student_params_b']:.2f}B params)\n\n")
        f.write(f"Compression: {comparison['model_size']['compression_ratio']:.2f}× ({comparison['model_size']['size_reduction_pct']:.1f}% reduction)\n")
        f.write(f"VRAM Savings: {comparison['memory']['vram_reduction_gb']:.2f} GB ({comparison['memory']['vram_reduction_pct']:.1f}%)\n")
        f.write(f"Speedup: {comparison['performance']['speedup']:.2f}×\n")
        f.write(f"Throughput Improvement: {comparison['performance']['throughput_improvement']:.2f}×\n")
    
    logger.info(f"✓ Summary saved to: {summary_file}")
    logger.info("\n" + "="*70)
    logger.info("Evaluation Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    import sys
    
    student_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/distillation/best_student_model"
    evaluate_distillation(student_model_path=student_path)

