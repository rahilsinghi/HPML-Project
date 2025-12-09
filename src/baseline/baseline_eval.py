"""
FirstSight: Baseline Evaluation Script
Runs zero-shot inference on EgoSchema with profiling
"""

import torch
import time
import json
import os
from pathlib import Path
from tqdm import tqdm
import gc
from datetime import datetime

def profile_baseline():
    """
    Quick baseline with Qwen2-VL-2B on small sample
    Get immediate numbers for mid-term report
    """
    
    print("="*70)
    print("FirstSight - Baseline Evaluation")
    print("="*70)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Model loading
    print("\n" + "="*70)
    print("Loading Model: Qwen2-VL-2B-Instruct")
    print("="*70)
    
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    load_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    load_time = time.time() - load_start
    load_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"✓ Model VRAM: {load_vram:.2f} GB")
    
    # Create dummy sample for profiling (since we may not have real data yet)
    print("\n" + "="*70)
    print("Running Inference Profiling")
    print("="*70)
    
    # Simulate egocentric QA
    questions = [
        "What object did I pick up?",
        "Where did I place the cup?",
        "What was I looking at?",
        "How many items are on the table?",
        "What color is the object?",
    ]
    
    # Profile with text-only (video processing will come later)
    results = {
        "model": model_name,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "timestamp": datetime.now().isoformat(),
        "model_load_time_s": load_time,
        "model_vram_gb": load_vram,
        "inference_metrics": []
    }
    
    print(f"\nRunning {len(questions)} test inferences...")
    
    for idx, question in enumerate(tqdm(questions)):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Prepare input (text-only for now)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"Question: {question}\nAnswer concisely:"}]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Metrics
        latency = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        
        results["inference_metrics"].append({
            "question": question,
            "latency_s": latency,
            "peak_vram_gb": peak_vram,
            "answer_length": len(answer.split())
        })
        
        # Cleanup
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Aggregate metrics
    latencies = [m["latency_s"] for m in results["inference_metrics"]]
    vrams = [m["peak_vram_gb"] for m in results["inference_metrics"]]
    
    results["summary"] = {
        "avg_latency_s": sum(latencies) / len(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "avg_peak_vram_gb": sum(vrams) / len(vrams),
        "max_peak_vram_gb": max(vrams),
        "throughput_queries_per_sec": 1.0 / (sum(latencies) / len(latencies))
    }
    
    # Save results
    output_dir = Path("experiments/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"\nModel Loading:")
    print(f"  Time: {results['model_load_time_s']:.2f}s")
    print(f"  VRAM: {results['model_vram_gb']:.2f} GB")
    print(f"\nInference Performance:")
    print(f"  Avg Latency: {results['summary']['avg_latency_s']:.3f}s")
    print(f"  Min Latency: {results['summary']['min_latency_s']:.3f}s")
    print(f"  Max Latency: {results['summary']['max_latency_s']:.3f}s")
    print(f"  Throughput: {results['summary']['throughput_queries_per_sec']:.2f} queries/sec")
    print(f"\nMemory Usage:")
    print(f"  Avg Peak VRAM: {results['summary']['avg_peak_vram_gb']:.2f} GB")
    print(f"  Max Peak VRAM: {results['summary']['max_peak_vram_gb']:.2f} GB")
    print("="*70)
    print(f"\n✓ Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = profile_baseline()


