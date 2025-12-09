"""
FirstSight: INT8 Quantization Evaluation
Quick optimization experiment for mid-term report
"""

import torch
import time
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import gc

def compare_precision():
    """
    Compare FP16 vs INT8 quantization
    Quick wins for the report!
    """
    
    print("="*70)
    print("FirstSight - Quantization Comparison")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    questions = [
        "What object did I pick up?",
        "Where did I place the cup?",
        "What was I looking at?",
        "How many items are on the table?",
        "What color is the object?",
    ]
    
    results_all = []
    
    # =============================
    # Test 1: FP16/BF16 Baseline
    # =============================
    print("\n" + "="*70)
    print("TEST 1: FP16/BF16 Baseline")
    print("="*70)
    
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    
    model_fp16 = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    load_time_fp16 = time.time() - load_start
    vram_fp16 = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"✓ FP16 Model loaded: {load_time_fp16:.2f}s, VRAM: {vram_fp16:.2f} GB")
    
    # Run inference
    latencies_fp16 = []
    for question in tqdm(questions, desc="FP16 Inference"):
        start = time.time()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"Question: {question}\nAnswer:"}]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model_fp16.generate(**inputs, max_new_tokens=50)
        
        latencies_fp16.append(time.time() - start)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    avg_latency_fp16 = sum(latencies_fp16) / len(latencies_fp16)
    
    results_fp16 = {
        "precision": "FP16/BF16",
        "load_time_s": load_time_fp16,
        "model_vram_gb": vram_fp16,
        "avg_latency_s": avg_latency_fp16,
        "throughput_qps": 1.0 / avg_latency_fp16
    }
    
    # Cleanup
    del model_fp16
    torch.cuda.empty_cache()
    gc.collect()
    
    # =============================
    # Test 2: INT8 Quantization
    # =============================
    print("\n" + "="*70)
    print("TEST 2: INT8 Quantization")
    print("="*70)
    
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    
    model_int8 = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    load_time_int8 = time.time() - load_start
    vram_int8 = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"✓ INT8 Model loaded: {load_time_int8:.2f}s, VRAM: {vram_int8:.2f} GB")
    print(f"✓ VRAM Reduction: {((vram_fp16 - vram_int8) / vram_fp16 * 100):.1f}%")
    
    # Run inference
    latencies_int8 = []
    for question in tqdm(questions, desc="INT8 Inference"):
        start = time.time()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"Question: {question}\nAnswer:"}]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model_int8.generate(**inputs, max_new_tokens=50)
        
        latencies_int8.append(time.time() - start)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    avg_latency_int8 = sum(latencies_int8) / len(latencies_int8)
    
    results_int8 = {
        "precision": "INT8",
        "load_time_s": load_time_int8,
        "model_vram_gb": vram_int8,
        "avg_latency_s": avg_latency_int8,
        "throughput_qps": 1.0 / avg_latency_int8,
        "vram_reduction_pct": (vram_fp16 - vram_int8) / vram_fp16 * 100
    }
    
    # Save results
    output_dir = Path("experiments/quantization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "fp16": results_fp16,
        "int8": results_int8,
        "comparison": {
            "vram_reduction_pct": ((vram_fp16 - vram_int8) / vram_fp16) * 100,
            "vram_reduction_gb": vram_fp16 - vram_int8,
            "latency_change_pct": ((avg_latency_int8 - avg_latency_fp16) / avg_latency_fp16) * 100,
            "speedup": avg_latency_fp16 / avg_latency_int8
        }
    }
    
    output_file = output_dir / f"quantization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison
    print("\n" + "="*70)
    print("QUANTIZATION COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'FP16':<15} {'INT8':<15} {'Change':<15}")
    print("-"*70)
    print(f"{'Model VRAM (GB)':<30} {vram_fp16:<15.2f} {vram_int8:<15.2f} {comparison['comparison']['vram_reduction_pct']:>+14.1f}%")
    print(f"{'Avg Latency (s)':<30} {avg_latency_fp16:<15.3f} {avg_latency_int8:<15.3f} {comparison['comparison']['latency_change_pct']:>+14.1f}%")
    print(f"{'Throughput (q/s)':<30} {results_fp16['throughput_qps']:<15.2f} {results_int8['throughput_qps']:<15.2f} {(results_int8['throughput_qps']/results_fp16['throughput_qps']-1)*100:>+14.1f}%")
    print("="*70)
    print(f"\n✓ VRAM Saved: {comparison['comparison']['vram_reduction_gb']:.2f} GB ({comparison['comparison']['vram_reduction_pct']:.1f}%)")
    print(f"✓ Results saved to: {output_file}")
    
    return comparison

if __name__ == "__main__":
    results = compare_precision()


