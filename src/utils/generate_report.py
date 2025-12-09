"""
FirstSight: Mid-Term Report Generator
Automatically generates report content from experimental results
"""

import json
import glob
from pathlib import Path
from datetime import datetime

def generate_midterm_report():
    """
    Generate mid-term report content from results
    """
    
    print("="*70)
    print("Generating Mid-Term Report")
    print("="*70)
    
    # Load baseline results
    baseline_dir = Path("experiments/baseline")
    quantization_dir = Path("experiments/quantization")
    
    baseline_files = list(baseline_dir.glob("baseline_*.json")) if baseline_dir.exists() else []
    quantization_files = list(quantization_dir.glob("quantization_*.json")) if quantization_dir.exists() else []
    
    report = []
    report.append("# FirstSight: Mid-Term Report")
    report.append("=" * 70)
    report.append("")
    report.append("**Project:** FirstSight: Efficient Egocentric Question Answering")
    report.append("**Team Members:**")
    report.append("- Sunidhi Tandel (sdt9243@nyu.edu)")
    report.append("- Rahil Singhi (rs9174@nyu.edu)")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    report.append("")
    report.append("=" * 70)
    report.append("")
    
    # Section 1: Milestones
    report.append("## 1. Project Milestones")
    report.append("")
    report.append("### Planned Timeline (6 Weeks)")
    report.append("")
    report.append("| Week | Milestone |")
    report.append("|------|-----------|")
    report.append("| 1-2  | Dataset selection, baseline establishment, profiling infrastructure |")
    report.append("| 3-4  | LoRA fine-tuning, FlashAttention-2, distributed training comparison |")
    report.append("| 5-6  | Quantization (INT8, W4A8), edge deployment, final evaluation |")
    report.append("")
    
    # Section 2: Completed Milestones
    report.append("## 2. Milestones Completed & Results")
    report.append("")
    report.append("### ✅ Completed:")
    report.append("")
    report.append("1. **Infrastructure Setup**")
    report.append("   - Configured NYU HPC environment with CUDA 11.8")
    report.append("   - Set up conda environment with PyTorch 2.1.0")
    report.append("   - Installed quantization tools (bitsandbytes, PEFT)")
    report.append("")
    report.append("2. **Model Selection & Baseline**")
    report.append("   - Selected Qwen2-VL-2B-Instruct as starting model")
    report.append("   - Implemented profiling infrastructure")
    report.append("   - Established baseline performance metrics")
    report.append("")
    report.append("3. **Initial Optimization Experiments**")
    report.append("   - Implemented INT8 quantization")
    report.append("   - Measured memory and latency improvements")
    report.append("")
    
    # Add baseline results
    if baseline_files:
        latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
        with open(latest_baseline) as f:
            baseline_data = json.load(f)
        
        report.append("### 📊 Baseline Results (Qwen2-VL-2B)")
        report.append("")
        report.append(f"**Model:** {baseline_data['model']}")
        report.append(f"**Device:** {baseline_data.get('gpu', 'N/A')}")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Model Load Time | {baseline_data['model_load_time_s']:.2f}s |")
        report.append(f"| Model VRAM | {baseline_data['model_vram_gb']:.2f} GB |")
        report.append(f"| Avg Inference Latency | {baseline_data['summary']['avg_latency_s']:.3f}s |")
        report.append(f"| Throughput | {baseline_data['summary']['throughput_queries_per_sec']:.2f} queries/sec |")
        report.append(f"| Peak VRAM | {baseline_data['summary']['max_peak_vram_gb']:.2f} GB |")
        report.append("")
    
    # Add quantization results
    if quantization_files:
        latest_quant = max(quantization_files, key=lambda p: p.stat().st_mtime)
        with open(latest_quant) as f:
            quant_data = json.load(f)
        
        report.append("### 🚀 Quantization Results (FP16 vs INT8)")
        report.append("")
        report.append("| Metric | FP16 | INT8 | Improvement |")
        report.append("|--------|------|------|-------------|")
        report.append(f"| Model VRAM (GB) | {quant_data['fp16']['model_vram_gb']:.2f} | {quant_data['int8']['model_vram_gb']:.2f} | {quant_data['comparison']['vram_reduction_pct']:.1f}% ↓ |")
        report.append(f"| Avg Latency (s) | {quant_data['fp16']['avg_latency_s']:.3f} | {quant_data['int8']['avg_latency_s']:.3f} | {quant_data['comparison']['latency_change_pct']:.1f}% |")
        report.append(f"| Throughput (q/s) | {quant_data['fp16']['throughput_qps']:.2f} | {quant_data['int8']['throughput_qps']:.2f} | {((quant_data['int8']['throughput_qps']/quant_data['fp16']['throughput_qps']-1)*100):.1f}% |")
        report.append("")
        report.append(f"**Key Finding:** INT8 quantization reduces VRAM by {quant_data['comparison']['vram_reduction_gb']:.2f} GB ({quant_data['comparison']['vram_reduction_pct']:.1f}%), enabling deployment on smaller GPUs.")
        report.append("")
    
    # Section 3: Remaining Work
    report.append("## 3. Remaining Milestones & Timeline")
    report.append("")
    report.append("### 🔄 In Progress / Planned:")
    report.append("")
    report.append("**Week 3-4 (Current):**")
    report.append("- [ ] Acquire EgoSchema dataset (5K videos)")
    report.append("- [ ] Implement LoRA fine-tuning on egocentric QA task")
    report.append("- [ ] Integrate FlashAttention-2 for memory optimization")
    report.append("- [ ] Compare DDP vs FSDP for distributed training")
    report.append("")
    report.append("**Week 5-6:**")
    report.append("- [ ] Test W4A8 quantization (4-bit weights, 8-bit activations)")
    report.append("- [ ] Implement modality-aware calibration (MBQ approach)")
    report.append("- [ ] Edge deployment with TensorRT-LLM")
    report.append("- [ ] Final evaluation on full EgoSchema benchmark")
    report.append("- [ ] Prepare demo and final presentation")
    report.append("")
    
    # Section 4: Bottlenecks
    report.append("## 4. Bottlenecks & Mitigation Strategies")
    report.append("")
    report.append("### Current Challenges:")
    report.append("")
    report.append("1. **Dataset Access**")
    report.append("   - *Issue:* EgoSchema download requires approval/authentication")
    report.append("   - *Mitigation:* Using Ego4D subset or synthetic egocentric samples for initial development")
    report.append("   - *Status:* Baseline established with text-only evaluation")
    report.append("")
    report.append("2. **Model Size vs Timeline**")
    report.append("   - *Issue:* Limited time to experiment with 8B/34B models")
    report.append("   - *Mitigation:* Progressive scaling strategy (2B → 8B → 34B)")
    report.append("   - *Status:* 2B model provides quick iteration and baseline metrics")
    report.append("")
    report.append("3. **Fine-tuning Data Requirements**")
    report.append("   - *Issue:* Need sufficient egocentric QA pairs for meaningful fine-tuning")
    report.append("   - *Mitigation:* Using pre-existing egocentric QA datasets (Ego4D, EgoSchema)")
    report.append("   - *Status:* Baseline infrastructure ready for fine-tuning once data is available")
    report.append("")
    
    # Section 5: Work Distribution
    report.append("## 5. Work Contribution by Team Member")
    report.append("")
    report.append("### Sunidhi Tandel:")
    report.append("- HPC environment setup and dependency installation")
    report.append("- Baseline evaluation script development")
    report.append("- Model profiling infrastructure")
    report.append("- Performance metrics collection")
    report.append("")
    report.append("### Rahil Singhi:")
    report.append("- SLURM job configuration and scheduling")
    report.append("- Quantization implementation and testing")
    report.append("- Result analysis and report generation")
    report.append("- Documentation and project organization")
    report.append("")
    report.append("### Joint Contributions:")
    report.append("- Project proposal refinement")
    report.append("- Experimental design and methodology")
    report.append("- Literature review and technical research")
    report.append("- Weekly coordination and planning meetings")
    report.append("")
    
    # Section 6: Next Steps
    report.append("## 6. Immediate Next Steps (Week 3-4)")
    report.append("")
    report.append("1. **Data Acquisition** (Priority 1)")
    report.append("   - Download EgoSchema or prepare Ego4D subset")
    report.append("   - Implement video preprocessing pipeline")
    report.append("   - Validate data loading with Qwen2-VL")
    report.append("")
    report.append("2. **LoRA Fine-Tuning** (Priority 2)")
    report.append("   - Implement PEFT with LoRA on attention layers")
    report.append("   - Fine-tune on egocentric QA samples")
    report.append("   - Measure accuracy improvement vs zero-shot")
    report.append("")
    report.append("3. **Memory Optimization** (Priority 3)")
    report.append("   - Integrate FlashAttention-2")
    report.append("   - Enable gradient checkpointing")
    report.append("   - Profile memory savings")
    report.append("")
    report.append("4. **Distributed Training** (Priority 4)")
    report.append("   - Scale to 2-4 GPUs with DDP")
    report.append("   - Compare with FSDP and ZeRO-3")
    report.append("   - Measure training throughput")
    report.append("")
    
    # Save report
    output_file = Path("MIDTERM_REPORT_DRAFT.md")
    with open(output_file, "w") as f:
        f.write("\n".join(report))
    
    print(f"✓ Report generated: {output_file}")
    print(f"✓ Total lines: {len(report)}")
    print("")
    print("="*70)
    print("Report Preview:")
    print("="*70)
    for line in report[:30]:
        print(line)
    print("...")
    print("="*70)
    
    return output_file

if __name__ == "__main__":
    generate_midterm_report()


