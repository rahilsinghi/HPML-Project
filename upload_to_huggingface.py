#!/usr/bin/env python3
"""
Upload distilled FirstSight model to Hugging Face Hub

This script uploads the distilled Qwen2-VL-2B model with comprehensive
documentation and metrics from the production training run.
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datetime import datetime

def create_model_card(eval_results: dict) -> str:
    """Create a comprehensive model card with all metrics"""
    
    comparison = eval_results["comparison"]
    teacher_summary = eval_results["teacher"]["summary"]
    student_summary = eval_results["student"]["summary"]
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
- vision
- vision-language
- knowledge-distillation
- egocentric-qa
- qwen2-vl
- edge-deployment
- model-compression
datasets:
- custom-egocentric-qa
metrics:
- latency
- throughput
- vram-usage
model-index:
- name: FirstSight-Qwen2-VL-2B-Distilled
  results:
  - task:
      type: visual-question-answering
      name: Egocentric Visual Question Answering
    metrics:
    - type: compression-ratio
      value: {comparison['model_size']['compression_ratio']:.2f}
      name: Model Compression
    - type: speedup
      value: {comparison['performance']['speedup']:.2f}
      name: Inference Speedup
    - type: vram-reduction
      value: {comparison['memory']['vram_reduction_pct']:.1f}
      name: VRAM Reduction (%)
---

# FirstSight: Distilled Qwen2-VL-2B for Efficient Egocentric QA

## Model Description

FirstSight is a **knowledge-distilled vision-language model** optimized for efficient egocentric question answering on edge devices. This model is distilled from [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) using advanced distillation techniques, achieving **{comparison['model_size']['compression_ratio']:.2f}× compression** with minimal performance degradation.

### Key Highlights

- 🚀 **{comparison['performance']['speedup']:.2f}× faster inference** than teacher model
- 💾 **{comparison['memory']['vram_reduction_pct']:.1f}% VRAM reduction** ({comparison['memory']['vram_reduction_gb']:.2f} GB savings)
- 📦 **{comparison['model_size']['size_reduction_pct']:.1f}% smaller model size** ({comparison['model_size']['student_params_b']:.2f}B vs {comparison['model_size']['teacher_params_b']:.2f}B parameters)
- ⚡ **Optimized for edge deployment** on resource-constrained devices
- 🎯 **Specialized for egocentric scenarios** (first-person perspective)

### Model Architecture

- **Base Model**: [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **Teacher Model**: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- **Student Parameters**: {comparison['model_size']['student_params_b']:.2f}B
- **Precision**: BFloat16 mixed precision
- **Distillation Method**: Logit-based knowledge distillation with KL divergence

## Training Details

### Training Data

- **Dataset**: Synthetic egocentric QA dataset with 5,000 training samples
- **Validation Set**: 1,000 samples
- **Question Types**: Object recognition, spatial reasoning, action understanding, temporal queries, environment understanding
- **Scenarios**: Kitchen, living room, office, outdoor, workshop

### Training Procedure

- **Framework**: PyTorch with Hugging Face Transformers
- **Epochs**: 10
- **Batch Size**: 2 per GPU with 4× gradient accumulation
- **Learning Rate**: 1e-5 (AdamW optimizer)
- **Scheduler**: Cosine annealing with 100 warmup steps
- **Loss Function**: Weighted combination of distillation loss (α=0.7) and hard label loss (α=0.3)
- **Temperature**: 2.0 for knowledge distillation
- **Hardware**: NVIDIA Quadro RTX 8000 (48GB)
- **Training Time**: ~4 hours

### Training Hyperparameters

```python
{{
    "learning_rate": 1e-5,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "temperature": 2.0,
    "alpha": 0.7,
    "epochs": 10
}}
```

## Performance Metrics

### Inference Speed

| Metric | Teacher (7B) | Student (2B) | Improvement |
|--------|-------------|-------------|-------------|
| **Avg Latency** | {teacher_summary['avg_latency_s']:.3f}s | {student_summary['avg_latency_s']:.3f}s | **{comparison['performance']['speedup']:.2f}×** |
| **Throughput (samples/s)** | {teacher_summary['throughput_samples_per_sec']:.2f} | {student_summary['throughput_samples_per_sec']:.2f} | **{comparison['performance']['throughput_improvement']:.2f}×** |
| **Throughput (tokens/s)** | {teacher_summary['throughput_tokens_per_sec']:.2f} | {student_summary['throughput_tokens_per_sec']:.2f} | **{comparison['performance']['throughput_improvement']:.2f}×** |

### Memory Usage

| Metric | Teacher (7B) | Student (2B) | Savings |
|--------|-------------|-------------|---------|
| **Model Size** | {comparison['model_size']['teacher_params_b']:.2f}B params | {comparison['model_size']['student_params_b']:.2f}B params | **{comparison['model_size']['size_reduction_pct']:.1f}%** |
| **Peak VRAM** | {comparison['memory']['teacher_vram_gb']:.2f} GB | {comparison['memory']['student_vram_gb']:.2f} GB | **{comparison['memory']['vram_reduction_gb']:.2f} GB** |
| **VRAM Reduction** | - | - | **{comparison['memory']['vram_reduction_pct']:.1f}%** |

### Model Compression

- **Compression Ratio**: {comparison['model_size']['compression_ratio']:.2f}×
- **Parameter Reduction**: {comparison['model_size']['size_reduction_pct']:.1f}%
- **From**: {comparison['model_size']['teacher_params_b']:.2f}B parameters
- **To**: {comparison['model_size']['student_params_b']:.2f}B parameters

## Usage

### Installation

```bash
pip install transformers torch pillow qwen-vl-utils
```

### Inference Example

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model_name = "YOUR_USERNAME/firstsight-qwen2-vl-2b-distilled"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Prepare image and question
image = Image.open("egocentric_image.jpg")
question = "What object am I holding in my right hand?"

# Create conversation template
messages = [
    {{
        "role": "system",
        "content": "You are a helpful assistant."
    }},
    {{
        "role": "user",
        "content": [
            {{"type": "image", "image": image}},
            {{"type": "text", "text": f"Question: {{question}}\\nAnswer concisely:"}}
        ]
    }}
]

# Prepare inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

# Generate answer
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )

# Decode response
response = processor.batch_decode(
    outputs[:, inputs['input_ids'].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"Answer: {{response}}")
```

## Intended Use

### Primary Use Cases

- **Egocentric Visual Question Answering**: Answer questions about first-person perspective images/videos
- **Edge Device Deployment**: Run VLM inference on resource-constrained hardware (mobile, IoT, AR/VR)
- **Real-time Assistive Systems**: Power low-latency visual assistants for wearable cameras
- **Smart Glasses Applications**: Enable efficient VLM capabilities on AR/VR headsets

### Supported Question Types

1. **Object Recognition**: "What object did I just pick up?"
2. **Spatial Reasoning**: "Where is the nearest door?"
3. **Action Understanding**: "What action am I performing?"
4. **Temporal Queries**: "What was I looking at 5 seconds ago?"
5. **Environment Understanding**: "What room am I in?"
6. **Counting**: "How many items are on the table?"
7. **Attribute Recognition**: "What color is the object I'm holding?"

## Limitations

- Model is specialized for **egocentric scenarios** and may perform worse on third-person images
- Trained on **synthetic data** - real-world performance may vary
- **No multimodal training** - relies solely on knowledge distillation
- May inherit biases from the teacher model (Qwen2-VL-7B)
- Limited to **short-form QA** - not optimized for long conversations

## Ethical Considerations

- **Privacy**: Egocentric images often contain sensitive personal information. Ensure proper consent and data protection.
- **Bias**: Model may exhibit biases from training data and teacher model. Evaluate on diverse datasets.
- **Misuse**: Could be used for unauthorized surveillance. Deploy responsibly with user consent.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{firstsight2024,
  title={{FirstSight: Efficient Knowledge Distillation for Vision-Language Models on Edge Devices}},
  author={{NYU HPML Project Team}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/YOUR_USERNAME/firstsight-qwen2-vl-2b-distilled}}}},
  note={{Distilled from Qwen2-VL-7B-Instruct for egocentric question answering}}
}}
```

## Model Card Authors

NYU High Performance Machine Learning (HPML) Project Team

## Model Card Contact

For questions or feedback, please open an issue on the [GitHub repository](https://github.com/rahils/firstsight).

---

**Training Date**: December 8-9, 2024  
**Evaluation Date**: {eval_results['student']['timestamp']}  
**Framework**: PyTorch 2.3.0, Transformers 4.57.3, BitsAndBytes 0.48.2
"""
    
    return model_card


def upload_model_to_hub(
    model_path: str,
    eval_results_path: str,
    repo_name: str,
    organization: str = None,
    private: bool = False
):
    """
    Upload model to Hugging Face Hub with comprehensive documentation
    
    Args:
        model_path: Path to the saved model directory
        eval_results_path: Path to evaluation results JSON
        repo_name: Name for the HuggingFace repository
        organization: Optional organization name
        private: Whether to make the repo private
    """
    
    # Initialize HF API
    api = HfApi()
    
    # Load evaluation results
    print(f"📊 Loading evaluation results from: {eval_results_path}")
    with open(eval_results_path, 'r') as f:
        eval_results = json.load(f)
    
    # Create model card
    print("📝 Creating model card...")
    model_card = create_model_card(eval_results)
    
    # Full repo ID
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        # Get username from token
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{repo_name}"
    
    print(f"🎯 Target repository: {repo_id}")
    
    # Create repository
    print(f"🔨 Creating repository...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Upload model files
    print(f"📤 Uploading model files from: {model_path}")
    model_path = Path(model_path)
    
    files_to_upload = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "video_preprocessor_config.json"
    ]
    
    uploaded_files = []
    for filename in files_to_upload:
        file_path = model_path / filename
        if file_path.exists():
            print(f"  ⬆️  Uploading: {filename}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model"
            )
            uploaded_files.append(filename)
        else:
            print(f"  ⚠️  Skipping (not found): {filename}")
    
    # Upload README (model card)
    print(f"📄 Uploading README.md (model card)...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Upload evaluation results as metadata
    print(f"📊 Uploading evaluation results...")
    api.upload_file(
        path_or_fileobj=eval_results_path,
        path_in_repo="evaluation_results.json",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\n{'='*70}")
    print(f"✅ MODEL UPLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"📦 Repository: https://huggingface.co/{repo_id}")
    print(f"📁 Files uploaded: {len(uploaded_files) + 2}")  # +2 for README and eval results
    print(f"🎉 Your model is now ready to use!")
    print(f"\nTo load the model:")
    print(f"```python")
    print(f"from transformers import Qwen2VLForConditionalGeneration, AutoProcessor")
    print(f"model = Qwen2VLForConditionalGeneration.from_pretrained('{repo_id}')")
    print(f"processor = AutoProcessor.from_pretrained('{repo_id}')")
    print(f"```")
    print(f"{'='*70}\n")


def main():
    """Main upload script"""
    
    # Configuration
    MODEL_PATH = "results/distillation_production_5k/best_student_model"
    EVAL_RESULTS_PATH = "results/distillation_production_5k/evaluation_20251209_003305.json"
    REPO_NAME = "firstsight-qwen2-vl-2b-distilled"
    PRIVATE = False  # Set to True if you want a private repo
    
    print(f"\n{'='*70}")
    print(f"🚀 FirstSight Model Upload to Hugging Face")
    print(f"{'='*70}\n")
    
    # Verify paths exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model path not found: {MODEL_PATH}")
        return
    
    if not Path(EVAL_RESULTS_PATH).exists():
        print(f"❌ Error: Evaluation results not found: {EVAL_RESULTS_PATH}")
        return
    
    # Upload model
    upload_model_to_hub(
        model_path=MODEL_PATH,
        eval_results_path=EVAL_RESULTS_PATH,
        repo_name=REPO_NAME,
        organization=None,  # Set to your org name if uploading to an organization
        private=PRIVATE
    )


if __name__ == "__main__":
    main()

