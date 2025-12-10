"""
Ego4D dataset loader for LLaVA-OneVision fine-tuning.
Handles video/audio loading and conversation format.
Supports loading from HuggingFace datasets or local JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from datasets import load_dataset as hf_load_dataset

decord.bridge.set_bridge("torch")


class Ego4DDataset(Dataset):
    """
    Dataset for Ego4D-style multimodal QA.
    
    Expected format:
    [
        {
            "id": "video_id",
            "video": "path/to/video.mp4",
            "audio": "path/to/audio.mp3",  # Optional
            "conversations": [
                {"from": "human", "value": "<speech>\n<image>\nQuestion"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
    ]
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        processor: AutoProcessor = None,
        video_ids_file: Optional[str] = None,
        max_frames: int = 8,
        fps: int = 1,
        video_folder: Optional[str] = None,
        audio_folder: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
    ):
        """
        Initialize Ego4D dataset.
        
        Args:
            data_path: Path to JSON file with dataset (if not using HuggingFace)
            processor: LLaVA processor for image/text processing
            video_ids_file: Optional .txt file with video IDs (one per line) to filter
            max_frames: Maximum number of frames to sample from video
            fps: Frames per second to sample
            video_folder: Base folder for video paths (if relative)
            audio_folder: Base folder for audio paths (if relative)
            hf_dataset_name: HuggingFace dataset name (e.g., "sunidhitandel/FirstSight-Dataset")
            hf_dataset_config: HuggingFace dataset config (e.g., "Ego4D")
        """
        self.processor = processor
        self.max_frames = max_frames
        self.fps = fps
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        
        # Load dataset from HuggingFace or local file
        # Priority: HuggingFace > local JSON file
        if hf_dataset_name:
            print(f"Loading dataset from HuggingFace: {hf_dataset_name}/{hf_dataset_config}")
            try:
                hf_dataset = hf_load_dataset(hf_dataset_name, hf_dataset_config)
                # Convert to list format
                if isinstance(hf_dataset, dict):
                    # If split dict, use train split
                    self.data = hf_dataset.get("train", list(hf_dataset.values())[0]).to_list()
                else:
                    self.data = hf_dataset.to_list()
                print(f"Loaded {len(self.data)} samples from HuggingFace")
            except Exception as e:
                print(f"Warning: Failed to load from HuggingFace: {e}")
                print("Falling back to local JSON file...")
                if data_path and os.path.exists(data_path):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self.data = list(data.values())
                    else:
                        self.data = data
                else:
                    raise ValueError(f"Failed to load from HuggingFace and data_path not found: {data_path}")
        elif data_path and os.path.exists(data_path):
            # Load from local JSON file
            print(f"Loading dataset from local file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to list if dict
            if isinstance(data, dict):
                self.data = list(data.values())
            else:
                self.data = data
            print(f"Loaded {len(self.data)} samples from local file")
        else:
            raise ValueError("Must provide either hf_dataset_name or data_path")
        
        # Filter by video IDs if provided
        if video_ids_file and os.path.exists(video_ids_file):
            with open(video_ids_file, 'r') as f:
                allowed_ids = set(line.strip() for line in f if line.strip())
            self.data = [item for item in self.data if item.get("id") in allowed_ids]
            print(f"Filtered to {len(self.data)} samples using {video_ids_file}")
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_video(self, video_path: str) -> List[Image.Image]:
        """Load and sample frames from video."""
        if self.video_folder and not os.path.isabs(video_path):
            video_path = os.path.join(self.video_folder, video_path)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            # Return dummy frames
            return [Image.new("RGB", (224, 224), color="gray") for _ in range(self.max_frames)]
        
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
            total_frames = len(vr)
            
            # Sample frames
            if total_frames == 0:
                return [Image.new("RGB", (224, 224), color="gray") for _ in range(self.max_frames)]
            
            avg_fps = vr.get_avg_fps()
            frame_interval = max(1, int(avg_fps / self.fps))
            frame_indices = list(range(0, total_frames, frame_interval))
            
            # Limit to max_frames
            if len(frame_indices) > self.max_frames:
                uniform_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
                frame_indices = uniform_indices.tolist()
            
            # Get frames
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Convert to PIL Images
            images = []
            for frame in frames:
                img = Image.fromarray(frame)
                images.append(img)
            
            return images
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return [Image.new("RGB", (224, 224), color="gray") for _ in range(self.max_frames)]
    
    def _load_audio(self, audio_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load audio if available."""
        if not audio_path:
            return None
        
        if self.audio_folder and not os.path.isabs(audio_path):
            audio_path = os.path.join(self.audio_folder, audio_path)
        
        if not os.path.exists(audio_path):
            return None
        
        try:
            import librosa
            speech, sr = librosa.load(audio_path, sr=16000)
            if speech.ndim > 1:
                speech = np.mean(speech, axis=1)
            # Return as tensor (will be processed by model)
            return torch.from_numpy(speech.astype(np.float32))
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Load video frames
        video_path = sample.get("video", "")
        images = self._load_video(video_path)
        
        # Load audio if available
        audio_path = sample.get("audio")
        audio = self._load_audio(audio_path)
        
        # Get conversations
        conversations = sample.get("conversations", [])
        
        # Format conversations for LLaVA
        # Combine human messages and GPT responses
        prompt_parts = []
        answer_parts = []
        
        for conv in conversations:
            if conv["from"] == "human":
                value = conv["value"]
                # Ensure <image> and <speech> tokens are present
                if "<image>" not in value:
                    value = "<image>\n" + value
                if audio is not None and "<speech>" not in value:
                    value = "<speech>\n" + value
                prompt_parts.append(value)
            elif conv["from"] == "gpt":
                answer_parts.append(conv["value"])
        
        prompt = "\n".join(prompt_parts)
        answer = "\n".join(answer_parts) if answer_parts else ""
        
        return {
            "images": images,
            "audio": audio,
            "prompt": prompt,
            "answer": answer,
            "sample_id": sample.get("id", f"sample_{idx}"),
            "video_path": video_path,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch for training.
        Processes images and creates input_ids with proper loss masking.
        """
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]
        images_list = [item["images"] for item in batch]
        audio_list = [item["audio"] for item in batch]
        
        # Process images
        all_images = []
        image_counts = []
        for images in images_list:
            all_images.extend(images)
            image_counts.append(len(images))
        
        # Process all images at once
        pixel_values = self.processor.image_processor(
            all_images,
            return_tensors="pt"
        )["pixel_values"]
        
        # Split back per sample
        pixel_values_list = []
        idx = 0
        for count in image_counts:
            pixel_values_list.append(pixel_values[idx:idx+count])
            idx += count
        
        # Process text: prompt + answer
        full_texts = []
        for prompt, answer in zip(prompts, answers):
            full_text = f"{prompt}\n{answer}"
            full_texts.append(full_text)
        
        # Tokenize
        text_inputs = self.processor.tokenizer(
            full_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        # Create labels: mask prompt tokens, keep answer tokens
        labels = text_inputs["input_ids"].clone()
        
        # Find where prompt ends and answer begins
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            # Tokenize prompt separately to find its length
            prompt_tokens = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            
            # Mask prompt tokens (set to -100)
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100
        
        # Also mask padding tokens
        attention_mask = text_inputs["attention_mask"]
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": attention_mask,
            "pixel_values": pixel_values_list,
            "labels": labels,
            "sample_ids": [item["sample_id"] for item in batch],
            "audio": audio_list if any(a is not None for a in audio_list) else None,
        }

