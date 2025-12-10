"""
Lightweight profiling utilities for training.
Uses CUDA events for timing and tracks memory/throughput metrics.
"""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


class TrainingProfiler:
    """
    Lightweight profiler for training metrics.
    Tracks timing, memory, and throughput.
    """
    
    def __init__(self, output_dir: str = "experiments/profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing metrics
        self.step_times: List[float] = []
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []
        self.optimizer_times: List[float] = []
        
        # CUDA events for precise timing
        self.forward_start = torch.cuda.Event(enable_timing=True)
        self.forward_end = torch.cuda.Event(enable_timing=True)
        self.backward_start = torch.cuda.Event(enable_timing=True)
        self.backward_end = torch.cuda.Event(enable_timing=True)
        self.optimizer_start = torch.cuda.Event(enable_timing=True)
        self.optimizer_end = torch.cuda.Event(enable_timing=True)
        
        # Memory tracking
        self.peak_memory: float = 0.0
        self.optimizer_memory: float = 0.0
        
        # Throughput
        self.tokens_processed: int = 0
        self.samples_processed: int = 0
        
        # Epoch tracking
        self.epoch_start_time: Optional[float] = None
        self.epoch_times: List[float] = []
        
        # Wall clock
        self.wall_clock_start: float = time.time()
    
    def start_epoch(self):
        """Mark start of epoch."""
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        """Mark end of epoch and record duration."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            self.epoch_start_time = None
    
    def start_forward(self):
        """Mark start of forward pass."""
        self.forward_start.record()
    
    def end_forward(self):
        """Mark end of forward pass and record duration."""
        self.forward_end.record()
        torch.cuda.synchronize()
        forward_time = self.forward_start.elapsed_time(self.forward_end) / 1000.0  # Convert to seconds
        self.forward_times.append(forward_time)
    
    def start_backward(self):
        """Mark start of backward pass."""
        self.backward_start.record()
    
    def end_backward(self):
        """Mark end of backward pass and record duration."""
        self.backward_end.record()
        torch.cuda.synchronize()
        backward_time = self.backward_start.elapsed_time(self.backward_end) / 1000.0  # Convert to seconds
        self.backward_times.append(backward_time)
    
    def start_optimizer(self):
        """Mark start of optimizer step."""
        self.optimizer_start.record()
    
    def end_optimizer(self):
        """Mark end of optimizer step and record duration."""
        self.optimizer_end.record()
        torch.cuda.synchronize()
        optimizer_time = self.optimizer_start.elapsed_time(self.optimizer_end) / 1000.0
        self.optimizer_times.append(optimizer_time)
    
    def record_step(self, batch_size: int, num_tokens: int):
        """
        Record a complete training step.
        
        Args:
            batch_size: Batch size for this step
            num_tokens: Number of tokens processed
        """
        # Calculate step time from forward + backward + optimizer
        if len(self.forward_times) > 0 and len(self.backward_times) > 0:
            step_time = (
                self.forward_times[-1] +
                self.backward_times[-1] +
                (self.optimizer_times[-1] if len(self.optimizer_times) > 0 else 0.0)
            )
            self.step_times.append(step_time)
        
        # Update memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, current_memory)
        
        # Update throughput
        self.tokens_processed += num_tokens
        self.samples_processed += batch_size
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        def mean(data: List[float]) -> float:
            return sum(data) / len(data) if data else 0.0
        
        total_time = time.time() - self.wall_clock_start
        
        stats = {
            "wall_clock_time_total": total_time,
            "epoch_time_mean": mean(self.epoch_times),
            "epoch_time_total": sum(self.epoch_times),
            "step_time_mean": mean(self.step_times),
            "step_time_median": percentile(self.step_times, 0.5),
            "step_time_p95": percentile(self.step_times, 0.95),
            "forward_time_mean": mean(self.forward_times),
            "backward_time_mean": mean(self.backward_times),
            "optimizer_time_mean": mean(self.optimizer_times),
            "tokens_processed": self.tokens_processed,
            "samples_processed": self.samples_processed,
            "tokens_per_second": self.tokens_processed / total_time if total_time > 0 else 0.0,
            "samples_per_second": self.samples_processed / total_time if total_time > 0 else 0.0,
            "peak_memory_gb": self.peak_memory,
            "optimizer_memory_gb": self.optimizer_memory,
        }
        
        return stats
    
    def save(self, filename: str = "profiling_metrics.json"):
        """Save profiling metrics to JSON."""
        stats = self.get_stats()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Profiling metrics saved to {output_path}")
        return stats
    
    def save_csv(self, filename: str = "profiling_metrics.csv"):
        """Save profiling metrics to CSV."""
        stats = self.get_stats()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in stats.items():
                writer.writerow([key, value])
        
        print(f"Profiling metrics saved to CSV: {output_path}")
        return stats

