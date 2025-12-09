"""
FirstSight: Data Preparation for VLM Distillation
Prepares egocentric QA samples for knowledge distillation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging
from PIL import Image
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EgocentricQADataset(Dataset):
    """
    Dataset for egocentric question answering.
    Can load from JSON files or use synthetic/fallback data.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 500,
        use_synthetic: bool = True
    ):
        """
        Initialize egocentric QA dataset.
        
        Args:
            data_path: Path to data JSON file (if available)
            num_samples: Number of samples to use
            use_synthetic: Use synthetic questions if no data file
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.samples = []
        
        if data_path and Path(data_path).exists():
            logger.info(f"Loading data from {data_path}")
            self._load_from_file(data_path)
        elif use_synthetic:
            logger.info("Using synthetic egocentric QA samples")
            self._create_synthetic_data()
        else:
            raise ValueError("No data available and synthetic data disabled")
        
        # Limit to num_samples
        if len(self.samples) > num_samples:
            self.samples = random.sample(self.samples, num_samples)
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
    
    def _load_from_file(self, data_path: str):
        """Load data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Expect format: [{"question": "...", "answer": "...", "image_path": "..."}, ...]
        for item in data:
            self.samples.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "image_path": item.get("image_path", None)
            })
    
    def _create_synthetic_data(self):
        """
        Create synthetic egocentric QA samples for distillation.
        Uses text-only questions that simulate egocentric scenarios.
        """
        # Egocentric question templates
        templates = [
            # Object interaction
            ("What object did I just pick up?", "mug"),
            ("What am I holding in my right hand?", "phone"),
            ("What tool am I currently using?", "hammer"),
            ("What item did I place on the table?", "book"),
            ("What object is closest to my hand?", "keys"),
            
            # Spatial reasoning
            ("Where did I put the cup?", "on the kitchen counter"),
            ("Where am I looking at?", "at the computer screen"),
            ("What direction am I facing?", "towards the window"),
            ("Where is the nearest door?", "to my left"),
            ("What room am I in?", "living room"),
            
            # Action understanding
            ("What action am I performing?", "typing on keyboard"),
            ("What task am I currently doing?", "cooking dinner"),
            ("What did I just finish doing?", "washing dishes"),
            ("What am I about to do?", "open the refrigerator"),
            ("How many steps have I taken?", "approximately 10 steps"),
            
            # Attention/focus
            ("What was I looking at 5 seconds ago?", "the clock on the wall"),
            ("What caught my attention?", "a notification on my phone"),
            ("What am I focused on?", "the document I'm reading"),
            ("What did I glance at?", "the person walking by"),
            
            # Counting and enumeration
            ("How many items are on the table?", "five items"),
            ("How many people are in the room?", "three people"),
            ("How many steps did I climb?", "twelve steps"),
            ("How many objects did I touch?", "four objects"),
            
            # Attributes and details
            ("What color is the object I'm holding?", "blue"),
            ("What size is the cup?", "medium sized"),
            ("What material is this made of?", "plastic"),
            ("What shape is this object?", "rectangular"),
            
            # Temporal reasoning
            ("When did I enter this room?", "about 2 minutes ago"),
            ("How long have I been here?", "approximately 10 minutes"),
            ("What time did I start this task?", "around 3 PM"),
            ("How long ago did I see that object?", "about 30 seconds ago"),
            
            # Scene understanding
            ("What type of environment am I in?", "indoor kitchen"),
            ("What is the lighting condition?", "bright natural light"),
            ("What is the temperature like?", "comfortable room temperature"),
            ("What sounds can I hear?", "ambient background noise"),
        ]
        
        # Expand templates with variations
        for question, answer in templates:
            self.samples.append({
                "question": question,
                "answer": answer,
                "image_path": None  # Text-only for now
            })
        
        # Duplicate samples to reach desired count
        while len(self.samples) < self.num_samples:
            self.samples.extend(self.samples[:min(len(templates), self.num_samples - len(self.samples))])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn_distillation(batch: List[Dict], processor) -> Dict:
    """
    Collate function for distillation dataloader.
    Prepares inputs for both teacher and student models.
    
    Args:
        batch: List of samples
        processor: Model processor
    
    Returns:
        Dictionary with collated inputs
    """
    questions = [sample["question"] for sample in batch]
    answers = [sample["answer"] for sample in batch]
    
    # Prepare messages for chat template
    messages_list = []
    for question in questions:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": f"Question: {question}\nAnswer concisely:"}]
        }]
        messages_list.append(messages)
    
    # Apply chat template
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    
    # Process inputs
    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt"
    )
    
    # Add answers for reference (not used in forward pass, but useful for evaluation)
    inputs["answers"] = answers
    inputs["questions"] = questions
    
    return inputs


def create_egocentric_dataloader(
    data_path: Optional[str] = None,
    batch_size: int = 2,
    num_samples: int = 500,
    num_workers: int = 4,
    shuffle: bool = True,
    processor = None
) -> DataLoader:
    """
    Create DataLoader for egocentric QA distillation.
    
    Args:
        data_path: Path to data file (optional)
        batch_size: Batch size
        num_samples: Number of samples to use
        num_workers: Number of data loading workers
        shuffle: Shuffle data
        processor: Model processor for collation
    
    Returns:
        DataLoader instance
    """
    logger.info("Creating egocentric QA dataloader")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num samples: {num_samples}")
    logger.info(f"  Shuffle: {shuffle}")
    
    dataset = EgocentricQADataset(
        data_path=data_path,
        num_samples=num_samples,
        use_synthetic=True
    )
    
    if processor is None:
        logger.warning("No processor provided, using default collation")
        collate_fn = None
    else:
        collate_fn = lambda batch: collate_fn_distillation(batch, processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"✓ DataLoader created with {len(dataloader)} batches")
    
    return dataloader


def create_train_val_split(
    data_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    train_samples: int = 400,
    val_samples: int = 100,
    batch_size: int = 2,
    processor = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to data file (used for both if train/val paths not specified)
        train_data_path: Path to training data file (overrides data_path)
        val_data_path: Path to validation data file (overrides data_path)
        train_samples: Number of training samples
        val_samples: Number of validation samples
        batch_size: Batch size
        processor: Model processor
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Creating train/val split")
    
    # Use specific paths if provided, otherwise fall back to data_path
    train_path = train_data_path or data_path
    val_path = val_data_path or data_path
    
    train_loader = create_egocentric_dataloader(
        data_path=train_path,
        batch_size=batch_size,
        num_samples=train_samples,
        shuffle=True,
        processor=processor
    )
    
    val_loader = create_egocentric_dataloader(
        data_path=val_path,
        batch_size=batch_size,
        num_samples=val_samples,
        shuffle=False,
        processor=processor
    )
    
    logger.info(f"✓ Train batches: {len(train_loader)}")
    logger.info(f"✓ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing data loading...")
    
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    dataloader = create_egocentric_dataloader(
        batch_size=4,
        num_samples=20,
        processor=processor
    )
    
    logger.info("\nTesting batch loading:")
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i+1}:")
        logger.info(f"  Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"  Questions: {batch['questions'][:2]}")
        logger.info(f"  Answers: {batch['answers'][:2]}")
        if i >= 2:
            break
    
    logger.info("\n✓ Data loading test complete!")

