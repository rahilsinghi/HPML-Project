"""
Create a realistic synthetic egocentric QA dataset
Mimics characteristics of EgoQA/Ego4D datasets
"""

import json
import random
from pathlib import Path

# Realistic egocentric activities and scenarios from Ego4D-style datasets
ACTIVITIES = [
    "cooking", "cleaning", "gardening", "repair work", "assembling furniture",
    "grocery shopping", "organizing", "packing", "exercising", "crafting"
]

OBJECTS = [
    "knife", "cutting board", "pot", "pan", "spatula", "bowl", "plate", "cup",
    "spoon", "fork", "towel", "sponge", "soap", "screwdriver", "hammer", "wrench",
    "tape", "scissors", "box", "bag", "bottle", "can", "jar", "carton", "package"
]

LOCATIONS = [
    "kitchen counter", "dining table", "sink", "stove", "refrigerator",
    "drawer", "cabinet", "shelf", "workbench", "floor", "chair", "desk"
]

ACTIONS = [
    "picking up", "putting down", "cutting", "stirring", "washing", "wiping",
    "opening", "closing", "moving", "assembling", "holding", "carrying"
]

# Question templates based on real EgoQA patterns
QUESTION_TEMPLATES = [
    # Object-centric questions
    "What object did I {action} before {object2}?",
    "Where did I put the {object}?",
    "What did I pick up from the {location}?",
    "How many {object}s are on the {location}?",
    
    # Activity-centric questions  
    "What was I doing before {activity}?",
    "Where was I {activity}?",
    "What did I use for {activity}?",
    
    # Temporal questions
    "What did I do after picking up the {object}?",
    "What was the first thing I did in the {location}?",
    "What did I do right before this?",
    
    # State questions
    "Did I close the {object}?",
    "Was the {object} on the {location}?",
    "Is the {object} still in my hand?",
]

def generate_realistic_sample(idx):
    """Generate a single realistic QA sample"""
    activity = random.choice(ACTIVITIES)
    obj1 = random.choice(OBJECTS)
    obj2 = random.choice([o for o in OBJECTS if o != obj1])
    location = random.choice(LOCATIONS)
    action = random.choice(ACTIONS)
    
    # Select question template and fill it
    template = random.choice(QUESTION_TEMPLATES)
    question = template.format(
        action=action,
        object=obj1,
        object2=obj2,
        location=location,
        activity=activity
    )
    
    # Generate contextual answer
    answer_templates = [
        f"the {obj2}",
        f"on the {location}",
        f"I was {action} the {obj1}",
        f"Yes, I did",
        f"No, I didn't",
        f"two {obj1}s",
        f"three {obj1}s",
    ]
    answer = random.choice(answer_templates)
    
    # Add some context/rationale
    context = f"While {activity}, I {action} {obj1} near the {location}."
    
    return {
        "id": f"ego_qa_{idx:05d}",
        "question": question,
        "answer": answer,
        "context": context,
        "activity": activity,
        "objects": [obj1, obj2],
        "location": location,
        "difficulty": random.choice(["easy", "medium", "hard"]),
        "question_type": random.choice([
            "object_recognition", "action_sequence", "spatial_reasoning", 
            "temporal_reasoning", "counting", "state_verification"
        ])
    }

def create_dataset(num_samples, output_path):
    """Create a realistic synthetic dataset"""
    print(f"Generating {num_samples} realistic egocentric QA samples...")
    
    samples = []
    for i in range(num_samples):
        sample = generate_realistic_sample(i)
        samples.append(sample)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"  Total samples: {len(samples)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print statistics
    question_types = {}
    difficulties = {}
    for sample in samples:
        qt = sample['question_type']
        diff = sample['difficulty']
        question_types[qt] = question_types.get(qt, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\n  Question Types:")
    for qt, count in sorted(question_types.items()):
        print(f"    {qt}: {count}")
    
    print(f"\n  Difficulty Distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"    {diff}: {count}")

if __name__ == "__main__":
    # Create training set (5000 samples)
    create_dataset(
        num_samples=5000,
        output_path="data/egoqa_train_5k.json"
    )
    
    # Create validation set (1000 samples)
    create_dataset(
        num_samples=1000,
        output_path="data/egoqa_val_1k.json"
    )
    
    print("\n✓ Dataset generation complete!")

