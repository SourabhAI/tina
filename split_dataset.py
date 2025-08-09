#!/usr/bin/env python3
"""
Split questions.json into train/validation/test sets (80/10/10)
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def split_dataset(input_file='questions.json', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the questions dataset into train, validation, and test sets.
    
    Args:
        input_file: Path to the input JSON file
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # First split: separate out test set (10%)
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_ratio, 
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation from remaining 90%
    # Calculate validation size relative to train+val
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_relative,
        random_state=42,
        shuffle=True
    )
    
    # Print split statistics
    print(f"\nDataset split:")
    print(f"Training set: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    # Verify no overlap between sets
    train_ids = set([q.get('id', str(q)) for q in train_data])
    val_ids = set([q.get('id', str(q)) for q in val_data])
    test_ids = set([q.get('id', str(q)) for q in test_data])
    
    assert len(train_ids & val_ids) == 0, "Overlap found between train and validation sets"
    assert len(train_ids & test_ids) == 0, "Overlap found between train and test sets"
    assert len(val_ids & test_ids) == 0, "Overlap found between validation and test sets"
    
    print("\n✓ No overlap between sets confirmed")
    
    # Save the splits
    output_files = {
        'train_questions.json': train_data,
        'val_questions.json': val_data,
        'test_questions.json': test_data
    }
    
    for filename, data_split in output_files.items():
        with open(filename, 'w') as f:
            json.dump(data_split, f, indent=2)
        print(f"✓ Saved {filename}")
    
    # Also save a summary file with statistics
    summary = {
        'total_samples': len(data),
        'split_ratios': {
            'train': train_ratio,
            'validation': val_ratio,
            'test': test_ratio
        },
        'split_counts': {
            'train': len(train_data),
            'validation': len(val_data),
            'test': len(test_data)
        },
        'split_percentages': {
            'train': round(len(train_data)/len(data)*100, 2),
            'validation': round(len(val_data)/len(data)*100, 2),
            'test': round(len(test_data)/len(data)*100, 2)
        },
        'random_seed': 42
    }
    
    with open('dataset_split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Saved dataset_split_summary.json")
    
    # Print sample questions from each set
    print("\n" + "="*60)
    print("SAMPLE QUESTIONS FROM EACH SET:")
    print("="*60)
    
    for set_name, data_split in [('TRAINING', train_data), ('VALIDATION', val_data), ('TEST', test_data)]:
        print(f"\n### {set_name} SET SAMPLES ###")
        samples = random.sample(data_split, min(3, len(data_split)))
        for i, sample in enumerate(samples, 1):
            print(f"{i}. {sample['question_text'][:100]}{'...' if len(sample['question_text']) > 100 else ''}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Run the split
    train, val, test = split_dataset()
    
    print("\n" + "="*60)
    print("Dataset splitting completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("- train_questions.json (80% of data)")
    print("- val_questions.json (10% of data)")
    print("- test_questions.json (10% of data)")
    print("- dataset_split_summary.json (split statistics)")
