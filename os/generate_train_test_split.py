import os
import json
import random
import numpy as np

def generate_data_split(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Generate train/val/test split for dataset.
    
    Args:
        data_dir: Directory containing the data files
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all base filenames (without extensions) that have all required files
    all_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.png') and not file.endswith('_keypoints.png') and not file.endswith('_road_mask.png'):
            base_name = file[:-4]  # Remove .png extension
            
            # Check if all required files exist
            required_files = [
                f"{base_name}.png",
                f"{base_name}_keypoints.png",
                f"{base_name}_road_mask.png",
                f"{base_name}_graph.json"
            ]
            
            if all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
                all_files.append(base_name)
    
    # Shuffle the files
    random.shuffle(all_files)
    
    # Calculate split sizes
    total_size = len(all_files)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split the data
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    # Create split dictionary
    split_dict = {
        'train': sorted(train_files),
        'validation': sorted(val_files),
        'test': sorted(test_files)
    }
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(data_dir), 'data_split.json')
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)
    
    print(f"Split created with {len(train_files)} train, {len(val_files)} validation, and {len(test_files)} test samples")
    print(f"Split saved to {output_path}")
    
    return split_dict

if __name__ == "__main__":
    data_dir = "./os/data"
    split = generate_data_split(data_dir,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1)