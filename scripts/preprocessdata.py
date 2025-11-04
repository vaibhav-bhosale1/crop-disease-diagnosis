"""
Script to preprocess the raw Indian Pines dataset.

1.  Loads the HSI cube and ground truth (GT) mask from .mat files.
2.  Normalizes the HSI cube.
3.  Applies padding to the HSI cube.
4.  Iterates through the GT mask:
    - For each *labeled* pixel (class > 0), extract its 3D patch.
5.  Aggregates all patches (X) and labels (y).
6.  Splits (X, y) into training, validation, and test sets.
7.  Saves the processed sets to `data/processed/` as .npy files.
"""

import sys
import os
from pathlib import Path
import numpy as np
import scipy.io  # To load .mat files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from crop_disease_hsi import preprocess

def load_mat_files(data_path, gt_path):
    """Loads .mat files for data and ground truth."""
    try:
        hsi_cube_mat = scipy.io.loadmat(data_path)
        gt_mask_mat = scipy.io.loadmat(gt_path)
    except FileNotFoundError:
        print(f"Error: Dataset files not found in {config.RAW_DATA_DIR}")
        print(f"Please download '{os.path.basename(config.DATA_FILE_PATH)}' and")
        print(f"'{os.path.basename(config.GT_FILE_PATH)}'")
        print("from: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes")
        return None, None
        
    # Extract the data using the keys specified in config
    hsi_cube = hsi_cube_mat[config.DATA_KEY]
    gt_mask = gt_mask_mat[config.GT_KEY]
    
    # Normalize the HSI cube (scale each band to 0-1)
    # Shape is (H, W, C), so scale along axes (0, 1)
    hsi_cube = hsi_cube.astype(np.float32)
    for i in range(hsi_cube.shape[2]):
        hsi_cube[:, :, i] = minmax_scale(hsi_cube[:, :, i].ravel()).reshape(hsi_cube.shape[:2])

    print(f"Loaded HSI cube. Shape: {hsi_cube.shape}")
    print(f"Loaded GT mask. Shape: {gt_mask.shape}")
    
    return hsi_cube, gt_mask

def extract_patches_from_mask(hsi_cube, gt_mask, window_size):
    """
    Extracts 3D patches *only* for labeled pixels.
    Ignores pixels where the mask is 0 (Unclassified).
    """
    patch_radius = window_size // 2
    
    # 1. Pad the HSI cube
    padded_cube = preprocess.apply_padding(hsi_cube, padding_width=patch_radius)
    
    all_patches = []
    all_labels = []
    
    height, width = gt_mask.shape
    
    print(f"Extracting patches for {np.count_nonzero(gt_mask)} labeled pixels...")
    # Iterate over the original (unpadded) mask dimensions
    for r in tqdm(range(height)):
        for c in range(width):
            label = gt_mask[r, c]
            
            # --- This is the key change ---
            # Only extract a patch if it's NOT background
            if label > 0:
                # The corresponding center pixel in the *padded* cube
                padded_r = r + patch_radius
                padded_c = c + patch_radius
                
                # Extract the 3D patch
                patch = padded_cube[
                    padded_r - patch_radius : padded_r + patch_radius + 1,
                    padded_c - patch_radius : padded_c + patch_radius + 1,
                    :
                ]
                
                all_patches.append(patch)
                all_labels.append(label)
                
    # Convert lists to numpy arrays
    # Note: We subtract 1 from labels so that class "1" becomes "0", "16" becomes "15"
    # This is because PyTorch CrossEntropyLoss expects classes from 0 to C-1
    # Our model will now have 16 classes (0-15)
    X = np.array(all_patches, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64) - 1 
    
    return X, y

def main():
    print("--- Starting Data Preprocessing for Indian Pines ---")
    
    # 1. Load Data
    hsi_cube, gt_mask = load_mat_files(config.DATA_FILE_PATH, config.GT_FILE_PATH)
    if hsi_cube is None:
        return
        
    # 2. Extract Patches
    X, y = extract_patches_from_mask(
        hsi_cube, 
        gt_mask, 
        config.SPATIAL_WINDOW_SIZE
    )
    
    print(f"\nTotal labeled patches extracted: {X.shape[0]}")
    print(f"Patch shape: {X.shape[1:]}")
    print(f"Labels shape: {y.shape} (Labels shifted to 0-15)")
    print(f"Unique labels: {np.unique(y)}")

    # --- 3. Split the data ---
    print("Splitting data into train, validation, and test sets...")
    
    # First split: Train + Val vs. Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # Essential for imbalanced datasets
    )
    
    # Second split: Train vs. Val
    val_size_adjusted = config.VAL_SPLIT_SIZE / (1.0 - config.TEST_SPLIT_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=config.RANDOM_STATE,
        stratify=y_train_val
    )
    
    print(f"  Training set:   {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set:       {X_test.shape[0]} samples")
    
    # --- 4. Save processed data ---
    print(f"Saving processed data to {config.PROCESSED_DATA_DIR}...")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    print("-" * 30)
    print("Preprocessing complete!")
    print("-" * 30)

if __name__ == "__main__":
    main()