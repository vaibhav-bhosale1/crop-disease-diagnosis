"""
Inference script to make a full classification map for the Indian Pines dataset.

1.  Loads the trained model.
2.  Loads the *entire* `Indian_pines_corrected.mat` HSI cube.
3.  Preprocesses the cube (normalize, pad, extract patches for *every* pixel).
4.  Runs inference on all patches.
5.  Reconstructs the full 145x145 classification map.
6.  Saves the map as a .png image in `outputs/plots/`.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

# --- Add this block to fix the import error ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# ----------------------------------------------

import config
from crop_disease_hsi import model as model_def
from crop_disease_hsi import preprocess
from crop_disease_hsi.data_loader import HSIDataset

def load_hsi_cube(data_path, data_key):
    """Loads and normalizes the full HSI cube."""
    try:
        hsi_cube_mat = scipy.io.loadmat(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}")
        print("Please ensure the file exists in `data/raw/`.")
        return None
        
    hsi_cube = hsi_cube_mat[data_key]
    
    # Normalize the HSI cube (scale each band to 0-1)
    hsi_cube = hsi_cube.astype(np.float32)
    for i in range(hsi_cube.shape[2]):
        hsi_cube[:, :, i] = minmax_scale(hsi_cube[:, :, i].ravel()).reshape(hsi_cube.shape[:2])
    
    return hsi_cube

def predict(model, dataloader, device):
    """Runs the model on the data and returns all predictions."""
    model.eval()
    all_preds = []
    
    progress_bar = tqdm(dataloader, desc="Generating Map", leave=False)
    
    with torch.no_grad():
        for inputs, _ in progress_bar: # We don't have labels, so use _
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_preds)

def reconstruct_full_map(preds, height, width):
    """Reshapes the 1D array of predictions back into the full 2D map."""
    return preds.reshape((height, width))

def save_classification_map(pred_map, gt_mask, save_path):
    """
    Saves the 2D prediction map as a color image.
    We also load the ground truth mask to show "unclassified" areas.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # We add 1 to our predictions (0-15) to match the GT mask's classes (1-16)
    # Then, we set all "unclassified" pixels (where GT is 0) to 0.
    final_map = pred_map + 1
    final_map[gt_mask == 0] = 0
    
    plt.figure(figsize=(10, 8))
    
    # Use a colormap with enough colors for all 17 classes (0-16)
    # 'nipy_spectral' is a good choice
    cmap = plt.cm.get_cmap('nipy_spectral', config.NUM_CLASSES + 1)
    
    plt.imshow(final_map, cmap=cmap)
    plt.title("Indian Pines Classification Map")
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    # Create a color bar
    cbar = plt.colorbar(ticks=range(config.NUM_CLASSES + 1))
    # Get a list of class names, including "Unclassified" at 0
    class_names_list = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES + 1)]
    cbar.set_ticklabels(class_names_list)
    
    plt.savefig(save_path)
    print(f"Full classification map saved to {save_path}")

def main():
    print("--- Starting Full Map Inference ---")
    print(f"Using device: {config.DEVICE}")
    
    # --- 1. Load Model ---
    print("Loading trained model...")
    model = model_def.HSI_CNN_3D(
        num_classes=config.NUM_CLASSES, # Should be 16
        num_bands=config.NUM_BANDS,
        window_size=config.SPATIAL_WINDOW_SIZE
    ).to(config.DEVICE)
    
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("Please run `python scripts/train.py` first.")
        return

    # --- 2. Load and Preprocess Full HSI Cube ---
    print(f"Loading full HSI cube from {config.DATA_FILE_PATH}...")
    hsi_cube = load_hsi_cube(config.DATA_FILE_PATH, config.DATA_KEY)
    if hsi_cube is None:
        return
        
    # Also load GT mask just to set background pixels to 0
    gt_mask = scipy.io.loadmat(config.GT_FILE_PATH)[config.GT_KEY]
    
    height, width, _ = hsi_cube.shape
    
    padding_width = config.SPATIAL_WINDOW_SIZE // 2
    
    hsi_cube_padded = preprocess.apply_padding(hsi_cube, padding_width)
    
    # Create a dummy mask of ALL ZEROS.
    # This tricks the 'extract_patches' function into extracting a patch
    # for *every single pixel* in the 145x145 image.
    dummy_mask = np.zeros((height, width), dtype=int)
    
    X_patches, y_dummy_labels = preprocess.extract_patches(
        hsi_cube_padded,
        dummy_mask,
        config.SPATIAL_WINDOW_SIZE
    )
    
    print(f"Extracted {X_patches.shape[0]} total patches for full map.")
    
    # --- 3. Create DataLoader ---
    inference_dataset = HSIDataset(X_patches, y_dummy_labels)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Must be False
        num_workers=2
    )
    
    # --- 4. Get Predictions ---
    predictions = predict(model, inference_loader, config.DEVICE)
    
    # --- 5. Reconstruct and Save Map ---
    prediction_map = reconstruct_full_map(predictions, height, width)
    
    save_path = os.path.join(
        config.PLOTS_DIR, "full_classification_map.png"
    )
    
    save_classification_map(prediction_map, gt_mask, save_path)
    
    print("\n--- Full Map Inference Finished ---")

if __name__ == "__main__":
    # This script now runs without arguments
    main()