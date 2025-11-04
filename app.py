"""
Streamlit Web App for Indian Pines Classification.

Loads the trained 3D-CNN model and runs inference on the
entire Indian Pines HSI cube to generate a full classification map.
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
import streamlit as st

# --- Add this block to fix the import error ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
# ----------------------------------------------

import config
from crop_disease_hsi import model as model_def
from crop_disease_hsi import preprocess
from crop_disease_hsi.data_loader import HSIDataset

# --- 1. Caching Functions ---

@st.cache_resource  # Caches the model for performance
def load_trained_model():
    """Loads and returns the trained 3D-CNN model."""
    print("Loading model...")
    model_path = config.MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run `python scripts/train.py` first.")
        return None

    model = model_def.HSI_CNN_3D(
        num_classes=config.NUM_CLASSES,
        num_bands=config.NUM_BANDS,
        window_size=config.SPATIAL_WINDOW_SIZE
    ).to(config.DEVICE)
    
    map_location = torch.device(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")
    return model

@st.cache_data  # Caches the loaded data
def load_indian_pines_data():
    """Loads and returns the HSI cube and GT mask."""
    print("Loading Indian Pines data...")
    try:
        hsi_cube_mat = scipy.io.loadmat(config.DATA_FILE_PATH)
        gt_mask_mat = scipy.io.loadmat(config.GT_FILE_PATH)
    except FileNotFoundError:
        st.error(f"Error: Dataset files not found in {config.RAW_DATA_DIR}. Please download them first.")
        return None, None
        
    hsi_cube = hsi_cube_mat[config.DATA_KEY]
    gt_mask = gt_mask_mat[config.GT_KEY]
    
    # Normalize the HSI cube
    hsi_cube = hsi_cube.astype(np.float32)
    for i in range(hsi_cube.shape[2]):
        hsi_cube[:, :, i] = minmax_scale(hsi_cube[:, :, i].ravel()).reshape(hsi_cube.shape[:2])
    
    return hsi_cube, gt_mask

# --- 2. Helper Functions (Adapted from predict.py) ---

def predict(model, dataloader, device):
    """Runs the model on the data and returns all predictions."""
    model.eval()
    all_preds = []
    
    # Use st.progress for a dynamic progress bar
    progress_bar = st.progress(0)
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_batches)
            
    progress_bar.empty() # Clear the progress bar
    return np.array(all_preds)

def reconstruct_full_map(preds, height, width):
    """Reshapes the 1D array of predictions back into the full 2D map."""
    return preds.reshape((height, width))

def create_map_figure(map_data, title):
    """Creates a Matplotlib figure for a classification map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use 'nipy_spectral' colormap with 17 discrete colors (0-16)
    cmap = plt.cm.get_cmap('nipy_spectral', config.NUM_CLASSES + 1)
    
    im = ax.imshow(map_data, cmap=cmap, vmin=-0.5, vmax=16.5)
    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    
    # Create a color bar
    cbar = fig.colorbar(im, ticks=range(config.NUM_CLASSES + 1))
    class_names_list = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES + 1)]
    cbar.set_ticklabels(class_names_list)
    
    return fig

# --- 3. Streamlit App UI ---

st.title("🛰️ Indian Pines HSI Classification")
st.subheader("Using a 3D-CNN Model in PyTorch")

# Load model
model = load_trained_model()

if model:
    # Load data
    hsi_cube, gt_mask = load_indian_pines_data()
    
    if hsi_cube is not None and gt_mask is not None:
        st.info(f"Loaded Indian Pines dataset. Cube: {hsi_cube.shape}, Mask: {gt_mask.shape}")
        
        # --- Generate Map Button ---
        if st.button("Generate Full Classification Map"):
            with st.spinner('Preprocessing data... This may take a moment.'):
                
                # --- 1. Preprocess Full HSI Cube ---
                height, width, _ = hsi_cube.shape
                padding_width = config.SPATIAL_WINDOW_SIZE // 2
                hsi_cube_padded = preprocess.apply_padding(hsi_cube, padding_width)
                
                # Create patches for *every* pixel
                dummy_mask = np.zeros((height, width), dtype=int)
                X_patches, y_dummy_labels = preprocess.extract_patches(
                    hsi_cube_padded,
                    dummy_mask,
                    config.SPATIAL_WINDOW_SIZE
                )
                
                # --- 2. Create DataLoader ---
                inference_dataset = HSIDataset(X_patches, y_dummy_labels)
                inference_loader = DataLoader(
                    inference_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False
                )
            
            # --- 3. Get Predictions ---
            st.write("Running model inference on all 21,025 pixels...")
            predictions = predict(model, inference_loader, config.DEVICE)
            
            # --- 4. Reconstruct and Plot ---
            st.write("Generating maps...")
            prediction_map = reconstruct_full_map(predictions, height, width)
            
            # Add 1 to predictions (0-15) to match GT classes (1-16)
            # Then, mask out the unclassified pixels (where GT == 0)
            final_pred_map = prediction_map + 1
            final_pred_map[gt_mask == 0] = 0
            
            # --- 5. Display Results ---
            st.subheader("Classification Results")
            
            # Create figures
            fig_gt = create_map_figure(gt_mask, "Ground Truth Map")
            fig_pred = create_map_figure(final_pred_map, "Model Prediction Map")
            
            # Display side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_gt)
            with col2:
                st.pyplot(fig_pred)

else:
    st.warning("Model is not loaded. Please train the model first by running `python scripts/train.py`.")