"""
Script to evaluate the trained model on the test dataset.

1.  Loads the test data (X_test, y_test).
2.  Loads the best saved model state.
3.  Creates a test DataLoader.
4.  Performs inference on the test set.
5.  Calculates and prints metrics:
    - Accuracy
    - Precision, Recall, F1-score (per class)
    - Confusion Matrix
6.  Saves a plot of the confusion matrix.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Add this block to fix the import error ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# ----------------------------------------------

import config
from crop_disease_hsi import model as model_def
from crop_disease_hsi import data_loader as loader

def get_predictions(model, dataloader, device):
    """
    Runs the model on the test data and returns all predictions and true labels.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get the predicted class (index with the highest score)
            _, preds = torch.max(outputs, 1)
            
            # Move preds and labels to CPU and store them
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plots and saves the confusion matrix.
    """
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',  # Integer format
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Ensure the plots directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def main():
    print("--- Starting Model Evaluation ---")
    print(f"Using device: {config.DEVICE}")
    
    # --- 1. Load Test Data ---
    print("Loading test data...")
    try:
        X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    except FileNotFoundError:
        print(f"Error: Test data not found in {config.PROCESSED_DATA_DIR}")
        print("Please run `python scripts/preprocess_data.py` first.")
        return
        
    # --- 2. Create Test DataLoader ---
    # We create a placeholder dataset for train/val, as they are not needed
    # but the create_dataloaders function expects them.
    _, _, test_loader = loader.create_dataloaders(
        X_test, y_test, # Placeholder for train
        X_test, y_test, # Placeholder for val
        X_test, y_test  # The actual test data
    )

    # --- 3. Initialize and Load Model ---
    print("Initializing and loading model...")
    model = model_def.HSI_CNN_3D(
        num_classes=config.NUM_CLASSES,
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

    # --- 4. Get Predictions ---
    print("Generating predictions on test set...")
    y_true, y_pred = get_predictions(model, test_loader, config.DEVICE)
    
    # --- 5. Report Metrics ---
    print("\n" + "="*30)
    print("      Classification Report")
    print("="*30)
    
    # Get class names from config
    class_names = [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)]
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # --- 6. Plot Confusion Matrix ---
    cm_save_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_save_path)
    
    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    main()