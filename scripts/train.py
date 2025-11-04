"""
Main training script for the 3D-CNN HSI model.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Add this block to fix the import error ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# ----------------------------------------------

import config
from crop_disease_hsi import model as model_def
from crop_disease_hsi import data_loader as loader
from crop_disease_hsi import utils

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one full training pass over the dataset."""
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        # Move data to the configured device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 3. Backward pass
        loss.backward()
        
        # 4. Optimize
        optimizer.step()
        
        # --- Statistics ---
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix(
            loss=running_loss / total_samples,
            acc=correct_preds.double().item() / total_samples
        )
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def validate_one_epoch(model, dataloader, criterion, device):
    """Performs one full validation pass."""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    # No gradients needed for validation
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # --- Statistics ---
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=running_loss / total_samples,
                acc=correct_preds.double().item() / total_samples
            )
            
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def main():
    print("--- Starting Model Training ---")
    print(f"Using device: {config.DEVICE}")
    torch.manual_seed(config.RANDOM_STATE) # For reproducibility
    
    # --- 1. Load Data ---
    print("Loading processed data...")
    try:
        X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.npy'))
        # X_test, y_test are not needed for training
    except FileNotFoundError:
        print(f"Error: Processed data not found in {config.PROCESSED_DATA_DIR}")
        print("Please run `python scripts/preprocess_data.py` first.")
        return

    # --- 2. Create DataLoaders ---
    train_loader, val_loader, _ = loader.create_dataloaders(
        X_train, y_train, X_val, y_val, X_val, y_val # Using val as placeholder for test
    )
    
    # --- 3. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    model = model_def.HSI_CNN_3D(
        num_classes=config.NUM_CLASSES,
        num_bands=config.NUM_BANDS,
        window_size=config.SPATIAL_WINDOW_SIZE
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 4. Training Loop ---
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Log results
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_model(model, history, config.MODEL_SAVE_PATH)
            
    # --- 5. Post-Training ---
    print("-" * 30)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot and save history
    utils.plot_and_save_history(history, config.PLOTS_DIR)
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()