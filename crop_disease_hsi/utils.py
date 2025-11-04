"""
Utility functions for the project.
- Saving and loading models
- Plotting training history
"""

import torch
import matplotlib.pyplot as plt
import os
import config

def save_model(model, history, file_path=config.MODEL_SAVE_PATH):
    """
    Saves the model's state_dict and the training history.
    """
    print(f"Saving model to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, file_path)
    print("Model saved successfully.")

def plot_and_save_history(history, save_dir=config.PLOTS_DIR):
    """
    Plots training & validation loss and accuracy, and saves the plot.
    """
    print("Plotting training history...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data from history dict
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- Plot 1: Loss ---
    ax1.plot(epochs, train_loss, 'bo-', label='Training loss')
    ax1.plot(epochs, val_loss, 'ro-', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot 2: Accuracy ---
    ax2.plot(epochs, train_acc, 'bo-', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save the figure
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    
    print(f"Training history plot saved to {plot_path}")
    # plt.show() # Optional: uncomment to display the plot