"""
PyTorch Dataset and DataLoader definitions.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config

class HSIDataset(Dataset):
    """
    PyTorch Dataset for HSI 3D patches.
    """
    def __init__(self, patches, labels):
        """
        Args:
            patches (np.array): The 3D patches (N, H, W, C).
            labels (np.array): The corresponding labels (N,).
        """
        self.patches = patches
        self.labels = labels
        self.num_bands = patches.shape[3]
        self.window_size = patches.shape[1]

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns one sample (patch, label).
        
        PyTorch 3D-CNNs expect input in the shape:
        (BatchSize, Channels, Depth, Height, Width)
        
        Here, our "Channels" is 1, and "Depth" is the number of spectral bands.
        So we reshape the patch from (H, W, C) to (1, C, H, W).
        """
        
        # Get the patch and label
        patch = self.patches[idx]  # Shape: (window_size, window_size, num_bands)
        label = self.labels[idx]
        
        # 1. Transpose to (num_bands, window_size, window_size)
        # This treats spectral bands as the "Depth"
        patch_transposed = np.transpose(patch, (2, 0, 1))
        
        # 2. Add a singleton "channel" dimension
        # Shape becomes (1, num_bands, window_size, window_size)
        patch_with_channel = np.expand_dims(patch_transposed, axis=0)
        
        # Convert to PyTorch tensors
        # Use float32 for data and long for labels (required by CrossEntropyLoss)
        patch_tensor = torch.tensor(patch_with_channel, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.
    """
    # Create Dataset objects
    train_dataset = HSIDataset(X_train, y_train)
    val_dataset = HSIDataset(X_val, y_val)
    test_dataset = HSIDataset(X_test, y_test)
    
    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle training data
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"DataLoaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader