"""
Defines the 3D-CNN model architecture using PyTorch.
"""

import torch
import torch.nn as nn
import config

class HSI_CNN_3D(nn.Module):
    """
    A 3D Convolutional Neural Network for HSI classification.
    
    Input shape: (BatchSize, 1, NumBands, WindowSize, WindowSize)
    Output shape: (BatchSize, NumClasses)
    """
    
    def __init__(self, num_classes, num_bands, window_size):
        super(HSI_CNN_3D, self).__init__()
        
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.window_size = window_size
        
        # --- 1. Convolutional Blocks ---
        # A block = 3D-Conv -> 3D-BatchNorm -> ReLU
        
        # Block 1
        # Kernel: (7 spectral, 3 spatial, 3 spatial)
        # Input: (B, 1, 200, 5, 5) -> Output: (B, 8, 194, 3, 3)
        self.conv1 = nn.Conv3d(
            in_channels=1, 
            out_channels=8, 
            kernel_size=(7, 3, 3), 
            stride=1, 
            padding=0
        )
        self.bn1 = nn.BatchNorm3d(8) # Batch normalization for stability
        self.relu1 = nn.ReLU()
        
        # Block 2
        # Kernel: (5 spectral, 3 spatial, 3 spatial)
        # Input: (B, 8, 194, 3, 3) -> Output: (B, 16, 190, 1, 1)
        self.conv2 = nn.Conv3d(
            in_channels=8, 
            out_channels=16, 
            kernel_size=(5, 3, 3), 
            stride=1, 
            padding=0
        )
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        
        # Block 3
        # Kernel: (3 spectral, 1 spatial, 1 spatial)
        # Input: (B, 16, 190, 1, 1) -> Output: (B, 32, 188, 1, 1)
        self.conv3 = nn.Conv3d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3, 1, 1), 
            stride=1, 
            padding=0
        )
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU()
        
        # --- 2. Fully Connected (Classifier) Layers ---
        
        # Flatten the output from the conv blocks
        self.flatten = nn.Flatten()
        
        # We need to dynamically calculate the flattened size
        self.fc_input_size = self._get_fc_input_size()
        
        # Classifier
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4) # Dropout for regularization
        self.fc2 = nn.Linear(128, self.num_classes)
        

    def _get_fc_input_size(self):
        """
        Helper function to calculate the input size for the first
        fully connected layer. It performs a "dummy" forward pass.
        """
        # Create a dummy input tensor
        # Shape: (1, 1, NumBands, WindowSize, WindowSize)
        dummy_input = torch.randn(
            1, 1, self.num_bands, self.window_size, self.window_size
        )
        
        # Pass it through the convolutional layers
        x = self.conv1(dummy_input)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # The shape of the flattened vector is our FC input size
        return x.shape[1]

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # x shape: (B, 1, 200, 5, 5)
        
        # Pass through Conv Block 1
        x = self.relu1(self.bn1(self.conv1(x)))
        # x shape: (B, 8, 194, 3, 3)
        
        # Pass through Conv Block 2
        x = self.relu2(self.bn2(self.conv2(x)))
        # x shape: (B, 16, 190, 1, 1)
        
        # Pass through Conv Block 3
        x = self.relu3(self.bn3(self.conv3(x)))
        # x shape: (B, 32, 188, 1, 1)
        
        # Flatten for the classifier
        x = self.flatten(x)
        # x shape: (B, 32 * 188 * 1 * 1) = (B, 6016)
        
        # Pass through Classifier
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Raw output scores (logits)
        # x shape: (B, NumClasses)
        
        return x

# --- You can uncomment this block to test the model ---
# if __name__ == "__main__":
#     print("Testing model architecture...")
    
#     # Load parameters from config
#     window = config.SPATIAL_WINDOW_SIZE
#     bands = config.NUM_BANDS
#     classes = config.NUM_CLASSES
    
#     # Create a model instance
#     model = HSI_CNN_3D(num_classes=classes, num_bands=bands, window_size=window)
    
#     # Create a dummy batch of data
#     # (BatchSize, 1, Bands, Height, Width)
#     dummy_batch = torch.randn(config.BATCH_SIZE, 1, bands, window, window)
    
#     print(f"Model created. FC input size: {model.fc_input_size}")
#     print(f"Dummy input batch shape: {dummy_batch.shape}")
    
#     # Perform a forward pass
#     output = model(dummy_batch)
    
#     print(f"Output batch shape: {output.shape}")
#     print("Model test successful!")
#     assert output.shape == (config.BATCH_SIZE, classes)