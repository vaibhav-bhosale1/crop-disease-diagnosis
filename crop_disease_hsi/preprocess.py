"""
Data preprocessing functions.

Includes:
- Normalization
- Padding
- 3D Patch Extraction
"""

import numpy as np
import config

def apply_padding(hsi_cube, padding_width):
    """
    Applies mirror padding to the spatial dimensions of the HSI cube.
    
    Args:
        hsi_cube (np.array): The HSI cube (H, W, C).
        padding_width (int): The amount of padding to add (e.g., 2 for a 5x5 window).
        
    Returns:
        np.array: The padded HSI cube.
    """
    # Pad only the spatial dimensions (axis 0 and 1), not the spectral (axis 2)
    # 'reflect' mode mirrors the data at the edges.
    return np.pad(
        hsi_cube, 
        ((padding_width, padding_width), (padding_width, padding_width), (0, 0)), 
        mode='reflect'
    )

def extract_patches(hsi_cube, mask, window_size):
    """
    Extracts 3D patches from a single HSI cube.
    
    For each pixel (x, y) in the mask, it extracts a 3D patch of
    shape (window_size, window_size, num_bands) from the HSI cube,
    centered at (x, y).
    
    Args:
        hsi_cube (np.array): The *padded* HSI cube (H_padded, W_padded, C).
        mask (np.array): The *original* (unpadded) annotation mask (H, W).
        window_size (int): The spatial size of the patch (e.g., 5 for 5x5).
        
    Returns:
        (np.array, np.array): A tuple of (patches, labels)
            - patches: (N, window_size, window_size, num_bands) where N = H * W
            - labels: (N,)
    """
    patch_radius = window_size // 2
    height, width = mask.shape
    num_bands = hsi_cube.shape[2]
    
    # Calculate total number of patches
    num_patches = height * width
    
    # Initialize arrays to store the patches and labels
    # Shape for patches: (NumPatches, Window, Window, Bands)
    patches = np.zeros((num_patches, window_size, window_size, num_bands), dtype=hsi_cube.dtype)
    # Shape for labels: (NumPatches,)
    labels = np.zeros(num_patches, dtype=mask.dtype)
    
    # Iterate over each pixel in the *original* mask
    patch_index = 0
    for r in range(height):
        for c in range(width):
            # The corresponding center pixel in the *padded* cube
            padded_r = r + patch_radius
            padded_c = c + patch_radius
            
            # Extract the 3D patch
            # Slicing from [center - radius] to [center + radius + 1]
            patch = hsi_cube[
                padded_r - patch_radius : padded_r + patch_radius + 1,
                padded_c - patch_radius : padded_c + patch_radius + 1,
                :
            ]
            
            patches[patch_index] = patch
            labels[patch_index] = mask[r, c]
            patch_index += 1
            
    return patches, labels

def normalize(hsi_cube):
    """
    Performs simple min-max normalization on the HSI cube.
    """
    min_val = np.min(hsi_cube)
    max_val = np.max(hsi_cube)
    if max_val - min_val == 0:
        return hsi_cube # Avoid division by zero if flat
    return (hsi_cube - min_val) / (max_val - min_val)