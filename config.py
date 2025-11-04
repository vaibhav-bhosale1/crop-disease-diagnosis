"""
Central configuration file for the HSI project.
NOW CONFIGURED FOR THE "INDIAN PINES" DATASET.
"""

import torch

# --- 1. System & Paths ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
OUTPUT_DIR = "outputs"
MODEL_DIR = f"{OUTPUT_DIR}/models"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
MODEL_SAVE_PATH = f"{MODEL_DIR}/best_model.pth"

# --- 2. Data Parameters (Indian Pines) ---
# We will download these files into `data/raw/`
DATA_FILE_PATH = f"{RAW_DATA_DIR}/Indian_pines_corrected.mat"
GT_FILE_PATH = f"{RAW_DATA_DIR}/Indian_pines_gt.mat"

# The .mat files use these keys to store the data
DATA_KEY = 'indian_pines_corrected'
GT_KEY = 'indian_pines_gt'

# --- 3. Data Preprocessing & Loading (Step 3) ---
# We'll use a 9x9 spatial window for this dataset
SPATIAL_WINDOW_SIZE = 9
TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2  # From the training set
RANDOM_STATE = 42

# --- 4. Model & Training Parameters (Step 4 & 5) ---
NUM_BANDS = 200  # Indian Pines (corrected) has 200 bands
NUM_CLASSES = 16 # 16 classes + 1 for "Unclassified"
BATCH_SIZE = 64  # Increased batch size
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # We may need more epochs for real data

# Class 0 is "Unclassified" and we will ignore it.
CLASS_NAMES = {
    0: "Unclassified",
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers"
}