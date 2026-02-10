import os

# Get project root (parent of brats directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Root path of BraTS 2020 training data
TRAIN_DATASET_PATH = os.environ.get(
    "BRATS20_TRAIN_PATH",
    os.path.join(PROJECT_ROOT, "data", "MICCAI_BraTS2020_TrainingData"),
)

# Fix the misnamed segmentation file in BraTS20_Training_355 if present
APPLY_355_RENAME_FIX = True

# Validation data path
VAL_DATASET_PATH = os.environ.get(
    "BRATS20_VAL_PATH",
    os.path.join(PROJECT_ROOT, "data", "MICCAI_BraTS2020_ValidationData"),
)

# Input shape and sampling
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
NUM_CHANNELS = 2  # FLAIR + T1ce

# Classes
SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "NECROTIC/CORE",
    2: "EDEMA",
    3: "ENHANCING",  # original 4 -> mapped to 3
}

# Training
EPOCHS = 35
LEARNING_RATE = 1e-3
BATCH_SIZE = 1

# Checkpoints and logs
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
CHECKPOINT_PATTERN = os.path.join(CHECKPOINT_DIR, "model_{epoch:02d}-{val_loss:.6f}.keras")
TRAIN_LOG_PATH = os.path.join(PROJECT_ROOT, "models", "training.log")
BEST_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "best.weights.h5")  # optional

# Pre-trained model paths
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "my_model.keras")
USE_PRETRAINED = True  # Set to True if you want to use pre-trained model

# Random seed
RANDOM_SEED = 1337

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "generated_outputs")
