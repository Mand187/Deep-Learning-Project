import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
PADDING_TOKEN = -1
NUM_INPUT_FEATURES = 5  # Frame, X, Y, Width, Height
SEQUENCE_LENGTH = 100  # Number of frames in input sequence
PREDICTION_LENGTH = 30  # Number of future frames to predict

# DataLoader parameters
NUM_WORKERS = 0
NUM_BATCHES_TO_PREFETCH = 2
BATCH_SIZE = 64

# Training parameters
EPOCHS = 50
MINIMUM_TEST_ACCURACY = 0
SAVE_CHECKPOINTS = False

# Model parameters
HIDDEN_SIZE = 128
NUM_HEADS = 8
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5