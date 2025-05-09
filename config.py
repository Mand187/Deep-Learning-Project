import torch

# Device configuration

# if torch.cuda.is_available():
#     print("Using GPU")
#     DEVICE = torch.device("cuda")
# else:
#     print("Using CPU")
#     DEVICE = torch.device("cpu")

DEVICE = torch.device('cuda:2')

# Data parameters
PADDING_TOKEN = -1
NUM_INPUT_FEATURES = 4  # Frame, X, Y, Width, Height
SEQUENCE_LENGTH = 100  # Number of frames in input sequence
PREDICTION_LENGTH = 30  # Number of future frames to predict
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 0.01
PIN_MEMORY = True
NUM_GPUS_TO_USE = 4

# DataLoader parameters
NUM_WORKERS = 40
NUM_BATCHES_TO_PREFETCH = 2
NUM_TRAIN_BATCHES_TO_PREFETCH = 2
NUM_TEST_BATCHES_TO_PREFETCH = 2
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

# Training parameters
EPOCHS = 50
MINIMUM_TEST_ACCURACY = 0
SAVE_CHECKPOINTS = False

# Model parameters
HIDDEN_SIZE = 104
NUM_HEADS = 8
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5