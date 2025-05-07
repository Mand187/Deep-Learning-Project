# %%
# ===============================================================================================================
#                                          Data and Library Loading
# ===============================================================================================================
import torch
import torch.nn as nn
import torch.utils.data as data
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtnt.utils.data import CudaDataPrefetcher
from torchprofile import profile_macs

from sklearn.preprocessing import MinMaxScaler # , LabelEncoder, StandardScaler
import pandas as pd

from sklearn.model_selection import train_test_split
# import numpy as np
import math

import matplotlib.pyplot as plt

import time
import signal
import os

assert torch.cuda.is_available(), "ERR: No GPU available"

DEVICE:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available(): torch.cuda.empty_cache()

# ================================ GET CSV DATA ======================================
# Path to the Preprocessed_CSVs folder
csv_folder = './Preprocessed_CSVs'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Initialize an empty DataFrame to store all data
df = pd.DataFrame()

# Read each CSV and append a CSV_ID
for csv_id, csv_file in enumerate(csv_files):
    temp_df = pd.read_csv(os.path.join(csv_folder, csv_file))
    temp_df['CSV_ID'] = csv_id  # Add CSV_ID column
    
    # Check for duplicate IDs within the same frame before concatenating
    frame_duplicates = temp_df.groupby(['Frame', 'ID_Norm']).size().reset_index(name='counts')
    duplicate_frames = frame_duplicates[frame_duplicates['counts'] > 1]
    assert duplicate_frames.empty, f"Warning: Found duplicate IDs within the same frame in {csv_file}:\n{duplicate_frames}"
    
    df = pd.concat([df, temp_df], ignore_index=True)

# Find minimum frame count across all CSVs
min_frames = df.groupby('CSV_ID')['Frame'].nunique().min()

# For each CSV, remove frames beyond the minimum frame count
for csv_id in df['CSV_ID'].unique():
    # Get all frames for this CSV, sorted
    csv_frames = df[df['CSV_ID'] == csv_id]['Frame'].unique()
    csv_frames_sorted = sorted(csv_frames)
    
    # Determine cutoff frame (min_frames-th frame)
    if len(csv_frames_sorted) > min_frames:
        cutoff_frame = csv_frames_sorted[min_frames]
        # Remove all rows with frame >= cutoff_frame
        df = df[~((df['CSV_ID'] == csv_id) & (df['Frame'] >= cutoff_frame))]

# Verify all CSVs now have same number of frames
frames_per_csv = df.groupby('CSV_ID')['Frame'].nunique()
assert frames_per_csv.nunique() == 1, "Not all CSVs have same number of frames after trimming"
print(f"\nAll CSVs now have {min_frames} frames after trimming")

# Get counts of records per ID
id_counts = df['ID_Norm'].value_counts()

# Calculate statistics
print(f"Minimum records per ID: {id_counts.min()}")
print(f"Average records per ID: {id_counts.mean():.2f}")
print(f"Maximum records per ID: {id_counts.max()}")

# Get counts of IDs per frame
frame_id_counts = df.groupby('Frame')['ID_Norm'].nunique()

# Calculate statistics
print(f"\nMinimum IDs (Vehicles) per frame: {frame_id_counts.min()}")
print(f"Average IDs (Vehicles) per frame: {frame_id_counts.mean():.2f}")
print(f"Maximum IDs (Vehicles) per frame: {frame_id_counts.max()}")

TRANSFORMER_MAX_IDS_PER_FRAME:int = int(frame_id_counts.max())

# Initialize MinMaxScaler for each coordinate column
misc_feature_scaler = MinMaxScaler(feature_range=(0, 5))
xy_scaler = MinMaxScaler(feature_range=(0, 5))

# Columns to normalize
misc_fields_to_normalize = ['Height', 'Width']
xy_fields_to_normalize = ['X', 'Y']

# Normalize each coordinate column between 0 and 1
# df[misc_fields_to_normalize] = misc_feature_scaler.fit_transform(df[misc_fields_to_normalize])
# df[xy_fields_to_normalize] = xy_scaler.fit_transform(df[xy_fields_to_normalize])

# Normalize Frame field separately since we need to preserve original mapping
# frame_scaler = MinMaxScaler(feature_range=(0, 5))
# original_frames = df['Frame'].values.reshape(-1, 1)
# normalized_frames = frame_scaler.fit_transform(original_frames)
# df['Frame'] = normalized_frames

# Function to get original frame value from normalized value
# def get_original_frame(normalized_frame):
#     """Convert normalized frame value back to original frame number"""
#     return int(frame_scaler.inverse_transform([[normalized_frame]])[0][0])

# Verify normalization
print("\nAfter normalization:")
print(f"X range: {df['X'].min():.4f} to {df['X'].max():.4f}")
print(f"Y range: {df['Y'].min():.4f} to {df['Y'].max():.4f}")
print(f"Height range: {df['Height'].min():.4f} to {df['Height'].max():.4f}")
print(f"Width range: {df['Width'].min():.4f} to {df['Width'].max():.4f}")
# print(f"Frame range: {df['Frame'].min():.4f} to {df['Frame'].max():.4f}")

# ================================ CREATE TENSOR FROM CSV DATA ======================================
# Desired Output Shape: [CSV, Sequence/Frame, ID (Padded), features (Frame, X, Y, Width, Height)]

# Sort the dataframe by CSV_ID, then Frame, then ID_Norm to ensure consistent ordering
df = df.sort_values(by=['CSV_ID', 'Frame', 'ID_Norm'])

# Group by frame and create sequences
frames_grouped = df.groupby('Frame')
NUM_INPUT_FEATURES:int = 5  # Frame, X, Y, Width, Height
PADDING_TOKEN:int = -1 # If ID does not exist, all feature values given this

# Group by CSV_ID and Frame
grouped = df.groupby(['CSV_ID', 'Frame'])

# Initialize list to store CSV tensors
csv_tensors = []

# Iterate over each CSV_ID
for csv_id in df['CSV_ID'].unique():
    csv_data = df[df['CSV_ID'] == csv_id]
    frames_grouped = csv_data.groupby('Frame')
    
    # Initialize list to store frame tensors for this CSV
    frame_tensors = []
    
    # Create padded tensors for each frame in this CSV
    frames = sorted(csv_data['Frame'].unique())
    for frame in frames:
        frame_data = frames_grouped.get_group(frame)
        
        # Get IDs and features for current frame
        frame_ids = frame_data['ID_Norm'].values
        frame_features = frame_data[['Frame', 'X', 'Y', 'Width', 'Height']].values
        
        # Create padded tensor for current frame
        frame_tensor = torch.full((TRANSFORMER_MAX_IDS_PER_FRAME, NUM_INPUT_FEATURES), PADDING_TOKEN, dtype=torch.float32)
        frame_tensor[frame_ids] = torch.from_numpy(frame_features).float()
        
        frame_tensors.append(frame_tensor)
    
    # Stack all frames for this CSV into a single tensor
    frames_tensor = torch.stack(frame_tensors)  # [Sequence, ID, Features]
    csv_tensors.append(frames_tensor)

# Stack all CSVs into a single tensor
all_data_tensor = torch.stack(csv_tensors)  # [CSV, Sequence, ID, Features]

print(f"All data tensor shape: {all_data_tensor.shape}")

# ================================ Create Sequences and Dataloader ======================================
# Create sequences of frames and their corresponding next n frames, ensuring no cross-CSV sequences
SEQUENCE_LENGTH:int = 100  # Number of frames in input sequence     100
PREDICTION_LENGTH:int = 30  # Number of future frames to predict   30

X = []
Y = []

for csv_idx in range(all_data_tensor.shape[0]):
    csv_data = all_data_tensor[csv_idx]  # [Sequence, ID, Features]
    
    for i in range(len(csv_data) - SEQUENCE_LENGTH - PREDICTION_LENGTH + 1):
        # Input sequence (SEQUENCE_LENGTH frames)
        x_seq = csv_data[i:i+SEQUENCE_LENGTH]
        # Target sequence (next PREDICTION_LENGTH frames) - Only include X and Y features (indices 1 and 2)
        y_seq = csv_data[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+PREDICTION_LENGTH, :, 1:3]  # Slice to get X and Y only
        
        X.append(x_seq)
        Y.append(y_seq)

# Convert to tensors
X = torch.stack(X)  # [Num_sequences, SEQUENCE_LENGTH, ID, Features]
Y = torch.stack(Y)  # [Num_sequences, PREDICTION_LENGTH, ID, 2] (only X and Y)

# Split data into train and test sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)

class VehiclePositionDataset(data.Dataset):
    def __init__(self, features, labels, padding_token = PADDING_TOKEN):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset:VehiclePositionDataset = VehiclePositionDataset(X_Train, Y_Train)
test_dataset:VehiclePositionDataset = VehiclePositionDataset(X_Test, Y_Test)

NUM_WORKERS:int = 0 #int(os.cpu_count() / 2)
NUM_BATCHES_TO_PREFETCH:int = 2
BATCH_SIZE:int = 64

train_loader:data.DataLoader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

test_loader:data.DataLoader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, pin_memory=True)

train_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=train_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)
test_prefetcher:CudaDataPrefetcher = CudaDataPrefetcher(data_iterable=test_loader, device=DEVICE, num_prefetch_batches=NUM_BATCHES_TO_PREFETCH)

# %%
class FrameTransformer(nn.Module):
    def __init__(self, input_feature_size=NUM_INPUT_FEATURES, num_ids=TRANSFORMER_MAX_IDS_PER_FRAME, sequence_length=SEQUENCE_LENGTH, prediction_length=PREDICTION_LENGTH):
        super().__init__()
        
        HIDDEN_SIZE = 104
        NUM_HEADS = 8
        DROPOUT_RATE = 0.1
        self.prediction_length = prediction_length
        
        # Sinusoidal positional encoding for frames - creates a unique positional encoding for each frame in the sequence
        # that helps the model understand the order/temporal relationships between frames. Uses alternating sin/cos waves
        # of different frequencies to encode position information.
        positions = torch.arange(sequence_length).unsqueeze(1)
        feature_frequency = torch.exp(torch.arange(0, HIDDEN_SIZE, 2) * (-math.log(10000.0) / HIDDEN_SIZE))
        
        positional_encoder = torch.zeros(1, sequence_length, HIDDEN_SIZE)
        
        positional_encoder[0, :, 0::2] = torch.sin(positions * feature_frequency)
        positional_encoder[0, :, 1::2] = torch.cos(positions * feature_frequency)
        self.register_buffer('frame_pos_encoder', positional_encoder)  # register as buffer so it moves with model to GPU
        
        # Input feature projection
        self.input_proj = nn.Linear(input_feature_size, HIDDEN_SIZE)
        
        # Multihead attention across IDs in a frame
        self.id_attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE*sequence_length,
            num_heads=NUM_HEADS,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        
        # Multihead attention across frames
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE * num_ids,
            num_heads=NUM_HEADS,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        
        # Temporal convolution to map sequence length to prediction length
        self.temporal_conv = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=prediction_length,
            kernel_size=1
        )
        
        # Output feature projection
        self.output_proj = nn.Linear(HIDDEN_SIZE, 2)
        
        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE*sequence_length)
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE * num_ids)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        """
        x.shape: [batch_size, sequence_length, num_ids, input_feature_size]
        
        return.shape: [batch_size, prediction_length, num_ids, 2 (X, Y)]
        """
        batch_size, seq_len, num_ids, input_feat_dim = x.shape
        
        # Project input features to HIDDEN_SIZE
        x = self.input_proj(x)  # [batch_size, sequence_length, num_ids, HIDDEN_SIZE]
        
        # Add frame positional encoding
        x = x + self.frame_pos_encoder.unsqueeze(2)
        
        # Reshape back for frame attention
        x_frame = x.reshape(batch_size, seq_len, -1) # [batch_size, sequence_length, num_ids * HIDDEN_SIZE]
        
        # Self attention across frames with residual
        frame_attn_out, _ = self.frame_attention(x_frame, x_frame, x_frame)
        frame_attn_out = self.dropout(frame_attn_out)
        frame_attn_out = self.norm2(x_frame + frame_attn_out)
        
        # Reshape for ID attention
        x_id = frame_attn_out.reshape(batch_size, seq_len, num_ids, -1) # [batch_size, sequence_length, num_ids, HIDDEN_SIZE]
        x_id = x_id.permute(0,2,1,3) # [batch_size, num_ids, sequence_length, HIDDEN_SIZE]
        x_id = x_id.reshape(batch_size, num_ids, -1) # [batch_size, num_ids, sequence_length * HIDDEN_SIZE]
        
        # Self attention across IDs with residual
        id_attn_out, _ = self.id_attention(x_id, x_id, x_id)
        id_attn_out = self.dropout(id_attn_out)
        id_attn_out = self.norm1(x_id + id_attn_out)
        
        # Reshape for temporal convolution
        x_conv = id_attn_out.reshape(batch_size, num_ids, seq_len, -1) # [batch_size, num_ids, sequence_length, HIDDEN_SIZE]
        x_conv = x_conv.permute(0,2,1,3) # [batch_size, sequence_length, num_ids, HIDDEN_SIZE]
        x_conv = x_conv.reshape(batch_size, seq_len, -1) # [batch_size, sequence_length, num_ids * HIDDEN_SIZE]
        
        # Apply temporal convolution
        output = self.temporal_conv(x_conv) # [batch_size, prediction_length, num_ids * HIDDEN_SIZE]
        
        # Reshape back and project to input feature size
        output = output.reshape(batch_size, self.prediction_length, num_ids, -1) # [batch_size, prediction_length, num_ids, HIDDEN_SIZE]
        output = self.output_proj(output)  # [batch_size, prediction_length, num_ids, 2]
        
        return output


# model.load_state_dict(torch.load('Saved_Models/best_model.pth'))

# ========== Model Parameters ==========
model:FrameTransformer = FrameTransformer().to(DEVICE)
total_params = sum([p.numel() for p in model.parameters()])
print(f"Total Num Params in loaded model: {total_params:,}")

# Calculate MACs (Multiply-Accumulate Operations)
# Create sample inputs for profiling
firstBatch = next(iter(train_loader))
sample_X, sample_Y = firstBatch
sample_X, sample_Y = sample_X.to(DEVICE), sample_Y.to(DEVICE)

# Profile the model
macs = profile_macs(model, (sample_X, ))
print(f"Computational complexity: {macs:,} MACs")
print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")

# * Training Loop Reinitialization
epochIterator:int = 0
bestTestAccuracy:float = 0

avgTrainBatchLossPerEpoch:list = []
avgTestBatchLossPerEpoch:list = []
trainAccuracyPerEpoch:list = []
testAccuracyPerEpoch:list = []

# %%
# ================================================ Shape Testing ================================================
firstBatch = next(iter(test_loader))
X, Y = firstBatch
X, Y = X.to(DEVICE), Y.to(DEVICE)

print(f"X: {X.shape}")
logits = model(X.to(DEVICE))
print(f"Logits: {logits.shape}")
print(f"Expected: {Y.shape}")

# %%
# ===============================================================================================================
#                                                   Load Model
# ===============================================================================================================
# model = torch.load('Saved_Models/20_Epoch_CIFAR.pt')
# %% 
# ===============================================================================================================
#                                                    Training
# ===============================================================================================================
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("Interrupt received. Flag set...")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def linearOffset(input, offset, target):
    # max() ensures offset is always positive or 0
    # min() returns the smaller offset between target - input and default offset
    return max(0, min(offset, target - input))


Loss_Function:nn.CrossEntropyLoss = nn.CrossEntropyLoss()
# Optimizer_Function:torch.optim.Adam = torch.optim.Adam(params=model.parameters())
# Optimizer_Function:torch.optim.SGD = torch.optim.SGD(params=model.parameters(),
#                                                      lr=0.0001)
Optimizer_Function:torch.optim.Adam = torch.optim.Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-5
)

EPOCHS:int = 50

MINIMUM_TEST_ACCURACY:int = 0
SAVE_CHECKPOINTS:bool = False

trainStartTime:float = time.time()
while not interrupted and ((epochIterator < EPOCHS or EPOCHS == -1) or trainEpochAccuracy < testEpochAccuracy + linearOffset(input=testEpochAccuracy, offset=3, target=99) or bestTestAccuracy < MINIMUM_TEST_ACCURACY):
    epochStartTime:float = time.time()
    model.train()
    
    numCorrectInEpoch:int = 0
    totalTrainLossInEpoch:float = 0
    for X_train_batch, Y_train_batch in train_prefetcher:
        X_train_batch:torch.Tensor = X_train_batch.to(DEVICE, non_blocking=True)
        Y_train_batch:torch.Tensor = Y_train_batch.to(DEVICE, non_blocking=True)
        
        Y_train_pred_logits:torch.Tensor = model(X_train_batch)
        
        trainBatchLoss = Loss_Function(Y_train_pred_logits, Y_train_batch.type(torch.int64))
        
        Optimizer_Function.zero_grad()
        trainBatchLoss.backward()
        Optimizer_Function.step()
        
        numCorrectInEpoch += torch.eq(Y_train_pred_logits.argmax(dim=1), Y_train_batch).sum().item()
        totalTrainLossInEpoch += trainBatchLoss
        
    
    model.eval()
    
    with torch.inference_mode():
        trainEpochAverageBatchLoss:float = totalTrainLossInEpoch/len(train_loader)
        avgTrainBatchLossPerEpoch += [trainEpochAverageBatchLoss]
        
        trainEpochAccuracy:float = numCorrectInEpoch/len(train_loader.dataset) * 100 # accuracy is calculated per item in a batch instead of per batch
        trainAccuracyPerEpoch += [trainEpochAccuracy]
        
        numCorrectInEpoch:int = 0
        totalTestLossInEpoch:float = 0
        for X_test_batch, Y_test_batch in test_prefetcher:
            X_test_batch:torch.Tensor = X_test_batch.to(DEVICE, non_blocking=True)
            Y_test_batch:torch.Tensor = Y_test_batch.to(DEVICE, non_blocking=True)
        
            Y_test_pred_logits:torch.Tensor = model(X_test_batch)
        
            testBatchLoss = Loss_Function(Y_test_pred_logits, Y_test_batch.type(torch.int64))
    
            numCorrectInEpoch += torch.eq(Y_test_pred_logits.argmax(dim=1), Y_test_batch).sum().item()
            
            totalTestLossInEpoch += testBatchLoss
        
        testEpochAverageBatchLoss:float = totalTestLossInEpoch/len(test_loader)
        avgTestBatchLossPerEpoch += [testEpochAverageBatchLoss]
        
        testEpochAccuracy:float = numCorrectInEpoch/len(test_loader.dataset) * 100
        testAccuracyPerEpoch += [testEpochAccuracy]
    
        epochTime:float = time.time() - epochStartTime
        estRemainingTime:float = (EPOCHS - epochIterator - 1)*epochTime / 60
        print(f"epoch: {epochIterator} \t| train loss: {trainEpochAverageBatchLoss:.5f}, train accuracy: {trainEpochAccuracy:.2f}% \t| test loss: {testEpochAverageBatchLoss:.5f}, test accuracy: {testEpochAccuracy:.2f}% \t| TTG: {int(estRemainingTime):02}:{int((estRemainingTime - int(estRemainingTime))*60):02}")
        
        newBestModel:bool = testEpochAccuracy > MINIMUM_TEST_ACCURACY and testEpochAccuracy > bestTestAccuracy
        if newBestModel: 
            bestTestAccuracy:float = testEpochAccuracy
            print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑ NEW BEST MODEL ↑↑↑↑↑↑↑↑↑↑↑↑↑")
            
        if SAVE_CHECKPOINTS and newBestModel: 
            torch.save(model.state_dict(), 'Saved_Models/best_model.pth')
            print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ SAVED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
        
        epochIterator += 1
        
totalTrainTime:float = (time.time() - trainStartTime)/60
averageEpochTime:float = totalTrainTime / epochIterator

print(f"Total Training Time: {int(totalTrainTime):02}:{int((totalTrainTime - int(totalTrainTime))*60):02}")
print(f"Average Epoch Time: {int(averageEpochTime):02}:{int((averageEpochTime - int(averageEpochTime))*60):02}")
print(f"({model.PATCH_SIZE}x{model.PATCH_SIZE}) -- {model.EMBEDDING_SIZE} -- ({model.NUM_LAYERS},{model.NUM_HEADS})")
        
# %%
# ===============================================================================================================
#                                                   Plot Loss
# ===============================================================================================================
with torch.inference_mode():
    avgTrainBatchLossPerEpoch1:list = torch.tensor(avgTrainBatchLossPerEpoch).cpu()
    avgTestBatchLossPerEpoch1:list = torch.tensor(avgTestBatchLossPerEpoch).cpu()
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # First subplot
    axs[0].scatter(x=[x for x in range(len(avgTrainBatchLossPerEpoch1))], y=avgTrainBatchLossPerEpoch1, label="Training Loss")
    axs[0].scatter(x=[x for x in range(len(avgTestBatchLossPerEpoch1))], y=avgTestBatchLossPerEpoch1, label="Test / Validation Loss")
    axs[0].set_title('Loss Per Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Second subplot
    axs[1].scatter(x=[x for x in range(len(trainAccuracyPerEpoch))], y=trainAccuracyPerEpoch, label="Training Accuracy")
    axs[1].scatter(x=[x for x in range(len(testAccuracyPerEpoch))], y=testAccuracyPerEpoch, label="Test / Validation Accuracy")
    axs[1].set_title('Accuracy Per Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy %')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adjust layout and display the plot
    plt.tight_layout()  # Avoid overlap between subplots
    plt.plot()
    plt.show()
    
# %%
# ===============================================================================================================
#                                                   Save Model
# ===============================================================================================================
# torch.save(model, 'Saved_Models/ResNet11_Baseline_CIFAR10.pt')