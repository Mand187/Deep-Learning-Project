import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchtnt.utils.data import CudaDataPrefetcher

from config import *

def load_and_preprocess_data(csv_folder='./Preprocessed_CSVs'):
    """Load and preprocess CSV data from folder"""
    # Path to the Preprocessed_CSVs folder
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
    
    transformer_max_ids_per_frame = int(frame_id_counts.max())
    
    # Initialize MinMaxScaler for each coordinate column
    scaler = MinMaxScaler(feature_range=(0, 5))
    
    # Columns to normalize
    fields_to_normalize = ['X', 'Y', 'Height', 'Width']
    
    # Normalize each coordinate column between 0 and 1
    df[fields_to_normalize] = scaler.fit_transform(df[fields_to_normalize])
    
    # Normalize Frame field separately since we need to preserve original mapping
    frame_scaler = MinMaxScaler(feature_range=(0, 5))
    original_frames = df['Frame'].values.reshape(-1, 1)
    normalized_frames = frame_scaler.fit_transform(original_frames)
    df['Frame'] = normalized_frames
    
    # Verify normalization
    print("\nAfter normalization:")
    print(f"X range: {df['X'].min():.4f} to {df['X'].max():.4f}")
    print(f"Y range: {df['Y'].min():.4f} to {df['Y'].max():.4f}")
    print(f"Height range: {df['Height'].min():.4f} to {df['Height'].max():.4f}")
    print(f"Width range: {df['Width'].min():.4f} to {df['Width'].max():.4f}")
    print(f"Frame range: {df['Frame'].min():.4f} to {df['Frame'].max():.4f}")
    
    return df, transformer_max_ids_per_frame, frame_scaler


def create_tensor_from_dataframe(df, transformer_max_ids_per_frame): # Keep arg for compatibility if needed elsewhere
    """Create a tensor from dataframe for model input"""
    # Group by frame and create sequences
    frames_grouped = df.groupby('Frame')

    # Group by CSV_ID and Frame
    grouped = df.groupby(['CSV_ID', 'Frame'])

    # *** Determine the required size based on the maximum ID_Norm value ***
    max_id_norm_value = df['ID_Norm'].max()
    tensor_id_dimension_size = max_id_norm_value + 1 # Add 1 because IDs are 0-based indices
    print(f"Determined tensor ID dimension size based on max(ID_Norm): {tensor_id_dimension_size}")

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

            # Create padded tensor for current frame using the calculated dimension size
            frame_tensor = torch.full((tensor_id_dimension_size, NUM_INPUT_FEATURES), PADDING_TOKEN, dtype=torch.float32)
            frame_tensor[frame_ids] = torch.from_numpy(frame_features).float()

            frame_tensors.append(frame_tensor)

        # Stack all frames for this CSV into a single tensor
        frames_tensor = torch.stack(frame_tensors)  # [Sequence, ID, Features]
        csv_tensors.append(frames_tensor)

    # Stack all CSVs into a single tensor
    all_data_tensor = torch.stack(csv_tensors)  # [CSV, Sequence, ID, Features]

    print(f"All data tensor shape: {all_data_tensor.shape}")
    return all_data_tensor


def create_sequences(all_data_tensor):
    """Create input-output sequences from tensor data"""
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
    
    return X, Y


class VehiclePositionDataset(data.Dataset):
    def __init__(self, features, labels, padding_token=PADDING_TOKEN):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataloaders(X, Y):
    """Create train and test dataloaders"""
    # Split data into train and test sets
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_dataset = VehiclePositionDataset(X_Train, Y_Train)
    test_dataset = VehiclePositionDataset(X_Test, Y_Test)
    
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, 
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, 
        pin_memory=True
    )
    
    train_prefetcher = CudaDataPrefetcher(
        data_iterable=train_loader, 
        device=DEVICE, 
        num_prefetch_batches=NUM_BATCHES_TO_PREFETCH
    )
    
    test_prefetcher = CudaDataPrefetcher(
        data_iterable=test_loader, 
        device=DEVICE, 
        num_prefetch_batches=NUM_BATCHES_TO_PREFETCH
    )
    
    return train_loader, test_loader, train_prefetcher, test_prefetcher


def get_original_frame(normalized_frame, frame_scaler):
    """Convert normalized frame value back to original frame number"""
    return int(frame_scaler.inverse_transform([[normalized_frame]])[0][0])