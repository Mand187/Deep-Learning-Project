import os
import warnings
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchtnt.utils.data import CudaDataPrefetcher
import shutil # For rmtree
from datetime import datetime # For success marker

from config import *
from Training.jutils import ColorPrinter, Colors # Assuming ColorPrinter is in jutils

printer = ColorPrinter() # Assuming Colors can be used like this, or use ColorPrinter

# 687,223,758 parameters

def load_and_preprocess_data(csv_folder='./Preprocessed_CSVs'):
    """Load and preprocess CSV data from folder"""
    df_path = os.path.join(csv_folder, 'df.pt')
    if os.path.exists(df_path):
        print(f"Loading preprocessed DataFrame from {df_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = torch.load(df_path)
        # Infer max_ids_per_frame from the loaded DataFrame
        max_ids_per_frame = df['ID_Norm'].max() + 1  # Add 1 because IDs are 0-based indices
        return df, max_ids_per_frame
        
    
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
    
    
    print(f"\nBefore normalization:")
    print(f"X range: {df['X'].min():.4f} to {df['X'].max():.4f}")
    print(f"Y range: {df['Y'].min():.4f} to {df['Y'].max():.4f}")
    print(f"Height range: {df['Height'].min():.4f} to {df['Height'].max():.4f}")
    print(f"Width range: {df['Width'].min():.4f} to {df['Width'].max():.4f}")
    print(f"Frame range: {df['Frame'].min():.4f} to {df['Frame'].max():.4f}")
    
    
    
    transformer_max_ids_per_frame = int(frame_id_counts.max())
    
    torch.save(df, df_path)  # Save the DataFrame for future use
    
    
    return df, transformer_max_ids_per_frame


def create_tensor_from_dataframe(df, *args, **kwargs): # Keep arg for compatibility if needed elsewhere
    """Create a tensor from dataframe for model input"""
    #! This is the slow function
    csv_dir = kwargs.get('csv_dir', './Preprocessed_CSVs')
    tensor_path = os.path.join(csv_dir, 'all_data_tensor.pt')
    if os.path.exists(tensor_path):
        print(f"Loading tensor from {tensor_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_data_tensor = torch.load(tensor_path)
        
        #infer tensor_id_dimension_size and len(features) from the loaded tensor
        tensor_id_dimension_size = all_data_tensor.shape[2]
        num_features = all_data_tensor.shape[3]
        print(f"Loaded tensor shape: {all_data_tensor.shape}")
        
        return all_data_tensor, num_features, tensor_id_dimension_size # Return tensor_id_dimension_size
    # Group by frame and create sequences
    frames_grouped = df.groupby('Frame')

    # Group by CSV_ID and Frame

    # *** Determine the required size based on the maximum ID_Norm value ***
    max_id_norm_value = df['ID_Norm'].max()
    tensor_id_dimension_size = max_id_norm_value + 1 # Add 1 because IDs are 0-based indices
    print(f"Determined tensor ID dimension size (max_ids_per_frame): {tensor_id_dimension_size}")

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
            features = ['X', 'Y', 'Width', 'Height']
            # Get IDs and features for current frame
            frame_ids = frame_data['ID_Norm'].values
            frame_features = frame_data[features].values

            # Create padded tensor for current frame using the calculated dimension size
            frame_tensor = torch.full((tensor_id_dimension_size, len(features)), PADDING_TOKEN, dtype=torch.float32)
            frame_tensor[frame_ids] = torch.from_numpy(frame_features).float()

            frame_tensors.append(frame_tensor)

        # Stack all frames for this CSV into a single tensor
        frames_tensor = torch.stack(frame_tensors)  # [Sequence, ID, Features]
        csv_tensors.append(frames_tensor)

    # Stack all CSVs into a single tensor
    all_data_tensor = torch.stack(csv_tensors)  # [CSV, Sequence, ID, Features]

    print(f"All data tensor shape: {all_data_tensor.shape}")
    torch.save(all_data_tensor, tensor_path)  # Save the tensor for future use
    return all_data_tensor, len(features), tensor_id_dimension_size # Return tensor_id_dimension_size


def create_sequences(all_data_tensor, sequence_offset = 1, sequence_length=SEQUENCE_LENGTH, prediction_length=PREDICTION_LENGTH):
    """Create input-output sequences from tensor data"""
    X = []
    Y = []
    
    for csv_idx in range(all_data_tensor.shape[0]): # Iterate over CSVs
        csv_data = all_data_tensor[csv_idx]  # [Sequence, ID, Features]
        
        for i in range(0, len(csv_data) - sequence_length - prediction_length + 1, sequence_offset): # Iterate over rows of CSV
            # Input sequence (SEQUENCE_LENGTH frames)
            x_seq = csv_data[i:i+sequence_length]
            # Target sequence (next PREDICTION_LENGTH frames) - Only include X and Y features (indices 1 and 2)
            y_seq = csv_data[i+sequence_length:i+sequence_length+prediction_length, :, :2]  # Slice to get X and Y only
            # print(x_seq.shape)
            X.append(x_seq)
            Y.append(y_seq)
    
    # Convert to tensors
    X = torch.stack(X)  # [Num_sequences, SEQUENCE_LENGTH, ID, Features]
    Y = torch.stack(Y)  # [Num_sequences, PREDICTION_LENGTH, ID, 2] (only X and Y)
    
    return X, Y


class VehiclePositionDataset(data.Dataset):
    def __init__(
        self,
        features,
        labels,
        padding_token=PADDING_TOKEN,
        feauture_range=(0,5),
        num_features=NUM_INPUT_FEATURES,
        normalize=False,
        max_ids_per_frame=None # Added new parameter
    ):
        try:
            self.features = features # [Num_sequences, SEQUENCE_LENGTH, ID, Features]
            self.labels = labels
            self.padding_token = padding_token
            self.max_ids_per_frame = max_ids_per_frame # Store the value
            self.x_scaler = MinMaxScaler(feature_range=(0, 16))
            self.y_scaler = MinMaxScaler(feature_range=(0, 9))
            self.other_scaler = MinMaxScaler(feauture_range)
            if not normalize:
                print(f"Normalization skipped for features and labels")
                return
            # Ensure features is a CPU tensor before attempting to reshape and convert to numpy
            features_cpu = features.cpu()
            
            features_x = features_cpu[..., 0].reshape(-1, 1)
            features_y = features_cpu[..., 1].reshape(-1, 1)
            features_other = features_cpu[..., 2:].reshape(-1, num_features-2)
            
            features_x = self.x_scaler.fit_transform(features_x.numpy()) # Convert to numpy after reshape
            features_y = self.y_scaler.fit_transform(features_y.numpy()) # Convert to numpy after reshape
            #features_other = self.other_scaler.fit_transform(features_other.numpy()) # Convert to numpy
            
            print(f"\n After Normalization:")
            print(f"X range: {features_x.min():.4f} to {features_x.max():.4f}")
            print(f"Y range: {features_y.min():.4f} to {features_y.max():.4f}")
            # Ensure features_other is 2D before accessing columns
            if features_other.ndim == 2 and features_other.shape[1] > 0:
                 print(f"Height range: {features_other[:, 0].min():.4f} to {features_other[:, 0].max():.4f}")
                 if features_other.shape[1] > 1:
                     print(f"Width range: {features_other[:, 1].min():.4f} to {features_other[:, 1].max():.4f}")
                 else:
                     print("Width range: Not available (features_other has only 1 column)")
            else:
                print("Height/Width range: Not available (features_other is not 2D or is empty)")

            print(f"Target X range: {labels[..., 0].min():.4f} to {labels[..., 0].max():.4f}")
            print(f"Target Y range: {labels[..., 1].min():.4f} to {labels[..., 1].max():.4f}")
            
            # Concatenate numpy arrays
            features_np = np.concatenate((features_x, features_y, features_other), axis=-1)
            # Convert the final processed numpy array back to a tensor
            self.features = torch.tensor(features_np, dtype=torch.float32).reshape(features.shape)

        except Exception as e:
            print(f"ERROR in VehiclePositionDataset __init__: {type(e).__name__}: {e}")
            import traceback
            print(traceback.format_exc())
            raise # Re-raise the exception to make it visible

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            return self.features[idx], self.labels[idx]
        except Exception as e:
            print(f"ERROR in VehiclePositionDataset __getitem__ for idx {idx}: {type(e).__name__}: {e}")
            import traceback
            print(traceback.format_exc())
            # In a multiprocessing DataLoader context, raising here might terminate the worker.
            # It's often better to return a sentinel value or handle it in the collation function,
            # but for debugging, re-raising can help identify the issue.
            raise

def create_datasets(X, Y, num_features=NUM_INPUT_FEATURES, normalize=False, save=True, save_dir='Data/', dataset_name='', max_ids_per_frame=None): # Removed success_marker_path
    """Create datasets for training and testing"""
    
    # Split into train and test sets
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = VehiclePositionDataset(X_Train, Y_Train, num_features=num_features, normalize=normalize, max_ids_per_frame=max_ids_per_frame) # Pass to constructor
    test_dataset = VehiclePositionDataset(X_Test, Y_Test, num_features=num_features, normalize=normalize, max_ids_per_frame=max_ids_per_frame)   # Pass to constructor
    
    if save:    
        tld = os.path.join(save_dir, dataset_name)
        os.makedirs(tld, exist_ok=True) # Ensure directory exists
        train_path = os.path.join(tld, 'train.pt')
        test_path = os.path.join(tld, 'test.pt')
        
        printer.print(f"Saving train dataset to {train_path}", Colors.BLUE)
        torch.save(train_dataset, train_path)
        printer.print(f"Train dataset saved to {train_path}", Colors.GREEN)
        
        printer.print(f"Saving test dataset to {test_path}", Colors.BLUE)
        torch.save(test_dataset, test_path)
        printer.print(f"Test dataset saved to {test_path}", Colors.GREEN)
        
        # Removed success marker creation
        
    return train_dataset, test_dataset

def _clean_dataset_files(tld, train_path, test_path): # Removed success_marker_path
    printer.print(f"Cleaning up dataset files in {tld}", Colors.YELLOW)
    paths_to_remove = [train_path, test_path]
    for item_path in paths_to_remove:
        if os.path.exists(item_path):
            try:
                os.remove(item_path)
                printer.print(f"Removed file {item_path}", Colors.YELLOW)
            except OSError as e:
                printer.print(f"Error removing file {item_path}: {e}", Colors.RED)

    # Clean up the common 'data' directory within 'tld'
    common_data_dir = os.path.join(tld, "data")
    if os.path.isdir(common_data_dir):
        try:
            shutil.rmtree(common_data_dir)
            printer.print(f"Removed directory {common_data_dir}", Colors.YELLOW)
        except OSError as e_rmtree:
            printer.print(f"Error removing directory {common_data_dir}: {e_rmtree}", Colors.RED)

def get_datasets(
    csv_folder='./Preprocessed_CSVs',
    sequence_offset = 1,
    sequence_length=SEQUENCE_LENGTH, 
    prediction_length=PREDICTION_LENGTH,
    padding_token=PADDING_TOKEN,
    feauture_range=(0,5), # Typo 'feauture_range' kept as it's in original signature
    num_features=NUM_INPUT_FEATURES,
    normalize=False,
    recompute=False,
    save=True,
    save_dir='Data/',
    dataset_name='',
):
    train_set, test_set = None, None
    tld = os.path.join(save_dir, dataset_name)
    train_path = os.path.join(tld, 'train.pt')
    test_path = os.path.join(tld, 'test.pt')
    # Removed success_marker_path

    if recompute:
        printer.print(f"Recompute=True for dataset '{dataset_name}'. Cleaning up old files if any.", Colors.YELLOW)
        _clean_dataset_files(tld, train_path, test_path)
    else:
        if os.path.exists(train_path) and os.path.exists(test_path):
            printer.print(f"Attempting to load datasets from {tld}.", Colors.BLUE)
            try:
                train_set = torch.load(train_path)
                test_set = torch.load(test_path)
                
                if not (hasattr(train_set, 'max_ids_per_frame') and train_set.max_ids_per_frame is not None and \
                        hasattr(test_set, 'max_ids_per_frame') and test_set.max_ids_per_frame is not None):
                    printer.print(f"Loaded dataset from {tld} appears incomplete (missing max_ids_per_frame). Will recompute.", Colors.YELLOW)
                    train_set, test_set = None, None 
                    _clean_dataset_files(tld, train_path, test_path) # Clean up before recompute
                else:
                    printer.print(f"Train dataset loaded successfully from {train_path}", Colors.GREEN)
                    printer.print(f"Test dataset loaded successfully from {test_path}", Colors.GREEN)
            except RuntimeError as e:
                if "PytorchStreamReader failed locating file" in str(e) or \
                   "Invalid argument passed to Caffe2" in str(e):
                    printer.print(f"Error loading dataset from {tld} (likely corrupted or incomplete save - RuntimeError): {e}. Will recompute.", Colors.RED)
                else:
                    printer.print(f"Unhandled RuntimeError loading dataset from {tld}: {e}. Will recompute.", Colors.RED)
                train_set, test_set = None, None
                _clean_dataset_files(tld, train_path, test_path) # Clean up before recompute
            except Exception as e: # Catch any other exception during loading
                printer.print(f"Generic error loading dataset from {tld}: {type(e).__name__}: {e}. Will recompute.", Colors.RED)
                train_set, test_set = None, None
                _clean_dataset_files(tld, train_path, test_path) # Clean up
        else: 
            if not os.path.exists(train_path):
                 printer.print(f"Train dataset not found at {train_path}.", Colors.YELLOW)
            if not os.path.exists(test_path):
                 printer.print(f"Test dataset not found at {test_path}.", Colors.YELLOW)
            printer.print(f"Proceeding to compute dataset '{dataset_name}'.", Colors.BLUE)
            # train_set, test_set remain None, triggering recomputation

    if train_set is None or test_set is None: # Condition to recompute
        printer.print(f"Recomputing dataset: {dataset_name}", Colors.BLUE)
        
        os.makedirs(tld, exist_ok=True)

        df, _ = load_and_preprocess_data(csv_folder)
        all_data_tensor, num_features_from_tensor, max_ids_val = create_tensor_from_dataframe(df)
        X, Y = create_sequences(all_data_tensor, sequence_offset, sequence_length, prediction_length)
        
        train_set, test_set = create_datasets(
            X, Y, 
            num_features=num_features_from_tensor, 
            normalize=normalize, 
            save=save, 
            save_dir=save_dir, 
            dataset_name=dataset_name, 
            max_ids_per_frame=max_ids_val
            # Removed success_marker_path argument
        )
    return train_set, test_set


def create_dataloaders(X, Y, num_features=NUM_INPUT_FEATURES, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE):
    """Create train and test dataloaders"""
    # Split data into train and test sets
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_dataset = VehiclePositionDataset(X_Train, Y_Train, num_features=num_features)
    test_dataset = VehiclePositionDataset(X_Test, Y_Test, num_features=num_features)
    
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True, 
        num_workers=NUM_WORKERS * 2 // 3, 
        prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, 
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset, 
        batch_size=test_batch_size,
        shuffle=False, 
        num_workers=NUM_WORKERS // 3, 
        prefetch_factor=NUM_BATCHES_TO_PREFETCH if NUM_WORKERS > 0 else None, 
        pin_memory=True
    )  
    return train_loader, test_loader


def get_original_frame(normalized_frame, frame_scaler):
    """Convert normalized frame value back to original frame number"""
    return int(frame_scaler.inverse_transform([[normalized_frame]])[0][0])