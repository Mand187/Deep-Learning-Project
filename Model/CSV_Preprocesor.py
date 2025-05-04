# %%
import pandas as pd
import os
import time

def process_csv(input_csv_path, output_csv_path):
    """Process a single CSV file to apply Temporal ID Recycling."""
    df = pd.read_csv(input_csv_path)
    
    # Filter the 'Class' field to only keep specified vehicle types
    valid_classes = ['bus', 'car', 'truck']
    df = df[df['Class'].isin(valid_classes)]
    df = df.drop(['Class'], axis=1)
    
    # Check for duplicate IDs within the same frame
    frame_duplicates = df.groupby(['Frame', 'ID']).size().reset_index(name='counts')
    duplicate_frames = frame_duplicates[frame_duplicates['counts'] > 1]
    
    if not duplicate_frames.empty: 
        print(f"ERROR SKIPPING CSV! CSV: {input_csv_path} has the following duplicate Frame ID Combos\n{duplicate_frames}")
        return

    # Get counts of IDs per frame
    frame_id_counts = df.groupby('Frame')['ID'].nunique()

    # Calculate statistics
    min_ids_per_frame = frame_id_counts.min()
    max_ids_per_frame = frame_id_counts.max()
    avg_ids_per_frame = frame_id_counts.mean()
    
    print(f"\nMinimum IDs (Vehicles) per frame: {min_ids_per_frame}")
    print(f"Average IDs (Vehicles) per frame: {avg_ids_per_frame:.2f}")
    print(f"Maximum IDs (Vehicles) per frame: {max_ids_per_frame}")

    # Implement temporal ID recycling to avoid collisions
    MAX_NORMALIZED_IDS_PER_FRAME:int = int(max_ids_per_frame * 1.2)
    
    # Move ID column to the rightmost position
    cols = [col for col in df.columns if col != 'ID'] + ['ID']
    df = df[cols]
    
    # Sort by frame and ID to ensure temporal consistency
    df = df.sort_values(['Frame', 'ID'])

    # Initialize ID tracker and temporal ID mapping
    current_ids = set()
    id_mapping = {}
    df['ID_Norm'] = -1  # Initialize with invalid value

    # Process each frame in order
    start_time = time.time()
    total_frames = len(df['Frame'].unique())
    for i, frame in enumerate(df['Frame'].unique()):
        frame_data = df[df['Frame'] == frame]
        current_ids = set()
        used_ids_in_frame = set()  # Track which IDs are used in the current frame

        for _, row in frame_data.iterrows():
            original_id = row['ID']

            # If ID is already being tracked, reuse its temporal ID
            if original_id in id_mapping:
                temp_id = id_mapping[original_id]
                # Ensure the temp_id is not already in use in the current frame
                assert temp_id not in current_ids, f"Temp ID {temp_id} for original ID {original_id} is already in use in frame {frame}"
            else:
                # Find next available temporal ID, ensuring it's not in id_mapping or current_ids
                temp_id = 0
                while temp_id in current_ids or temp_id in {v for k, v in id_mapping.items() if k != original_id}:
                    temp_id += 1
                id_mapping[original_id] = temp_id

            # Assign the temporal ID
            df.loc[(df['Frame'] == frame) & (df['ID'] == original_id), 'ID_Norm'] = temp_id
            assert temp_id < MAX_NORMALIZED_IDS_PER_FRAME, f"Normalized ID {temp_id} exceeds MAX_NORMALIZED_IDS_PER_FRAME ({MAX_NORMALIZED_IDS_PER_FRAME})"
            
            current_ids.add(temp_id)
            used_ids_in_frame.add(original_id)  # Mark this ID as used in the current frame

        # Clean up id_mapping: Remove entries not used in the current frame
        unused_ids = set(id_mapping.keys()) - used_ids_in_frame
        for original_id in unused_ids:
            del id_mapping[original_id]

        # Print progress every 100 frames
        if i % 100 == 0 or i == total_frames - 1:
            elapsed = time.time() - start_time
            frames_processed = i + 1
            frames_remaining = total_frames - frames_processed
            if frames_processed > 0:
                time_per_frame = elapsed / frames_processed
                eta = time_per_frame * frames_remaining

                # Convert to hours:minutes:seconds
                elapsed_hours = int(elapsed // 3600)
                elapsed_minutes = int((elapsed % 3600) // 60)
                elapsed_seconds = int(elapsed % 60)

                eta_hours = int(eta // 3600)
                eta_minutes = int((eta % 3600) // 60)
                eta_seconds = int(eta % 60)

                print(f"\rProcessing frame {frames_processed}/{total_frames} | "
                      f"Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d} | "
                      f"ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}", end="", flush=True)

    # Save the processed CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nProcessed CSV saved to: {output_csv_path}")

    # Check for duplicate normalized IDs per frame
    duplicate_norm_ids = df.groupby(['Frame', 'ID_Norm']).size().reset_index(name='counts')
    duplicate_norm_ids = duplicate_norm_ids[duplicate_norm_ids['counts'] > 1]
    assert duplicate_norm_ids.empty, f"ERROR: Duplicate normalized IDs found in the following frames:\n{duplicate_norm_ids}"

    # Calculate and print statistics for normalized IDs
    norm_id_stats = df['ID_Norm'].describe()
    print(f"\nNormalized ID Statistics:")
    print(f"Minimum Normalized ID: {norm_id_stats['min']}")
    print(f"Average Normalized ID: {norm_id_stats['mean']:.2f}")
    print(f"Maximum Normalized ID: {norm_id_stats['max']}")

def process_all_csvs(raw_csvs_dir, preprocessed_csvs_dir):
    """Process all CSV files in the raw_csvs_dir and save them to preprocessed_csvs_dir."""
    if not os.path.exists(preprocessed_csvs_dir):
        os.makedirs(preprocessed_csvs_dir)

    csv_files = [f for f in os.listdir(raw_csvs_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files to process")
    
    for i, filename in enumerate(csv_files, 1):
        input_path = os.path.join(raw_csvs_dir, filename)
        output_path = os.path.join(preprocessed_csvs_dir, filename)
        print(f"\nProcessing file {i}/{total_files}: {filename}")
        process_csv(input_path, output_path)

if __name__ == "__main__":
    RAW_CSVS_DIR = "Raw_CSVs"
    PREPROCESSED_CSVS_DIR = "Preprocessed_CSVs"
    process_all_csvs(RAW_CSVS_DIR, PREPROCESSED_CSVS_DIR)
