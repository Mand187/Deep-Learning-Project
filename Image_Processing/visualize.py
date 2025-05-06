import time
from tqdm import tqdm
import cv2
import pandas as pd
import os
import numpy as np
import concurrent.futures # Added import


# --- Configuration ---
video_path = 'Image_Processing/visualization/michael_10s.mp4' # Relative path from script location
prediction_path = 'Image_Processing/visualization/predictions.csv'
output_path = 'Image_Processing/visualization/prediction_visualization_future.mp4' # Where to save the output video

# ID 11 shows model does not predict car going off screen
# ID 12 shows model failing to handle car just coming into detection range
IDs_To_Visualize = list(range(17)) # IDs to visualize

frame_offset = 5

"""
Will only operate on first 130 frames of the video
First create an overlay of the predictions
    - Optionally set a max id to visualize
    - Optionally set a min id to visualize
    - For the first 100 frames of the predictions, create red lines using the points
    - For the next 30 frames, create green lines using the true points and purple lines using the predicted points

Create a 130 frame video with the overlay
"""

# --- Load video ---
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise Exception(f"Could not open video file: {video_path}")
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}")
print(f"Video Frame Count: {frame_count}")
print(f"Video Width: {width}")
print(f"Video Height: {height}")

# --- Load predictions ---
df = pd.read_csv(prediction_path)
# Check if the CSV has the expected columns
expected_columns = ['Frame', 'ID', 'X_pred', 'Y_pred', 'X_true', 'Y_true']
for col in expected_columns:
    if col not in df.columns:
        raise Exception(f"Missing expected column: {col} in {prediction_path}")

# Start video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# New function to process a single frame
def process_frame_for_video(task_args):
    frame_idx, original_frame, predictions_data = task_args
    processed_frame = original_frame.copy()

    # Shapes: (len id_range, 6)
    cur_pos = predictions_data[predictions_data['Frame'] == frame_idx] # len id_range
    future_pos = predictions_data[predictions_data['Frame'] == frame_idx + frame_offset] # len id_range
    offset_temp = frame_offset
    while future_pos.empty and frame_idx + offset_temp < 130:
        offset_temp -= 1
        future_pos = predictions_data[predictions_data['Frame'] == frame_idx + offset_temp]
        if offset_temp == 0:
            break
    if future_pos.empty:
        return frame_idx, processed_frame
    
    
    # Draw red arrow towards future position
    if frame_idx < 100-frame_offset:
        return frame_idx, processed_frame
        for _, row in cur_pos.iterrows():
            color = (0, 0, 255)
            if row['ID'] in future_pos['ID'].values:
                future_row = future_pos[future_pos['ID'] == row['ID']]
                if not future_row.empty:
                    cv2.arrowedLine(
                        img = processed_frame,
                        pt1 = (int(row['X_true']), int(row['Y_true'])), 
                        pt2 = (int(future_row.iloc[0]['X_true']), int(future_row.iloc[0]['Y_true'])), 
                        color = color,
                        thickness = 2,
                        tipLength=0.1
                    )
    else:
        for _, row in cur_pos.iterrows():
            pred_color = (255, 0, 255)  # Purple for prediction
            true_color = (0, 255, 0)    # Green for true
            if row['ID'] in future_pos['ID'].values:
                future_row = future_pos[future_pos['ID'] == row['ID']]
                if not future_row.empty:
                    
                    # skip if any coordinates are negative
                    if (row['X_true'] < 0 or row['Y_true'] < 0 or
                        future_row.iloc[0]['X_true'] < 0 or future_row.iloc[0]['Y_true'] < 0 or
                        row['X_pred'] < 0 or row['Y_pred'] < 0 or
                        future_row.iloc[0]['X_pred'] < 0 or future_row.iloc[0]['Y_pred'] < 0):
                        continue
                    
                    # Draw red dot at current position
                    # cv2.circle(
                    #     img = processed_frame,
                    #     center = (int(row['X_true']), int(row['Y_true'])), 
                    #     radius = 5, 
                    #     color = (0, 0, 255), 
                    #     thickness = -1
                    # )
                    
                    
                    # Draw predicted trajectory
                    cv2.arrowedLine(
                        img = processed_frame,
                        pt1 = (int(row['X_true']), int(row['Y_true'])), 
                        pt2 = (int(future_row.iloc[0]['X_pred']), int(future_row.iloc[0]['Y_pred'])), 
                        color = pred_color,
                        thickness = 2,
                        tipLength=0.1
                    )
                    # Draw true trajectory
                    cv2.arrowedLine(
                        img = processed_frame,
                        pt1 = (int(row['X_true']), int(row['Y_true'])), 
                        pt2 = (int(future_row.iloc[0]['X_true']), int(future_row.iloc[0]['Y_true'])), 
                        color = true_color,
                        thickness = 3,
                        tipLength=0.1
                    )
            
    return frame_idx, processed_frame

# Filter predictions once based on ID range
predictions_filtered_df = df[df['ID'].isin(IDs_To_Visualize)]

# Read all relevant video frames into memory
video_frames_to_process = []
print("Reading video frames into memory...")
for i in tqdm(range(130), desc="Reading Frames"):
    ret, frame = video.read()
    if not ret:
        print(f"Warning: Could not read video frame {i}. Processing only {len(video_frames_to_process)} frames.")
        break
    video_frames_to_process.append(frame)

num_frames_read = len(video_frames_to_process)
processed_frames_ordered = [None] * num_frames_read

# Prepare tasks for the executor
tasks_to_submit = []
for i in range(num_frames_read):
    tasks_to_submit.append((i, video_frames_to_process[i], predictions_filtered_df))

# Use ThreadPoolExecutor for parallel processing
# Adjust max_workers as needed; os.cpu_count() is a common choice.
# If None, it defaults to min(32, os.cpu_count() + 4) for Python 3.8+
num_workers = os.cpu_count() 
print(f"Starting parallel processing with up to {num_workers} worker threads...")

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_idx = {executor.submit(process_frame_for_video, task): task[0] for task in tasks_to_submit}
    
    for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks_to_submit), desc="Applying Overlay"):
        idx = future_to_idx[future]
        try:
            _, processed_frame = future.result()
            processed_frames_ordered[idx] = processed_frame
        except Exception as exc:
            print(f'Frame {idx} generated an exception: {exc}')
            # Optionally, store the original frame or a black frame in case of error
            processed_frames_ordered[idx] = video_frames_to_process[idx] 


# Write processed frames to output video
print("Writing processed frames to video...")
for i in tqdm(range(num_frames_read), desc="Writing Video"):
    if i < 100 - frame_offset:
        continue
    if processed_frames_ordered[i] is not None:
        out.write(processed_frames_ordered[i])
    else:
        # This should only happen if reading frames failed initially for some indices
        # or if an exception occurred and wasn't handled by placing original frame.
        print(f"Warning: Frame {i} was not available for writing.")


video.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {output_path}")



