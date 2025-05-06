import time
import tqdm
import cv2
import pandas as pd
import os
import numpy as np

"""


"""


# --- Configuration ---
video_path = 'visualization/cars-10s.mp4' # Relative path from script location
prediction_path = 'visualization/predictions.csv'
csv_path = 'visualization/cars-10s_detections.csv' # Relative path
output_path = 'visualization/prediction_visualization_future.mp4' # Where to save the output video

# Column names
frame_col = 'Frame' # Column name for frame number
id_col = 'ID_Norm'    # Column name for the car ID to use
current_x_col = 'X' # Column name for current X position
current_y_col = 'Y' # Column name for current Y position

# Prediction settings
frame_skip = 5 # Number of frames into the future to predict (e.g., 30 frames @ 60fps = 0.5s)

# --- Ensure output directory exists ---
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Load Data ---
try:
    df = pd.read_csv(csv_path)
    # Verify required columns exist
    required_cols = [frame_col, id_col, current_x_col, current_y_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}. Required: {required_cols}")

    # Optimize for future lookups: Create a multi-index
    # Convert ID column to a consistent type if necessary (e.g., int or float)
    # Check if id_col is numeric, if not, maybe skip conversion or handle appropriately
    if pd.api.types.is_numeric_dtype(df[id_col]):
         # Decide on int or float based on data. If it has decimals, use float.
         if df[id_col].apply(lambda x: x == int(x) if pd.notnull(x) else True).all():
             df[id_col] = df[id_col].astype(int)
         else:
              df[id_col] = df[id_col].astype(float) # Or handle non-integer IDs differently

    # Ensure frame column is integer
    df[frame_col] = df[frame_col].astype(int)

    # Set index for faster lookups
    grouped_data = df.set_index([frame_col, id_col])
    grouped_data.sort_index(inplace=True) # Sorting can potentially speed up lookups

except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error reading or processing CSV: {e}")
    exit()


# --- Video Processing ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing video: {video_path}")
print(f"Outputting to: {output_path}")
print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")
print(f"Using ID column: '{id_col}', Position columns: '{current_x_col}', '{current_y_col}'")
print(f"Predicting {frame_skip} frames into the future.")
