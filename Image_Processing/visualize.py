import time
import tqdm
import cv2
import pandas as pd
import os
import numpy as np

# --- Configuration ---
video_path = 'visualization/cars-10s.mp4' # Relative path from script location
prediction_path = 'visualization/predictions.csv'
csv_path = 'visualization/cars-10s_detections.csv' # Relative path
output_path = 'visualization/prediction_visualization_future.mp4' # Where to save the output video

# Column names
frame_col = 'Frame' # Column name for frame number
id_col = 'ID'    # Column name for the car ID to use
current_x_col = 'X' # Column name for current X position
current_y_col = 'Y' # Column name for current Y position

# Prediction settings
future_frame_offset = 30 # Number of frames into the future to predict (e.g., 30 frames @ 60fps = 0.5s)

# --- Ensure output directory exists ---
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Load Data ---
try:
    df = pd.read_csv(csv_path)
    # Verify required columns exist
    required_cols = [frame_col, id_col, current_x_col, current_y_col, future_x_col, future_y_col]
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
print(f"Predicting {future_frame_offset} frames into the future.")

"""
Predictions start at frame 20
Use frames 0-19 to predict frames 20-49
Use frames 50-69 to predict frames 70-89
Frames 20-49 are in sequence_index 0
Frames 

For frames 0-19 do nothing
For frames 20-59:
    For each car ID in the current frame:
        Draw a circle at the current position
        Draw a green arrow to the actual position in the future frame
        Draw a red arrow to the predicted position in the future frame
"""


# --- Frame-by-Frame Processing ---
# Use tqdm for progress bar
for frame_num in tqdm.tqdm(range(total_frames), desc="Processing Frames"):
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Stopped reading early at frame {frame_num}.")
        break
    pred_loop_idx = frame_num % 50
    if pred_loop_idx < 20:
        # Skip drawing for frames 0-19
        out.write(frame)
        continue
    

    # Get detections for the current frame
    # Use direct boolean indexing which is usually efficient
    frame_detections = df[df[frame_col] == frame_num]

    # Draw prediction lines for each detection
    for _, row in frame_detections.iterrows():
        # Get current position and ID
        pt1 = (int(row[current_x_col]), int(row[current_y_col]))
        car_id_norm = row[id_col] # Already converted type during loading

        # Draw circle at current position
        cv2.circle(frame, pt1, 3, (0, 0, 255), -1) # Red circle, filled

        # Calculate target future frame
        future_frame = frame_num + future_frame_offset

        # Find the car's data in the future frame using the index
        future_index = (future_frame, car_id_norm)
        try:
            # Use .loc for index-based lookup
            future_row = grouped_data.loc[future_index]

            # If future_row is a Series (single match)
            if isinstance(future_row, pd.Series):
                pt2 = (int(future_row[future_x_col]), int(future_row[future_y_col]))
                # Draw the prediction line
                cv2.arrowedLine(frame, pt1, pt2, (0, 0, 255), 2) # Red line, thickness 2
            # Handle case if multiple rows match (shouldn't happen with unique frame, id pairs)
            elif isinstance(future_row, pd.DataFrame) and not future_row.empty:
                 # Take the first match if duplicates exist (log warning?)
                 first_future_row = future_row.iloc[0]
                 pt2 = (int(first_future_row[current_x_col]), int(first_future_row[current_y_col]))
                 cv2.arrowedline(frame, pt1, pt2, (0, 255, 0), 2)
                 print(f"Warning: Multiple entries found for Frame {future_frame}, ID {car_id_norm}. Using first.")


            # Optional: Put ID text near the start point
            # cv2.putText(frame, str(car_id_norm), (pt1[0] + 5, pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except KeyError:
            # Data for this car_id_norm not found at future_frame
            # No line is drawn, only the circle at pt1 remains
            pass
        except Exception as e:
             print(f"Error during future lookup for Frame {frame_num}, ID {car_id_norm}: {e}")


    # Write the frame with drawings to the output video
    out.write(frame)

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nProcessing complete. Output saved to {output_path}")
