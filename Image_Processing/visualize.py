import time
import tqdm
import cv2
import pandas as pd
import os
import numpy as np



# --- Configuration ---
video_path = 'visualization/michael_10s.mp4' # Relative path from script location
prediction_path = 'visualization/predictions.csv'
output_path = 'visualization/prediction_visualization_future.mp4' # Where to save the output video
min_id = 0
max_id = 17

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

# Iterate through frames to use the cv2 library to draw the predictions
# Even though the overlay will be the same for all frames, we will still iterate through the frames
# to create a video with the overlay
for row_frame in tqdm(range(130), desc="Processing Frames"):
    ret, frame = video.read()
    if not ret:
        print(f"Error reading frame {row_frame}")
        break
    # get all predictions for ids within the range
    predictions = (df['ID'].between(min_id, max_id))
    for i, row in df[predictions].iterrows():
        row_frame = int(row['Frame'])
        if row_frame < 100:
            # Draw red lines for the first 100 frames
            color = (0, 0, 255)
            cv2.circle(frame, (int(row['X_pred']), int(row['Y_pred'])), 5, color, -1)
        elif row_frame < 130:
            color_pred = (255, 0, 255)
            color_true = (0, 255, 0)
            # Draw green lines for the true points
            cv2.circle(frame, (int(row['X_true']), int(row['Y_true'])), 5, color_true, -1)
            # Draw purple lines for the predicted points
            cv2.circle(frame, (int(row['X_pred']), int(row['Y_pred'])), 5, color_pred, -1)
    # Write the frame with the overlay to the output video
    out.write(frame)

video.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {output_path}")
            
    

