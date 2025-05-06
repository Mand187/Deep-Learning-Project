import torch
from ultralytics import YOLO
import pandas as pd
import os # Make sure os is imported
import cv2
import multiprocessing as mp
from pathlib import Path
import time # Optional: for timing
from tqdm import tqdm # Import tqdm
import signal # Import signal module
import sys # Import sys module
import concurrent.futures # Import ThreadPoolExecutor
import threading # For potential Lock if needed, though futures might suffice

model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file

data_dir = 'visualization'
output_dir = 'visualization'

#create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def process_video(path):
    
    #use cv2 to get the total number of frames

    try:
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        print(f"Error opening video file: {path}. Error: {e}")
        total_frames = 0
    
    
    results = model.track(
        source=str(path),
        show=False,
        save=False,
        half=True,
        stream=True,
        device='cuda',
        verbose=False,
        batch=64
    )
    df = pd.DataFrame(columns=['Frame', 'ID', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])
    progress_bar = tqdm(
        enumerate(results),
        desc=f"Processing video {path}",
        total=total_frames,
        unit = ' frames'
    )
    for frame, result in progress_bar:
        try:
            ids = result.boxes.id.tolist()
            classes = result.boxes.cls.tolist()
            xywh = result.boxes.xywh.tolist()
            confs = result.boxes.conf.tolist()
            
            for v_class, v_id, xywh, conf in zip(classes, ids, xywh, confs):
                x, y, w, h = xywh
                label = model.names[int(v_class)]
                df.loc[len(df)] = [frame, int(v_id), label, conf, int(x), int(y), int(w), int(h)]
        except Exception as e:
            print(f"Error processing frame {frame} of video {path}. Error: {e}")
                #print(f"Frame: {frame}, ID: {int(v_id)}, Class: {label}, X: {int(x)}, Y: {int(y)}, Width: {int(w)}, Height: {int(h)}")
    # Save the DataFrame to a CSV file
    csv_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Processed {path} and saved to {csv_path}")

if __name__ == "__main__":
    # Get a list of all video files in the directory
    video_files = list(Path(data_dir).glob('*.mp4')) + list(Path(data_dir).glob('*.MP4'))  # Handle both lowercase and uppercase extensions
    video_files.sort(key=lambda x: x.name, reverse=True)
    for video_file in video_files:
        print(f"Processing {video_file}...")
        process_video(video_file)
        print(f"Finished processing {video_file}.")
    print("All videos processed.")
    
    
        

