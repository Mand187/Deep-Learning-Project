import torch
from ultralytics import YOLO
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import time

csv_path = 'cars.csv'

df = pd.DataFrame(columns=['Frame', 'ID', 'Class', 'X', 'Y', 'Width', 'Height'])

video_path = 'visualization/michael_10s.mp4'


model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file

start = time.perf_counter()
results = model.track(
    source = video_path,
    show=False,
    save=True,
    half=True,
    stream=False,
    batch=64,
    verbose=False
)
end = time.perf_counter()

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    # Ensure executor is shut down even on early exit
    exit()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

#print(f"Frame: {frame}, ID: {int(v_id)}, Class: {label}, X: {int(x)}, Y: {int(y)}, Width: {int(w)}, Height: {int(h)}")
print(f"Processing time: {end - start:.2f} seconds")
print(f"Average frame time: {((end - start) / total_frames)*1000:.3f} ms")
#df.to_csv(csv_path, index=False)