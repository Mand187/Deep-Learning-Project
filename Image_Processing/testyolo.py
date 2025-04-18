import torch
from ultralytics import YOLO
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import time

csv_path = 'cars.csv'

df = pd.DataFrame(columns=['Frame', 'ID', 'Class', 'X', 'Y', 'Width', 'Height'])

video_path = 'dataset/cars-10s.mp4'


model = YOLO('yolo12l.pt').to('cuda', non_blocking=True)  # load a model from file

start = time.perf_counter()
results = model.track(
    source = video_path,
    show=False,
    save=False,
    half=True,
    stream=False,
    batch=256,
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


progress_bar = tqdm(
    enumerate(results),
    desc = f"Processing video {video_path}",
    total = total_frames,
)

for frame, result in progress_bar:
    for v_class, v_id, xywh, conf in zip(result.boxes.cls, result.boxes.id, result.boxes.xywh, result.boxes.conf):
        x, y, w, h = xywh
        label = model.names[int(v_class)]
        df.loc[len(df)] = [frame, int(v_id), label, int(x), int(y), int(w), int(h)]
        #print(f"Frame: {frame}, ID: {int(v_id)}, Class: {label}, X: {int(x)}, Y: {int(y)}, Width: {int(w)}, Height: {int(h)}")
print(f"Processing time: {end - start:.2f} seconds")
print(f"Average frame time: {((end - start) / total_frames)*1000:.3f} ms")
#df.to_csv(csv_path, index=False)