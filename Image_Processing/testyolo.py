import torch
from ultralytics import YOLO
import pandas as pd
import torch

csv_path = 'cars.csv'

df = pd.DataFrame(columns=['Frame', 'ID', 'Class', 'X', 'Y', 'Width', 'Height'])



model = YOLO('yolo12l.pt').to('cuda', non_blocking=True)  # load a model from file

results = model.track(
    source = 'dataset/cars-10s.mp4',
    show=False,
    save=False,
    half=True,
    stream=True,
    batch=32,
)




for frame, result in enumerate(results):
    for v_class, v_id, xywh, conf in zip(result.boxes.cls, result.boxes.id, result.boxes.xywh, result.boxes.conf):
        x, y, w, h = xywh
        label = model.names[int(v_class)]
        df.loc[len(df)] = [frame, int(v_id), label, int(x), int(y), int(w), int(h)]
        #print(f"Frame: {frame}, ID: {int(v_id)}, Class: {label}, X: {int(x)}, Y: {int(y)}, Width: {int(w)}, Height: {int(h)}")
df.to_csv(csv_path, index=False)