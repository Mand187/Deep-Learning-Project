
from ultralytics import YOLO
import time

path = 'dataset/cars-1m.mp4'

model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file
results = model.track(
    source = path,
    show=False,
    save=True,
    half=True,
    stream=False,
    batch=256,
    verbose=False
)
