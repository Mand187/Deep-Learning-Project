
from ultralytics import YOLO
import time

path = 'videos/cars-1m.mp4'
save_path = 'videos/cars-1m-processed'

model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file
start = time.perf_counter()
results = model.track(
    source = path,
    save=True,
    save_txt=False,
    half=True,
    stream=False,
    batch=512,
    verbose=False,

)
duration = time.perf_counter() - start
print(f"Duration: {duration:.2f} seconds")
