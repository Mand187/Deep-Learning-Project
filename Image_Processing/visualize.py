
from ultralytics import YOLO
import time

path = 'dataset/cars-1m.mp4'
save_path = 'cars-1m-processed'

model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file
start = time.perf_counter()
results = model.track(
    source = path,
    save=True,
    save_txt=True,
    half=True,
    stream=False,
    batch=1024,
    verbose=True,
    visualize = True,
    line_width = 1,
    font_size = 8,

)
duration = time.perf_counter() - start
print(f"Duration: {duration:.2f} seconds")