
from ultralytics import YOLO
import time
import tqdm
import cv2

path = 'dataset/cars-1m.mp4'
save_path = 'dataset/cars-1m-processed'

model = YOLO('yolo12x.pt').to('cuda', non_blocking=True)  # load a model from file
start = time.perf_counter()
results = model.track(
    source = path,
    save=True,
    save_txt=False,
    half=True,
    stream=True,
    batch=128,
    verbose=False,

)

# Get frame count
cap = cv2.VideoCapture(path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f"Total frames: {frame_count}")

# create a progress bar
pbar = tqdm.tqdm(iterable=enumerate(results), total=frame_count, desc="Processing frames", unit="frame")
full_results = []
for frame, result in pbar:
    # Process each result
    #pbar.set_postfix({"frame": result.frame})
    full_results.append(result)



duration = time.perf_counter() - start
print(f"Duration: {duration:.2f} seconds")
