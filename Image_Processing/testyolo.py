import torch.nn
from ultralytics import YOLO



image = 'dataset/frames/vid1.MOV/0000000125.jpg'


model = YOLO('yolo12l.pt')

results = model(
    source = image,
    show=True,
    save=True
    
)