import torch
from ultralytics.utils.metrics import box_iou
from ultralytics import YOLO
from Performance import benchmark
from Results import save_csv
from pathlib import Path



model = YOLO("yolo26n.pt") 

def detect_yolo(image_path):
    results = model(image_path, save=True)
    return results


if __name__ == "__main__":
    
    dataset_path = Path("Datasets/Pilot")

    image_paths = [str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]]

    results = benchmark(image_paths, detect_yolo, warmups=5)

    save_csv("output/raw/pilot/yolo_latency.csv", results["latency"])
    save_csv("output/raw/pilot/yolo_resources.csv", results["resources"])