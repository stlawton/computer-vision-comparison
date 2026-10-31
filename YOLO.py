from ultralytics.utils.metrics import box_iou
from ultralytics import YOLO
from Performance import benchmark
from Results import save_csv
from pathlib import Path
import cv2 as cv



model = YOLO("yolo26n.pt") 

def detect_yolo(image):
    results = model.predict(
        source = image,
        imgsz = 640,
        conf = 0.25,
        iou = 0.7,
        device = "cpu",
        save = False,
        verbose = False
    )

    predictions = []

    for box in results[0].boxes:

        class_id = int(box.cls[0].item())

        predictions.append({
            "bbox": box.xyxy[0].tolist(),
            "class_id": class_id,
            "class_name": results[0].names[class_id],
            "confidence": float(box.conf[0].item())
        })

    return predictions


if __name__ == "__main__":
    
    dataset_path = Path("Datasets/Pilot")

    image_paths = [str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]]

    results = benchmark(image_paths, cv.imread, detect_yolo, warmups=5, method="yolo26n")

    save_csv("output/raw/pilot/yolo_latency.csv", results["latency"])
    save_csv("output/raw/pilot/yolo_resources.csv", results["resources"])
    save_csv("output/raw/pilot/yolo_predictions.csv", results["predictions"])
    save_csv("output/raw/pilot/yolo_run_info.csv", [results["run_info"]])