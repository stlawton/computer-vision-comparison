from pathlib import Path
import cv2 as cv
import sys
import numpy as np


from Results import save_csv
from Performance import benchmark


def detect_blue(img):
    # Convert the image to the HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define the range of blue colors in HSV
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([135, 255, 255])
    
    # Create a mask for blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    predictions = []

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        

        if cv.contourArea(contour) > 100:  # Filter out small contours
            x, y, w, h = cv.boundingRect(contour)

            predictions.append({
                "bbox": [x, y, x + w, y + h],
                "class_id": None,
                "class_name": "blue_candidate",
                "confidence": None
            })


    return predictions



if __name__ == "__main__":
    
    dataset_path = Path("Datasets/Pilot")

    image_paths = [str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]]

    results = benchmark(image_paths, cv.imread, detect_blue, warmups=5, method="openCV")

    save_csv("output/raw/pilot/openCV_latency.csv", results["latency"])
    save_csv("output/raw/pilot/openCV_resources.csv", results["resources"])
    save_csv("output/raw/pilot/openCV_predictions.csv", results["predictions"])
    save_csv("output/raw/pilot/openCV_run_info.csv", [results["run_info"]])


