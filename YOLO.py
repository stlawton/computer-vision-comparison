from ultralytics.utils.metrics import box_iou
from ultralytics import YOLO
from Performance import benchmark
from Results import save_csv
from pathlib import Path
import cv2 as cv

# This script runs the Yolo26n model on the selected dataset and saves the results to CSV files. It is built to be run independently so that
# when testing the different moethods, the Raspberry Pi can be run with a clean slate between tests. The main method runs the script allowing
# the user to choose the inputs and output locations. The detect_yolo function is used to run the model on each image and return the predictions
# in a structured format.

model = YOLO("yolo26n.pt")                              # Load the desired YOLO26n model.

# This function takes an image as input, runs the YOLO26n model on it, and returns a list of predictions.
# Each prediction is a dictionary containing the bounding box coordinates, class ID, class name, and confidence score.

def detect_yolo(image):

    results = model.predict(                            # Run the YOLO26n model on the input image and set the parameters
        source = image,                                 # for image size, confidence threshold, IoU threshold, device, and verbosity.
        imgsz = 640,
        conf = 0.25,
        iou = 0.7,
        device = "cpu",
        save = False,
        verbose = False
    )

    predictions = []                                    # Initialize an empty list to store the predictions.

    for box in results[0].boxes:                        # Iterate through the bounding boxes in the results.

        class_id = int(box.cls[0].item())               # Get the class ID of the detected object.

        predictions.append({                            # Append a dictionary containing the bounding box coordinates, 
            "bbox": box.xyxy[0].tolist(),               # class ID, class name, and confidence score to the predictions list.
            "class_id": class_id,
            "class_name": results[0].names[class_id],
            "confidence": float(box.conf[0].item())
        })

    return predictions


# This main method runs the script, allowing the user to choose the inputs and output locations. It calls the benchmark function from Perfomance.py to
# collect latency, resource usage, and prediction results for the YOLO26n model on the selected dataset. It also calls the save_csv function from Results.py
# to save the results to CSV files for further analysis.

if __name__ == "__main__":
    
    dataset_path = Path("Datasets/Pilot")                                                                               # Set the path to the dataset directory containing the images to be processed.

    image_paths = [str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]]      # Create a list of image paths found in the dataset directory.

    results = benchmark(image_paths, cv.imread, detect_yolo, warmups=5, method="yolo26n")                               # Call the benchmark function on the list of image paths

    save_csv("output/pilot/raw/yolo_latency.csv", results["latency"])
    save_csv("output/pilot/raw/yolo_resources.csv", results["resources"])
    save_csv("output/pilot/raw/yolo_predictions.csv", results["predictions"])
    save_csv("output/pilot/raw/yolo_run_info.csv", [results["run_info"]])