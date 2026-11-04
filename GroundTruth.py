from pathlib import Path
import csv
import cv2 as cv
import subprocess

def transfer_images(dataset_path):
    dataset_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(["rsync", "-av", "slawton@raspberrypi:research_datasets/Pilot/", str(dataset_path) + "/"],check=True)

def transfer_csv(csv_path):

    # Create the destination directory on the Pi
    subprocess.run(
        [
            "ssh",
            "slawton@raspberrypi",
            "mkdir -p ~/research_datasets/GroundTruth/Pilot"
        ],
        check=True
    )

    # Transfer the CSV from desktop to Pi
    subprocess.run(
        [
            "rsync",
            "-av",
            str(csv_path),
            "slawton@raspberrypi:research_datasets/GroundTruth/Pilot/"
        ],
        check=True
    )
    
def save_csv(filename, measurements):
    if not measurements:
        return

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=measurements[0].keys()
        )

        writer.writeheader()
        writer.writerows(measurements)

def select_ground_truth(img):
    # Manually draw rectangle
    x, y, w, h = cv.selectROI("Ground Truth", img, showCrosshair=True, fromCenter=False)

    cv.destroyWindow("Ground Truth")

    if w == 0 or h == 0:
        return None

    return (x, y, x + w, y + h)


def review_ground_truth(img, reference_bbox):

    current_bbox = reference_bbox

    while True:
        preview = img.copy()

        if current_bbox is not None:
            cv.rectangle(preview, current_bbox[:2], current_bbox[2:], (0, 255, 0), 2)

        cv.imshow("Review", preview)

        # A = accept, R = redraw, N = no object present
        key = cv.waitKey(0) & 0xFF

        if key == ord("a"):
            if current_bbox is not None:
                cv.destroyWindow("Review")
                return current_bbox

        elif key == ord("r"):
            new_bbox = select_ground_truth(img)

            if new_bbox is not None:
                current_bbox = new_bbox

        elif key == ord("n"):
            cv.destroyWindow("Review")
            return None


if __name__ == "__main__":

    dataset_path = Path("Datasets/Pilot")

    transfer_images(dataset_path)

    image_paths = [str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]]
    
    first_image = cv.imread(image_paths[0])

    reference_bbox = select_ground_truth(first_image)

    measurements = []

    for image_path in image_paths:
        img = cv.imread(str(image_path))

        approved_bbox = review_ground_truth(img, reference_bbox)

        if approved_bbox is None:
            object_present = False
            coordinates = [None, None, None, None]
        else:
            object_present = True
            coordinates = list(approved_bbox)
            reference_bbox = approved_bbox

        measurements.append({
            "image_id": Path(image_path).stem,
            "object_present": object_present,
            "gt_x1": coordinates[0],
            "gt_y1": coordinates[1],
            "gt_x2": coordinates[2],
            "gt_y2": coordinates[3]
        })

    csv_path = Path("output/pilot/ground_truth/ground_truth.csv")

    save_csv(csv_path, measurements)

    transfer_csv(csv_path)
