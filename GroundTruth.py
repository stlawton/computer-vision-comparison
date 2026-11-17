from pathlib import Path
import csv
import cv2 as cv
import subprocess

def transfer_images(dataset_path, dataset):
    dataset_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(["rsync", "-av", f"slawton@raspberrypi:research_datasets/{dataset}/", str(dataset_path) + "/"],check=True)

def transfer_csv(csv_path, dataset):

    # Create the destination directory on the Pi
    subprocess.run(
        [
            "ssh",
            "slawton@raspberrypi",
            f"mkdir -p ~/research_datasets/GroundTruth/{dataset}/"
        ],
        check=True
    )

    # Transfer the CSV from desktop to Pi
    subprocess.run(
        [
            "rsync",
            "-av",
            str(csv_path),
            f"slawton@raspberrypi:research_datasets/GroundTruth/{dataset}/"
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


def transfer_yolo_labels(label_dir, dataset):

    remote_dir = f"research_datasets/{dataset}"

    subprocess.run(
        [
            "ssh",
            "slawton@raspberrypi",
            f"mkdir -p ~/{remote_dir}"
        ],
        check=True
    )

    subprocess.run(
        [
            "rsync",
            "-av",
            str(label_dir) + "/",
            f"slawton@raspberrypi:{remote_dir}/"
        ],
        check=True
    )


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



def save_yolo_label(image_path, bbox, class_id, label_dir, image_shape):

    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    label_path = label_dir / (Path(image_path).stem + ".txt")

    # Empty image: create an empty label file.
    if bbox is None:
        label_path.write_text("")
        return

    x1, y1, x2, y2 = bbox

    img_height, img_width = image_shape[:2]

    # Convert pixel coordinates to normalized YOLO coordinates.
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    if not (0 <= x1 < x2 <= img_width and
            0 <= y1 < y2 <= img_height):
        raise ValueError(f"Invalid bounding box: {bbox}")

    with open(label_path, "w") as file:
        file.write(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        )


if __name__ == "__main__":

    dataset = input("Enter the dataset name (e.g., 'Pilot'): ")

    dataset_path = Path(f"Datasets/{dataset}")

    transfer_images(dataset_path, dataset)

    image_paths = sorted([str(img) for img in dataset_path.iterdir() if img.is_file() and img.suffix in [".jpg", ".png"]])
    
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

        if dataset == "Training":
       
            save_yolo_label(
                image_path=image_path,
                bbox=approved_bbox,
                class_id=0,
                label_dir=Path(f"Datasets/{dataset}/{dataset}_labels"),
                image_shape=img.shape
            )


    csv_path = Path(f"output/{dataset}/ground_truth/ground_truth.csv")

    save_csv(csv_path, measurements)

    transfer_csv(csv_path, dataset)

    if dataset == "Training":

        transfer_yolo_labels(Path(f"Datasets/{dataset}/{dataset}_labels"), dataset)

