import torch
from torchvision import ops
import cv2 as cv
import pandas as pd
from pathlib import Path

def get_iou(method, prediction_csv, ground_truth_csv):

    ground_truth_df = pd.read_csv(ground_truth_csv)
    prediction_df = pd.read_csv(prediction_csv)

    results = []

    image_dir = Path("Datasets/Pilot")
    output_dir = Path(f"output/pilot/iou_image/{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, gt in ground_truth_df.iterrows():

        image_id = gt["image_id"]

        # Find predictions belonging to this image.
        predictions = prediction_df[
            prediction_df["image_id"] == image_id
        ]

        # Load the original image.
        img = cv.imread(str(image_dir / f"{image_id}.jpg"))

        if img is None:
            raise FileNotFoundError(f"Image not found: {image_id}")

        output_image = img.copy()

        # Convert predictions to bounding-box tensors.
        pred_boxes = None

        if not predictions.empty:
            pred_boxes = torch.tensor(
                predictions[
                    ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]
                ].to_numpy(),
                dtype=torch.float32
            )

        # Check whether the object is present.
        object_present = str(
            gt["object_present"]
        ).lower() == "true"

        if not object_present:
            # IoU is not applicable when no object exists.
            iou = None

        else:
            # Ground-truth bounding box.
            gt_box = torch.tensor([[
                gt["gt_x1"],
                gt["gt_y1"],
                gt["gt_x2"],
                gt["gt_y2"]
            ]], dtype=torch.float32)

            # Draw ground truth in red.
            gt_x1, gt_y1, gt_x2, gt_y2 = (
                gt_box[0].int().tolist()
            )

            cv.rectangle(
                output_image,
                (gt_x1, gt_y1),
                (gt_x2, gt_y2),
                (0, 0, 255),
                2
            )

            if pred_boxes is None:
                # Object present, but no prediction.
                iou = 0.0

            else:
                # Calculate IoU for every prediction.
                ious = ops.box_iou(gt_box, pred_boxes)

                # Select the prediction with highest IoU.
                best_index = torch.argmax(ious[0]).item()

                iou = ious[0, best_index].item()

                # Draw only the best prediction in green.
                p_x1, p_y1, p_x2, p_y2 = (
                    pred_boxes[best_index].int().tolist()
                )

                cv.rectangle(
                    output_image,
                    (p_x1, p_y1),
                    (p_x2, p_y2),
                    (0, 255, 0),
                    2
                )

        # Record the calculated IoU.
        results.append({
            "image_id": image_id,
            "iou": iou
        })

        # Save the annotated image.
        cv.imwrite(
            str(output_dir / f"{method}_{image_id}_IOUResult.jpg"),
            output_image
        )

    return results
