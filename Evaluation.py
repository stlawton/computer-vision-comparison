import torch
from torchvision import ops
import cv2 as cv


def get_iou(img, p_x1, p_y1, p_x2, p_y2):
    # Bounding box coordinates.
    ground_truth_bbox = torch.tensor([[831.45, 318.78, 1053.28, 833.18]], dtype=torch.float)
    prediction_bbox = torch.tensor([[p_x1, p_y1, p_x2, p_y2]], dtype=torch.float)

    # Get iou.
    iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
    print('IOU : ', iou.numpy()[0][0])

    output_image = img.copy()

    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_bbox[0].int().tolist()

    cv.rectangle(output_image, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
    cv.rectangle(output_image, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 0, 255), 2)

    cv.imwrite("output/config_tests/IOUResult.jpg", output_image)