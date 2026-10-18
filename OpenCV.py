import cv2 as cv
import sys
import numpy as np
import torch
from torchvision import ops

img = cv.imread("output/config_tests/config_test1.jpg")
if img is None:
    sys.exit("Could not read the image.")

cv.imwrite("output/config_tests/CVTest.jpg", img)

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


def detect_blue(img):
    # Convert the image to the HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define the range of blue colors in HSV
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([135, 255, 255])
    
    # Create a mask for blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    output_image = img.copy()
    mask_img = img.copy()

    cv.imwrite("output/config_tests/CVTestMask.jpg", mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)


        if cv.contourArea(contour) > 100:  # Filter out small contours
            cv.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            get_iou(output_image, x, y, x + w, y + h)


    return output_image

cv.imwrite("output/config_tests/CVTestBox.jpg", detect_blue(img))

