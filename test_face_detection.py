import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, ".")
from configs.load_configs import configs

framework = configs["framework"]

# ONNX -------------------------------------------------------------------------------------------------------------

def check_bbox(bbox, img):
    h, w = img.shape[:2]
    bbox[0] = 0 if bbox[0] < 0 else bbox[0]
    bbox[1] = 0 if bbox[1] < 0 else bbox[1]
    bbox[2] = w if bbox[2] > w else bbox[2]
    bbox[3] = h if bbox[3] > h else bbox[3]
    return bbox


def transform_to_square_bbox(bbox, img, size_expand=1.55):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * -0.15
    size = int(old_size * size_expand)
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    roi_box = check_bbox(roi_box, img)
    return np.uint32(roi_box)

# scrfd - face detection
scrfd_weight = configs['weights'][framework]
if os.path.isfile(scrfd_weight):
    from face_detection.SCRFD_ONNX import SCRFD_ONNX

    provider = ["CPUExecutionProvider"]
    face_detection_model = SCRFD_ONNX(model_file=scrfd_weight,
                                      providers=provider,
                                      input_size=(320, 320))

if __name__ == "__main__":
    # Detect bounding boxes
    img = cv2.imread(configs['path_image'])
    bboxes = face_detection_model.run(img)
    # Get the biggest bounding box
    bboxes_sizes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    biggest_bbox = bboxes[np.argmax(bboxes_sizes)][:4]
    square_bbox = transform_to_square_bbox(biggest_bbox, img)
    x1, y1, x2, y2 = square_bbox
    print(square_bbox)

    # Visualize
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imshow("face detection", img)
    cv2.waitKey(0)

    # print(biggest_bbox)