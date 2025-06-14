# visualize.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def visualize_annotations(image: np.ndarray, annotations: List[Dict], figsize=(12, 8)):
    img = image.copy()
    for ann in annotations:
        xmin, ymin, xmax, ymax = map(int, ann['bbox'])
        color = (0, 255, 0) if ann['cutting'] else (255, 0, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{ann['label']} (cut-in)" if ann['cutting'] else ann['label']
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
