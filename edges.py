import cv2
import numpy as np
from scipy.stats import mode

def get_quantized_edges(label_img):
    H, W = label_img.shape
    edges = np.zeros((H, W), dtype=np.uint8)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(label_img, shift=(dy, dx), axis=(0, 1))
        edges |= (shifted != label_img).astype(np.uint8)

    return edges * 255  # make it binary image

def remove_small_regions(label_img, min_size=30):
    cleaned = label_img.copy()
    unique_labels = np.unique(label_img)

    for color in unique_labels:
        mask = (label_img == color).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
                component_mask = (labels == i)

                # Find surrounding pixels (more robust)
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(component_mask.astype(np.uint8), kernel)
                border = dilated & (~component_mask)

                surrounding_labels = label_img[border.astype(bool)]
                if surrounding_labels.size == 0:
                    continue  # skip isolated pixels with no valid border

                m = mode(surrounding_labels, axis=None, keepdims=True)
                if m.count.size == 0 or m.count[0] == 0:
                    continue  # no mode found
                new_color = m.mode[0]
                cleaned[component_mask] = new_color

    return cleaned
