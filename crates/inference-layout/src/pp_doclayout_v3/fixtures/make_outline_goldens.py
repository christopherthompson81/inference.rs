#!/usr/bin/env python3
"""Region outline goldens for outline.rs: transformers' own PPDocLayoutV3ImageProcessor._extract_polygon_points_by_masks
on synthetic masks (shapes drawn with OpenCV), so the Rust port can be held to its exact vertices.

Usage: python3 make_outline_goldens.py > outline_goldens.json  (needs transformers >= 5 with PP-DocLayoutV3, numpy, cv2)
"""

import json

import cv2
import numpy as np
from transformers.models.pp_doclayout_v3.image_processing_pp_doclayout_v3 import PPDocLayoutV3ImageProcessor

MASK_H, MASK_W = 50, 60
# model input 4x the mask, and a page of another size: (width, height)
MODEL_SIZE = (MASK_W * 4, MASK_H * 4)
PAGE_SIZE = (900, 700)


def shape(kind):
    m = np.zeros((MASK_H, MASK_W), np.uint8)
    if kind == "rect":
        m[10:30, 12:45] = 1
    elif kind == "skewed":
        cv2.fillPoly(m, [np.array([[8, 12], [50, 6], [54, 30], [12, 38]], np.int32)], 1)
    elif kind == "ellipse":
        cv2.ellipse(m, (30, 25), (22, 14), 20, 0, 360, 1, -1)
    elif kind == "l_shape":
        m[5:40, 5:15] = 1
        m[30:40, 5:50] = 1
    elif kind == "hole":
        m[5:45, 5:55] = 1
        m[15:35, 20:40] = 0
    elif kind == "two_parts":
        m[5:15, 5:20] = 1
        m[20:45, 25:55] = 1
    elif kind == "thin":
        m[24:26, 3:57] = 1
    elif kind == "dot":
        m[20, 20] = 1
    elif kind == "empty":
        pass
    elif kind == "triangle":
        cv2.fillPoly(m, [np.array([[10, 40], [50, 40], [30, 5]], np.int32)], 1)
    elif kind == "notched":
        cv2.fillPoly(m, [np.array([[5, 5], [55, 5], [55, 45], [30, 25], [5, 45]], np.int32)], 1)
    return m


KINDS = ["rect", "skewed", "ellipse", "l_shape", "hole", "two_parts", "thin", "dot", "empty", "triangle", "notched"]


def main():
    proc = PPDocLayoutV3ImageProcessor()
    sx, sy = PAGE_SIZE[0] / MODEL_SIZE[0], PAGE_SIZE[1] / MODEL_SIZE[1]
    # boxes in page pixels covering most of the mask, and one off by a margin so the crop is partial
    boxes = []
    for i, _ in enumerate(KINDS):
        margin = 3 * (i % 3)
        boxes.append([margin * 4 * sx + 1.7, margin * 4 * sy + 0.4, (MASK_W * 4 - margin * 4) * sx - 0.3, (MASK_H * 4 - margin * 4) * sy - 1.2])
    masks = np.stack([shape(k) for k in KINDS])
    polygons = proc._extract_polygon_points_by_masks(
        np.array(boxes, np.float32), masks, [MODEL_SIZE[0] / PAGE_SIZE[0], MODEL_SIZE[1] / PAGE_SIZE[1]]
    )
    cases = [
        {"kind": k, "mask": m.astype(int).tolist(), "bbox": b, "polygon": np.asarray(p, np.float64).tolist()}
        for k, m, b, p in zip(KINDS, masks, boxes, polygons)
    ]
    print(json.dumps({"model_size": MODEL_SIZE, "page_size": PAGE_SIZE, "cases": cases}))


if __name__ == "__main__":
    main()
