"""The layout detector through the package; needs INFERENCE_TEST_LAYOUT_MODEL (HF dir), INFERENCE_TEST_LAYOUT_IMAGE
and Pillow to decode the image."""

import math
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_rs as ir

MODEL_VARIABLE = "INFERENCE_TEST_LAYOUT_MODEL"
IMAGE_VARIABLE = "INFERENCE_TEST_LAYOUT_IMAGE"
MIN_POLYGON_VERTICES = 4

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipUnless(
    os.environ.get(MODEL_VARIABLE) and os.environ.get(IMAGE_VARIABLE) and Image,
    f"{MODEL_VARIABLE}, {IMAGE_VARIABLE} and Pillow are needed",
)
class LayoutTest(unittest.TestCase):
    def test_detections_carry_boxes_and_outlines(self):
        image = Image.open(os.environ[IMAGE_VARIABLE]).convert("RGB")
        with ir.LayoutModel(os.environ[MODEL_VARIABLE], backend="cpu") as model:
            labels = model.labels
            detections = model.detect(ir.LayoutImage(image.tobytes(), image.width, image.height, ir.PixelFormat.RGB8))
        self.assertTrue(detections)
        for detection in detections:
            self.assertEqual(detection.label, labels[detection.class_id])
            self.assertGreaterEqual(len(detection.polygon), MIN_POLYGON_VERTICES)
            self.assertTrue(all(math.isfinite(v) for point in detection.polygon for v in point))


if __name__ == "__main__":
    unittest.main()
