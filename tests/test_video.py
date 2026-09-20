import cv2
import numpy as np

from models import Region
from video import _inpaint_with_context


def test_rectangle_inpainting_uses_pixels_outside_the_region() -> None:
    frame = np.full((80, 120, 3), (30, 80, 140), dtype=np.uint8)
    region = Region(45, 30, 30, 20)
    frame[region.y : region.y2, region.x : region.x2] = (245, 245, 245)
    mask = np.full((region.height, region.width), 255, dtype=np.uint8)

    _inpaint_with_context(frame, region, mask, radius=4)

    repaired = frame[region.y : region.y2, region.x : region.x2]
    assert float(cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY).mean()) < 200
