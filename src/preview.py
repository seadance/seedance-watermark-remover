from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dynamic import DynamicDetection
from models import Detection, Region


def write_preview(
    frame: np.ndarray,
    output_path: Path,
    region: Region,
    mask: np.ndarray,
    *,
    detection: Detection | None = None,
) -> None:
    preview = frame.copy()
    overlay = preview[region.y : region.y2, region.x : region.x2].copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(overlay[selected], 0.45, red[selected], 0.55, 0)
    preview[region.y : region.y2, region.x : region.x2] = overlay
    cv2.rectangle(preview, (region.x, region.y), (region.x2, region.y2), (0, 220, 255), 2)
    if detection is not None:
        label = f"{detection.corner.value} confidence={detection.confidence:.3f}"
        text_y = max(24, region.y - 8)
        cv2.putText(
            preview,
            label,
            (region.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"could not write preview: {output_path}")


def write_dynamic_preview(
    frame: np.ndarray,
    output_path: Path,
    detections: list[DynamicDetection],
) -> None:
    preview = frame.copy()
    for detection in detections:
        region = detection.region
        overlay = preview[region.y : region.y2, region.x : region.x2].copy()
        red = np.zeros_like(overlay)
        red[..., 2] = 255
        selected = detection.mask > 0
        overlay[selected] = cv2.addWeighted(
            overlay[selected], 0.45, red[selected], 0.55, 0
        )
        preview[region.y : region.y2, region.x : region.x2] = overlay
        cv2.rectangle(
            preview, (region.x, region.y), (region.x2, region.y2), (0, 220, 255), 2
        )
        label = f"{detection.anchor} confidence={detection.confidence:.3f}"
        cv2.putText(
            preview,
            label,
            (region.x, max(24, region.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"could not write preview: {output_path}")
