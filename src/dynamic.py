from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from models import Region

_DOLA_TEMPLATE_DATA = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/gAAAAAAAAAAAAAAAAAAf/8B//gAAAAB/AAAAAAAAH//gf/8AAAAAfwAAAAAAAD//4H//wAAAAH8AAAAAAAA5w+B//+AAAAB/AAAAAAAAc8PAf//wAAAAfgAAAAAAAOPDwHwP8AAAAH4AA4AAAADjw8B8B/gB/gA/AD/gAAAAx4GAfAH4D//APwH//AAAAM+AgHwB+B//4D8D//4AAACfAAB8APw///A/D///AAABnwAAfAD8P4/wPwfgfwAAAT4AAHwA/H4H+D8AAF8AAAE+AAB8APx+Afg/AAA/AAACfAAAfAD8fgH4fwAD9wAABkcAAHwA/H4B+H8A//cAAAYD/8B8APh+Afh/A///AAAPg//AfAP4fgH4fwf/9wAADw//wHwD+H4B+H8P4HcAAA4//8B8H/B+A/h/D8B3AAAOP//Af//wPwf4fx/AfwAAD+AAAH//4B/f8H8f4P8AAB3gAAB//8AP/+B/D///AAAf4AAAf/8AB//Afw///4AAP+AAAH/4AAP/AH8H//+AAD/gAAAAAAAAAAAAAP4/wAB/4AAAAAAAAAAAAAAAAAAAf+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
)
_DOLA_TEMPLATE = np.unpackbits(
    np.frombuffer(base64.b64decode(_DOLA_TEMPLATE_DATA), dtype=np.uint8)
).reshape(32, 128)


def _wordmark_template(text: str) -> np.ndarray:
    canvas = np.zeros((64, 320), dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (4, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        255,
        2,
        cv2.LINE_AA,
    )
    return _normalize_shape(canvas)


@dataclass(frozen=True)
class DynamicDetection:
    region: Region
    mask: np.ndarray
    confidence: float
    shape_similarity: float
    anchor: str


def detect_edge_watermarks(
    frame: np.ndarray,
    *,
    min_confidence: float = 0.45,
    min_shape_similarity: float = 0.55,
    max_detections: int = 2,
) -> list[DynamicDetection]:
    """Detect pale, text-like badges near eight positions along the frame edge."""
    frame_height, frame_width = frame.shape[:2]
    candidates: list[DynamicDetection] = []

    for anchor, outer in edge_regions(frame_width, frame_height):
        crop = frame[outer.y : outer.y2, outer.x : outer.x2]
        result = _text_candidate(
            crop, min_shape_similarity=min_shape_similarity
        )
        if result is None:
            continue
        confidence, local_region, local_mask, shape_similarity = result
        if confidence < min_confidence:
            continue
        candidates.append(
            DynamicDetection(
                region=Region(
                    outer.x + local_region.x,
                    outer.y + local_region.y,
                    local_region.width,
                    local_region.height,
                ),
                mask=local_mask,
                confidence=confidence,
                shape_similarity=shape_similarity,
                anchor=anchor,
            )
        )

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    selected: list[DynamicDetection] = []
    for candidate in candidates:
        if any(_overlap_ratio(candidate.region, item.region) > 0.25 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_detections:
            break
    return selected


def edge_regions(width: int, height: int) -> list[tuple[str, Region]]:
    crop_width = min(width, max(140, round(width * 0.21)))
    crop_height = min(height, max(68, round(height * 0.13)))
    center_x = max(0, (width - crop_width) // 2)
    center_y = max(0, (height - crop_height) // 2)
    right = max(0, width - crop_width)
    bottom = max(0, height - crop_height)
    return [
        ("top-left", Region(0, 0, crop_width, crop_height)),
        ("top-center", Region(center_x, 0, crop_width, crop_height)),
        ("top-right", Region(right, 0, crop_width, crop_height)),
        ("middle-left", Region(0, center_y, crop_width, crop_height)),
        ("middle-right", Region(right, center_y, crop_width, crop_height)),
        ("bottom-left", Region(0, bottom, crop_width, crop_height)),
        ("bottom-center", Region(center_x, bottom, crop_width, crop_height)),
        ("bottom-right", Region(right, bottom, crop_width, crop_height)),
    ]


def _text_candidate(
    crop: np.ndarray, *, min_shape_similarity: float
) -> tuple[float, Region, np.ndarray, float] | None:
    height, width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local_background = cv2.GaussianBlur(gray, (0, 0), 3)
    bright_detail = cv2.subtract(gray, local_background)

    raw_mask = np.where(
        (hsv[..., 2] > 130) & (hsv[..., 1] < 100) & (bright_detail > 4),
        255,
        0,
    ).astype(np.uint8)
    mask = raw_mask.copy()
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
    )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components: list[tuple[int, int, int, int, int, int]] = []
    min_height = max(5, round(height * 0.10))
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if (
            area >= 8
            and component_height >= min_height
            and component_height <= height * 0.58
            and component_width <= width * 0.84
        ):
            components.append(
                (component_id, x, y, component_width, component_height, area)
            )

    best: tuple[float, Region, np.ndarray, float] | None = None
    for component in components:
        target_y = component[2] + component[4] / 2
        aligned = [
            item
            for item in components
            if abs((item[2] + item[4] / 2) - target_y) <= height * 0.18
        ]
        x1 = min(item[1] for item in aligned)
        y1 = min(item[2] for item in aligned)
        x2 = max(item[1] + item[3] for item in aligned)
        y2 = max(item[2] + item[4] for item in aligned)
        box_width = x2 - x1
        box_height = y2 - y1
        width_fraction = box_width / width
        height_fraction = box_height / height
        density = sum(item[5] for item in aligned) / max(1, box_width * box_height)
        if len(aligned) < 3 and width_fraction < 0.40:
            continue

        confidence = _candidate_confidence(
            len(aligned), width_fraction, height_fraction, density
        )
        if confidence <= 0:
            continue

        pad = max(6, round(min(width, height) * 0.04))
        region = Region(
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(width - max(0, x1 - pad), box_width + 2 * pad),
            min(height - max(0, y1 - pad), box_height + 2 * pad),
        )
        component_mask = np.zeros((height, width), dtype=np.uint8)
        for component_id, *_ in aligned:
            component_mask[labels == component_id] = 255
        component_mask = cv2.dilate(
            component_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        local_mask = component_mask[region.y : region.y2, region.x : region.x2]
        shape_mask = raw_mask[region.y : region.y2, region.x : region.x2]
        shape_similarity = _shape_similarity(shape_mask)
        if shape_similarity < min_shape_similarity:
            continue
        confidence = float(np.sqrt(confidence * shape_similarity))
        candidate = (confidence, region, local_mask, shape_similarity)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _candidate_confidence(
    component_count: int,
    width_fraction: float,
    height_fraction: float,
    density: float,
) -> float:
    count_score = min(component_count / 6, 1.0)
    if width_fraction >= 0.40:
        count_score = max(count_score, 0.65)
    width_score = _ramp(width_fraction, 0.20, 0.50) * _falloff(
        width_fraction, 0.72, 0.84
    )
    height_score = _ramp(height_fraction, 0.08, 0.20) * _falloff(
        height_fraction, 0.45, 0.65
    )
    density_score = _ramp(density, 0.04, 0.16) * _falloff(density, 0.60, 0.80)
    return float(count_score * width_score * height_score * density_score)


def _ramp(value: float, start: float, end: float) -> float:
    return max(0.0, min(1.0, (value - start) / (end - start)))


def _falloff(value: float, start: float, end: float) -> float:
    return max(0.0, min(1.0, (end - value) / (end - start)))


def _overlap_ratio(left: Region, right: Region) -> float:
    overlap_width = max(0, min(left.x2, right.x2) - max(left.x, right.x))
    overlap_height = max(0, min(left.y2, right.y2) - max(left.y, right.y))
    overlap = overlap_width * overlap_height
    if overlap == 0:
        return 0.0
    return overlap / min(left.width * left.height, right.width * right.height)


def _shape_similarity(mask: np.ndarray) -> float:
    normalized = _normalize_shape(mask)
    return max(_dice_similarity(normalized, template) for template in _WORDMARK_TEMPLATES)


def _normalize_shape(mask: np.ndarray) -> np.ndarray:
    y_indices, x_indices = np.where(mask > 0)
    if not len(x_indices):
        return np.zeros((32, 128), dtype=bool)
    glyphs = mask[
        y_indices.min() : y_indices.max() + 1,
        x_indices.min() : x_indices.max() + 1,
    ]
    target_height, target_width = 32, 128
    scale = min(
        (target_width - 4) / glyphs.shape[1],
        (target_height - 4) / glyphs.shape[0],
    )
    resized_width = max(1, round(glyphs.shape[1] * scale))
    resized_height = max(1, round(glyphs.shape[0] * scale))
    resized = cv2.resize(
        glyphs,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    ) > 64
    normalized = np.zeros((target_height, target_width), dtype=bool)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    normalized[y : y + resized_height, x : x + resized_width] = resized
    return normalized


def _dice_similarity(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    return float(2 * intersection / max(1, left.sum() + right.sum()))


_WORDMARK_TEMPLATES = (
    _DOLA_TEMPLATE > 0,
    _wordmark_template("seeora.app"),
)
