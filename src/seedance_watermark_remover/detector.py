from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .models import Corner, Detection, Region, VideoInfo

_CORNER_ORDER = (
    Corner.TOP_LEFT,
    Corner.TOP_RIGHT,
    Corner.BOTTOM_LEFT,
    Corner.BOTTOM_RIGHT,
)


def corner_region(corner: Corner, width: int, height: int) -> Region:
    crop_width = min(width, max(160, round(width * 0.24)))
    crop_height = min(height, max(80, round(height * 0.18)))
    positions = {
        Corner.TOP_LEFT: (0, 0),
        Corner.TOP_RIGHT: (width - crop_width, 0),
        Corner.BOTTOM_LEFT: (0, height - crop_height),
        Corner.BOTTOM_RIGHT: (width - crop_width, height - crop_height),
    }
    if corner not in positions:
        raise ValueError("corner_region requires a concrete corner")
    x, y = positions[corner]
    return Region(x, y, crop_width, crop_height)


def sample_corner_frames(
    capture: cv2.VideoCapture,
    info: VideoInfo,
    *,
    sample_count: int = 32,
) -> dict[Corner, list[np.ndarray]]:
    if info.frame_count <= 0:
        raise ValueError("video has no readable frames")

    count = min(sample_count, info.frame_count)
    indices = np.linspace(0, info.frame_count - 1, num=count, dtype=int)
    regions = {corner: corner_region(corner, info.width, info.height) for corner in _CORNER_ORDER}
    samples: dict[Corner, list[np.ndarray]] = {corner: [] for corner in _CORNER_ORDER}

    for frame_index in np.unique(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        for corner, region in regions.items():
            samples[corner].append(frame[region.y : region.y2, region.x : region.x2].copy())

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not any(samples.values()):
        raise ValueError("could not sample frames from the video")
    return samples


def detect_watermark(
    samples: dict[Corner, list[np.ndarray]],
    frame_width: int,
    frame_height: int,
    *,
    allowed_corners: Iterable[Corner] | None = None,
    min_confidence: float = 0.18,
    min_separation: float = 1.15,
) -> Detection | None:
    corners = tuple(allowed_corners or _CORNER_ORDER)
    scored: list[tuple[float, Corner, Region, np.ndarray]] = []

    for corner in corners:
        corner_samples = samples.get(corner, [])
        if len(corner_samples) < 2:
            continue
        score, local_region, local_mask = _score_corner(corner_samples)
        outer = corner_region(corner, frame_width, frame_height)
        region = Region(
            outer.x + local_region.x,
            outer.y + local_region.y,
            local_region.width,
            local_region.height,
        )
        scored.append((score, corner, region, local_mask))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    confidence, corner, region, mask = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    separation = confidence / max(runner_up, 1e-6)
    if confidence < min_confidence or (len(scored) > 1 and separation < min_separation):
        return None

    return Detection(region, corner, confidence, runner_up, mask)


def mask_for_crops(
    crops: list[np.ndarray],
    *,
    rectangle: bool = False,
) -> np.ndarray:
    if not crops:
        raise ValueError("at least one crop is required to build a mask")
    height, width = crops[0].shape[:2]
    if rectangle:
        return np.full((height, width), 255, dtype=np.uint8)

    _, _, mask = _build_text_mask(crops)
    return mask


def _score_corner(frames: list[np.ndarray]) -> tuple[float, Region, np.ndarray]:
    score_map, median_bgr, text_mask = _build_text_mask(frames)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask)
    best_component: tuple[float, Region] | None = None
    crop_height, crop_width = text_mask.shape

    for component_id in range(1, component_count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 18 or width < 5 or height < 4:
            continue
        if width > crop_width * 0.9 or height > crop_height * 0.8:
            continue

        component_pixels = labels == component_id
        mean_signal = float(score_map[component_pixels].mean())
        size_factor = min(1.0, area / max(40.0, crop_width * crop_height * 0.012))
        aspect_factor = 1.0 if width >= height else 0.75
        component_score = mean_signal * (0.55 + 0.45 * size_factor) * aspect_factor
        pad = max(4, round(min(crop_width, crop_height) * 0.02))
        region = Region(
            max(0, x - pad),
            max(0, y - pad),
            min(crop_width - max(0, x - pad), width + 2 * pad),
            min(crop_height - max(0, y - pad), height + 2 * pad),
        )
        if best_component is None or component_score > best_component[0]:
            best_component = (component_score, region)

    if best_component is None:
        empty_region = Region(0, 0, crop_width, crop_height)
        return 0.0, empty_region, np.zeros_like(text_mask)

    confidence, region = best_component
    local_mask = text_mask[region.y : region.y2, region.x : region.x2]
    # Slightly reward pale overlay pixels without making white scene objects sufficient.
    hsv = cv2.cvtColor(
        median_bgr[region.y : region.y2, region.x : region.x2], cv2.COLOR_BGR2HSV
    )
    pale_fraction = float(((hsv[..., 1] < 120) & (hsv[..., 2] > 135)).mean())
    confidence = min(1.0, confidence * (0.85 + min(pale_fraction, 0.5)))
    return confidence, region, local_mask


def _build_text_mask(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(frames, axis=0).astype(np.float32)
    median_bgr = np.median(stack, axis=0).astype(np.uint8)
    gray_stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames], axis=0
    ).astype(np.float32)
    temporal_std = np.std(gray_stack, axis=0)
    median_gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(median_gray, 35, 100)
    stable_limit = min(16.0, max(3.0, float(np.percentile(temporal_std, 45))))
    stability = np.clip(1.0 - temporal_std / max(stable_limit * 2.0, 1.0), 0.0, 1.0)
    edge_signal = edges.astype(np.float32) / 255.0
    score_map = edge_signal * stability

    stable_edges = np.where((edge_signal > 0) & (stability > 0.45), 255, 0).astype(np.uint8)
    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    text_mask = cv2.morphologyEx(stable_edges, cv2.MORPH_CLOSE, join_kernel, iterations=1)
    text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask)
    filtered = np.zeros_like(text_mask)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if area >= 14 and width >= 4 and height >= 3:
            filtered[labels == component_id] = 255

    return score_map, median_bgr, filtered
