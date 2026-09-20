import cv2
import numpy as np

from detector import detect_watermark
from dynamic import detect_edge_watermarks
from models import Corner


def _corner_samples(with_watermark: bool) -> dict[Corner, list[np.ndarray]]:
    height, width = 130, 310
    samples = {corner: [] for corner in Corner if corner is not Corner.AUTO}
    for index in range(24):
        for corner in samples:
            rng = np.random.default_rng(index * 10 + list(samples).index(corner))
            image = rng.integers(30, 90, size=(height, width, 3), dtype=np.uint8)
            image = cv2.GaussianBlur(image, (15, 15), 0)
            cv2.circle(image, (70 + index * 3, 80), 22, (160, 100, 70), -1)
            if with_watermark and corner is Corner.TOP_LEFT:
                cv2.putText(
                    image,
                    "AI GENERATED",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
            samples[corner].append(image)
    return samples


def test_detects_static_corner_text() -> None:
    result = detect_watermark(
        _corner_samples(True),
        1280,
        720,
        min_confidence=0.05,
        min_separation=1.05,
    )
    assert result is not None
    assert result.corner is Corner.TOP_LEFT
    assert np.any(result.mask)


def test_rejects_scene_without_text_at_normal_threshold() -> None:
    result = detect_watermark(_corner_samples(False), 1280, 720)
    assert result is None


def test_detects_dynamic_badge_at_middle_left_edge() -> None:
    frame = np.full((720, 1280, 3), (35, 45, 60), dtype=np.uint8)
    cv2.putText(
        frame,
        "Dola AI",
        (24, 374),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (230, 230, 230),
        3,
        cv2.LINE_AA,
    )

    results = detect_edge_watermarks(frame, min_confidence=0.30)

    assert results
    assert results[0].anchor == "middle-left"
    assert np.any(results[0].mask)


def test_detects_seadance_app_badge_at_bottom_right_edge() -> None:
    frame = np.full((720, 1280, 3), (35, 45, 60), dtype=np.uint8)
    cv2.putText(
        frame,
        "seadance.app",
        (1070, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    results = detect_edge_watermarks(frame, min_confidence=0.30)

    assert results
    assert results[0].anchor == "bottom-right"
    assert np.any(results[0].mask)


def test_dynamic_detector_rejects_plain_frame() -> None:
    frame = np.full((720, 1280, 3), (35, 45, 60), dtype=np.uint8)
    assert detect_edge_watermarks(frame) == []


def test_dynamic_detector_rejects_unrelated_edge_title() -> None:
    frame = np.full((720, 1280, 3), (35, 45, 60), dtype=np.uint8)
    cv2.putText(
        frame,
        "Seedance 2.5",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (235, 235, 235),
        3,
        cv2.LINE_AA,
    )

    assert detect_edge_watermarks(frame, min_shape_similarity=0.30) == []
