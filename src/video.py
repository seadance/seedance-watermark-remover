from __future__ import annotations

import shutil
import subprocess
import tempfile
from bisect import bisect_left
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from dynamic import DynamicDetection, detect_edge_watermarks
from models import Region, VideoInfo


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found on PATH")


def open_video(input_path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {input_path}")

    info = VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.width <= 0 or info.height <= 0 or info.fps <= 0 or info.frame_count <= 0:
        capture.release()
        raise ValueError("video metadata is incomplete or invalid")
    return capture, info


def process_video(
    input_path: Path,
    output_path: Path,
    region: Region,
    local_mask: np.ndarray,
    *,
    radius: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    require_ffmpeg()
    capture, info = open_video(input_path)
    region.validate(info.width, info.height)
    if local_mask.shape != (region.height, region.width):
        capture.release()
        raise ValueError("mask dimensions do not match the selected region")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".seedance-wm-", dir=output_path.parent
    ) as temp_dir:
        silent_path = Path(temp_dir) / "silent.mp4"
        completed_path = Path(temp_dir) / "completed.mp4"
        encoder = subprocess.Popen(
            _encode_command(info, silent_path), stdin=subprocess.PIPE
        )
        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                _inpaint_with_context(frame, region, local_mask, radius)
                if encoder.stdin is None:
                    raise RuntimeError("FFmpeg encoder stdin is unavailable")
                encoder.stdin.write(frame.tobytes())
                processed += 1
                if progress is not None:
                    progress(processed, info.frame_count)
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg stopped while encoding video frames") from exc
        finally:
            capture.release()
            if encoder.stdin is not None:
                encoder.stdin.close()

        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the repaired video")
        if processed == 0:
            raise RuntimeError("no video frames were processed")

        _restore_audio(silent_path, input_path, completed_path)
        completed_path.replace(output_path)


def process_video_dynamic(
    input_path: Path,
    output_path: Path,
    *,
    radius: int = 4,
    min_confidence: float = 0.45,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Detect and repair a moving pale edge watermark on each video frame."""
    require_ffmpeg()
    capture, info = open_video(input_path)
    timeline = _build_dynamic_timeline(capture, info, min_confidence)
    capture, _ = open_video(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".seedance-wm-", dir=output_path.parent
    ) as temp_dir:
        silent_path = Path(temp_dir) / "silent.mp4"
        completed_path = Path(temp_dir) / "completed.mp4"
        encoder = subprocess.Popen(
            _encode_command(info, silent_path), stdin=subprocess.PIPE
        )
        processed = 0
        repaired_frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = timeline[processed] if processed < len(timeline) else []
                if detections:
                    repaired_frames += 1
                for detection in detections:
                    _inpaint_with_context(
                        frame, detection.region, detection.mask, radius
                    )
                if encoder.stdin is None:
                    raise RuntimeError("FFmpeg encoder stdin is unavailable")
                encoder.stdin.write(frame.tobytes())
                processed += 1
                if progress is not None:
                    progress(processed, info.frame_count)
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg stopped while encoding video frames") from exc
        finally:
            capture.release()
            if encoder.stdin is not None:
                encoder.stdin.close()

        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the repaired video")
        if processed == 0:
            raise RuntimeError("no video frames were processed")
        _restore_audio(silent_path, input_path, completed_path)
        completed_path.replace(output_path)
    return repaired_frames, processed


def _build_dynamic_timeline(
    capture: cv2.VideoCapture,
    info: VideoInfo,
    min_confidence: float,
) -> list[list[DynamicDetection]]:
    """Use strong matches as temporal anchors for faint and transitional frames."""
    candidates: list[list[DynamicDetection]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        candidates.append(
            detect_edge_watermarks(
                frame,
                min_confidence=min_confidence,
                min_shape_similarity=0.30,
            )
        )
    capture.release()

    strong_by_anchor: dict[str, list[tuple[int, DynamicDetection]]] = {}
    all_strong: list[tuple[int, DynamicDetection]] = []
    for frame_index, detections in enumerate(candidates):
        for detection in detections:
            if detection.shape_similarity < 0.55:
                continue
            strong_by_anchor.setdefault(detection.anchor, []).append(
                (frame_index, detection)
            )
            all_strong.append((frame_index, detection))

    all_strong.sort(key=lambda item: item[0])
    canonical = {
        anchor: _solid_detection(
            max(items, key=lambda item: item[1].confidence)[1]
        )
        for anchor, items in strong_by_anchor.items()
    }
    continuity = max(3, round(info.fps * 1.25))
    timeline: list[list[DynamicDetection]] = []
    for frame_index, detections in enumerate(candidates):
        accepted = [
            _edge_transition_detection(_solid_detection(detection), info)
            for detection in detections
            if _nearest_detection(
                strong_by_anchor.get(detection.anchor, []), frame_index, continuity
            )
            is not None
        ]
        if not accepted:
            nearest = _next_or_previous_detection(
                all_strong, frame_index, continuity
            )
            if nearest is not None:
                accepted = [
                    _edge_transition_detection(canonical[nearest.anchor], info)
                ]
        timeline.append(accepted)
    return timeline


def _solid_detection(detection: DynamicDetection) -> DynamicDetection:
    """Fill the learned wordmark bounds so faint strokes and fades are covered."""
    y_indices, x_indices = np.where(detection.mask > 0)
    if not len(x_indices):
        return detection
    mask = np.zeros_like(detection.mask)
    pad = 4
    x1 = max(0, int(x_indices.min()) - pad)
    y1 = max(0, int(y_indices.min()) - pad)
    x2 = min(mask.shape[1], int(x_indices.max()) + pad + 1)
    y2 = min(mask.shape[0], int(y_indices.max()) + pad + 1)
    mask[y1:y2, x1:x2] = 255
    return DynamicDetection(
        region=detection.region,
        mask=mask,
        confidence=detection.confidence,
        shape_similarity=detection.shape_similarity,
        anchor=detection.anchor,
    )


def _edge_transition_detection(
    detection: DynamicDetection, info: VideoInfo
) -> DynamicDetection:
    """Cover the short edge-to-wordmark animation when glyphs cannot be read."""
    region = detection.region
    horizontal_pad = max(8, round(info.width * 0.02))
    if detection.anchor.endswith("left"):
        region = Region(
            0,
            region.y,
            min(info.width, region.x2 + horizontal_pad),
            region.height,
        )
    elif detection.anchor.endswith("right"):
        x = max(0, region.x - horizontal_pad)
        region = Region(
            x,
            region.y,
            info.width - x,
            region.height,
        )
    return DynamicDetection(
        region=region,
        mask=np.full((region.height, region.width), 255, dtype=np.uint8),
        confidence=detection.confidence,
        shape_similarity=detection.shape_similarity,
        anchor=detection.anchor,
    )


def _nearest_detection(
    detections: list[tuple[int, DynamicDetection]],
    frame_index: int,
    max_distance: int,
) -> DynamicDetection | None:
    if not detections:
        return None
    insertion = bisect_left(detections, frame_index, key=lambda item: item[0])
    nearby = detections[max(0, insertion - 1) : insertion + 1]
    nearest_index, nearest = min(
        nearby, key=lambda item: (abs(item[0] - frame_index), -item[0])
    )
    if abs(nearest_index - frame_index) > max_distance:
        return None
    return nearest


def _next_or_previous_detection(
    detections: list[tuple[int, DynamicDetection]],
    frame_index: int,
    max_distance: int,
) -> DynamicDetection | None:
    """Prefer the incoming anchor while a moving watermark changes position."""
    if not detections:
        return None
    insertion = bisect_left(detections, frame_index, key=lambda item: item[0])
    if insertion < len(detections):
        next_index, next_detection = detections[insertion]
        if next_index - frame_index <= max_distance:
            return next_detection
    if insertion:
        previous_index, previous_detection = detections[insertion - 1]
        if frame_index - previous_index <= max_distance:
            return previous_detection
    return None


def _inpaint_with_context(
    frame: np.ndarray,
    region: Region,
    local_mask: np.ndarray,
    radius: int,
) -> None:
    """Inpaint a mask while retaining known pixels around the selected region."""
    frame_height, frame_width = frame.shape[:2]
    pad = max(12, radius * 3)
    x1 = max(0, region.x - pad)
    y1 = max(0, region.y - pad)
    x2 = min(frame_width, region.x2 + pad)
    y2 = min(frame_height, region.y2 + pad)
    expanded = frame[y1:y2, x1:x2].copy()
    expanded_mask = np.zeros(expanded.shape[:2], dtype=np.uint8)
    offset_x = region.x - x1
    offset_y = region.y - y1
    expanded_mask[
        offset_y : offset_y + region.height,
        offset_x : offset_x + region.width,
    ] = local_mask
    if np.any(expanded_mask):
        repaired = cv2.inpaint(expanded, expanded_mask, radius, cv2.INPAINT_TELEA)
        frame[y1:y2, x1:x2] = repaired


def _encode_command(info: VideoInfo, silent_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.8f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(silent_path),
    ]


def _restore_audio(silent_path: Path, input_path: Path, completed_path: Path) -> None:
    mux_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        "-y",
        str(completed_path),
    ]
    completed = subprocess.run(mux_command, check=False)
    if completed.returncode != 0:
        mux_command[mux_command.index("copy", mux_command.index("-c:a"))] = "aac"
        completed = subprocess.run(mux_command, check=False)
    if completed.returncode != 0:
        raise RuntimeError("FFmpeg failed to restore the original audio")
