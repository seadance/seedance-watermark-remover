from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .models import Region, VideoInfo


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
        encode_command = [
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

        encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                roi = frame[region.y : region.y2, region.x : region.x2]
                repaired = cv2.inpaint(roi, local_mask, radius, cv2.INPAINT_TELEA)
                frame[region.y : region.y2, region.x : region.x2] = repaired
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
            # Some source audio codecs cannot be copied into MP4. Re-encode audio
            # to AAC while keeping the already encoded video stream unchanged.
            mux_command[mux_command.index("copy", mux_command.index("-c:a"))] = "aac"
            completed = subprocess.run(mux_command, check=False)
        if completed.returncode != 0:
            raise RuntimeError("FFmpeg failed to restore the original audio")

        completed_path.replace(output_path)
