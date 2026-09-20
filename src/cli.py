from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from detector import detect_watermark, mask_for_crops, sample_corner_frames
from dynamic import DynamicDetection, detect_edge_watermarks
from models import Corner, Detection, Region
from preview import write_dynamic_preview, write_preview
from version import __version__
from video import open_video, process_video, process_video_dynamic


def parse_region(value: str) -> Region:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "region values must be integers: x,y,width,height"
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("region must contain four values: x,y,width,height")
    region = Region(*values)
    if region.x < 0 or region.y < 0 or region.width <= 0 or region.height <= 0:
        raise argparse.ArgumentTypeError(
            "region coordinates must be non-negative and sizes positive"
        )
    return region


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedance-watermark-remover",
        description="Remove a small visible corner watermark from a video locally.",
    )
    parser.add_argument("input", type=Path, help="input video")
    parser.add_argument("-o", "--output", type=Path, help="output MP4 path")
    parser.add_argument("--region", type=parse_region, help="manual x,y,width,height region")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="track a pale watermark that moves between edge positions",
    )
    parser.add_argument(
        "--corner",
        choices=[corner.value for corner in Corner],
        default=Corner.AUTO.value,
        help="restrict automatic detection to one corner",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="minimum automatic-detection confidence (default: 0.18)",
    )
    parser.add_argument(
        "--dynamic-threshold",
        type=float,
        default=0.45,
        help="minimum per-frame dynamic-detection confidence (default: 0.45)",
    )
    parser.add_argument(
        "--mask",
        choices=("text", "rectangle"),
        default="text",
        help="mask style (default: text)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=4,
        help="OpenCV inpainting radius in pixels (default: 4)",
    )
    parser.add_argument("--preview", type=Path, help="detection preview image path")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="write a preview without processing the video",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


def run(args: argparse.Namespace) -> None:
    input_path: Path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= args.dynamic_threshold <= 1.0:
        raise ValueError("dynamic threshold must be between 0 and 1")
    if not 1 <= args.radius <= 20:
        raise ValueError("radius must be between 1 and 20")
    if args.dynamic and args.region is not None:
        raise ValueError("--dynamic and --region cannot be used together")
    if args.dynamic and args.mask == "rectangle":
        raise ValueError("--dynamic builds its own per-frame masks; omit --mask rectangle")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}-clean.mp4")
    )
    preview_path = (
        args.preview.expanduser().resolve()
        if args.preview
        else input_path.with_name(f"{input_path.stem}-detection.jpg")
    )
    if output_path == input_path:
        raise ValueError("output must not overwrite the input video")
    if output_path.exists() and not args.overwrite and not args.preview_only:
        raise ValueError(f"output already exists; pass --overwrite: {output_path}")

    if args.dynamic:
        _run_dynamic(args, input_path, output_path, preview_path)
        return

    capture, info = open_video(input_path)
    samples = sample_corner_frames(capture, info)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("could not read the first video frame")

    detection: Detection | None = None
    if args.region is not None:
        region = args.region
        region.validate(info.width, info.height)
        manual_crops = _read_manual_crops(input_path, info.frame_count, region)
        mask = mask_for_crops(
            manual_crops,
            rectangle=args.mask == "rectangle",
        )
    else:
        selected_corner = Corner(args.corner)
        allowed = None if selected_corner is Corner.AUTO else (selected_corner,)
        detection = detect_watermark(
            samples,
            info.width,
            info.height,
            allowed_corners=allowed,
            min_confidence=args.threshold,
            min_separation=1.0 if allowed is not None else 1.15,
        )
        if detection is None:
            raise ValueError(
                "automatic detection was not confident enough; use --region x,y,width,height"
            )
        region = detection.region
        mask = (
            np.full((region.height, region.width), 255, dtype=np.uint8)
            if args.mask == "rectangle"
            else detection.mask
        )

    if not np.any(mask):
        raise ValueError("the selected region produced an empty mask; try --mask rectangle")

    write_preview(first_frame, preview_path, region, mask, detection=detection)
    print(f"Preview: {preview_path}")
    print(f"Region: {region.x},{region.y},{region.width},{region.height}")
    if detection is not None:
        print(
            f"Detection: {detection.corner.value}, confidence={detection.confidence:.3f}, "
            f"separation={detection.separation:.2f}x"
        )
    if args.preview_only:
        return

    last_percent = -1

    def report(processed: int, total: int) -> None:
        nonlocal last_percent
        percent = min(100, round(processed * 100 / max(total, 1)))
        if percent != last_percent and (percent % 5 == 0 or processed == total):
            print(f"Processing: {percent}%")
            last_percent = percent

    process_video(input_path, output_path, region, mask, radius=args.radius, progress=report)
    print(f"Output: {output_path}")


def _run_dynamic(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    preview_path: Path,
) -> None:
    preview = _find_dynamic_preview(input_path, args.dynamic_threshold)
    if preview is None:
        raise ValueError(
            "dynamic detection found no edge watermark; lower --dynamic-threshold "
            "or use --region x,y,width,height"
        )
    preview_frame, preview_detections = preview
    write_dynamic_preview(preview_frame, preview_path, preview_detections)
    print(f"Preview: {preview_path}")
    print(
        "Dynamic detection: "
        + ", ".join(
            f"{item.anchor} confidence={item.confidence:.3f}"
            for item in preview_detections
        )
    )
    if args.preview_only:
        return

    last_percent = -1

    def report(processed: int, total: int) -> None:
        nonlocal last_percent
        percent = min(100, round(processed * 100 / max(total, 1)))
        if percent != last_percent and (percent % 5 == 0 or processed == total):
            print(f"Processing: {percent}%")
            last_percent = percent

    repaired, total = process_video_dynamic(
        input_path,
        output_path,
        radius=args.radius,
        min_confidence=args.dynamic_threshold,
        progress=report,
    )
    print(f"Dynamic masks applied: {repaired}/{total} frames")
    print(f"Output: {output_path}")


def _find_dynamic_preview(
    input_path: Path,
    min_confidence: float,
) -> tuple[np.ndarray, list[DynamicDetection]] | None:
    capture, info = open_video(input_path)
    step = max(1, round(info.fps / 5))
    limit = min(info.frame_count, round(info.fps * 10))
    try:
        for frame_index in range(0, limit, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            detections = detect_edge_watermarks(
                frame, min_confidence=min_confidence
            )
            if detections:
                return frame, detections
    finally:
        capture.release()
    return None


def _read_manual_crops(
    input_path: Path,
    frame_count: int,
    region: Region,
    count: int = 24,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(input_path))
    indices = np.linspace(0, frame_count - 1, num=min(count, frame_count), dtype=int)
    crops: list[np.ndarray] = []
    for frame_index in np.unique(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            crops.append(frame[region.y : region.y2, region.x : region.x2].copy())
    capture.release()
    if not crops:
        raise ValueError("could not sample frames for the manual region")
    return crops


if __name__ == "__main__":
    main(sys.argv[1:])
