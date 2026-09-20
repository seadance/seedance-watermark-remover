# Detection and Removal Algorithm

## Design goals

The MVP prioritizes predictable behavior over claiming universal watermark
removal. Automatic detection is allowed to fail safely. A user-provided region
is the deterministic fallback.

## Bounded sampling

The detector samples frames across the video and retains only four corner crops.
This bounds memory usage by the corner area and sample count rather than by the
video duration or full-frame resolution.

## Corner detection

For each corner, the detector combines:

- temporal stability across sampled crops;
- Canny edge density;
- low-saturation, high-value pixels commonly found in white overlay text;
- connected-component geometry; and
- separation between the best and second-best corner scores.

The detector returns no result when confidence is below the configured minimum.
This is intentional: an incorrect automatic removal is worse than asking for a
manual rectangle.

## Mask construction

Text-mask mode intersects stable pixels with expanded edges, filters tiny
components, closes gaps inside glyphs, and adds a small dilation margin for
anti-aliased edges. Rectangle mode masks the entire selected region and should
only be used when a sparse mask leaves visible remnants.

## Frame processing

Frames are decoded one at a time with OpenCV. The selected pixels are repaired
with the Telea fast-marching inpainting method. Repaired BGR frames are written
to an FFmpeg stdin pipe and encoded as H.264. A second FFmpeg pass copies the
input audio stream into the final MP4.

This avoids a directory of lossless frames and keeps memory approximately
constant as video duration increases.

## Known limitations

- Frame-independent inpainting can flicker on detailed moving backgrounds.
- A watermark outside the inspected corner area requires `--region`.
- Static scene edges can resemble watermark text; confidence checks reduce but
  cannot eliminate this ambiguity.
- The tool cannot reconstruct details that were never visible under an opaque
  mark. It generates a plausible local fill.
- Visible removal does not imply removal of invisible provenance signals.

