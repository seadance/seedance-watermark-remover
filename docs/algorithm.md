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

## Dynamic edge detection

`--dynamic` pre-scans every frame at eight edge anchors: the four corners plus
the center of each edge. It looks for aligned components formed by bright,
low-saturation local detail and compares their silhouette with the `Dola AI`
wordmark. Strong matches become temporal anchors. Nearby faint matches and
short edge-transition frames inherit the trusted position, so a fade does not
leave half of the wordmark behind. Up to two masks may be repaired during a
cross-fade between positions.

Dynamic mode is opt-in because bright scene text or a logo near an edge can
have similar geometry. Raising `--dynamic-threshold` reduces false positives;
lowering it retains more faint transition frames.

## Frame processing

Static frames are decoded once. Dynamic mode performs a detection pre-scan and
then decodes again for repair; it retains only small masks and metadata, not
full frames. Each mask is placed inside a padded crop so Telea has known pixels
surrounding the repair area. This also makes a fully masked manual rectangle
functional. Repaired BGR frames are written to an FFmpeg stdin pipe and encoded
as H.264. A second FFmpeg pass copies the input audio stream into the final MP4.

This avoids a directory of lossless frames. Frame memory remains bounded;
dynamic timeline metadata grows modestly with video duration.

## Known limitations

- Spatial inpainting can flicker on detailed moving backgrounds.
- Static mode only inspects corners; dynamic mode inspects eight edge anchors.
- A watermark outside those areas requires `--region`.
- Static scene edges can resemble watermark text; confidence checks reduce but
  cannot eliminate this ambiguity.
- The tool cannot reconstruct details that were never visible under an opaque
  mark. It generates a plausible local fill.
- Visible removal does not imply removal of invisible provenance signals.
