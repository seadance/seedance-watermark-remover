# Seedance Watermark Remover

An open-source **Seedance watermark remover** for cleaning small, visible corner
watermarks from videos locally. It is designed for creators who need to remove
a Seedance watermark from video they own or are authorized to edit.

The tool runs on CPU, keeps video processing on your machine, preserves the
original audio stream, and provides a manual selection fallback when automatic
detection is not confident enough.

> [!IMPORTANT]
> Only remove watermarks from content you own or have permission to modify.
> This project removes visible pixels; it does not attempt to remove provenance,
> invisible watermarks, Content Credentials, or C2PA metadata.

## Features

- Automatically searches all four corners for a small, static watermark.
- Refuses low-confidence detections instead of modifying the wrong region.
- Accepts a manual `x,y,width,height` region as a reliable fallback.
- Uses a sparse text mask to preserve more of the original frame.
- Processes frames as a stream instead of exporting thousands of PNG files.
- Copies the original audio into the cleaned MP4.
- Runs locally with OpenCV and FFmpeg; no cloud API or GPU is required.
- Supports landscape and portrait video.

## Scope of the MVP

This release targets a fixed `AI-generated` style badge in a video corner. It
works best when the watermark is small and the surrounding background is flat
or softly textured. It is not intended for moving watermarks, center-screen
logos, large overlays, or marks covering faces and detailed objects.

## Installation

Requirements:

- Python 3.10 or newer
- FFmpeg available on `PATH`

Install from source:

```bash
git clone https://github.com/seadance/seedance-watermark-remover.git
cd seedance-watermark-remover
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Usage

Try automatic detection:

```bash
seedance-watermark-remover input.mp4 -o clean.mp4
```

Automatic mode writes a detection preview before processing. If confidence is
too low, no output video is created and the command asks for a manual region.

Specify the watermark region manually:

```bash
seedance-watermark-remover input.mp4 -o clean.mp4 \
  --region 18,12,150,58
```

Create only a preview and inspect the mask:

```bash
seedance-watermark-remover input.mp4 --preview-only \
  --preview detection-preview.jpg
```

Useful options:

```text
--region X,Y,W,H       Use an exact watermark rectangle
--corner POSITION      Restrict detection to one corner
--threshold FLOAT      Minimum automatic-detection confidence
--mask text|rectangle  Sparse text mask or full rectangle
--radius INTEGER       OpenCV inpainting radius
--preview PATH         Detection preview image path
--overwrite            Replace an existing output file
```

Run `seedance-watermark-remover --help` for the complete CLI reference.

## Remove a Seedance 2.5 Watermark

To **remove a Seedance 2.5 watermark**, first try automatic detection. If the
badge is inset farther from the corner, changes position, or the preview covers
real scene details, pass a manual region. Manual confirmation is deliberately
preferred over silently damaging the video.

The current release uses spatial inpainting independently on each frame. Very
fast motion beneath a watermark can therefore produce visible texture changes.
A temporally consistent restoration engine is planned after a representative
set of real-world samples is available.

## How It Works

1. Sample a bounded number of frames without loading the full video into RAM.
2. Inspect corner regions for stable, text-like edges.
3. Require both an absolute confidence threshold and separation from the
   second-best corner.
4. Build a mask from stable edges and morphological components.
5. Inpaint only the masked pixels with OpenCV Telea.
6. Stream repaired frames to FFmpeg and remux the original audio.

See [docs/algorithm.md](docs/algorithm.md) for implementation details and known
limitations.

## Development

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

## Privacy

The CLI does not upload videos or call an external service. Temporary encoded
video data is created in the operating system's temporary directory and removed
after a successful or failed run.

## License

MIT. See [LICENSE](LICENSE).

