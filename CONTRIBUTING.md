# Contributing

Bug reports should include the operating system, Python version, FFmpeg version,
video resolution, watermark corner, and the command used. Do not upload video
you do not have permission to share.

Before opening a pull request:

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

Keep new detection behavior covered by synthetic tests. Real video fixtures must
be redistributable and should be kept small.

