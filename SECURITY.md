# Security Policy

Please report security issues privately through GitHub's security advisory
feature instead of opening a public issue.

The tool invokes the local FFmpeg executable without a shell. Inputs are passed
as argument arrays, and output paths are validated before processing.

