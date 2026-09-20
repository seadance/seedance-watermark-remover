from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Corner(str, Enum):
    AUTO = "auto"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def validate(self, frame_width: int, frame_height: int) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("region must contain non-negative coordinates and positive dimensions")
        if self.x2 > frame_width or self.y2 > frame_height:
            raise ValueError(
                f"region {self} exceeds the {frame_width}x{frame_height} video frame"
            )


@dataclass(frozen=True)
class Detection:
    region: Region
    corner: Corner
    confidence: float
    runner_up_confidence: float
    mask: object

    @property
    def separation(self) -> float:
        if self.runner_up_confidence <= 0:
            return float("inf")
        return self.confidence / self.runner_up_confidence


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

