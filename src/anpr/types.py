"""Shared records for frames and plate results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    stream_name: str
    data: object
    monotonic_ns: int
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class PlateBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class PlateText:
    text: str
    confidence: float
