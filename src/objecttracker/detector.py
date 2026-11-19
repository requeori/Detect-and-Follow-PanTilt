"""Contrato entre detectores y evaluador"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Resultado de detectar la pelota en un fotograma.
        Center y bbox están en píxeles de la imagen recibida."""

    found: bool
    center: tuple[float, float] | None = None
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.found and (self.center is None or self.bbox is None):
            raise ValueError("Una detección con found=True necesita center y bbox")

    @classmethod
    def none(cls) -> Detection:
        return cls(found=False)


@runtime_checkable
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> Detection: ...
