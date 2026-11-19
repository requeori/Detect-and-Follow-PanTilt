"""Tipos de datos compartidos entre módulos."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, uint8
    timestamp_ns: int  # reloj monótono en el instante en que devolvió el fotograma
    index: int  # nº de fotograma desde que se abrió la fuente
