from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraMode:
    backend: str  # msmf | dshow | v4l2 | any
    width: int
    height: int
    fps: float
    focal_px: float | None = None  # focal en píxeles


@dataclass(frozen=True)
class CameraProfile:
    name: str
    device_index: int
    default_mode: str
    modes: dict[str, CameraMode]
    flip_vertical: bool = False
    flip_horizontal: bool = False

    def mode(self, name: str | None = None) -> CameraMode:
        """Modo pedido, o el modo por defecto si no se indica."""
        return self.modes[name or self.default_mode]


def load_camera_profile(path: str | Path) -> CameraProfile:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    modes = {key: CameraMode(**value) for key, value in data.pop("modes").items()}
    return CameraProfile(modes=modes, **data)
