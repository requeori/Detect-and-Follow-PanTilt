"""Lectura de clips grabados (vídeo + JSON) como fuente de fotogramas."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2

from objecttracker.types import Frame


class ClipError(RuntimeError):
    """El clip o su JSON no se pueden leer o son incoherentes."""


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    scenario: str
    split: str
    ball_present: str  # "always" o "never"
    fps: float
    expected_frames: int
    video_path: Path
    json_path: Path

    @classmethod
    def from_json(cls, json_path: str | Path) -> ClipMeta:
        json_path = Path(json_path)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            recording = data["recording"]
            return cls(
                clip_id=data["clip_id"],
                scenario=data["scenario"],
                split=data["split"],
                ball_present=data["ball"]["present"],
                fps=float(data["camera"]["fps_measured"]),
                expected_frames=int(recording["frames"]),
                video_path=json_path.with_name(recording["file"]),
                json_path=json_path,
            )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ClipError(f"JSON de clip inválido {json_path}: {exc!r}") from exc


class VideoFileSource:
    """Itera los fotogramas de un clip como `Frame`"""

    def __init__(self, meta: ClipMeta) -> None:
        self._meta = meta

    def __iter__(self) -> Iterator[Frame]:
        path = self._meta.video_path
        if not path.exists():
            raise ClipError(f"No existe el vídeo {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ClipError(f"OpenCV no pudo abrir {path}")
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(
                    image=image, timestamp_ns=round(index / self._meta.fps * 1e9), index=index
                )
                index += 1
        finally:
            cap.release()


def find_clips(root: str | Path) -> list[ClipMeta]:
    """Todos los clips bajo root, ordenados por nombre."""
    clips = []
    for json_path in sorted(Path(root).rglob("*.json")):
        try:
            clips.append(ClipMeta.from_json(json_path))
        except ClipError:
            continue  # otros JSON (resultados, config) no son clips
    return clips
