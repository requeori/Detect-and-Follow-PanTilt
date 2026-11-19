"""Graba clip para dataset en vídeo MJPG con metadatos.

Flujo: vista previa -> ESPACIO para empezar -> graba N segundos -> guarda clip y JSON.
ESC - cancela.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import objecttracker
from objecttracker.capture import CameraSource, CaptureError
from objecttracker.config import load_camera_profile


def make_clip_id(scenario: str, name: str, split: str, when: datetime) -> str:
    return f"{scenario}_{name}_{split}_{when:%Y%m%d-%H%M}"


def fps_from_timestamps(timestamps_ns: list[int]) -> float | None:
    """fps medio real a partir de las marcas temporales de los fotogramas."""
    if len(timestamps_ns) < 2:
        return None
    return round((len(timestamps_ns) - 1) / ((timestamps_ns[-1] - timestamps_ns[0]) / 1e9), 2)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Escritor de vídeo MJPG (decisión D-020: los demás códecs no aguantan 30 fps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), size
    )
    if not writer.isOpened():
        raise CaptureError(f"No se pudo crear el vídeo {path}")
    return writer


def build_metadata(
    *,
    clip_id: str,
    args: argparse.Namespace,
    when: datetime,
    mode_name: str,
    mode,
    timestamps_ns: list[int],
    failed_reads: int,
) -> dict:
    """Metadatos de un clip (ver docs/protocolo_grabacion.md §3)."""
    return {
        "clip_id": clip_id,
        "scenario": args.scenario,
        "split": args.split,
        "date": when.isoformat(timespec="seconds"),
        "camera": {
            "profile": Path(args.profile).stem,
            "mode": mode_name,
            "backend": mode.backend,
            "width": mode.width,
            "height": mode.height,
            "fps_measured": fps_from_timestamps(timestamps_ns),
        },
        "ball": {"present": args.ball, "distance_m": args.distance_m},
        "recording": {
            "file": f"{clip_id}.avi",
            "codec": "MJPG",
            "frames": len(timestamps_ns),
            "failed_reads": failed_reads,
        },
        "software": {"objecttracker": objecttracker.__version__, "opencv": cv2.__version__},
        "notes": args.notes,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenario", required=True, help="S0, S1, ...")
    p.add_argument("--name", required=True, help="descripción corta sin espacios")
    p.add_argument("--split", required=True, choices=("A", "V"), help="ajuste o validación")
    p.add_argument("--ball", required=True, choices=("always", "never", "intermittent"))
    p.add_argument("--distance-m", type=float, default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--profile", default="configs/cameras/aukey_pc_lm1e.yaml")
    p.add_argument("--mode", default=None, help="modo del perfil (por defecto, el default_mode)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--out", default="data/clips")
    return p


def wait_for_start(cam: CameraSource, clip_id: str) -> bool:
    """Vista previa hasta ESPACIO (True) o ESC (False)."""
    while True:
        frame = cam.read()
        if frame is not None:
            cv2.putText(
                frame.image,
                f"{clip_id} - ESPACIO: grabar, ESC: salir",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("record_clip", frame.image)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return True
        if key == 27:
            return False


def main() -> int:
    args = build_parser().parse_args()
    profile = load_camera_profile(args.profile)
    mode = profile.mode(args.mode)
    when = datetime.now()
    clip_id = make_clip_id(args.scenario, args.name, args.split, when)
    video_path = Path(args.out) / f"{clip_id}.avi"
    timestamps: list[int] = []
    failed = 0
    with CameraSource(profile, args.mode) as cam:
        if not wait_for_start(cam, clip_id):
            cv2.destroyAllWindows()
            print("Cancelado.")
            return 1
        writer = open_writer(video_path, mode.fps, (mode.width, mode.height))
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            frame = cam.read()
            if frame is None:
                failed += 1
                continue
            writer.write(frame.image)
            timestamps.append(frame.timestamp_ns)
            cv2.imshow("record_clip", frame.image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        writer.release()
    cv2.destroyAllWindows()
    meta = build_metadata(
        clip_id=clip_id,
        args=args,
        when=when,
        mode_name=args.mode or profile.default_mode,
        mode=mode,
        timestamps_ns=timestamps,
        failed_reads=failed,
    )
    json_path = video_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Clip: {video_path}  ({len(timestamps)} fotogramas, "
        f"{meta['camera']['fps_measured']} fps medidos, {failed} lecturas fallidas)"
    )
    print(f"Metadatos: {json_path}")
    return 0


def run() -> int:
    try:
        return main()
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
