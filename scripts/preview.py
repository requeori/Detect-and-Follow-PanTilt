"""Guarda una imagen o un clip anotado con la mascara y la deteccion, para ver por que falla.

Uso:
    python scripts/preview.py ENTRADA [--config params.yaml] [--out salida] [--every N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from objecttracker.clips import ClipMeta, VideoFileSource
from objecttracker.color_detector import make_red_ball
from objecttracker.detector import Detection

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def annotate(image: np.ndarray, mask: np.ndarray, detection: Detection) -> np.ndarray:
    """Mascara en verde (40 %), caja y centro de la deteccion, y el estado en texto."""
    out = image.copy()
    out[mask > 0] = (0.6 * out[mask > 0] + 0.4 * np.array((0, 255, 0))).astype(np.uint8)
    if detection.found:
        x, y, w, h = detection.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 128, 0), 2)
        cx, cy = (round(c) for c in detection.center)
        cv2.drawMarker(out, (cx, cy), (255, 128, 0), cv2.MARKER_CROSS, 12, 2)
        text = f"found score={detection.score:.2f} bbox_w={w}"
    else:
        text = "NO DETECTION"
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--every", type=int, default=1, help="procesar 1 de cada N fotogramas")
    args = parser.parse_args(argv)
    detector = make_red_ball(args.config)

    if args.input.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(args.input))
        if image is None:
            print(f"No se pudo leer {args.input}")
            return 1
        out = args.out or args.input.with_name(args.input.stem + "_preview.png")
        cv2.imwrite(str(out), annotate(image, detector.mask(image), detector.detect(image)))
        print(f"Escrito {out}")
        return 0

    meta = ClipMeta.from_json(args.input)
    out = args.out or args.input.with_name(args.input.stem + "_preview.avi")
    writer = None
    found = total = 0
    for frame in VideoFileSource(meta):
        if frame.index % args.every:
            continue
        detection = detector.detect(frame.image)
        found += detection.found
        total += 1
        annotated = annotate(frame.image, detector.mask(frame.image), detection)
        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*"MJPG"), meta.fps / args.every, (w, h)
            )
        writer.write(annotated)
    if writer is not None:
        writer.release()
    print(f"Escrito {out}: {found}/{total} fotogramas con deteccion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
