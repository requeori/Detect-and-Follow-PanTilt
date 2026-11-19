"""Guarda una imagen o un clip anotado con la mascara y la deteccion, para ver por que falla.

Uso:
    python scripts/preview.py ENTRADA [--config params.yaml] [--out salida] [--every N]
                              [--dump-misses DIR [--max-dumps N]]
ENTRADA: imagen (png/jpg) o el JSON de un clip. Salida por defecto: <entrada>_preview.png / .avi.
--dump-misses: (solo clips) guarda en DIR un PNG por fotograma sin deteccion (original | anotado),
repartidos a lo largo del clip, e imprime por que se rechaza cada candidato de la mascara.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from objecttracker.clips import ClipMeta, VideoFileSource
from objecttracker.color_detector import ColorParams, make_red_ball
from objecttracker.detector import Detection

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def annotate(
    image: np.ndarray, mask: np.ndarray, detection: Detection, index: int | None = None
) -> np.ndarray:
    """Mascara en verde (40 %), caja y centro de la deteccion, y el estado en texto.

    Con `index` escribe el numero de fotograma (sirve para anotar ball_visible en el JSON).
    """
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
    if index is not None:
        text = f"#{index} {text}"
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def diagnose(mask: np.ndarray, params: ColorParams) -> list[str]:
    """Una linea por candidato de la mascara con el filtro que lo rechaza (solo depuracion).

    Repite los filtros de pick_ball; si se cambian alli hay que cambiarlos aqui.
    """
    height = mask.shape[0]
    min_d, max_d = params.min_diameter_rel * height, params.max_diameter_rel * height
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if area <= 0 or radius <= 0:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        diameter = 2 * radius
        fill = area / (math.pi * radius * radius)
        aspect = max(w, h) / min(w, h)
        reasons = []
        if diameter < min_d:
            reasons.append(f"diametro<{min_d:.0f}")
        if diameter > max_d:
            reasons.append(f"diametro>{max_d:.0f}")
        if fill < params.min_fill:
            reasons.append(f"relleno<{params.min_fill}")
        if aspect > params.max_aspect:
            reasons.append(f"aspecto>{params.max_aspect}")
        lines.append(
            f"    area={area:.0f} d={diameter:.0f}px relleno={fill:.2f} aspecto={aspect:.2f} -> "
            + (", ".join(reasons) or "pasa los filtros")
        )
    return lines or ["    mascara vacia (ningun pixel supera los umbrales de color)"]


def _side_by_side(raw: np.ndarray, annotated: np.ndarray, max_height: int = 480) -> np.ndarray:
    pair = np.hstack([raw, annotated])
    if pair.shape[0] > max_height:
        scale = max_height / pair.shape[0]
        pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return pair


def dump_misses(
    meta: ClipMeta, detector, out_dir: Path, max_dumps: int, every: int = 1
) -> tuple[int, int]:
    """Guarda hasta `max_dumps` fotogramas sin deteccion, repartidos por todo el clip.

    Dos pasadas para no guardar imagenes en RAM: la primera solo apunta los indices fallidos.
    Devuelve (fotogramas sin deteccion, fotogramas procesados).
    """
    missed: list[int] = []
    total = 0
    for frame in VideoFileSource(meta):
        if frame.index % every:
            continue
        total += 1
        if not detector.detect(frame.image).found:
            missed.append(frame.index)
    if not missed:
        return 0, total

    step = max(1, len(missed) // max_dumps)
    chosen = set(missed[::step][:max_dumps])
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in VideoFileSource(meta):
        if frame.index not in chosen:
            continue
        mask = detector.mask(frame.image)
        annotated = annotate(frame.image, mask, Detection.none(), frame.index)
        path = out_dir / f"{meta.clip_id}_miss_{frame.index:04d}.png"
        cv2.imwrite(str(path), _side_by_side(frame.image, annotated))
        print(f"  {path.name}")
        print("\n".join(diagnose(mask, detector.params)))
    return len(missed), total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--every", type=int, default=1, help="procesar 1 de cada N fotogramas")
    parser.add_argument(
        "--dump-misses", type=Path, help="carpeta donde guardar fotogramas fallidos"
    )
    parser.add_argument("--max-dumps", type=int, default=6, help="maximo de PNG con --dump-misses")
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
    if args.dump_misses:
        missed, total = dump_misses(meta, detector, args.dump_misses, args.max_dumps, args.every)
        print(f"{meta.clip_id}: {missed}/{total} fotogramas sin deteccion")
        return 0
    out = args.out or args.input.with_name(args.input.stem + "_preview.avi")
    writer = None
    found = total = 0
    for frame in VideoFileSource(meta):
        if frame.index % args.every:
            continue
        detection = detector.detect(frame.image)
        found += detection.found
        total += 1
        annotated = annotate(frame.image, detector.mask(frame.image), detection, frame.index)
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
