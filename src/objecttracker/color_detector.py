"""Detector de color de la pelota roja: HSV + morfologia + contornos.

Dos pasos separados:
    segment_hsv(image, params) -> mascara binaria
    pick_ball(mask, params)    -> Detection
Todos los tamanos son relativos a la altura de la imagen: la misma configuracion vale a 720p y 480p.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from objecttracker.detector import Detection

DEFAULT_CONFIG = Path("configs/objects/red_ball.yaml")


@dataclass(frozen=True)
class ColorParams:
    hue_ranges: tuple[tuple[int, int], ...]
    s_min: int
    v_min: int
    open_kernel_rel: float
    close_kernel_rel: float
    min_diameter_rel: float
    max_diameter_rel: float
    min_fill: float
    max_aspect: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> ColorParams:
        """Lee los parametros con yaml.safe_load. Falta o sobra una clave -> TypeError."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        data["hue_ranges"] = tuple((int(lo), int(hi)) for lo, hi in data["hue_ranges"])
        return cls(**data)


def _kernel_px(rel: float, height: int) -> int:
    return max(1, round(rel * height))


def segment_hsv(image: np.ndarray, params: ColorParams) -> np.ndarray:
    """Mascara (uint8, 0/255) de los pixeles rojos, limpiada con apertura y cierre."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lo, hi in params.hue_ranges:
        mask |= cv2.inRange(hsv, (lo, params.s_min, params.v_min), (hi, 255, 255))
    height = image.shape[0]
    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_kernel_px(params.open_kernel_rel, height),) * 2
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_kernel_px(params.close_kernel_rel, height),) * 2
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)


def pick_ball(mask: np.ndarray, params: ColorParams) -> Detection:
    """Elige entre los contornos externos el de mayor area que parezca una esfera.

    Filtros: diametro del circulo minimo dentro del rango esperado (0,5-3 m), relleno de ese
    circulo y relacion de aspecto de la caja. score = relleno (0-1).
    """
    height = mask.shape[0]
    min_d, max_d = params.min_diameter_rel * height, params.max_diameter_rel * height
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, Detection] | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if not min_d <= 2 * radius <= max_d or area <= 0:
            continue
        fill = area / (math.pi * radius * radius)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / min(w, h)
        if fill < params.min_fill or aspect > params.max_aspect:
            continue
        m = cv2.moments(contour)
        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        if best is None or area > best[0]:
            best = (area, Detection(True, center, (x, y, w, h), float(min(fill, 1.0))))
    return best[1] if best else Detection.none()


class ColorDetector:
    """Cumple el protocolo Detector: detect(image BGR) -> Detection."""

    def __init__(self, params: ColorParams) -> None:
        self.params = params

    def detect(self, image: np.ndarray) -> Detection:
        return pick_ball(segment_hsv(image, self.params), self.params)

    def mask(self, image: np.ndarray) -> np.ndarray:
        """Mascara intermedia, para depurar (preview.py)."""
        return segment_hsv(image, self.params)


def make_red_ball(config: str | Path | None = None) -> ColorDetector:
    """Fabrica para `evaluate.py --detector objecttracker.color_detector:make_red_ball`."""
    return ColorDetector(ColorParams.from_yaml(config or DEFAULT_CONFIG))
