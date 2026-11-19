"""Captura desde cámara según un perfil"""

from __future__ import annotations
import time
import cv2
from objecttracker.config import CameraProfile
from objecttracker.types import Frame

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "v4l2": cv2.CAP_V4L2,
            "any": cv2.CAP_ANY}

class CaptureError(RuntimeError):
    """No se pudo abrir la cámara o el driver no entregó el modo pedido."""

class CameraSource:
    """Lee fotogramas de una cámara con el modo de un perfil (backend, tamaño y fps).

    La marca temporal es la del retorno de `read()`: aproxima la llegada del fotograma, no el
    instante de captura del sensor.
    """

    def __init__(self, profile: CameraProfile, mode: str | None = None) -> None:
        self._profile = profile
        self._mode = profile.mode(mode)
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    def open(self) -> CameraSource:
        mode = self._mode
        cap = cv2.VideoCapture(self._profile.device_index, BACKENDS[mode.backend])
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"No se pudo abrir la cámara {self._profile.device_index} "
                                f"con el backend {mode.backend}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
        cap.set(cv2.CAP_PROP_FPS, mode.fps)
        ok, image = cap.read()  # el primer fotograma confirma el tamaño real y se descarta
        if not ok:
            cap.release()
            raise CaptureError("La cámara se abrió pero no entregó ningún fotograma")
        height, width = image.shape[:2]
        if (width, height) != (mode.width, mode.height):
            cap.release()
            raise CaptureError(f"El driver entrega {width}x{height} y el perfil pide "
                                f"{mode.width}x{mode.height}")
        self._cap = cap
        self._index = 0
        return self

    def read(self) -> Frame | None:
        """Siguiente fotograma, o None si la lectura falló."""
        if self._cap is None:
            raise CaptureError("La cámara no está abierta (usa open() o 'with')")
        ok, image = self._cap.read()
        timestamp = time.perf_counter_ns()
        if not ok:
            return None
        if self._profile.flip_vertical and self._profile.flip_horizontal:
            image = cv2.flip(image, -1)
        elif self._profile.flip_vertical:
            image = cv2.flip(image, 0)
        elif self._profile.flip_horizontal:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, timestamp_ns=timestamp, index=self._index)
        self._index += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraSource:
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.release()
