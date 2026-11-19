"""Métricas de un detector sobre clips grabados (B1-04).

Sin anotaciones de posición solo se pueden medir cuatro cosas, y solo donde tienen sentido:
- detection_rate: fracción de fotogramas con found, en clips donde la pelota está siempre presente
  (si el JSON anota annotations.ball_visible, solo cuentan esos fotogramas).
- false_positive_rate: fracción de fotogramas con found, en clips sin pelota.
- timing: coste de detect() por fotograma (excluye la decodificación del vídeo).
- jitter: dispersión del centro detectado, solo si la pelota está quieta (lo decide el llamante).
No miden error de localización ni si el detector elige la pelota entre distractores (D-012).
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from objecttracker.clips import ClipMeta, VideoFileSource
from objecttracker.detector import Detection, Detector

WARMUP_FRAMES = 5  # primeras llamadas excluidas del tiempo (cachés, asignaciones)


@dataclass
class ClipResult:
    clip_id: str
    scenario: str
    condition: str  # clip_id sin sesion ni fecha: "S0_estatica_1m"; une repeticiones de la toma
    split: str
    ball_present: str
    n_frames: int
    expected_frames: int
    n_found: int
    found_rate: float  # sobre todos los fotogramas del clip
    rate_frames: int  # fotogramas que cuentan para detection/false_positive_rate
    rate_found: int  # de ellos, con found
    # Solo una de las dos tiene valor, según ball_present; None = no aplicable.
    detection_rate: float | None
    false_positive_rate: float | None
    time_ms_mean: float
    time_ms_p95: float
    time_ms_max: float
    fps_detector: float
    jitter_px: float | None  # sqrt(var_x + var_y) del centro; None si no se pidió o hay <2 puntos
    median_bbox_w: float | None
    frame_count_ok: bool
    detections: list[Detection] = field(default_factory=list, repr=False)

    def to_dict(self, with_detections: bool = False) -> dict:
        data = asdict(self)
        if not with_detections:
            data.pop("detections")
        return data


def condition_of(clip_id: str, split: str) -> str:
    """Quita el sufijo `_<split>_<fecha>` que añade record_clip.py; si no lo hay, el id entero.

    Evita mezclar en una misma tasa clips distintos del mismo escenario (S0 estática y S0
    acercamiento no deben promediarse).
    """
    return clip_id.rsplit(f"_{split}_", 1)[0]


def _jitter(centers: list[tuple[float, float]]) -> float | None:
    if len(centers) < 2:
        return None
    xs, ys = zip(*centers, strict=True)
    return float(np.sqrt(np.var(xs) + np.var(ys)))


def evaluate_clip(
    detector: Detector, meta: ClipMeta, measure_jitter: bool = False, warmup: int = WARMUP_FRAMES
) -> ClipResult:
    """Pasa todos los fotogramas del clip por el detector y calcula las métricas."""
    detections: list[Detection] = []
    times_ns: list[int] = []
    for frame in VideoFileSource(meta):
        start = time.perf_counter_ns()
        detection = detector.detect(frame.image)
        elapsed = time.perf_counter_ns() - start
        detections.append(detection)
        if frame.index >= warmup:
            times_ns.append(elapsed)

    n = len(detections)
    found = [d for d in detections if d.found]
    found_rate = len(found) / n if n else 0.0
    # Con la pelota siempre presente solo cuentan los fotogramas anotados como visibles.
    counted = [d for i, d in enumerate(detections) if meta.is_visible(i)]
    rate_frames = len(counted)
    rate_found = sum(d.found for d in counted)
    rate = rate_found / rate_frames if rate_frames else None
    times_ms = [t / 1e6 for t in times_ns] or [0.0]
    mean_ms = statistics.fmean(times_ms)
    centers = [d.center for d in found]
    widths = [d.bbox[2] for d in found]

    return ClipResult(
        clip_id=meta.clip_id,
        scenario=meta.scenario,
        condition=condition_of(meta.clip_id, meta.split),
        split=meta.split,
        ball_present=meta.ball_present,
        n_frames=n,
        expected_frames=meta.expected_frames,
        n_found=len(found),
        found_rate=found_rate,
        rate_frames=rate_frames,
        rate_found=rate_found,
        detection_rate=rate if meta.ball_present == "always" else None,
        false_positive_rate=found_rate if meta.ball_present == "never" else None,
        time_ms_mean=mean_ms,
        time_ms_p95=float(np.percentile(times_ms, 95)),
        time_ms_max=max(times_ms),
        fps_detector=1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        jitter_px=_jitter(centers) if measure_jitter else None,
        median_bbox_w=float(np.median(widths)) if widths else None,
        frame_count_ok=(n == meta.expected_frames),
        detections=detections,
    )


def aggregate(results: list[ClipResult]) -> list[dict]:
    """Agrega por (condición, sesión). Las tasas son medias ponderadas por fotogramas.

    Una condición es una toma repetible (p. ej. S0_estatica_1m): sus repeticiones se juntan,
    pero tomas distintas del mismo escenario no.
    """
    groups: dict[tuple[str, str], list[ClipResult]] = defaultdict(list)
    for r in results:
        groups[(r.condition, r.split)].append(r)

    rows = []
    for (condition, split), items in sorted(groups.items()):
        pos = [r for r in items if r.detection_rate is not None]
        neg = [r for r in items if r.false_positive_rate is not None]
        frames_all = sum(r.n_frames for r in items)
        rows.append(
            {
                "scenario": items[0].scenario,
                "condition": condition,
                "split": split,
                "clips": len(items),
                "frames": frames_all,
                "detection_rate": _weighted(pos),
                "false_positive_rate": _weighted(neg),
                "time_ms_mean": (
                    sum(r.time_ms_mean * r.n_frames for r in items) / frames_all
                    if frames_all
                    else None
                ),
                "all_frame_counts_ok": all(r.frame_count_ok for r in items),
            }
        )
    return rows


def _weighted(items: list[ClipResult]) -> float | None:
    frames = sum(r.rate_frames for r in items)
    return sum(r.rate_found for r in items) / frames if frames else None
