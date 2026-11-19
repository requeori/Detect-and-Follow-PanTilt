"""Evalúa un detector sobre los clips de una carpeta.

Uso:
    python scripts/evaluate.py CLIPS_DIR --detector paquete.modulo:fabrica [--out resultados.json]
        --split A|V [--config params.yaml] [--static S0_estatica S3] [--only S0 S8]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from objecttracker.clips import find_clips
from objecttracker.detector import Detector
from objecttracker.evaluation import aggregate, evaluate_clip


def load_detector(spec: str, config: Path | None = None) -> Detector:
    module_name, _, factory_name = spec.partition(":")
    if not factory_name:
        raise SystemExit("--detector debe tener la forma modulo:fabrica")
    factory = getattr(importlib.import_module(module_name), factory_name)
    detector = factory(config) if config else factory()
    if not isinstance(detector, Detector):
        raise SystemExit(f"{spec} no devuelve un objeto con detect(image)")
    return detector


def pct(value: float | None) -> str:
    return "  -  " if value is None else f"{100 * value:5.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clips_dir", type=Path)
    parser.add_argument("--detector", required=True)
    parser.add_argument(
        "--config", type=Path, help="YAML de parametros; se pasa a la fabrica del detector"
    )
    parser.add_argument("--out", type=Path, help="JSON con resultados por clip y agregados")
    parser.add_argument(
        "--static",
        nargs="*",
        default=[],
        help="prefijos de clip_id con la pelota quieta, para medir jitter. Sin esta opcion no se "
        "mide jitter. Ej.: --static S0_estatica S3",
    )
    parser.add_argument("--only", nargs="*", help="evaluar solo estos escenarios")
    parser.add_argument(
        "--split",
        choices=["A", "V"],
        required=True,
        help="sesion a evaluar: A (ajuste) o V (validacion); obligatorio",
    )
    args = parser.parse_args(argv)

    clips = [
        c
        for c in find_clips(args.clips_dir)
        if c.split == args.split and (not args.only or c.scenario in args.only)
    ]
    if not clips:
        print(f"No hay clips de la sesion {args.split} en {args.clips_dir}", file=sys.stderr)
        return 1
    detector = load_detector(args.detector, args.config)

    results = []
    print(f"{'clip':44s} {'det%':>5s} {'FP%':>5s} {'ms':>6s} {'jit px':>7s}")
    for meta in clips:
        r = evaluate_clip(
            detector,
            meta,
            measure_jitter=meta.clip_id.startswith(tuple(args.static)),
        )
        results.append(r)
        jitter = "  -  " if r.jitter_px is None else f"{r.jitter_px:7.2f}"
        flag = "" if r.frame_count_ok else f"  ! {r.n_frames}/{r.expected_frames} frames"
        print(
            f"{r.clip_id:44s} {pct(r.detection_rate)} {pct(r.false_positive_rate)} "
            f"{r.time_ms_mean:6.2f} {jitter:>7s}{flag}"
        )

    rows = aggregate(results)
    print("\nPor condición y sesión (tasas ponderadas por fotogramas)")
    for row in rows:
        print(
            f"{row['condition']:30s} {row['split']:2s} clips={row['clips']:2d} "
            f"det%={pct(row['detection_rate'])} FP%={pct(row['false_positive_rate'])}"
        )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "detector": args.detector,
                    "clips": [r.to_dict() for r in results],
                    "aggregate": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nEscrito {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
