# objecttracker

Seguimiento de objetos con **visión artificial clásica** (sin redes neuronales ni modelos entrenados). Primer objeto: una pelota roja de ≈6 cm (medida con regla; la cámara sugiere 5,4 cm). Hardware inicial: cámara USB AUKEY PC-LM1E, Pan-Tilt con 2×SG90 y ESP32.

> Estado: **Bloque 0 (definición)**. Todavía no hay detector. Ver `docs/planificacion/`.

## Puesta en marcha

```bash
conda env create -f environment.yml     # Python 3.10 + paquete en modo editable
conda activate objecttracker
pytest                                  # pruebas
ruff check .                            # análisis estático
```

**Si `import objecttracker` falla:** `pip install -e .` dentro del entorno.

**WSL:** si falla `import cv2` con `libGL.so.1`, instala `sudo apt install libgl1`. La cámara USB normalmente **no** es visible desde WSL; la captura y la caracterización de la cámara se hacen en Windows nativo (ver `docs/decisiones.md`, D-005).

## Grabar un clip (Windows)

```bash
python scripts/record_clip.py --scenario S0 --name estatica_1m --split A --ball always
```

Protocolo y metadatos: `docs/protocolo_grabacion.md`. 

## Estructura

| Ruta | Contenido |
|---|---|
| `src/objecttracker/` | Paquete (se irá llenando por bloques) |
| `configs/cameras/` | Perfiles de cámara en YAML |
| `scripts/` | `record_clip.py` (graba clips del dataset) |
| `experiments/` | `b0_camara/` (medición de cámara y evidencia), `b0_codec/` (comparación de códecs) |
| `data/` | Vídeos y resultados (**no se versiona**) |
| `docs/` | Planificación, decisiones, protocolo, bloque 0 |
| `tests/` | Pruebas |

## Reglas del repositorio

- Solo `yaml.safe_load` para leer configuración; nada de `pickle`, `eval` ni `exec` sobre ficheros externos (D-004). `ruff` lo comprueba.
- Los datos pesados no se suben a git; se registran solo sus metadatos.
- Los commits describen lo que realmente hace el cambio.
