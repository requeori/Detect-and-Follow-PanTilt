# objecttracker

Seguimiento de objetos con **visión artificial clásica** (sin redes neuronales ni modelos entrenados). Primer objeto: una pelota roja de ≈6 cm (medida con regla; la cámara sugiere 5,4 cm). Hardware inicial: cámara USB AUKEY PC-LM1E, Pan-Tilt con 2×SG90 y ESP32.


## Puesta en marcha

```bash
conda env create -f environment.yml     # Python 3.10 + paquete en modo editable
conda activate objecttracker
pytest                                  # pruebas
ruff check .                            # análisis estático
```

**Si `import objecttracker` falla:** `pip install -e .` dentro del entorno.

**WSL:** si falla `import cv2` con `libGL.so.1`, instala `sudo apt install libgl1`.La captura y la caracterización de la cámara se hacen en Windows nativo.

## Grabar un clip (Windows)

```bash
python scripts/record_clip.py --scenario S0 --name estatica_1m --split A --ball always
```

## Estructura

| Ruta | Contenido |
|---|---|
| `src/objecttracker/` | Paquete (se irá llenando por bloques) |
| `configs/cameras/` | Perfiles de cámara en YAML |
| `scripts/` | `record_clip.py` (graba clips del dataset) |
| `experiments/` | `b0_camara/` (medición de cámara y evidencia), `b0_codec/` (comparación de códecs) |
| `data/` | Vídeos y resultados |
| `docs/` | Planificación, decisiones, protocolo |
| `tests/` | Pruebas |

## Reglas del repositorio

- Solo `yaml.safe_load` para leer configuración; nada de `pickle`, `eval` ni `exec` sobre ficheros externos. `ruff` lo comprueba.
