# app/core/config.py
# Centraliza todas las constantes, variables de entorno y configuración
# de la aplicación en un único lugar.

import os
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# ── Servidor ───────────────────────────────────────────────────────────────────
APP_VERSION = "1.4.0"
TEMP_DIR    = "temp"

# ── CORS ───────────────────────────────────────────────────────────────────────
CORS_ORIGINS = ["*"]

# ── Tipos de archivos permitidos ───────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/jpg")
ALLOWED_VIDEO_TYPES = ("video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime")

# ── Calidad de placa ───────────────────────────────────────────────────────────
# Mapa: clave interna del clasificador → nombre en la base de datos
QUALITY_LABEL_MAP: dict[str, str] = {
    "oclusion": "Oclusión",
    "reflejo":  "Reflejo",
    "sucia":    "Suciedad",
    "legible":  "Legibilidad",
}

# Valores por defecto cuando el clasificador no retorna resultados
DEFAULT_QUALITY_LABELS: dict[str, str] = {
    "legible":  "Ilegible",
    "oclusion": "No",
    "reflejo":  "No",
    "sucia":    "No",
}

# ── Procesamiento de video (WebSocket) ─────────────────────────────────────────
VIDEO_FRAME_SKIP         = 10    # procesar 1 de cada N frames
VIDEO_MAX_CENTROID_DIST  = 120   # px máximos para asociar un track
VIDEO_MAX_FRAMES_MISSING = 6     # frames sin detección antes de cerrar un track
VIDEO_JPEG_QUALITY       = 75    # calidad de compresión del frame enviado al WS