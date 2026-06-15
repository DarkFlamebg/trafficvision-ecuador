# app/ai/plate_detector.py
# Detecta placas usando YOLOv11n entrenado localmente con dataset ecuatoriano

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO
from app.ai.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.abspath(os.path.join(_BASE_DIR, "../.."))

MODEL_PATH            = os.path.join(_ROOT_DIR, "app", "models", "trained", "yolo11n_combined_all", "best.pt")
CONFIDENCE_THRESHOLD  = 0.45

# Autos/camiones: ratio > 1.5 (placa horizontal)
# Motos EC:       ratio < 1.0 (placa vertical ~10x15cm)
# Rango combinado: 0.3 – 6.0, el giro se maneja en crop_utils._rotate_if_moto
ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 6.0

_model = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
    return _model


def _load_image(input_image) -> np.ndarray:
    """
    Carga la imagen respetando la orientación EXIF del celular.
    Retorna array NumPy BGR compatible con OpenCV.
    """
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_plate(input_image) -> list:
    """
    Detecta placas vehiculares usando YOLOv11n local.

    Filtros aplicados:
    - Confianza mínima: 0.45
    - Proporción ancho/alto: entre 1.5 y 6.0 (forma de placa real)

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa (con padding + deskew)
          - "bbox":       [x1, y1, x2, y2] en píxeles originales (int)
          - "confidence": float 0.0 – 1.0
    """
    image  = _load_image(input_image)
    model  = _get_model()
    ih, iw = image.shape[:2]

    results = model(image, verbose=False)[0]
    plates  = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0,  int(x1))
        y1 = max(0,  int(y1))
        x2 = min(iw, int(x2))
        y2 = min(ih, int(y2))

        w_box = x2 - x1
        h_box = y2 - y1

        if h_box == 0:
            continue

        aspect_ratio = w_box / h_box
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            print(f"[yolo] Bbox descartado por proporción: {w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        # ── CAMBIO: crop con padding adaptativo + deskew ──────────────────────
        crop = extract_plate_crop(image, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],   # bbox original sin padding (para visualización)
            "confidence": round(conf, 4),
        })

    return plates