# app/ai/plate_detector.py
# Detecta placas usando YOLOv8n entrenado localmente con dataset ecuatoriano

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_PATH           = "runs/detect/yolov8n_plates_ecuador43/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25  # bajo para maximizar detecciones

# Cargar modelo una sola vez al importar
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
    Detecta placas vehiculares usando YOLOv8n local.

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa
          - "bbox":       [x1, y1, x2, y2] en píxeles (int)
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

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
        })

    return plates