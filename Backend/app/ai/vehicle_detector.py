# app/ai/vehicle_detector.py
# Detecta vehículos usando YOLOv8n preentrenado con COCO
# No requiere entrenamiento adicional — COCO incluye las clases vehiculares

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO

# Modelo base COCO — se descarga automáticamente si no existe
MODEL_PATH           = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

# Clases de vehículos en COCO y sus IDs
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Traducción al español para el frontend
VEHICLE_LABELS_ES = {
    "car":        "Automóvil",
    "motorcycle": "Motocicleta",
    "bus":        "Autobús",
    "truck":      "Camión",
}

# Cargar modelo una sola vez
_model = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def _load_image(input_image) -> np.ndarray:
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_vehicles(input_image) -> list:
    """
    Detecta vehículos en una imagen usando YOLOv8n preentrenado COCO.

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "type":       tipo de vehículo en inglés (car, truck, bus, motorcycle)
          - "type_es":    tipo en español (Automóvil, Camión, etc.)
          - "bbox":       [x1, y1, x2, y2] en píxeles
          - "confidence": float 0.0 – 1.0
          - "image":      recorte NumPy BGR del vehículo
    """
    image  = _load_image(input_image)
    model  = _get_model()
    ih, iw = image.shape[:2]

    results  = model(image, verbose=False)[0]
    vehicles = []

    for box in results.boxes:
        class_id = int(box.cls[0])

        # Solo procesar clases vehiculares
        if class_id not in VEHICLE_CLASSES:
            continue

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

        vehicle_type = VEHICLE_CLASSES[class_id]

        vehicles.append({
            "type":       vehicle_type,
            "type_es":    VEHICLE_LABELS_ES[vehicle_type],
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "image":      crop,
        })

    return vehicles