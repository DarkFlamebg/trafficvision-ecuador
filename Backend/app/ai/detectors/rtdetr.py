# app/ai/detectors/rtdetr.py
# Detecta placas usando RT-DETR entrenado localmente con dataset ecuatoriano.

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import RTDETR
from app.ai.detectors.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_PATH           = os.path.join(_ROOT_DIR, "app", "models", "trained", "rtdetr_combined_all", "best.pt")
CONFIDENCE_THRESHOLD = 0.45

ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 6.0

_model = None


def _get_model() -> RTDETR:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo RT-DETR no encontrado: {MODEL_PATH}")
        _model = RTDETR(MODEL_PATH)
    return _model


def _load_image(input_image) -> np.ndarray:
    """Carga la imagen respetando orientación EXIF. Retorna array NumPy BGR."""
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_plate_rtdetr(input_image) -> list:
    """
    Detecta placas vehiculares usando RT-DETR local.

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts con image, bbox, confidence, detector.
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
            print(f"[rtdetr] Bbox descartado por proporción: {w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        crop = extract_plate_crop(image, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "detector":   "rtdetr",
        })

    return plates


def _apply_nms(detections: list, iou_threshold: float = 0.5) -> list:
    if len(detections) <= 1:
        return detections
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if _calculate_iou(best["bbox"], d["bbox"]) < iou_threshold]
    return keep


def _calculate_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0.0
