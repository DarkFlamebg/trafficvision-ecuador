# app/ai/detectors/vision_mamba.py
# Detector de placas usando Vision Mamba (Swin + SSM) vía MMDetection.
#
# Arquitectura: State Space Model (SSM) con backbone Swin-Mamba
# Checkpoint:   swin_r4/best_coco_bbox_mAP_epoch_51.pth
# Framework:    MMDetection (mmdet.apis)

import os
import numpy as np
import cv2
from PIL import Image, ImageOps

from app.ai.detectors.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_CONFIG     = os.path.join(_ROOT_DIR, "app", "models", "trained", "swin_r4", "vision_mamba_ecuaplacas.py")
MODEL_CHECKPOINT = os.path.join(_ROOT_DIR, "app", "models", "trained", "swin_r4", "best_coco_bbox_mAP_epoch_51.pth")

CONFIDENCE_THRESHOLD = 0.25
ASPECT_RATIO_MIN     = 0.3
ASPECT_RATIO_MAX     = 6.0

_model = None


def _get_model():
    """Inicializa el modelo Vision Mamba usando MMDetection (lazy, singleton)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_CHECKPOINT):
            raise FileNotFoundError(
                f"Checkpoint Vision Mamba no encontrado: {MODEL_CHECKPOINT}"
            )
        from mmdet.apis import init_detector
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = init_detector(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    return _model


def detect_plate_vision_mamba(input_image) -> list:
    """
    Detecta placas vehiculares usando Vision Mamba (Swin+SSM).

    Args:
        input_image: ruta (str) o array NumPy BGR (OpenCV)

    Returns:
        Lista de dicts con image, bbox, confidence, detector.
    """
    from mmdet.apis import inference_detector

    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    pil_img = ImageOps.exif_transpose(pil_img)
    image   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    ih, iw  = image.shape[:2]

    model  = _get_model()
    result = inference_detector(model, image)

    plates = []
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()

    for bbox, score in zip(bboxes, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = bbox
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
            print(f"[vision_mamba] Bbox descartado por proporción: {w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        crop = extract_plate_crop(image, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(float(score), 4),
            "detector":   "vision_mamba",
        })

    return plates


# Alias de compatibilidad
detect_plate = detect_plate_vision_mamba
