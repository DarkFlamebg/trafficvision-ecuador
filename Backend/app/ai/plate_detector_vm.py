# app/ai/plate_detector_vm.py
# Detector de placas usando Vision Mamba (Swin + SSM) vía MMDetection
# Interfaz idéntica a plate_detector.py — compatible con el resto del backend
#
# Arquitectura: State Space Model (SSM) con backbone Swin-Mamba
# Checkpoint:   swin_r4/best_coco_bbox_mAP_epoch_51.pth
# Framework:    MMDetection (mmdet.apis)

import os
import numpy as np
import cv2
from PIL import Image, ImageOps

from app.ai.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../.."))

MODEL_CONFIG     = os.path.join(_ROOT_DIR, "app", "models", "trained", "swin_r4", "vision_mamba_ecuaplacas.py")
MODEL_CHECKPOINT = os.path.join(_ROOT_DIR, "app", "models", "trained", "swin_r4", "best_coco_bbox_mAP_epoch_51.pth")

# Mismo umbral que YOLO y RT-DETR para comparativa simétrica
CONFIDENCE_THRESHOLD = 0.25

# Rango de proporciones válidas: cubre placas horizontales (autos/trucks)
# y placas verticales de moto ecuatoriana (~10x15 cm)
ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 6.0

_model = None


# ── Carga del modelo (singleton) ───────────────────────────────────────────────
def _get_model():
    """Inicializa el modelo Vision Mamba usando MMDetection (lazy, singleton)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_CHECKPOINT):
            raise FileNotFoundError(
                f"Checkpoint Vision Mamba no encontrado: {MODEL_CHECKPOINT}\n"
                f"Asegúrate de que best_coco_bbox_mAP_epoch_51.pth esté en swin_r4/."
            )
        from mmdet.apis import init_detector
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = init_detector(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
    return _model


# ── Función principal de detección ─────────────────────────────────────────────
def detect_plate_vision_mamba(input_image) -> list:
    """
    Detecta placas vehiculares usando Vision Mamba (Swin+SSM).

    Opera sobre el crop de región vehicular (igual que YOLO y RT-DETR),
    garantizando comparativa simétrica en el pipeline de TrafficVision.

    Args:
        input_image: ruta (str) o array NumPy BGR (OpenCV)

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa (con padding + deskew)
          - "bbox":       [x1, y1, x2, y2] en píxeles del crop de entrada (int)
          - "confidence": float 0.0 – 1.0
          - "detector":   str "vision_mamba"
    """
    from mmdet.apis import inference_detector

    # Carga de imagen con corrección EXIF
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
            print(f"[vision_mamba] Bbox descartado por proporción: "
                  f"{w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        # Extraer recorte con padding adaptativo + deskew (igual que plate_detector.py)
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


# ── Alias para compatibilidad con código que llame detect_plate ────────────────
detect_plate = detect_plate_vision_mamba


# ── Testing standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python plate_detector_vm.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Probando Vision Mamba Plate Detector")
    print(f"{'='*60}\n")

    detections = detect_plate_vision_mamba(image_path)
    print(f"Placas detectadas: {len(detections)}\n")

    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        print(f"Placa #{i}:")
        print(f"  Confianza : {d['confidence']:.2%}")
        print(f"  Bbox      : [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Tamaño    : {x2-x1}x{y2-y1}px")

    print(f"\n{'='*60}\n")
