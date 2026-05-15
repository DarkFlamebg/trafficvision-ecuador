# app/ai/plate_detector_rtdetr.py
# Detecta placas usando RT-DETR entrenado localmente con dataset ecuatoriano

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import RTDETR
from app.ai.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_PATH            = os.path.join(_ROOT_DIR, "ml", "models", "trained", "rtdetr_combined_all", "best.pt")
CONFIDENCE_THRESHOLD  = 0.45

# Autos/camiones: ratio > 1.5 (placa horizontal)
# Motos EC:       ratio < 1.0 (placa vertical ~10x15cm)
# Rango combinado: 0.3 – 6.0, el giro se maneja en crop_utils._rotate_if_moto
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


def detect_plate_rtdetr(input_image) -> list:
    """
    Detecta placas vehiculares usando RT-DETR local.

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
          - "detector":   str "rtdetr"
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

        # ── CAMBIO: crop con padding adaptativo + deskew ──────────────────────
        crop = extract_plate_crop(image, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],   # bbox original sin padding (para visualización)
            "confidence": round(conf, 4),
            "detector":   "rtdetr",
        })

    return plates


def detect_plate_ensemble(input_image, use_yolo: bool = True, use_rtdetr: bool = True) -> list:
    """
    Detector ensemble que combina YOLO y RT-DETR para mayor robustez.
    """
    all_plates = []

    if use_yolo:
        try:
            from plate_detector import detect_plate
            yolo_plates = detect_plate(input_image)
            for p in yolo_plates:
                p["detector"] = "yolo"
                all_plates.append(p)
        except Exception as e:
            print(f"[ensemble] Error en YOLO: {e}")

    if use_rtdetr:
        try:
            rtdetr_plates = detect_plate_rtdetr(input_image)
            all_plates.extend(rtdetr_plates)
        except Exception as e:
            print(f"[ensemble] Error en RT-DETR: {e}")

    if not all_plates:
        return []

    plates_nms = _apply_nms(all_plates, iou_threshold=0.5)
    plates_nms.sort(key=lambda x: x["confidence"], reverse=True)
    return plates_nms


def _apply_nms(detections: list, iou_threshold: float = 0.5) -> list:
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            det for det in detections
            if _calculate_iou(best["bbox"], det["bbox"]) < iou_threshold
        ]

    return keep


def _calculate_iou(box1: list, box2: list) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w    = max(0, x2_inter - x1_inter)
    inter_h    = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    box1_area  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ── Testing ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python plate_detector_rtdetr.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Probando RT-DETR Plate Detector")
    print(f"{'='*60}\n")

    print("Test 1: RT-DETR")
    plates_rtdetr = detect_plate_rtdetr(image_path)
    print(f"   Placas detectadas: {len(plates_rtdetr)}")
    for i, p in enumerate(plates_rtdetr, 1):
        print(f"   {i}. Confianza: {p['confidence']:.2%} | Bbox: {p['bbox']}")

    print("\nTest 2: Ensemble (YOLO + RT-DETR)")
    try:
        plates_ensemble = detect_plate_ensemble(image_path)
        print(f"   Placas detectadas: {len(plates_ensemble)}")
        for i, p in enumerate(plates_ensemble, 1):
            print(f"   {i}. Detector: {p.get('detector','?')} | Confianza: {p['confidence']:.2%} | Bbox: {p['bbox']}")
    except Exception as e:
        print(f"   Error en ensemble: {e}")

    print(f"\n{'='*60}\n")