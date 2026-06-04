# app/ai/plate_detector_efficientdet.py
# Detecta placas usando modelo entrenado localmente con dataset ecuatoriano
# Interfaz idéntica a plate_detector.py — compatible con el resto del backend

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO
from app.ai.crop_utils import extract_plate_crop

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_PATH = os.path.join(
    _ROOT_DIR, "runs", "detect", "ml", "runs",
    "detect", "efficientdet_d2_plates_v3",
    "weights", "best.pt"
)

CONFIDENCE_THRESHOLD = 0.25

# Autos/camiones: ratio > 1.5 (placa horizontal)
# Motos EC:       ratio < 1.0 (placa vertical ~10x15cm)
# Rango combinado: 0.3 – 6.0, el giro se maneja en crop_utils._rotate_if_moto
ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 6.0

_model = None


# ── Carga del modelo (singleton) ───────────────────────────────────────────────
def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo no encontrado: {MODEL_PATH}\n"
                f"Asegúrate de que best.pt esté en la ruta correcta."
            )
        _model = YOLO(MODEL_PATH)
        print(f"[efficientdet] Modelo cargado: {MODEL_PATH}")
    return _model


# ── Carga de imagen con corrección EXIF ────────────────────────────────────────
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


# ── Función principal de detección ─────────────────────────────────────────────
def detect_plate_efficientdet(input_image) -> list:
    """
    Detecta placas vehiculares en una imagen.

    Filtros aplicados:
    - Confianza mínima: 0.25
    - Proporción ancho/alto: entre 0.3 y 6.0 (forma de placa real)

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa (con padding + deskew)
          - "bbox":       [x1, y1, x2, y2] en píxeles originales (int)
          - "confidence": float 0.0 – 1.0
          - "detector":   str "efficientdet"
    """
    image  = _load_image(input_image)
    model  = _get_model()
    ih, iw = image.shape[:2]

    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    plates  = []

    for box in results.boxes:
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0,  int(x1))
        y1 = max(0,  int(y1))
        x2 = min(iw, int(x2))
        y2 = min(ih, int(y2))

        w_box = x2 - x1
        h_box = y2 - y1

        if h_box == 0 or w_box == 0:
            continue

        # Filtro de proporción — descarta detecciones que no tienen forma de placa
        aspect_ratio = w_box / h_box
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            print(f"[efficientdet] Bbox descartado por proporción: "
                  f"{w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        # Extraer recorte con padding adaptativo + deskew (igual que plate_detector.py)
        crop = extract_plate_crop(image, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "detector":   "efficientdet",
        })

    print(f"[efficientdet] Placas detectadas: {len(plates)}")
    return plates


# ── Alias para compatibilidad con código que llame detect_plate ────────────────
# Si algún módulo del backend importa detect_plate en vez de detect_plate_efficientdet
detect_plate = detect_plate_efficientdet


# ── Testing standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python plate_detector_efficientdet.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Probando EfficientDet Plate Detector")
    print(f"{'='*60}\n")

    img   = _load_image(image_path)
    model = _get_model()

    results   = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    annotated = results.plot()

    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Resultado guardado en: {output_path}")
    print(f"Placas detectadas: {len(results.boxes)}\n")

    for i, box in enumerate(results.boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        print(f"Placa #{i}:")
        print(f"  Confianza : {score:.2%}")
        print(f"  Bbox      : [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Tamaño    : {x2-x1}x{y2-y1}px")

    print(f"\n{'='*60}\n")