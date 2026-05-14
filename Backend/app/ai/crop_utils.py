# app/ai/crop_utils.py
# Utilidades compartidas para extracción de crops de placa
# Usado por plate_detector.py (YOLO) y plate_detector_rtdetr.py (RT-DETR)

import cv2
import numpy as np

# ── Constantes ─────────────────────────────────────────────────────────────────
PADDING_RATIO   = 0.08   # 8% del ancho/alto del bbox como padding extra
MIN_PADDING_PX  = 4      # Mínimo absoluto en píxeles


def extract_plate_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Extrae un crop de placa con padding adaptativo y corrección de perspectiva.

    Reemplaza el simple image[y1:y2, x1:x2] de los detectores.
    Mejora la calidad del crop que llega al OCR, especialmente en:
      - Placas pequeñas (evita cortar bordes de caracteres)
      - Placas con leve inclinación (deskew automático)
      - Placas con marco metálico brillante (padding absorbe el borde)

    Args:
        image:      imagen completa BGR (NumPy array)
        x1,y1,x2,y2: coordenadas del bbox detectado (ya clipeadas a imagen)

    Returns:
        Crop BGR listo para pasar a read_plate()
    """
    ih, iw = image.shape[:2]

    # ── 1. Padding adaptativo ──────────────────────────────────────────────────
    w_box = x2 - x1
    h_box = y2 - y1

    pad_x = max(MIN_PADDING_PX, int(w_box * PADDING_RATIO))
    pad_y = max(MIN_PADDING_PX, int(h_box * PADDING_RATIO))

    px1 = max(0,  x1 - pad_x)
    py1 = max(0,  y1 - pad_y)
    px2 = min(iw, x2 + pad_x)
    py2 = min(ih, y2 + pad_y)

    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        return image[y1:y2, x1:x2]  # fallback al crop original

    # ── 2. Corrección de perspectiva (deskew) ──────────────────────────────────
    crop = _deskew(crop)

    return crop


def _deskew(crop: np.ndarray) -> np.ndarray:
    """
    Detecta y corrige inclinación leve de la placa usando contornos.
    Si no puede corregir, devuelve el crop sin modificar.

    Funciona bien para ángulos de ±15°. Placas muy inclinadas
    quedan para trabajo futuro (homografía completa).
    """
    try:
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Umbral adaptativo para separar texto del fondo
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        # Morfología para conectar caracteres en una región
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop

        # Tomar el contorno más grande (debería ser la zona de texto)
        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)
        angle   = rect[2]

        # minAreaRect devuelve ángulos en (-90, 0]; normalizar a (-45, 45]
        if angle < -45:
            angle += 90

        # Solo corregir si la inclinación es significativa pero manejable
        if abs(angle) < 0.5 or abs(angle) > 20:
            return crop

        # Rotar el crop para enderezar la placa
        h, w   = crop.shape[:2]
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            crop, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    except Exception:
        return crop  # Nunca fallar: si algo sale mal, crop original