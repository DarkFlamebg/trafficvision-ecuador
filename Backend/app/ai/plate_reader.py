# app/ai/plate_reader.py
# Reconoce los caracteres de la placa usando EasyOCR

import re
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

_EC_PLATE     = re.compile(r'^[A-Z]{3}\d{4}$')
_EC_PLATE_FMT = re.compile(r'^[A-Z]{3}-\d{4}$')


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocesa el recorte de placa para mejorar la lectura OCR.

    Fixes:
    1. Recortar 25% superior — elimina etiquetas 'ECUA', 'PLACA PROVISIONAL'
    2. Escalar si es muy pequeño
    3. CLAHE para mejorar contraste
    4. Threshold adaptativo para binarizar
    """
    h, w = image.shape[:2]

    # Fix 1: recortar franja superior con etiqueta (ECUA, PLACA PROVISIONAL)
    # Ajuste dinámico según altura del recorte
    if h > 100:
        crop_top = int(h * 0.22)
    elif h > 60:
        crop_top = int(h * 0.15)
    else:
        crop_top = int(h * 0.10)
    image = image[crop_top:, :]
    h, w = image.shape[:2]

    # Fix 2: escalar si es muy pequeño
    if h < 80:
        scale = 120 / h
        image = cv2.resize(image, (int(w * scale), 120),
                           interpolation=cv2.INTER_CUBIC)
    elif w < 200:
        scale = 300 / w
        image = cv2.resize(image, (300, int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # Fix 3: escalar a mínimo 300px de ancho
    h, w = image.shape[:2]
    if w < 300:
        scale = 300 / w
        image = cv2.resize(image, (300, int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # Fix 4: detectar color de fondo y aplicar preprocesamiento adecuado
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue  = hsv[:, :, 0].mean()
    sat  = hsv[:, :, 1].mean()

    # Verde (placas ecuador estándar): hue 35-85
    # Naranja/amarillo (taxi): hue 15-35
    # Azul/celeste (provisional): hue 85-130
    # Blanco/gris (clásica): saturación baja < 40
    is_colored = sat > 40

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if is_colored:
        # Para fondos de color: extraer canal con mejor contraste texto/fondo
        b, g, r = cv2.split(image)
        # Texto blanco sobre fondo color — usar canal de valor (brillo)
        gray = hsv[:, :, 2]
        # CLAHE agresivo para separar texto del fondo
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        gray  = clahe.apply(gray)
    else:
        # Para fondos blancos/grises: CLAHE suave
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray   = cv2.filter2D(gray, -1, kernel)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _fix_ocr_errors(clean: str) -> str:
    if len(clean) != 7:
        return clean

    letters = (
        clean[:3]
        .replace('0', 'O').replace('1', 'I').replace('5', 'S')
        .replace('8', 'B').replace('2', 'Z').replace('6', 'G')
    )
    numbers = (
        clean[3:]
        .replace('O', '0').replace('I', '1').replace('S', '5')
        .replace('B', '8').replace('Z', '2').replace('G', '6')
    )
    return letters + numbers


def _format_plate(text: str) -> str:
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    clean = _fix_ocr_errors(clean)
    if _EC_PLATE.match(clean):
        return clean[:3] + '-' + clean[3:]
    return clean


def _join_results(results: list) -> str:
    texts = [r[1] for r in sorted(results, key=lambda x: x[0][0][0])]
    return ''.join(texts)


def read_plate(image: np.ndarray) -> dict | None:
    """
    Extrae el texto de la placa a partir de un recorte de imagen.

    Returns:
        Dict con 'plate' y 'confidence', o None si no detectó texto.
    """
    # Intento 1: preprocesado (sin header + contraste)
    processed = _preprocess(image)
    results   = reader.readtext(
        processed,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False
    )

    # Intento 2: imagen original como fallback
    if not results:
        results = reader.readtext(
            image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=1,
            paragraph=False
        )

    if not results:
        return None

    best        = max(results, key=lambda x: x[2])
    best_text   = _format_plate(best[1])
    best_conf   = best[2]

    joined_text = _format_plate(_join_results(results))
    avg_conf    = sum(r[2] for r in results) / len(results)

    if _EC_PLATE_FMT.match(joined_text):
        plate_text, confidence = joined_text, avg_conf
    elif _EC_PLATE_FMT.match(best_text):
        plate_text, confidence = best_text, best_conf
    else:
        if best_conf >= avg_conf:
            plate_text, confidence = best_text, best_conf
        else:
            plate_text, confidence = joined_text, avg_conf

    if not plate_text:
        return None

    return {
        "plate":      plate_text,
        "confidence": round(float(confidence), 4)
    }