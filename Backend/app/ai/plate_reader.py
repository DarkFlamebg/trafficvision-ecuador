# Detecta la placa en la imagen usando YOLOv8 personalizado
# Reconoce los caracteres de la placa usando EasyOCR
import re
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

_EC_PLATE     = re.compile(r'^[A-Z]{3}\d{4}$')
_EC_PLATE_FMT = re.compile(r'^[A-Z]{3}-\d{4}$')


def _preprocess(image):
    """Escala el recorte si es muy pequeño."""
    h, w = image.shape[:2]
    if h < 80:
        scale = 80 / h
        image = cv2.resize(image, (int(w * scale), 80),
                           interpolation=cv2.INTER_CUBIC)
    return image


def _fix_ocr_errors(clean):
    """
    Corrige errores comunes de OCR según la posición del carácter.
    Letras (pos 0-2): números que parecen letras
    Números (pos 3-6): letras que parecen números
    """
    if len(clean) != 7:
        return clean

    letters = (
        clean[:3]
        .replace('0', 'O')
        .replace('1', 'I')
        .replace('5', 'S')
        .replace('8', 'B')
        .replace('2', 'Z')
        .replace('6', 'G')
    )
    numbers = (
        clean[3:]
        .replace('O', '0')
        .replace('I', '1')
        .replace('S', '5')
        .replace('B', '8')
        .replace('Z', '2')
        .replace('G', '6')
    )
    return letters + numbers


def _format_plate(text):
    """Limpia y formatea como placa ecuatoriana AAA-0000."""
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    clean = _fix_ocr_errors(clean)

    if _EC_PLATE.match(clean):
        return clean[:3] + '-' + clean[3:]
    return clean


def _join_results(results):
    """
    Une todos los textos detectados en una sola cadena.
    Útil cuando EasyOCR divide la placa en varios bloques.
    """
    texts = [r[1] for r in sorted(results, key=lambda x: x[0][0][0])]
    return ''.join(texts)


def read_plate(image):
    """
    Extrae el texto de la placa a partir de un recorte de imagen.

    Args:
        image: array NumPy BGR con el recorte de la placa

    Returns:
        Dict con:
          - "plate":      texto formateado (ej: "GTP-6447")
          - "confidence": confianza del OCR (float 0-1)
        o None si no se detectó texto.
    """
    processed = _preprocess(image)

    results = reader.readtext(
        processed,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False
    )

    if not results:
        results = reader.readtext(
            image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=1,
            paragraph=False
        )

    if not results:
        return None

    # Estrategia 1: mejor resultado individual
    best        = max(results, key=lambda x: x[2])
    best_text   = _format_plate(best[1])
    best_conf   = best[2]

    # Estrategia 2: unir todos los bloques detectados
    joined_text = _format_plate(_join_results(results))
    avg_conf    = sum(r[2] for r in results) / len(results)

    # Preferir el resultado que sea una placa ecuatoriana válida
    if _EC_PLATE_FMT.match(joined_text):
        plate_text = joined_text
        confidence = avg_conf
    elif _EC_PLATE_FMT.match(best_text):
        plate_text = best_text
        confidence = best_conf
    else:
        # Ninguno es placa válida — devolver el de mayor confianza
        plate_text = best_text if best_conf >= avg_conf else joined_text
        confidence = max(best_conf, avg_conf)

    if not plate_text:
        return None

    return {
        "plate":      plate_text,
        "confidence": round(float(confidence), 4)
    }