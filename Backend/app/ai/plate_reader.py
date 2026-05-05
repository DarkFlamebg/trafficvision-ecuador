# app/ai/plate_reader.py
# Reconoce caracteres de placa usando EasyOCR + Super-Resolución (FSRCNN x4)

import os
import re
import cv2
import numpy as np
import easyocr

# ── Rutas SR ───────────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))
_SR_DIR      = os.path.join(_ROOT_DIR, "ml", "sr_models")
_FSRCNN_PATH = os.path.join(_SR_DIR, "FSRCNN_x4.pb")
_ESPCN_PATH  = os.path.join(_SR_DIR, "ESPCN_x4.pb")

# ── Constantes ─────────────────────────────────────────────────────────────────

# Placas ecuatorianas: 3 letras + 3 o 4 dígitos
# Ej: GTT-2178 (4 dígitos), PFJ-048 (3 dígitos — motos/especiales)
_EC_RAW = re.compile(r'[A-Z]{3}\d{3,4}')
_EC_FMT = re.compile(r'^[A-Z]{3}-\d{3,4}$')

MIN_CROP_W      = 60     # px — descartar crops inútiles
MIN_CROP_H      = 20     # px
SR_THRESHOLD    = 150    # px — aplicar SR si ancho < este valor
OCR_MIN_CONF    = 0.10   # confianza mínima — bajo esto → None ("No detectado")

INNER_MARGIN_X  = 4
INNER_MARGIN_Y  = 2

_reader = easyocr.Reader(['en'], gpu=False)

# ── Super-Resolución ───────────────────────────────────────────────────────────

_sr        = None
_sr_loaded = False


def _get_sr():
    global _sr, _sr_loaded
    if _sr_loaded:
        return _sr
    _sr_loaded = True
    for path, name in [(_FSRCNN_PATH, 'fsrcnn'), (_ESPCN_PATH, 'espcn')]:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(path)
                sr.setModel(name, 4)
                _sr = sr
                print(f"[plate_reader] SR cargado: {os.path.basename(path)}")
                return _sr
            except Exception as e:
                print(f"[plate_reader] SR fallido ({path}): {e}")
    print("[plate_reader] SR no disponible — usando escalado bicúbico")
    return None


def _upscale(image: np.ndarray) -> np.ndarray:
    """SR x4 si ancho < SR_THRESHOLD. Fallback bicúbico si no hay modelo."""
    h, w = image.shape[:2]
    if w >= SR_THRESHOLD:
        return image
    sr = _get_sr()
    if sr is not None:
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return sr.upsample(image)
        except Exception as e:
            print(f"[plate_reader] SR upsample error: {e}")
    return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


# ── Preprocesado ───────────────────────────────────────────────────────────────

def _trim_margins(image: np.ndarray) -> np.ndarray:
    """Elimina borde del marco metálico que genera caracteres espurios."""
    h, w = image.shape[:2]
    x1 = min(INNER_MARGIN_X, w // 6)
    y1 = min(INNER_MARGIN_Y, h // 6)
    x2 = max(w - INNER_MARGIN_X, w * 5 // 6)
    y2 = max(h - INNER_MARGIN_Y, h * 5 // 6)
    return image[y1:y2, x1:x2]


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Pipeline optimizado para crops de placa:
      1. SR x4 sobre imagen original (antes de cualquier filtro)
      2. Trim márgenes del marco
      3. Recorte franja superior 'ECUADOR'
      4. Escala mínima 300px ancho
      5. Gris → CLAHE suave → unsharp mask controlado
    """
    # 1. SR primero — sobre imagen sin modificar
    image = _upscale(image)

    # 2. Trim bordes del marco
    image = _trim_margins(image)
    h, w  = image.shape[:2]

    # 3. Recortar franja 'ECUADOR' superior
    if h > 100:
        crop_top = int(h * 0.22)
    elif h > 60:
        crop_top = int(h * 0.15)
    else:
        crop_top = int(h * 0.10)
    image = image[crop_top:, :]
    h, w  = image.shape[:2]

    # 4. Escala mínima
    if h < 80:
        scale = 120 / max(h, 1)
        image = cv2.resize(image, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)
    if w < 300:
        scale = 300 / max(w, 1)
        image = cv2.resize(image, (300, int(image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    # 5. Gris + CLAHE suave + unsharp controlado
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gray  = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Corrección OCR ─────────────────────────────────────────────────────────────

# Sustituciones en posición de LETRA (primeros 3 chars)
_LETTER_FIXES = str.maketrans('0158269', 'OISBZGG')

# Sustituciones en posición de DÍGITO (últimos 3-4 chars)
_DIGIT_FIXES = str.maketrans({
    'O': '0', 'I': '1', 'S': '5', 'B': '8',
    'Z': '2', 'G': '6', 'A': '4', 'T': '7',
    'E': '6', 'J': '1',
})

# Pares visuales entre letras — para corregir P↔F, H↔J, etc.
_VISUAL_PAIRS = [
    ('P', 'F'), ('F', 'P'),
    ('H', 'J'), ('J', 'H'),   # PFH-2048 → PFJ-2048 (antes que H↔N)
    ('H', 'N'), ('N', 'H'),
    ('U', 'V'), ('V', 'U'),
    ('C', 'G'), ('G', 'C'),
    ('I', 'J'), ('J', 'I'),
]


def _try_fix(candidate: str) -> str | None:
    """
    Intenta corregir un candidato de 6 o 7 chars al formato ABC[D]DDD.
    Retorna la placa corregida o None si no es posible.
    """
    # Separar en letras (3) + dígitos (3 o 4)
    for n_digits in (4, 3):
        if len(candidate) < 3 + n_digits:
            continue
        letters = candidate[:3].translate(_LETTER_FIXES)
        digits  = candidate[3:3 + n_digits].translate(_DIGIT_FIXES)
        fixed   = letters + digits
        if _EC_RAW.fullmatch(fixed):
            return fixed

    return None


def _clean_and_fix(text: str) -> str:
    """
    Limpia el texto OCR y extrae el patrón ABC(D)DDD.

    Estrategia:
      1. Limpiar caracteres no alfanuméricos
      2. Búsqueda directa del patrón
      3. Corrección posicional por ventana (letra/dígito fixes)
      4. Corrección por pares visuales si aún no hay match
    """
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not clean:
        return ''

    # 1. Búsqueda directa — si el texto ya tiene formato válido, aplicar
    #    fixes posicionales y corrección H→J solo en posición 3 (tercera letra)
    #    EasyOCR confunde J con H sistemáticamente en fuentes de placa
    m = _EC_RAW.search(clean)
    if m:
        base    = m.group()
        letters = list(base[:3].translate(_LETTER_FIXES))
        digits  = base[3:].translate(_DIGIT_FIXES)
        # H→J solo en posición 2 (índice 2, tercera letra)
        # Posiciones 0 y 1 se dejan intactas para no alterar letras válidas
        if letters[2] == 'H':
            letters[2] = 'J'
        return ''.join(letters) + digits

    # 2. Corrección posicional por ventana — para texto con chars mezclados
    for length in (7, 6):
        for start in range(len(clean) - length + 1):
            result = _try_fix(clean[start:start + length])
            if result:
                return result

    # 3. Pares visuales — último recurso, solo si pasos 1 y 2 fallaron
    #    En este punto el texto no matchea el patrón con ninguna corrección
    #    posicional, así que vale la pena probar confusiones entre letras
    for length in (7, 6):
        for start in range(len(clean) - length + 1):
            candidate = clean[start:start + length]
            letters   = candidate[:3].translate(_LETTER_FIXES)
            digits    = candidate[3:].translate(_DIGIT_FIXES)
            for pos in range(3):
                for orig, repl in _VISUAL_PAIRS:
                    if letters[pos] == orig:
                        variant = letters[:pos] + repl + letters[pos+1:]
                        fixed   = variant + digits
                        if _EC_RAW.fullmatch(fixed):
                            return fixed

    return clean


def _format_plate(raw: str) -> str:
    """'GTT2178' → 'GTT-2178', 'PFJ048' → 'PFJ-048'."""
    if _EC_RAW.fullmatch(raw):
        return raw[:3] + '-' + raw[3:]
    return raw


# ── OCR runner ────────────────────────────────────────────────────────────────

def _run_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    raw = _reader.readtext(
        image,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False,
    )
    return [(text, float(conf)) for (_, text, conf) in raw if text]


# ── Lector principal ───────────────────────────────────────────────────────────

def read_plate(image: np.ndarray) -> dict | None:
    """
    Extrae el texto de la placa a partir de un recorte BGR.

    Flujo:
      1. Validar tamaño mínimo (descarta <60×20 px)
      2. Preproceso (SR + CLAHE suave + unsharp controlado)
      3. OCR sobre imagen procesada
      4. Fallback a imagen original si no hay resultado
      5. Priorizar candidatos con formato ecuatoriano válido
      6. Descartar si confianza < OCR_MIN_CONF → None ("No detectado")

    Returns:
        {'plate': 'GTT-2178', 'confidence': 0.87} o None
    """
    h, w = image.shape[:2]
    if w < MIN_CROP_W or h < MIN_CROP_H:
        return None

    # OCR sobre imagen preprocesada
    processed = _preprocess(image)
    results   = _run_ocr(processed)

    # Fallback a imagen original
    if not results:
        results = _run_ocr(image)

    if not results:
        return None

    # Construir candidatos
    candidates = []
    for text, conf in results:
        fixed     = _clean_and_fix(text)
        formatted = _format_plate(fixed)
        candidates.append({
            'plate':      formatted,
            'confidence': round(conf, 4),
            'valid':      bool(_EC_FMT.match(formatted)),
        })

    # Priorizar válidas, luego mayor confianza
    valid_ones = [c for c in candidates if c['valid']]
    best = max(valid_ones or candidates, key=lambda c: c['confidence'])

    # Descartar ruido — confianza muy baja = texto inventado
    if best['confidence'] < OCR_MIN_CONF:
        return None

    return {
        'plate':      best['plate'],
        'confidence': best['confidence'],
    }