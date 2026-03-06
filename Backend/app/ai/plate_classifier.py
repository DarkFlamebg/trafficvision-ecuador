# Clasifica atributos de calidad de una placa usando Claude Vision API

import anthropic
import base64
import json
import numpy as np
import cv2
import os


SYSTEM_PROMPT = """Eres un sistema experto en análisis de calidad de imágenes de placas vehiculares.
Recibirás el recorte de una placa y debes clasificar exactamente 3 atributos visuales.
Responde ÚNICAMENTE con un JSON válido, sin texto adicional, sin markdown, sin explicaciones."""

USER_PROMPT = """Analiza esta imagen de placa vehicular y clasifica SOLO estos 3 atributos visuales.
La legibilidad ya fue evaluada por OCR, NO la incluyas.

1. oclusion: ¿Hay objetos físicos tapando parte de la placa (mano, sticker, objeto)?
   - "No"      → placa completamente visible
   - "Parcial" → menos del 50% tapado
   - "Severa"  → más del 50% tapado

2. reflejo: ¿Hay reflejos de luz o destellos sobre los caracteres?
   - "No" → sin reflejos que interfieran
   - "Sí" → hay reflejos visibles

3. sucia: ¿La placa tiene suciedad, barro, óxido o daño físico visible?
   - "No" → placa limpia
   - "Sí" → suciedad o daño visible

Responde SOLO con este JSON (sin markdown):
{
  "oclusion": "No" | "Parcial" | "Severa",
  "reflejo": "No" | "Sí",
  "sucia": "No" | "Sí"
}"""


def _upscale_if_small(crop: np.ndarray, min_w: int = 200, min_h: int = 80) -> np.ndarray:
    h, w = crop.shape[:2]
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return crop


def _sharpen(crop: np.ndarray) -> np.ndarray:
    kernel = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(crop, -1, kernel)


def _encode_crop(crop: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.standard_b64encode(buffer.tobytes()).decode("utf-8")


def classify_plate(crop: np.ndarray, ocr_confidence: float = 0.0) -> dict:
    """
    Clasifica atributos de calidad de un recorte de placa.
    Returns:
        dict con claves: legible, oclusion, reflejo, sucia
    """
    # Legibilidad basada en OCR
    legible = "Legible" if ocr_confidence >= 0.5 else "Ilegible"

    default = {
        "legible":  legible,
        "oclusion": "No",
        "reflejo":  "No",
        "sucia":    "No",
    }

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[plate_classifier] ANTHROPIC_API_KEY no encontrada en .env")
        return default

    try:
        processed = _upscale_if_small(crop, min_w=200, min_h=80)
        processed = _sharpen(processed)

        print(f"[plate_classifier] OCR confidence: {ocr_confidence:.2f} → legible: {legible}")
        print(f"[plate_classifier] Tamaño: {crop.shape[1]}x{crop.shape[0]} → {processed.shape[1]}x{processed.shape[0]}")

        # Claude evalúa solo oclusión, reflejo, suciedad
        client    = anthropic.Anthropic(api_key=api_key)
        image_b64 = _encode_crop(processed)

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       image_b64,
                        },
                    },
                    { "type": "text", "text": USER_PROMPT },
                ],
            }],
        )

        raw    = message.content[0].text.strip()
        result = json.loads(raw)

        # Combinar: legibilidad por OCR + visual por Claude
        return {
            "legible":  legible,
            "oclusion": result.get("oclusion", "No"),
            "reflejo":  result.get("reflejo",  "No"),
            "sucia":    result.get("sucia",     "No"),
        }

    except json.JSONDecodeError as e:
        print(f"[plate_classifier] Error parseando JSON: {e}")
        return default
    except Exception as e:
        print(f"[plate_classifier] Error: {e}")
        return default