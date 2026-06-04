import os
import json
import time
import numpy as np
import cv2
import PIL.Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "Eres un experto en visión artificial vehicular. Tu única función es devolver un objeto JSON puro."

USER_PROMPT = """Analiza la imagen de la placa y clasifica estos 3 atributos:
1. oclusion: ¿Hay algo físico tapando la placa? ("No", "Parcial", "Severa")
2. reflejo: ¿Hay destellos de luz que impidan ver? ("No", "Sí")
3. sucia: ¿La placa tiene lodo, polvo o está deteriorada? ("No", "Sí")

Responde estrictamente con este formato JSON:
{"oclusion": "No", "reflejo": "No", "sucia": "No"}"""

BATCH_SYSTEM_PROMPT = "Eres un experto en visión artificial vehicular. Tu única función es devolver un arreglo JSON puro."

def _get_batch_prompt(num_images: int) -> str:
    return f"""Analiza las {num_images} imágenes de placas proporcionadas (enviadas en orden del 1 al {num_images}).
Para CADA imagen, clasifica estos 3 atributos:
1. oclusion: ¿Hay algo físico tapando la placa? ("No", "Parcial", "Severa")
2. reflejo: ¿Hay destellos de luz que impidan ver? ("No", "Sí")
3. sucia: ¿La placa tiene lodo, polvo o está deteriorada? ("No", "Sí")

Responde estrictamente con un ARREGLO JSON que contenga exactamente {num_images} objetos en el mismo orden que las imágenes recibidas.
Ejemplo de formato:
[
  {{"oclusion": "No", "reflejo": "No", "sucia": "No"}},
  {{"oclusion": "Parcial", "reflejo": "Sí", "sucia": "No"}}
]"""


# ── Helper: parseo seguro de JSON ──────────────────────────────────────────────
def _parse_json(raw_text: str):
    """Limpia bloques ```json ... ``` y parsea el JSON."""
    text = raw_text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
        text = text.split("```")[0].strip()
    return json.loads(text)


# ── Helper: decide si el error es reintentable ─────────────────────────────────
def _is_retryable(error_msg: str) -> bool:
    """503 (demanda alta) y 429 (cuota) son transitorios y vale la pena reintentar."""
    return "503" in error_msg or "429" in error_msg


# ── Helper: cliente Gemini ─────────────────────────────────────────────────────
def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("No se encontró GEMINI_API_KEY en el entorno.")
    return genai.Client(api_key=api_key)


# ── Clasificación individual ───────────────────────────────────────────────────
def classify_plate(
    crop: np.ndarray,
    ocr_confidence: float = 0.0,
    retries: int = 3,
    _delay: float = 5.0,
) -> dict:
    """
    Clasifica la calidad de una placa con Gemini 2.5 Flash.
    Reintenta hasta `retries` veces ante errores 503 / 429 con backoff exponencial.
    """
    legible = "Legible" if ocr_confidence >= 0.10 else "Ilegible"
    default_response = {
        "legible":  legible,
        "oclusion": "No",
        "reflejo":  "No",
        "sucia":    "No",
    }

    try:
        client  = _get_client()
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb_img)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_img, f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"],
        )

        result = _parse_json(response.text)
        return {
            "legible":  legible,
            "oclusion": result.get("oclusion", "No"),
            "reflejo":  result.get("reflejo",  "No"),
            "sucia":    result.get("sucia",     "No"),
        }

    except EnvironmentError as e:
        print(f"[plate_classifier] {e}")
        return default_response

    except Exception as e:
        error_msg = str(e)

        if "403" in error_msg:
            print("[plate_classifier] CRÍTICO: API Key bloqueada (Leaked). Cámbiala en .env")
            return default_response

        if _is_retryable(error_msg) and retries > 0:
            code = "503" if "503" in error_msg else "429"
            print(
                f"[plate_classifier] Error {code} — reintentando en {_delay:.0f}s "
                f"({retries} intento/s restante/s)..."
            )
            time.sleep(_delay)
            return classify_plate(crop, ocr_confidence, retries - 1, _delay * 2)

        print(f"[plate_classifier] Error en clasificación: {error_msg}")
        return default_response


# ── Clasificación por lote ─────────────────────────────────────────────────────
def classify_plates_batch(
    crops: list[np.ndarray],
    ocr_confidences: list[float],
    retries: int = 3,
    _delay: float = 5.0,
) -> list[dict]:
    """
    Clasifica un lote de placas en una sola llamada a Gemini.
    Reintenta hasta `retries` veces ante errores 503 / 429 con backoff exponencial.
    """
    num_images = len(crops)
    if num_images == 0:
        return []

    legibilities = [
        "Legible" if conf >= 0.10 else "Ilegible"
        for conf in ocr_confidences
    ]
    default_responses = [
        {"legible": leg, "oclusion": "No", "reflejo": "No", "sucia": "No"}
        for leg in legibilities
    ]

    try:
        client = _get_client()

        pil_images = [
            PIL.Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for crop in crops
        ]

        prompt_text = f"{BATCH_SYSTEM_PROMPT}\n\n{_get_batch_prompt(num_images)}"
        contents    = pil_images + [prompt_text]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        results = _parse_json(response.text)

        if not isinstance(results, list):
            print("[plate_classifier_batch] Error: el modelo no devolvió una lista JSON.")
            return default_responses

        # Rellenar si Gemini devuelve menos elementos de los esperados
        while len(results) < num_images:
            results.append({"oclusion": "No", "reflejo": "No", "sucia": "No"})

        return [
            {
                "legible":  legibilities[i],
                "oclusion": results[i].get("oclusion", "No"),
                "reflejo":  results[i].get("reflejo",  "No"),
                "sucia":    results[i].get("sucia",     "No"),
            }
            for i in range(num_images)
        ]

    except EnvironmentError as e:
        print(f"[plate_classifier_batch] {e}")
        return default_responses

    except Exception as e:
        error_msg = str(e)

        if "403" in error_msg:
            print("[plate_classifier_batch] CRÍTICO: API Key bloqueada (Leaked). Cámbiala en .env")
            return default_responses

        if _is_retryable(error_msg) and retries > 0:
            code = "503" if "503" in error_msg else "429"
            print(
                f"[plate_classifier_batch] Error {code} — reintentando lote en {_delay:.0f}s "
                f"({retries} intento/s restante/s)..."
            )
            time.sleep(_delay)
            return classify_plates_batch(crops, ocr_confidences, retries - 1, _delay * 2)

        print(f"[plate_classifier_batch] Error en clasificación lote: {error_msg}")
        return default_responses