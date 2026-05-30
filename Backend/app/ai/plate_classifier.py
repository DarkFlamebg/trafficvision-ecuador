import os
import json
import time
import numpy as np
import cv2
import PIL.Image
from dotenv import load_dotenv
from google import genai

# Cargar variables de entorno (Asegúrate de tener la nueva GEMINI_API_KEY en tu .env)
load_dotenv()

# Configuración de Prompts (Consistentes con tu JSON de respuesta)
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

def classify_plate(crop: np.ndarray, ocr_confidence: float = 0.0, retries: int = 1) -> dict:
    """
    Clasifica la calidad de la placa usando Google Gemini 2.0 Flash.
    Incluye lógica de reintento para manejar el error 429 (Límite de cuota).
    """
    # Determinamos legibilidad basada en el EasyOCR local
    legible = "Legible" if ocr_confidence >= 0.10 else "Ilegible"
    
    # Respuesta por defecto en caso de error de red o API
    default_response = {
        "legible": legible,
        "oclusion": "No",
        "reflejo": "No",
        "sucia": "No"
    }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[plate_classifier] Error: No se encontró GEMINI_API_KEY en el entorno.")
        return default_response

    try:
        # 1. Preparación de imagen (BGR a RGB para correcta detección de suciedad/color)
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb_img)

        # 2. Inicialización del cliente (Configuración simplificada para evitar Error 400/404)
        client = genai.Client(api_key=api_key)

        # 3. Petición al modelo Multimodal
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_img, f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"]
        )
        
        # 4. Limpieza de la respuesta (Gemini a veces rodea el JSON con ```json ... ```)
        raw_text = response.text.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].replace("json", "").strip()
            raw_text = raw_text.split("```")[0].strip()

        # 5. Parseo del JSON
        result = json.loads(raw_text)

        return {
            "legible":  legible,
            "oclusion": result.get("oclusion", "No"),
            "reflejo":  result.get("reflejo",  "No"),
            "sucia":    result.get("sucia",     "No")
        }

    except Exception as e:
        error_msg = str(e)
        
        # Manejo de Límite de Cuota (429): Esperar y reintentar una vez
        if "429" in error_msg and retries > 0:
            print(f"[plate_classifier] Cuota excedida. Esperando 10 segundos para reintentar...")
            time.sleep(10)
            return classify_plate(crop, ocr_confidence, retries - 1)
        
        # Manejo de Error de Seguridad (403): La llave se filtró
        if "403" in error_msg:
            print("[plate_classifier] CRÍTICO: Tu API Key ha sido bloqueada (Leaked). Cámbiala en .env")
        
        print(f"[plate_classifier] Error en clasificación: {error_msg}")
        return default_response

def classify_plates_batch(crops: list[np.ndarray], ocr_confidences: list[float], retries: int = 1) -> list[dict]:
    """
    Clasifica un lote de placas usando Gemini Flash para evitar límites de RPM.
    Retorna una lista de diccionarios en el mismo orden que 'crops'.
    """
    num_images = len(crops)
    if num_images == 0:
        return []

    # Determinar legibilidad para cada placa
    legibilities = ["Legible" if conf >= 0.10 else "Ilegible" for conf in ocr_confidences]
    
    # Respuestas por defecto
    default_responses = [
        {"legible": leg, "oclusion": "No", "reflejo": "No", "sucia": "No"} 
        for leg in legibilities
    ]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[plate_classifier_batch] Error: No se encontró GEMINI_API_KEY en el entorno.")
        return default_responses

    try:
        # Preparar todas las imágenes en formato PIL
        pil_images = []
        for crop in crops:
            rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(PIL.Image.fromarray(rgb_img))

        client = genai.Client(api_key=api_key)

        # Construir contents list: [img1, img2, ..., SYSTEM_PROMPT + BATCH_USER_PROMPT]
        prompt_text = f"{BATCH_SYSTEM_PROMPT}\n\n{_get_batch_prompt(num_images)}"
        contents = pil_images + [prompt_text]

        # Usar gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        raw_text = response.text.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].replace("json", "").strip()
            raw_text = raw_text.split("```")[0].strip()

        results = json.loads(raw_text)

        # Validar que nos devolvió la cantidad esperada
        if not isinstance(results, list):
            print("[plate_classifier_batch] Error: El modelo no devolvió una lista JSON.")
            return default_responses
        
        # Rellenar con defaults si nos devolvió menos (a veces pasa)
        while len(results) < num_images:
            results.append({"oclusion": "No", "reflejo": "No", "sucia": "No"})

        # Combinar con las legibilidades
        final_results = []
        for i in range(num_images):
            final_results.append({
                "legible":  legibilities[i],
                "oclusion": results[i].get("oclusion", "No"),
                "reflejo":  results[i].get("reflejo", "No"),
                "sucia":    results[i].get("sucia", "No")
            })
            
        return final_results

    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg and retries > 0:
            print(f"[plate_classifier_batch] Cuota excedida. Esperando 10 segundos para reintentar lote...")
            time.sleep(10)
            return classify_plates_batch(crops, ocr_confidences, retries - 1)
        
        if "403" in error_msg:
            print("[plate_classifier_batch] CRÍTICO: Tu API Key ha sido bloqueada (Leaked). Cámbiala en .env")
        
        print(f"[plate_classifier_batch] Error en clasificación lote: {error_msg}")
        return default_responses