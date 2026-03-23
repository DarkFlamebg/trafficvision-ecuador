import os
import json
import numpy as np
import cv2
import PIL.Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

SYSTEM_PROMPT = "Eres un experto en visión artificial vehicular. Responde solo en JSON."
USER_PROMPT = """Analiza la placa y clasifica: 
1. oclusion (No, Parcial, Severa)
2. reflejo (No, Sí)
3. sucia (No, Sí)
Responde solo JSON: {"oclusion": "No", "reflejo": "No", "sucia": "No"}"""

def classify_plate(crop: np.ndarray, ocr_confidence: float = 0.0) -> dict:
    legible = "Legible" if ocr_confidence >= 0.10 else "Ilegible"
    default = {"legible": legible, "oclusion": "No", "reflejo": "No", "sucia": "No"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return default

    try:
        # 1. Preparar imagen
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb_img)

        # 2. Cliente forzando la versión estable 'v1'
        # Esto es lo que evita que busque en 'v1beta' y de el error 404
        client = genai.Client(
            api_key=api_key, 
            http_options={'api_version': 'v1'}
        )

        # 3. Llamada usando el nombre EXACTO de tu lista
        # Probamos con el 2.0 que es el más potente de tu lista
        response = client.models.generate_content(
            model='models/gemini-1.5-flash',
            contents=[pil_img, f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"]
        )
        
        # 4. Limpieza de texto
        raw_text = response.text.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1].replace("json", "").strip()
            raw_text = raw_text.split("```")[0].strip()

        result = json.loads(raw_text)

        return {
            "legible":  legible,
            "oclusion": result.get("oclusion", "No"),
            "reflejo":  result.get("reflejo",  "No"),
            "sucia":    result.get("sucia",    "No")
        }

    except Exception as e:
        # Si da 429 (Cuota), esperamos y devolvemos default
        print(f"[plate_classifier] Info: {e}")
        return default