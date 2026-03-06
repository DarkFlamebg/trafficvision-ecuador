# Detecta placas usando la API de Roboflow
import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from inference_sdk import InferenceHTTPClient

MODEL_ID             = "vehicle-registration-plates-trudk/2"
API_URL              = "https://serverless.roboflow.com"
CONFIDENCE_THRESHOLD = 0.45


def _get_client():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY no encontrada en .env")
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key)


def _load_image(input_image):
    """
    Carga la imagen respetando la orientación EXIF del celular.
    Pillow aplica la rotación automáticamente con ImageOps.exif_transpose().
    Retorna array NumPy BGR (compatible con OpenCV y EasyOCR).
    """
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    # Aplicar rotación EXIF automáticamente
    pil_img = ImageOps.exif_transpose(pil_img)

    # Convertir de vuelta a NumPy BGR para OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_plate(input_image):
    """
    Detecta placas vehiculares usando la API de Roboflow.
    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa
          - "bbox":       [x1, y1, x2, y2] en píxeles (int)
          - "confidence": float 0.0 – 1.0
        Lista vacía si no se detectó ninguna placa.
    """
    # Cargar con corrección EXIF
    image = _load_image(input_image)

    # Guardar imagen corregida para enviar a Roboflow
    tmp_path = "temp/_roboflow_tmp.jpg"
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite(tmp_path, image)

    client = _get_client()
    result  = client.infer(tmp_path, model_id=MODEL_ID)

    plates = []
    ih, iw = image.shape[:2]

    for pred in result.get("predictions", []):
        conf = pred["confidence"]

        if conf < CONFIDENCE_THRESHOLD:
            continue

        cx, cy = pred["x"], pred["y"]
        w,  h  = pred["width"], pred["height"]

        x1 = max(0,  int(cx - w / 2))
        y1 = max(0,  int(cy - h / 2))
        x2 = min(iw, int(cx + w / 2))
        y2 = min(ih, int(cy + h / 2))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(float(conf), 4)
        })

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return plates