# Ruta /detect — recibe imagen y retorna placas detectadas con texto OCR
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from app.ai.plate_detector import detect_plate
from app.ai.plate_reader import read_plate

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Recibe una imagen (jpg/png) y retorna las placas detectadas.

    Response:
        {
          "total": int,
          "results": [
            {
              "plate":            str,
              "ocr_confidence":   float,
              "yolo_confidence":  float,
              "bbox":             [x1, y1, x2, y2]
            }
          ]
        }
    """
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    contents = await file.read()

    # Decodificar bytes a array NumPy
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")

    # Detección YOLO
    plates = detect_plate(image)

    results = []

    for plate in plates:
        # OCR sobre el recorte de la placa
        ocr = read_plate(plate["image"])

        if ocr is None:
            continue  # descartar detecciones sin texto legible

        results.append({
            "plate":           ocr["plate"],
            "ocr_confidence":  ocr["confidence"],
            "yolo_confidence": plate["confidence"],
            "bbox":            plate["bbox"]
        })

    return {
        "total":   len(results),
        "results": results
    }