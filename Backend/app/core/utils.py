# app/core/utils.py
# Funciones utilitarias compartidas entre routers y servicios.
# Centraliza los helpers de imagen, archivos temporales y codificación.

import os
import uuid
import shutil

import cv2
import numpy as np
from fastapi import UploadFile, HTTPException

from app.core.config import TEMP_DIR, ALLOWED_IMAGE_TYPES


# ── Imagen ─────────────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convierte un array NumPy BGR a string base64 JPEG."""
    import base64
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def decode_upload_to_array(contents: bytes) -> np.ndarray:
    """
    Decodifica los bytes de un UploadFile a un array NumPy BGR.
    Lanza HTTPException 422 si la imagen no puede decodificarse.
    """
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")
    return image


# ── Archivos temporales ────────────────────────────────────────────────────────

def save_upload_to_temp(file: UploadFile, ext: str = "jpg") -> str:
    """
    Guarda un UploadFile en el directorio temporal con nombre UUID.
    Retorna la ruta absoluta del archivo guardado.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}.{ext}"
    path     = os.path.join(TEMP_DIR, filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path


def cleanup_temp(path: str) -> None:
    """Elimina un archivo temporal silenciosamente."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def validate_image_type(file: UploadFile) -> None:
    """Lanza HTTPException 400 si el Content-Type no es una imagen permitida."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan imágenes JPG o PNG"
        )
