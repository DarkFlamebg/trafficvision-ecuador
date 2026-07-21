# app/routes/detection.py
# Endpoints HTTP de detección de placas y vehículos.
# Versión: /api/v1/detection/...
#
# Endpoints:
#   POST /api/v1/detection/plate    — pipeline completo (vehículo + placa + OCR + calidad)
#   POST /api/v1/detection/vehicle  — solo detección de vehículos

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import ALLOWED_IMAGE_TYPES
from app.core.utils  import save_upload_to_temp, cleanup_temp
from app.services.detection_service   import run_plate_pipeline, run_vehicle_detection
from app.services.persistence_service import save_detections_to_db
from app.database.connection import SessionLocal

router = APIRouter(prefix="/detection", tags=["Detección v1"])


@router.post("/plate", summary="Pipeline completo: vehículo → placa → OCR → calidad")
async def detect_plate_endpoint(file: UploadFile = File(...)):
    """
    Pipeline completo de detección:
    1. Detecta vehículos en la imagen (YOLOv8n COCO)
    2. En cada vehículo detectado, busca su placa (YOLOv11n entrenado)
    3. Lee la placa con OCR (EasyOCR)
    4. Clasifica calidad de la placa (Gemini Flash Vision)
    5. Persiste resultados en PostgreSQL

    Si no se detectan vehículos, busca placas en la imagen completa.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    path = save_upload_to_temp(file)
    try:
        plates_found, num_vehicles = run_plate_pipeline(path)

        # ── Persistencia en BD ────────────────────────────────────────────────
        db = SessionLocal()
        try:
            save_detections_to_db(db, plates_found, path)
            db.commit()
            print(f"[BD] {len(plates_found)} placa(s) guardada(s) en Supabase.")
        except Exception as e:
            print(f"[BD] Error al guardar: {e}")
            db.rollback()
        finally:
            db.close()
        # ─────────────────────────────────────────────────────────────────────

        return {
            "total":    len(plates_found),
            "vehicles": num_vehicles,
            "plates":   plates_found,
        }
    finally:
        cleanup_temp(path)


@router.post("/vehicle", summary="Detecta vehículos sin procesar placas")
async def detect_vehicle_endpoint(file: UploadFile = File(...)):
    """
    Detecta únicamente vehículos en la imagen usando YOLOv8n COCO.
    No ejecuta OCR ni clasificación de placas.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    path = save_upload_to_temp(file)
    try:
        vehicles = run_vehicle_detection(path)
        return {
            "total":    len(vehicles),
            "vehicles": vehicles,
        }
    finally:
        cleanup_temp(path)
