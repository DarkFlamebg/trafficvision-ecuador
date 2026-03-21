# Punto de entrada de la API — TrafficVision Backend

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid

from app.ai.plate_detector   import detect_plate
from app.ai.plate_reader     import read_plate
from app.ai.plate_classifier import classify_plate
from app.routes.detect       import router as detect_router
from dotenv import load_dotenv

load_dotenv()

# Ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("temp", exist_ok=True)
    yield
    for f in os.listdir("temp"):
        os.remove(os.path.join("temp", f))

# Aplicación
app = FastAPI(
    title="TrafficVision API",
    description="API de detección, lectura y clasificación de placas vehiculares",
    version="1.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(detect_router, prefix="/api/v1", tags=["Detección"])

# Endpoints
@app.get("/")
def root():
    return {
        "message": "TrafficVision API activa",
        "docs":    "/docs",
        "version": "1.1.0"
    }


@app.post("/detect-plate")
async def detect_plate_api(file: UploadFile):
    """
    Detecta placas con YOLOv8, lee con EasyOCR y clasifica calidad con Google Gemini 2.0.
    La legibilidad se determina por la confianza del OCR local.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join("temp", filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Detección Local (YOLOv8)
        plates = detect_plate(path)
        results = []

        for plate in plates:
            # 2. Lectura Local (EasyOCR)
            ocr = read_plate(plate["image"])
            ocr_text = ocr["plate"] if ocr else "No detectado"
            ocr_confidence = ocr["confidence"] if ocr else 0.0

            # 3. Clasificación con IA (Google Gemini)
            # Pasamos ocr_confidence para que classify_plate determine si es 'Legible'
            labels = classify_plate(plate["image"], ocr_confidence=ocr_confidence)

            # 4. Consolidación de Resultados
            results.append({
                "bbox":            plate["bbox"],
                "yolo_confidence": plate["confidence"],
                "plate":           ocr_text,
                "ocr_confidence":  ocr_confidence,
                "labels":          labels # Ya viene con legible, oclusion, reflejo y sucia
            })

        return {
            "total": len(results), 
            "status": "success",
            "plates": results
        }

    except Exception as e:
        print(f"[main] Error crítico: {e}")
        raise HTTPException(status_code=500, detail="Error interno en el procesamiento de la imagen")

    finally:
        if os.path.exists(path):
            os.remove(path)