# Punto de entrada de la API — TrafficVision Backend
# Pipeline: Detección vehículo → Detección placa → OCR → Clasificación calidad

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid

from app.ai.vehicle_detector import detect_vehicles
from app.ai.plate_detector   import detect_plate
from app.ai.plate_reader     import read_plate
from app.ai.plate_classifier import classify_plate
from app.routes.detect       import router as detect_router
from dotenv import load_dotenv

load_dotenv()

# ── Ciclo de vida ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("temp", exist_ok=True)
    yield
    for f in os.listdir("temp"):
        os.remove(os.path.join("temp", f))

# ── Aplicación ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrafficVision API",
    description="Detección de vehículos y placas vehiculares con YOLOv8 + EasyOCR + Gemini",
    version="1.2.0",
    lifespan=lifespan
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(detect_router, prefix="/api/v1", tags=["Detección"])

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "TrafficVision API activa",
        "docs":    "/docs",
        "version": "1.2.0"
    }


@app.post("/detect-plate")
async def detect_plate_api(file: UploadFile):
    """
    Pipeline completo:
    1. Detecta vehículos en la imagen (YOLOv8n COCO)
    2. En cada vehículo detectado, busca su placa (YOLOv8n entrenado)
    3. Lee la placa con OCR (EasyOCR)
    4. Clasifica calidad de la placa (Gemini Flash Vision)

    Si no se detectan vehículos, busca placas en la imagen completa.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    filename = f"{uuid.uuid4()}.jpg"
    path     = os.path.join("temp", filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ── Etapa 1: Detectar vehículos ────────────────────────────────────
        vehicles = detect_vehicles(path)

        results      = []
        plates_found = []

        if vehicles:
            # ── Etapa 2a: Buscar placa dentro de cada vehículo ────────────
            for vehicle in vehicles:
                plates_in_vehicle = detect_plate(vehicle["image"])

                for plate in plates_in_vehicle:
                    ocr            = read_plate(plate["image"])
                    ocr_text       = ocr["plate"]      if ocr else "No detectado"
                    ocr_confidence = ocr["confidence"] if ocr else 0.0
                    labels         = classify_plate(plate["image"], ocr_confidence)

                    # Ajustar bbox de la placa al sistema de coordenadas original
                    vx1, vy1 = vehicle["bbox"][0], vehicle["bbox"][1]
                    abs_bbox = [
                        plate["bbox"][0] + vx1,
                        plate["bbox"][1] + vy1,
                        plate["bbox"][2] + vx1,
                        plate["bbox"][3] + vy1,
                    ]

                    plates_found.append({
                        "bbox":            abs_bbox,
                        "yolo_confidence": plate["confidence"],
                        "plate":           ocr_text,
                        "ocr_confidence":  ocr_confidence,
                        "labels":          labels,
                        "vehicle": {
                            "type":       vehicle["type"],
                            "type_es":    vehicle["type_es"],
                            "bbox":       vehicle["bbox"],
                            "confidence": vehicle["confidence"],
                        }
                    })
        else:
            # ── Etapa 2b: Sin vehículo detectado → buscar placa en imagen completa
            print("[main] No se detectaron vehículos — buscando placa en imagen completa")
            plates_in_image = detect_plate(path)

            for plate in plates_in_image:
                ocr            = read_plate(plate["image"])
                ocr_text       = ocr["plate"]      if ocr else "No detectado"
                ocr_confidence = ocr["confidence"] if ocr else 0.0
                labels         = classify_plate(plate["image"], ocr_confidence)

                plates_found.append({
                    "bbox":            plate["bbox"],
                    "yolo_confidence": plate["confidence"],
                    "plate":           ocr_text,
                    "ocr_confidence":  ocr_confidence,
                    "labels":          labels,
                    "vehicle":         None   # placa sin vehículo asociado
                })

        return {
            "total":    len(plates_found),
            "vehicles": len(vehicles),
            "plates":   plates_found,
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/detect-vehicle")
async def detect_vehicle_only(file: UploadFile):
    """
    Detecta solo vehículos sin procesar placas.
    Útil para contar vehículos o clasificar tipo de tráfico.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    filename = f"{uuid.uuid4()}.jpg"
    path     = os.path.join("temp", filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vehicles = detect_vehicles(path)

        return {
            "total": len(vehicles),
            "vehicles": [
                {
                    "type":       v["type"],
                    "type_es":    v["type_es"],
                    "bbox":       v["bbox"],
                    "confidence": v["confidence"],
                }
                for v in vehicles
            ]
        }

    finally:
        if os.path.exists(path):
            os.remove(path)