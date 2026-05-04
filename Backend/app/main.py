# main.py — TrafficVision Backend
# Pipeline: Detección vehículo → Detección placa (YOLO/RT-DETR/Ensemble) → OCR → Clasificación

from fastapi import FastAPI, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
from enum import Enum
import shutil
import os
import uuid
import base64
import asyncio
import cv2
import time
import numpy as np

from dotenv import load_dotenv

# ── Imports de módulos AI ──────────────────────────────────────────────────────
from app.ai.vehicle_detector import detect_vehicles
from app.ai.plate_detector import detect_plate as detect_plate_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr, detect_plate_ensemble
from app.ai.plate_reader import read_plate
from app.ai.plate_classifier import classify_plate
from app.ai.plate_detector_config import DetectorType, DetectorConfig

# ── Imports de routers ─────────────────────────────────────────────────────────
from app.routes.detect import router as detect_router
from app.routes.detect_multi import router as detect_multi_router

load_dotenv()

TEMP_DIR = "temp"


# ── Enums ──────────────────────────────────────────────────────────────────────
class DetectorTypeEnum(str, Enum):
    YOLO = "yolo"
    RTDETR = "rtdetr"
    ENSEMBLE = "ensemble"


# ── Ciclo de vida ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    print("🚀 TrafficVision API iniciada")
    print(f"📦 Detectores disponibles: {DetectorConfig.list_available_detectors()}")
    yield
    # Cleanup al cerrar
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except Exception:
            pass
    print("👋 TrafficVision API cerrada")


# ── Aplicación FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title="TrafficVision API",
    description="""
    Sistema de detección de vehículos y placas vehiculares.
    
    ## Detectores Disponibles
    
    - **YOLOv11n**: Detector rápido, ideal para tiempo real
    - **RT-DETR**: Detector transformer de alta precisión
    - **Ensemble**: Combinación de ambos con NMS para máxima robustez
    
    ## Pipeline de Detección
    
    1. Detección de vehículos (YOLOv8n COCO)
    2. Detección de placas (YOLO/RT-DETR/Ensemble)
    3. Lectura OCR (EasyOCR)
    4. Clasificación de calidad (Gemini Flash Vision)
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev
        "http://localhost:5173",      # Vite dev
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",       # Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Metrics", "X-Detections", "X-Processing-Time"],
)


# Tipos de archivo permitidos
ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/jpg")
ALLOWED_VIDEO_TYPES = ("video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime")


# ── Registrar Routers ──────────────────────────────────────────────────────────
app.include_router(detect_router, prefix="/api/v1", tags=["Detección Legacy"])
app.include_router(detect_multi_router, prefix="/api/v2", tags=["Detección Multi-Modelo"])


# ── Endpoints Raíz ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Sistema"])
def root():
    """Información del sistema y estado de salud."""
    return {
        "service": "TrafficVision API",
        "status": "active",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "legacy": "/api/v1/detect",
            "multi_model": "/api/v2/detect/full",
            "compare": "/api/v2/detect/compare",
            "quick": "/api/v2/detect/quick",
            "annotate": "/api/v2/detect/annotate",
            "models_info": "/api/v2/detect/models",
            "websocket_video": "/ws/detect-plate/video",
        },
        "detectors": DetectorConfig.list_available_detectors(),
        "default_detector": DetectorConfig.DEFAULT_DETECTOR.value,
    }


@app.get("/health", tags=["Sistema"])
def health_check():
    """Health check para monitoreo."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "detectors_available": len(DetectorConfig.list_available_detectors()),
    }


# ── Endpoint: Detección de Placa con Selector de Modelo ────────────────────────
@app.post("/detect-plate", tags=["Detección"])
async def detect_plate_api(
    file: UploadFile,
    detector: DetectorTypeEnum = Query(
        default=DetectorTypeEnum.ENSEMBLE,
        description="Modelo detector: yolo, rtdetr, ensemble"
    )
):
    """
    Pipeline completo de detección con selector de modelo.
    
    1. Detecta vehículos en la imagen (YOLOv8n COCO)
    2. En cada vehículo detectado, busca su placa usando el modelo seleccionado
    3. Lee la placa con OCR (EasyOCR)
    4. Clasifica calidad de la placa (Gemini Flash Vision)

    Si no se detectan vehículos, busca placas en la imagen completa.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_DIR, filename)
    t0 = time.time()

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Seleccionar función de detección
        if detector == DetectorTypeEnum.YOLO:
            detect_fn = detect_plate_yolo
            detector_name = "yolo"
        elif detector == DetectorTypeEnum.RTDETR:
            detect_fn = detect_plate_rtdetr
            detector_name = "rtdetr"
        else:
            detect_fn = detect_plate_ensemble
            detector_name = "ensemble"

        # ── Etapa 1: Detectar vehículos ────────────────────────────────────
        vehicles = detect_vehicles(path)

        plates_found = []

        if vehicles:
            # ── Etapa 2a: Buscar placa dentro de cada vehículo ────────────
            for vehicle in vehicles:
                plates_in_vehicle = detect_fn(vehicle["image"])

                for plate in plates_in_vehicle:
                    ocr = read_plate(plate["image"])
                    ocr_text = ocr["plate"] if ocr else "No detectado"
                    ocr_confidence = ocr["confidence"] if ocr else 0.0
                    labels = classify_plate(plate["image"], ocr_confidence)

                    # Ajustar bbox de la placa al sistema de coordenadas original
                    vx1, vy1 = vehicle["bbox"][0], vehicle["bbox"][1]
                    abs_bbox = [
                        plate["bbox"][0] + vx1,
                        plate["bbox"][1] + vy1,
                        plate["bbox"][2] + vx1,
                        plate["bbox"][3] + vy1,
                    ]

                    plates_found.append({
                        "bbox": abs_bbox,
                        "detector_confidence": plate["confidence"],
                        "detector": plate.get("detector", detector_name),
                        "plate": ocr_text,
                        "ocr_confidence": ocr_confidence,
                        "labels": labels,
                        "vehicle": {
                            "type": vehicle["type"],
                            "type_es": vehicle["type_es"],
                            "bbox": vehicle["bbox"],
                            "confidence": vehicle["confidence"],
                        }
                    })
        else:
            # ── Etapa 2b: Sin vehículo detectado → buscar placa en imagen completa
            print(f"[main] No se detectaron vehículos — buscando placa con {detector_name}")
            plates_in_image = detect_fn(path)

            for plate in plates_in_image:
                ocr = read_plate(plate["image"])
                ocr_text = ocr["plate"] if ocr else "No detectado"
                ocr_confidence = ocr["confidence"] if ocr else 0.0
                labels = classify_plate(plate["image"], ocr_confidence)

                plates_found.append({
                    "bbox": plate["bbox"],
                    "detector_confidence": plate["confidence"],
                    "detector": plate.get("detector", detector_name),
                    "plate": ocr_text,
                    "ocr_confidence": ocr_confidence,
                    "labels": labels,
                    "vehicle": None
                })

        processing_time = int((time.time() - t0) * 1000)

        return JSONResponse(
            content={
                "success": True,
                "total": len(plates_found),
                "vehicles": len(vehicles),
                "detector_used": detector_name,
                "processing_time_ms": processing_time,
                "plates": plates_found,
            },
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Detections": str(len(plates_found)),
            }
        )

    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Endpoint: Solo Detección de Vehículos ──────────────────────────────────────
@app.post("/detect-vehicle", tags=["Detección"])
async def detect_vehicle_only(file: UploadFile):
    """
    Detecta solo vehículos sin procesar placas.
    Útil para contar vehículos o clasificar tipo de tráfico.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(TEMP_DIR, filename)

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vehicles = detect_vehicles(path)

        return {
            "total": len(vehicles),
            "vehicles": [
                {
                    "type": v["type"],
                    "type_es": v["type_es"],
                    "bbox": v["bbox"],
                    "confidence": v["confidence"],
                }
                for v in vehicles
            ]
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


# ── WebSocket: Detección de Placas en Video ────────────────────────────────────
@app.websocket("/ws/detect-plate/video")
async def detect_plate_video_ws(
    websocket: WebSocket,
    detector: str = "ensemble"
):
    """
    WebSocket para detección de placas en video con selector de modelo.
    
    Query params:
        detector: yolo | rtdetr | ensemble (default: ensemble)
    
    Protocolo:
    1. Cliente envía video como bytes
    2. Servidor procesa frame a frame
    3. Servidor envía frames anotados + métricas en JSON
    """
    await websocket.accept()

    # Configuración
    FRAME_SKIP = 5  # Procesar cada N frames
    
    # Colores por detector
    COLOR_MAP = {
        "yolo": (0, 255, 136),      # Verde neón
        "rtdetr": (238, 211, 34),   # Cyan  
        "ensemble": (246, 130, 59), # Azul
    }
    DEFAULT_COLOR = (255, 255, 255)

    # Seleccionar detector
    detector = detector.lower()
    if detector == "yolo":
        detect_fn = detect_plate_yolo
    elif detector == "rtdetr":
        detect_fn = detect_plate_rtdetr
    else:
        detect_fn = detect_plate_ensemble
        detector = "ensemble"

    def draw_detections(frame, plates, detector_name):
        """Dibuja bounding boxes y texto de placas en el frame."""
        annotated = frame.copy()
        
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            det_type = plate.get("detector", detector_name)
            color = COLOR_MAP.get(det_type, DEFAULT_COLOR)
            
            # Dibujar bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # OCR
            ocr = read_plate(plate["image"])
            if ocr:
                label = f"{ocr['plate']} ({plate['confidence']*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                cv2.putText(annotated, label, (x1+3, y1-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated

    try:
        await websocket.send_json({
            "type": "status",
            "message": f"Conectado. Detector: {detector}. Esperando video..."
        })
        
        # Recibir video como bytes
        video_bytes = await websocket.receive_bytes()

        # Guardar en temp
        job_id = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        await websocket.send_json({
            "type": "status",
            "message": f"Procesando video con {detector.upper()}..."
        })

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Métricas
        all_plates = {}  # plate_text -> {count, max_conf, detector}
        frame_count = 0
        t0 = time.time()
        last_plates = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada N frames
            if frame_count % FRAME_SKIP == 0:
                plates = detect_fn(frame)
                last_plates = plates
                
                # Acumular placas únicas
                for plate in plates:
                    ocr = read_plate(plate["image"])
                    if ocr and ocr["plate"] != "No detectado":
                        plate_text = ocr["plate"]
                        det_type = plate.get("detector", detector)
                        
                        if plate_text not in all_plates:
                            all_plates[plate_text] = {
                                "count": 0,
                                "max_detector_conf": 0,
                                "max_ocr_conf": 0,
                                "detector": det_type,
                            }
                        
                        all_plates[plate_text]["count"] += 1
                        all_plates[plate_text]["max_detector_conf"] = max(
                            all_plates[plate_text]["max_detector_conf"],
                            plate["confidence"]
                        )
                        all_plates[plate_text]["max_ocr_conf"] = max(
                            all_plates[plate_text]["max_ocr_conf"],
                            ocr["confidence"]
                        )

            # Anotar y enviar frame
            annotated = draw_detections(frame, last_plates, detector)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_frame = base64.b64encode(buffer).decode("utf-8")

            progress = round((frame_count / max(total_frames, 1)) * 100)

            await websocket.send_json({
                "type": "frame",
                "frame": b64_frame,
                "progress": progress,
                "frame_num": frame_count,
                "current_detections": len(last_plates),
                "unique_plates": len(all_plates),
            })

            await asyncio.sleep(0.01)
            frame_count += 1

        cap.release()

        # Métricas finales
        duration = round(time.time() - t0, 2)
        
        # Ordenar placas por frecuencia
        sorted_plates = sorted(
            [
                {
                    "plate": plate,
                    "appearances": data["count"],
                    "max_detector_confidence": round(data["max_detector_conf"], 4),
                    "max_ocr_confidence": round(data["max_ocr_conf"], 4),
                    "detector": data["detector"],
                }
                for plate, data in all_plates.items()
            ],
            key=lambda x: x["appearances"],
            reverse=True
        )

        await websocket.send_json({
            "type": "done",
            "metrics": {
                "total_unique_plates": len(all_plates),
                "total_frames_processed": frame_count // FRAME_SKIP,
                "video_duration_s": round(total_frames / fps, 2),
                "processing_time_s": duration,
                "detector_used": detector,
                "plates": sorted_plates,
            },
        })

    except WebSocketDisconnect:
        print(f"[WS] Cliente desconectado")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)


# ── WebSocket: Detección de Vehículos en Video (existente) ─────────────────────
@app.websocket("/ws/detect-vehicle/video")
async def detect_vehicle_video_ws(websocket: WebSocket):
    """WebSocket para detección de vehículos en video (sin placas)."""
    await websocket.accept()

    FRAME_SKIP = 10
    MAX_CENTROID_DIST = 120
    MAX_FRAMES_MISSING = 6

    COLOR_MAP = {
        "Automóvil": (0, 200, 0),
        "Motocicleta": (0, 140, 255),
        "Autobús": (0, 0, 220),
        "Camión": (200, 0, 200),
    }
    DEFAULT_COLOR = (0, 200, 200)

    def get_centroid(bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def centroid_dist(c1, c2):
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

    def draw_frame(frame, detections, vehicle_counter):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["conf"]
            color = COLOR_MAP.get(label, DEFAULT_COLOR)
            text = f"{label} {conf*100:.0f}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(annotated, text, (x1+3, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        total = sum(vehicle_counter.values())
        panel_h = 30 + max(len(vehicle_counter), 1) * 22
        cv2.rectangle(annotated, (5, 5), (220, 5 + panel_h), (0, 0, 0), -1)
        cv2.putText(annotated, f"Total: {total}", (12, 26),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        for idx, (lbl, cnt) in enumerate(vehicle_counter.items()):
            color = COLOR_MAP.get(lbl, DEFAULT_COLOR)
            cv2.putText(annotated, f"  {lbl}: {cnt}", (12, 48 + idx*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        return annotated

    input_path = None
    
    try:
        await websocket.send_json({"type": "status", "message": "Esperando video..."})
        video_bytes = await websocket.receive_bytes()

        job_id = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        await websocket.send_json({"type": "status", "message": "Procesando video..."})

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        active_tracks = {}
        finished_tracks = []
        vehicle_counter = {}
        next_key = 0
        last_dets = []
        frame_count = 0
        processed_count = 0
        t0 = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                processed_count += 1
                proc_frame = frame_count // FRAME_SKIP
                frame_dets = []

                vehicles = detect_vehicles(frame)

                for v in vehicles:
                    frame_dets.append({
                        "bbox": v["bbox"],
                        "label": v["type_es"],
                        "conf": v["confidence"],
                        "matched": False,
                    })

                for key in list(active_tracks.keys()):
                    t = active_tracks[key]
                    best_dist, best_det = MAX_CENTROID_DIST, None
                    for det in frame_dets:
                        if det["matched"] or det["label"] != t["label"]:
                            continue
                        d = centroid_dist(get_centroid(det["bbox"]), t["centroid"])
                        if d < best_dist:
                            best_dist, best_det = d, det
                    if best_det:
                        best_det["matched"] = True
                        active_tracks[key].update({
                            "centroid": get_centroid(best_det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf": max(best_det["conf"], t["max_conf"]),
                        })

                for det in frame_dets:
                    if not det["matched"]:
                        active_tracks[next_key] = {
                            "label": det["label"],
                            "centroid": get_centroid(det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf": det["conf"],
                        }
                        vehicle_counter[det["label"]] = vehicle_counter.get(det["label"], 0) + 1
                        next_key += 1

                expired = [
                    k for k, t in active_tracks.items()
                    if proc_frame - t["last_seen_frame"] > MAX_FRAMES_MISSING
                ]
                for k in expired:
                    finished_tracks.append(active_tracks.pop(k))

                last_dets = frame_dets

            annotated = draw_frame(frame, last_dets, vehicle_counter)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_frame = base64.b64encode(buffer).decode("utf-8")

            progress = round((frame_count / max(total_frames, 1)) * 100)

            await websocket.send_json({
                "type": "frame",
                "frame": b64_frame,
                "progress": progress,
                "vehicle_counter": vehicle_counter,
                "frame_num": frame_count,
            })

            await asyncio.sleep(0.01)
            frame_count += 1

        cap.release()

        total_unique = len(finished_tracks) + len(active_tracks)
        duration = round(time.time() - t0, 2)
        type_stats = sorted([
            {
                "type": t,
                "count": c,
                "percent": round(c / total_unique * 100, 1) if total_unique > 0 else 0,
            }
            for t, c in vehicle_counter.items()
        ], key=lambda x: x["count"], reverse=True)

        await websocket.send_json({
            "type": "done",
            "metrics": {
                "total_unique_vehicles": total_unique,
                "total_raw_detections": processed_count,
                "video_duration_s": round(total_frames / fps, 2),
                "processing_time_ms": int(duration * 1000),
                "vehicles_per_minute": round(total_unique / (duration / 60), 2) if duration > 0 else 0,
                "by_type": type_stats,
            },
        })

    except WebSocketDisconnect:
        print("[WS] Cliente desconectado")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


# ── Ejecutar con Uvicorn ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
