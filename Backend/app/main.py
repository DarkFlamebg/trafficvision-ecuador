# Punto de entrada de la API — TrafficVision Backend
# Pipeline: Detección vehículo → Detección placa → OCR → Clasificación calidad

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.ai.vehicle_detector import detect_vehicles
from app.ai.plate_detector   import detect_plate
from app.ai.plate_reader     import read_plate
from app.ai.plate_classifier import classify_plate
from app.ai.model_loader     import load_all_models, get_status as model_status
from app.routes.detect       import router as detect_router
from app.routes.compare      import router as compare_router
from app.routes.benchmark    import router as benchmark_router
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect

from app.database.connection import SessionLocal
from app.database.models import PlateDetection, DetectionQuality, AuditLog, ModelIA, VehicleType, QualityLabel

import base64
import asyncio
import cv2
import time


load_dotenv()

TEMP_DIR = "temp"


# ── Ciclo de vida ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Carga y precalienta todos los modelos en paralelo (no bloquea el event loop)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_models)
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except Exception:
            pass

# ── Aplicación ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrafficVision API",
    description="Detección de vehículos y placas vehiculares con YOLOv8 + EasyOCR + Gemini",
    version="1.3.0",
    lifespan=lifespan
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Metrics", "X-Detections"],
)

ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/jpg")
ALLOWED_VIDEO_TYPES = ("video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime")

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(detect_router, prefix="/api/v1", tags=["Detección"])
app.include_router(compare_router, prefix="/api/v1", tags=["Comparativa"])
app.include_router(benchmark_router, prefix="/api/v1", tags=["Benchmark"])

# ── Endpoints base ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "TrafficVision API activa",
        "docs":    "/docs",
        "version": "1.3.0"
    }


@app.get("/health")
def health():
    """Estado de la API y tiempo de carga de modelos."""
    status = model_status()
    return {
        "status":        "ok" if status["models_ready"] else "loading",
        "models_ready":  status["models_ready"],
        "load_time_ms":  status["load_time_ms"],
        "version":       "1.3.0",
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

        vehicles     = detect_vehicles(path)
        plates_found = []

        if vehicles:
            for vehicle in vehicles:
                plates_in_vehicle = detect_plate(vehicle["image"])

                for plate in plates_in_vehicle:
                    ocr            = read_plate(plate["image"])
                    ocr_text       = ocr["plate"]      if ocr else "No detectado"
                    ocr_confidence = ocr["confidence"] if ocr else 0.0
                    labels         = classify_plate(plate["image"], ocr_confidence)

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
                    "vehicle":         None
                })

        # --- GUARDAR EN BASE DE DATOS (POSTGRESQL) ---
        db = SessionLocal()
        try:
            yolo_model = db.query(ModelIA).filter_by(name="YOLOv11n").first()
            
            for plate in plates_found:
                vtype_name = plate["vehicle"]["type_es"] if plate["vehicle"] else "Desconocido"
                vtype = db.query(VehicleType).filter_by(name=vtype_name).first()
                vtype_id = vtype.id if vtype else None
                
                new_detection = PlateDetection(
                    plate_text=plate["plate"][:15],  # Evitar overflow
                    confidence=plate["ocr_confidence"],
                    model_id=yolo_model.id if yolo_model else None,
                    vehicle_type_id=vtype_id,
                    inference_time_ms=120.0, # Ejemplo (calculable en el futuro)
                    image_path=path
                )
                db.add(new_detection)
                db.flush() # Obtener el ID
                
                # Guardar etiquetas de Gemini
                labels_dict = plate["labels"]
                if labels_dict:
                    quality_map = {
                        "oclusion": "Oclusión",
                        "reflejo": "Reflejo",
                        "suciedad": "Suciedad",
                        "is_legible": "Legibilidad"
                    }
                    for key, db_name in quality_map.items():
                        if key in labels_dict:
                            q_label = db.query(QualityLabel).filter_by(name=db_name).first()
                            if q_label:
                                db.add(DetectionQuality(
                                    detection_id=new_detection.id,
                                    quality_label_id=q_label.id,
                                    value=str(labels_dict[key])
                                ))
                
                # Log de auditoría
                db.add(AuditLog(
                    detection_id=new_detection.id,
                    checked_by="TrafficVision AI",
                    check_reason="Detección en tiempo real"
                ))
            
            db.commit()
            print(f"[BD] {len(plates_found)} placas guardadas en Supabase.")
        except Exception as e:
            print(f"[BD] Error al guardar: {e}")
            db.rollback()
        finally:
            db.close()
        # ---------------------------------------------

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
    """Detecta solo vehículos sin procesar placas."""
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


@app.websocket("/ws/detect-vehicle/video")
async def detect_vehicle_video_ws(websocket: WebSocket):
    await websocket.accept()

    FRAME_SKIP         = 10
    MAX_CENTROID_DIST  = 120
    MAX_FRAMES_MISSING = 6

    COLOR_MAP = {
        "Automóvil":   (0, 200, 0),
        "Motocicleta": (0, 140, 255),
        "Autobús":     (0, 0, 220),
        "Camión":      (200, 0, 200),
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
            label  = det["label"]
            conf   = det["conf"]
            color  = COLOR_MAP.get(label, DEFAULT_COLOR)
            text   = f"{label} {conf*100:.0f}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(annotated, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        total   = sum(vehicle_counter.values())
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

        job_id     = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        await websocket.send_json({"type": "status", "message": "Procesando video..."})

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        active_tracks   = {}
        finished_tracks = []
        vehicle_counter = {}
        next_key        = 0
        last_dets       = []
        frame_count     = 0
        processed_count = 0
        t0              = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                processed_count += 1
                proc_frame  = frame_count // FRAME_SKIP
                frame_dets  = []

                vehicles = detect_vehicles(frame)

                for v in vehicles:
                    frame_dets.append({
                        "bbox":    v["bbox"],
                        "label":   v["type_es"],
                        "conf":    v["confidence"],
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
                            "centroid":        get_centroid(best_det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        max(best_det["conf"], t["max_conf"]),
                        })

                for det in frame_dets:
                    if not det["matched"]:
                        active_tracks[next_key] = {
                            "label":           det["label"],
                            "centroid":        get_centroid(det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        det["conf"],
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
            _, buffer  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_frame  = base64.b64encode(buffer).decode("utf-8")
            progress   = round((frame_count / max(total_frames, 1)) * 100)

            await websocket.send_json({
                "type":            "frame",
                "frame":           b64_frame,
                "progress":        progress,
                "vehicle_counter": vehicle_counter,
                "frame_num":       frame_count,
            })

            await asyncio.sleep(0.01)
            frame_count += 1

        cap.release()

        total_unique = len(finished_tracks) + len(active_tracks)
        duration     = round(time.time() - t0, 2)
        type_stats   = sorted([
            {
                "type":    t,
                "count":   c,
                "percent": round(c / total_unique * 100, 1) if total_unique > 0 else 0,
            }
            for t, c in vehicle_counter.items()
        ], key=lambda x: x["count"], reverse=True)

        await websocket.send_json({
            "type": "done",
            "metrics": {
                "total_unique_vehicles": total_unique,
                "total_raw_detections":  processed_count,
                "video_duration_s":      round(total_frames / fps, 2),
                "processing_time_ms":    int(duration * 1000),
                "vehicles_per_minute":   round(total_unique / (duration / 60), 2) if duration > 0 else 0,
                "by_type":               type_stats,
            },
        })

    except WebSocketDisconnect:
        print("[WS] Cliente desconectado")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)