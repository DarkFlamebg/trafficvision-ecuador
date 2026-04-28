# app/routes/deteccion.py
# Procesamiento de video con STREAMING de frames via SocketIO
# Sin ffmpeg - frames JPEG enviados en tiempo real al frontend

import os
import uuid
import json
import shutil
import time
import base64
import asyncio

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
import socketio

from app.ai.vehicle_detector import detect_vehicles
from app.ai.plate_detector   import detect_plate
from app.ai.plate_reader     import read_plate

router = APIRouter()

# ── SocketIO Server ────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=['http://localhost:5173', '*']
)

TEMP_DIR    = "temp"
VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/x-msvideo",
    "video/quicktime", "video/webm",
}

# ── Colores BGR por tipo de vehículo ───────────────────────────────────────────
COLOR_MAP = {
    "Automóvil":   (0, 200, 0),
    "Motocicleta": (0, 140, 255),
    "Autobús":     (0, 0, 220),
    "Camión":      (200, 0, 200),
}
DEFAULT_COLOR = (0, 200, 200)
PLATE_COLOR   = (0, 220, 220)


# ── Helpers de tracking ────────────────────────────────────────────────────────
def _centroid(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def _dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5


# ── Anotación del frame ────────────────────────────────────────────────────────
def _draw_frame(frame, v_dets, p_dets, vehicle_counter):
    out = frame.copy()

    for det in v_dets:
        x1, y1, x2, y2 = det["bbox"]
        color = COLOR_MAP.get(det["label"], DEFAULT_COLOR)
        text  = f"{det['label']} {det['conf']*100:.0f}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, text, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    for pdet in p_dets:
        px1, py1, px2, py2 = pdet["bbox"]
        plate_text = pdet.get("plate", "")
        cv2.rectangle(out, (px1, py1), (px2, py2), PLATE_COLOR, 2)
        if plate_text:
            (tw, th), _ = cv2.getTextSize(
                plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(out, (px1, py2), (px1+tw+6, py2+th+8), PLATE_COLOR, -1)
            cv2.putText(out, plate_text, (px1+3, py2+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

    total   = sum(vehicle_counter.values())
    panel_h = 30 + max(len(vehicle_counter), 1) * 22
    cv2.rectangle(out, (5, 5), (220, 5+panel_h), (0, 0, 0), -1)
    cv2.putText(out, f"Total: {total}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for idx, (lbl, cnt) in enumerate(vehicle_counter.items()):
        color = COLOR_MAP.get(lbl, DEFAULT_COLOR)
        cv2.putText(out, f"  {lbl}: {cnt}", (12, 48 + idx*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    return out


def _frame_to_base64(frame, quality=80):
    """Convierte un frame OpenCV a base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


# ── POST /api/v1/video-analysis-stream ─────────────────────────────────────────
@router.post("/video-analysis-stream")
async def analyze_video_stream(file: UploadFile = File(...)):
    """
    Pipeline de video con STREAMING de frames via SocketIO.
    
    En lugar de guardar un video MP4 (que requiere ffmpeg para H.264),
    envía cada frame procesado como JPEG base64 al frontend en tiempo real.
    
    El frontend recibe:
      - 'frame': imagen base64 para mostrar en <img>
      - 'progress': porcentaje de progreso
      - 'metrics': métricas parciales
      - 'complete': métricas finales cuando termina
    """
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(400, "Solo se aceptan videos MP4, AVI o MOV")

    os.makedirs(TEMP_DIR, exist_ok=True)
    ext         = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    job_id      = str(uuid.uuid4())
    input_path  = os.path.join(TEMP_DIR, f"{job_id}{ext}")

    FRAME_SKIP         = 10
    MAX_CENTROID_DIST  = 120
    MAX_FRAMES_MISSING = 6

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        t0           = time.time()
        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Estado del tracker
        active_tracks    = {}
        finished_tracks  = []
        vehicle_counter  = {}
        next_key         = 0
        all_det_vehicles = []
        all_det_plates   = []
        last_v_dets      = []
        last_p_dets      = []
        frame_count      = 0
        processed_count  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                proc_frame = frame_count // FRAME_SKIP
                processed_count += 1
                frame_v_dets = []
                frame_p_dets = []

                vehicles = detect_vehicles(frame)

                for v in vehicles:
                    frame_v_dets.append({
                        "bbox":    v["bbox"],
                        "label":   v["type_es"],
                        "conf":    v["confidence"],
                        "matched": False,
                    })
                    all_det_vehicles.append({
                        "vehicle_type":    v["type_es"],
                        "confidence":      round(v["confidence"] * 100, 2),
                        "bbox":            v["bbox"],
                        "frame":           proc_frame,
                        "timestamp_video": round(frame_count / fps, 2),
                    })

                    plates_in_vehicle = detect_plate(v["image"])
                    for plate in plates_in_vehicle:
                        ocr      = read_plate(plate["image"])
                        ocr_text = ocr["plate"]      if ocr else ""
                        ocr_conf = ocr["confidence"] if ocr else 0.0

                        vx1, vy1 = v["bbox"][0], v["bbox"][1]
                        abs_bbox = [
                            plate["bbox"][0] + vx1,
                            plate["bbox"][1] + vy1,
                            plate["bbox"][2] + vx1,
                            plate["bbox"][3] + vy1,
                        ]

                        frame_p_dets.append({
                            "bbox":            abs_bbox,
                            "plate":           ocr_text,
                            "ocr_confidence":  round(ocr_conf, 4),
                            "yolo_confidence": plate["confidence"],
                        })
                        if ocr_text:
                            all_det_plates.append({
                                "plate":           ocr_text,
                                "ocr_confidence":  round(ocr_conf, 4),
                                "yolo_confidence": plate["confidence"],
                                "bbox":            abs_bbox,
                                "vehicle_type":    v["type_es"],
                                "frame":           proc_frame,
                                "timestamp_video": round(frame_count / fps, 2),
                            })

                # Tracking
                for key in list(active_tracks.keys()):
                    t = active_tracks[key]
                    best_dist, best_det = MAX_CENTROID_DIST, None
                    for det in frame_v_dets:
                        if det["matched"] or det["label"] != t["label"]:
                            continue
                        d = _dist(_centroid(det["bbox"]), t["centroid"])
                        if d < best_dist:
                            best_dist, best_det = d, det
                    if best_det:
                        best_det["matched"] = True
                        active_tracks[key].update({
                            "centroid":        _centroid(best_det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        max(best_det["conf"], t["max_conf"]),
                        })

                for det in frame_v_dets:
                    if not det["matched"]:
                        active_tracks[next_key] = {
                            "label":           det["label"],
                            "centroid":        _centroid(det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        det["conf"],
                        }
                        vehicle_counter[det["label"]] = (
                            vehicle_counter.get(det["label"], 0) + 1
                        )
                        next_key += 1

                expired = [
                    k for k, t in active_tracks.items()
                    if proc_frame - t["last_seen_frame"] > MAX_FRAMES_MISSING
                ]
                for k in expired:
                    finished_tracks.append(active_tracks.pop(k))

                last_v_dets = frame_v_dets
                last_p_dets = frame_p_dets

            # Anotar frame
            annotated = _draw_frame(frame, last_v_dets, last_p_dets, vehicle_counter)
            
            # ENVIAR frame via SocketIO cada N frames para no saturar
            if frame_count % 3 == 0:  # enviar 1 de cada 3 frames
                frame_b64 = _frame_to_base64(annotated, quality=75)
                progress = round((frame_count / total_frames) * 100, 1)
                
                await sio.emit('frame', {
                    'image': frame_b64,
                    'progress': progress,
                    'vehicles_count': sum(vehicle_counter.values()),
                    'plates_count': len(all_det_plates),
                })
                await asyncio.sleep(0.01)  # yield para no bloquear

            frame_count += 1

        cap.release()

        # Métricas finales
        total_unique = len(finished_tracks) + len(active_tracks)
        duration     = round(time.time() - t0, 2)

        type_stats = sorted([
            {
                "type":    t,
                "count":   c,
                "percent": round(c / total_unique * 100, 1) if total_unique > 0 else 0,
            }
            for t, c in vehicle_counter.items()
        ], key=lambda x: x["count"], reverse=True)

        unique_plates: dict[str, dict] = {}
        for p in all_det_plates:
            txt = p["plate"]
            if txt not in unique_plates or p["ocr_confidence"] > unique_plates[txt]["ocr_confidence"]:
                unique_plates[txt] = p

        metrics = {
            "total_unique_vehicles": total_unique,
            "total_raw_detections":  len(all_det_vehicles),
            "total_plates_detected": len(unique_plates),
            "frames_processed":      processed_count,
            "video_duration_s":      round(total_frames / fps, 2),
            "processing_time_ms":    int(duration * 1000),
            "vehicles_per_minute":   (
                round(total_unique / (duration / 60), 2) if duration > 0 else 0
            ),
            "by_type":               type_stats,
            "plates":                list(unique_plates.values()),
        }

        # Enviar evento de completado
        await sio.emit('complete', metrics)

        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": metrics
        }

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# ── SocketIO Events ────────────────────────────────────────────────────────────
@sio.event
async def connect(sid, environ):
    print(f"[SocketIO] Cliente conectado: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[SocketIO] Cliente desconectado: {sid}")