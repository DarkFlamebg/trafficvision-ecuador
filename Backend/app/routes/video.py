# app/routes/video.py
# WebSocket de análisis de video en tiempo real.
# Versión: /api/v1/detection/video
#
# Endpoints:
#   WS /api/v1/detection/video — streaming de frames con conteo de vehículos

import os
import uuid
import asyncio
import base64
import time

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai.vehicle_detector import detect_vehicles
from app.core.config import (
    TEMP_DIR,
    VIDEO_FRAME_SKIP,
    VIDEO_MAX_CENTROID_DIST,
    VIDEO_MAX_FRAMES_MISSING,
    VIDEO_JPEG_QUALITY,
)

router = APIRouter(prefix="/detection", tags=["Detección v1"])

# ── Colores por tipo de vehículo ───────────────────────────────────────────────
_COLOR_MAP = {
    "Automóvil":   (0, 200, 0),
    "Motocicleta": (0, 140, 255),
    "Autobús":     (0, 0, 220),
    "Camión":      (200, 0, 200),
}
_DEFAULT_COLOR = (0, 200, 200)


# ── Helpers internos ───────────────────────────────────────────────────────────

def _get_centroid(bbox: list) -> tuple[int, int]:
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def _centroid_dist(c1: tuple, c2: tuple) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _draw_frame(frame, detections: list, vehicle_counter: dict):
    """Dibuja bounding boxes y panel de conteo sobre el frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label  = det["label"]
        conf   = det["conf"]
        color  = _COLOR_MAP.get(label, _DEFAULT_COLOR)
        text   = f"{label} {conf * 100:.0f}%"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    total   = sum(vehicle_counter.values())
    panel_h = 30 + max(len(vehicle_counter), 1) * 22
    cv2.rectangle(annotated, (5, 5), (220, 5 + panel_h), (0, 0, 0), -1)
    cv2.putText(annotated, f"Total: {total}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    for idx, (lbl, cnt) in enumerate(vehicle_counter.items()):
        color = _COLOR_MAP.get(lbl, _DEFAULT_COLOR)
        cv2.putText(annotated, f"  {lbl}: {cnt}", (12, 48 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    return annotated


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@router.websocket("/video")
async def detect_vehicle_video_ws(websocket: WebSocket):
    """
    WebSocket de análisis de video en tiempo real.

    Protocolo:
      1. Cliente conecta → servidor envía {"type": "status", "message": "Esperando video..."}
      2. Cliente envía bytes del video MP4
      3. Servidor procesa y envía frames: {"type": "frame", "frame": "<base64>", ...}
      4. Al terminar: {"type": "done", "metrics": {...}}
    """
    await websocket.accept()

    input_path = None
    try:
        await websocket.send_json({"type": "status", "message": "Esperando video..."})
        video_bytes = await websocket.receive_bytes()

        job_id     = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
        os.makedirs(TEMP_DIR, exist_ok=True)

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

            if frame_count % VIDEO_FRAME_SKIP == 0:
                processed_count += 1
                proc_frame = frame_count // VIDEO_FRAME_SKIP
                frame_dets = []

                for v in detect_vehicles(frame):
                    frame_dets.append({
                        "bbox":    v["bbox"],
                        "label":   v["type_es"],
                        "conf":    v["confidence"],
                        "matched": False,
                    })

                # ── Tracking por centroide ─────────────────────────────────
                for key in list(active_tracks.keys()):
                    t = active_tracks[key]
                    best_dist, best_det = VIDEO_MAX_CENTROID_DIST, None
                    for det in frame_dets:
                        if det["matched"] or det["label"] != t["label"]:
                            continue
                        d = _centroid_dist(_get_centroid(det["bbox"]), t["centroid"])
                        if d < best_dist:
                            best_dist, best_det = d, det
                    if best_det:
                        best_det["matched"] = True
                        active_tracks[key].update({
                            "centroid":        _get_centroid(best_det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        max(best_det["conf"], t["max_conf"]),
                        })

                for det in frame_dets:
                    if not det["matched"]:
                        active_tracks[next_key] = {
                            "label":           det["label"],
                            "centroid":        _get_centroid(det["bbox"]),
                            "last_seen_frame": proc_frame,
                            "max_conf":        det["conf"],
                        }
                        vehicle_counter[det["label"]] = vehicle_counter.get(det["label"], 0) + 1
                        next_key += 1

                expired = [
                    k for k, t in active_tracks.items()
                    if proc_frame - t["last_seen_frame"] > VIDEO_MAX_FRAMES_MISSING
                ]
                for k in expired:
                    finished_tracks.append(active_tracks.pop(k))

                last_dets = frame_dets

            # ── Enviar frame anotado ───────────────────────────────────────
            annotated  = _draw_frame(frame, last_dets, vehicle_counter)
            _, buffer  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY])
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

        # ── Resumen final ──────────────────────────────────────────────────
        total_unique = len(finished_tracks) + len(active_tracks)
        duration     = round(time.time() - t0, 2)
        type_stats   = sorted(
            [
                {
                    "type":    t,
                    "count":   c,
                    "percent": round(c / total_unique * 100, 1) if total_unique > 0 else 0,
                }
                for t, c in vehicle_counter.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )

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
