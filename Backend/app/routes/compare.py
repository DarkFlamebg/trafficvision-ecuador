# app/routes/compare.py
# Endpoints de comparativa YOLO11n vs RT-DETR
# Requests separados — el front elige el modelo con ?model=yolo|rtdetr
#
# Imagen:  POST /api/v1/compare/image?model=yolo
# Video:   WS   /api/v1/compare/video?model=yolo
#
# Ambos retornan la misma estructura de métricas para facilitar
# la comparación simétrica en el frontend.

import os
import uuid
import time
import base64
import asyncio
import shutil
import re

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.ai.vehicle_detector    import detect_vehicles
from app.ai.plate_detector      import detect_plate      as detect_plate_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr
from app.ai.plate_reader        import read_plate

router = APIRouter(prefix="/compare", tags=["Comparativa"])

TEMP_DIR = "temp"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime", "video/webm"}

# ── Heurística para Placas ─────────────────────────────────────────────────────

def _is_valid_plate_text(text: str) -> bool:
    """
    Filtro heurístico inteligente:
    - Permite placas sin texto (para que aparezcan al final si el detector funcionó pero OCR falló).
    - Descarta letreros de buses conocidos ("ESCOLAR", "COMPANIA").
    - Si tiene letras y números, es válida.
    - Si solo tiene números o solo letras, la permite SOLO si es muy corta (lectura parcial ej. "GRY").
      Si tiene 4+ caracteres de un solo tipo (ej. "2923" o "ESCO"), la descarta.
    """
    if not text:
        return True
    
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Blacklist de textos de buses
    for w in ["ESCOLAR", "COMPANIA", "FURBUSA", "INSTITU"]:
        if w in text.upper():
            return False

    if len(clean_text) > 8:
        return False
        
    has_letters = any(c.isalpha() for c in clean_text)
    has_numbers = any(c.isdigit() for c in clean_text)
    
    if has_letters and has_numbers:
        return True
        
    # Si solo tiene números y es de 4+ caracteres (ej. "2923"), descartar
    if not has_letters and len(clean_text) >= 4:
        return False
        
    # Si solo tiene letras y es de 4+ caracteres (ej. "ESCO"), descartar
    if not has_numbers and len(clean_text) >= 4:
        return False
        
    # Lectura parcial corta (ej. "GRY" o "123") -> Permitir
    return True

# ── Selector de detector ───────────────────────────────────────────────────────

def _get_plate_detector(model: str):
    """
    Retorna la función de detección de placas según el modelo elegido.
    Lanza HTTPException 400 si el modelo no es válido.
    """
    if model == "yolo":
        return detect_plate_yolo, "YOLOv11n"
    if model == "rtdetr":
        return detect_plate_rtdetr, "RT-DETR"
    raise HTTPException(
        status_code=400,
        detail=f"Modelo '{model}' no válido. Usa 'yolo' o 'rtdetr'."
    )


# ── Estructura de métricas compartida ─────────────────────────────────────────

def _build_image_metrics(
    model_name: str,
    vehicles: list,
    plates_found: list,
    inference_ms: float,
) -> dict:
    """
    Construye el dict de métricas estándar para respuesta de imagen.
    Misma estructura independientemente del modelo — el front la consume igual.
    """
    confidences = [p["detector_confidence"] for p in plates_found]
    v_confidences = [v["confidence"] for v in vehicles]

    return {
        "model":                   model_name,
        "inference_ms":            round(inference_ms, 2),
        "vehicles_detected":       len(vehicles),
        "avg_vehicle_confidence":  round(sum(v_confidences) / len(v_confidences), 4) if v_confidences else 0.0,
        "plates_detected":         len(plates_found),
        "avg_plate_confidence":    round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "plates_with_ocr":         sum(1 for p in plates_found if p["plate"]),
        "vehicles_by_type":        _count_by_type(vehicles),
    }


def _count_by_type(vehicles: list) -> dict:
    counts = {}
    for v in vehicles:
        t = v["type_es"]
        counts[t] = counts.get(t, 0) + 1
    return counts


# ── POST /api/v1/compare/image?model=yolo|rtdetr ──────────────────────────────

@router.post("/image")
async def compare_image(
    file: UploadFile = File(...),
    model: str = Query("yolo", enum=["yolo", "rtdetr"]),
):
    """
    Corre el pipeline completo de detección sobre una imagen
    usando el modelo especificado.

    Pipeline:
      1. Detectar vehículos (YOLOv8n COCO — siempre el mismo)
      2. Por cada vehículo → detectar placa con el modelo elegido
      3. OCR con EasyOCR

    Response incluye métricas simétricas + detalle de placas para
    que el front construya la comparativa.

    Query params:
      model: "yolo" (YOLOv11n) | "rtdetr" (RT-DETR)
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, "Solo se aceptan imágenes JPG o PNG")

    detect_plates, model_name = _get_plate_detector(model)

    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")

    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ── Etapa 1: Vehículos (siempre YOLOv8n) ──────────────────────────
        vehicles = detect_vehicles(path)

        plates_found = []

        # ── Etapa 2: Placas con el modelo elegido ─────────────────────────
        t_start = time.perf_counter()

        search_targets = vehicles if vehicles else [{"bbox": [0, 0, 0, 0], "image": path, "type_es": None, "type": None, "confidence": 0.0}]

        for v in search_targets:
            v_image = v["image"] if vehicles else path
            vx1, vy1 = (v["bbox"][0], v["bbox"][1]) if vehicles else (0, 0)

            plates_in_v = detect_plates(v_image)

            for plate in plates_in_v:
                ocr      = read_plate(plate["image"])
                ocr_text = ocr["plate"]      if ocr else ""
                ocr_conf = ocr["confidence"] if ocr else 0.0

                abs_bbox = [
                    plate["bbox"][0] + vx1,
                    plate["bbox"][1] + vy1,
                    plate["bbox"][2] + vx1,
                    plate["bbox"][3] + vy1,
                ] if vehicles else plate["bbox"]

                # Añadir solo si pasa el filtro heurístico o si estamos en modo estricto
                # (Para la lista de imagen, mantendremos las que pasen el filtro para evitar ensuciar los resultados)
                if _is_valid_plate_text(ocr_text):
                    plates_found.append({
                        "bbox":               abs_bbox,
                        "detector_confidence": plate["confidence"],
                        "plate":              ocr_text,
                        "ocr_confidence":     round(ocr_conf, 4),
                        "vehicle_type":       v.get("type_es"),
                        "vehicle_bbox":       v["bbox"] if vehicles else None,
                    })

        inference_ms = (time.perf_counter() - t_start) * 1000

        metrics = _build_image_metrics(model_name, vehicles, plates_found, inference_ms)

        return {
            "model":    model_name,
            "metrics":  metrics,
            "vehicles": [
                {
                    "type":       v["type"],
                    "type_es":    v["type_es"],
                    "bbox":       v["bbox"],
                    "confidence": v["confidence"],
                }
                for v in vehicles
            ],
            "plates": plates_found,
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Helpers de tracking y anotación (compartidos por ambos WS) ────────────────

COLOR_MAP = {
    "Automóvil":   (0, 200, 0),
    "Motocicleta": (0, 140, 255),
    "Autobús":     (0, 0, 220),
    "Camión":      (200, 0, 200),
}
DEFAULT_COLOR = (0, 200, 200)
PLATE_COLOR   = (0, 220, 220)

FRAME_SKIP         = 10
MAX_CENTROID_DIST  = 120
MAX_FRAMES_MISSING = 6


def _centroid(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def _dist(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

def _draw_frame(frame, v_dets, p_dets, vehicle_counter, model_label):
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
            (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(out, (px1, py2), (px1+tw+6, py2+th+8), PLATE_COLOR, -1)
            cv2.putText(out, plate_text, (px1+3, py2+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

    # Panel superior izquierdo
    total   = sum(vehicle_counter.values())
    lines   = [f"[{model_label}]", f"Vehiculos: {total}"] + [
        f"  {lbl}: {cnt}" for lbl, cnt in vehicle_counter.items()
    ]
    panel_h = 12 + len(lines) * 22
    cv2.rectangle(out, (5, 5), (230, 5 + panel_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        color = (255, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(out, line, (12, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    return out


def _frame_to_b64(frame, quality=75):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def _build_video_metrics(
    model_name, vehicle_counter, finished_tracks, active_tracks,
    all_plates, processed_count, total_frames, fps, duration_s,
) -> dict:
    """Métricas finales de video — misma estructura para YOLO y RT-DETR."""
    total_unique = len(finished_tracks) + len(active_tracks)

    unique_plates: dict[str, dict] = {}
    for p in all_plates:
        txt = p["plate"]
        if txt not in unique_plates or p["ocr_confidence"] > unique_plates[txt]["ocr_confidence"]:
            unique_plates[txt] = p

    all_confs = [p["detector_confidence"] for p in all_plates]
    type_stats = sorted([
        {
            "type":    t,
            "count":   c,
            "percent": round(c / total_unique * 100, 1) if total_unique > 0 else 0,
        }
        for t, c in vehicle_counter.items()
    ], key=lambda x: x["count"], reverse=True)

    return {
        "model":                    model_name,
        "total_unique_vehicles":    total_unique,
        "total_raw_detections":     processed_count,
        "total_plates_detected":    len(unique_plates),
        "avg_plate_confidence":     round(sum(all_confs) / len(all_confs), 4) if all_confs else 0.0,
        "plates_with_ocr":          len(unique_plates),
        "frames_processed":         processed_count,
        "video_duration_s":         round(total_frames / fps, 2),
        "processing_time_ms":       int(duration_s * 1000),
        "vehicles_per_minute":      round(total_unique / (duration_s / 60), 2) if duration_s > 0 else 0,
        "by_type":                  type_stats,
        "plates":                   list(unique_plates.values()),
    }


# ── WS /api/v1/compare/video?model=yolo|rtdetr ────────────────────────────────

@router.websocket("/video")
async def compare_video(
    websocket: WebSocket,
    model: str = Query("yolo", enum=["yolo", "rtdetr"]),
):
    """
    WebSocket de comparativa de video.

    El cliente envía el video como bytes tras conectarse.
    El servidor retorna frames anotados + métricas en tiempo real.

    Mensajes del servidor:
      {"type": "status",   "message": str}
      {"type": "frame",    "frame": base64, "progress": int,
       "vehicle_counter": dict, "plates_count": int,
       "inference_ms": float}
      {"type": "done",     "metrics": dict}
      {"type": "error",    "message": str}

    Query params:
      model: "yolo" (YOLOv11n) | "rtdetr" (RT-DETR)
    """
    await websocket.accept()

    if model == "yolo":
        detect_plates = detect_plate_yolo
        model_name    = "YOLOv11n"
    else:
        detect_plates = detect_plate_rtdetr
        model_name    = "RT-DETR"

    input_path = None

    try:
        await websocket.send_json({"type": "status", "message": f"[{model_name}] Esperando video..."})
        video_bytes = await websocket.receive_bytes()

        job_id     = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
        os.makedirs(TEMP_DIR, exist_ok=True)

        with open(input_path, "wb") as f:
            f.write(video_bytes)

        await websocket.send_json({"type": "status", "message": f"[{model_name}] Procesando..."})

        cap          = cv2.VideoCapture(input_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames == 0:
            await websocket.send_json({"type": "error", "message": "No se pudo abrir el video"})
            return

        # Estado tracker
        active_tracks    = {}
        finished_tracks  = []
        vehicle_counter  = {}
        next_key         = 0
        all_plates       = []
        last_v_dets      = []
        last_p_dets      = []
        frame_count      = 0
        processed_count  = 0
        t0               = time.time()

        # Acumulador de inferencia para promedio en tiempo real
        inference_times = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                proc_frame   = frame_count // FRAME_SKIP
                processed_count += 1
                frame_v_dets = []
                frame_p_dets = []

                # ── Vehículos ──────────────────────────────────────────────
                vehicles = detect_vehicles(frame)

                for v in vehicles:
                    frame_v_dets.append({
                        "bbox":    v["bbox"],
                        "label":   v["type_es"],
                        "conf":    v["confidence"],
                        "matched": False,
                    })

                    # ── Placas con el modelo elegido ───────────────────────
                    t_inf = time.perf_counter()
                    plates_in_v = detect_plates(v["image"])
                    inference_times.append((time.perf_counter() - t_inf) * 1000)

                    vx1, vy1 = v["bbox"][0], v["bbox"][1]
                    for plate in plates_in_v:
                        ocr      = read_plate(plate["image"])
                        ocr_text = ocr["plate"]      if ocr else ""
                        ocr_conf = ocr["confidence"] if ocr else 0.0

                        abs_bbox = [
                            plate["bbox"][0] + vx1,
                            plate["bbox"][1] + vy1,
                            plate["bbox"][2] + vx1,
                            plate["bbox"][3] + vy1,
                        ]

                        p_entry = {
                            "bbox":                abs_bbox,
                            "plate":               ocr_text,
                            "ocr_confidence":      round(ocr_conf, 4),
                            "detector_confidence": plate["confidence"],
                            "vehicle_type":        v["type_es"],
                            "frame":               proc_frame,
                            "timestamp_video":     round(frame_count / fps, 2),
                            "image_base64":        _frame_to_b64(plate["image"]) if "image" in plate else None,
                        }
                        frame_p_dets.append(p_entry)
                        
                        # Solo consideramos la placa como "válida" si pasa el filtro heurístico
                        if _is_valid_plate_text(ocr_text):
                            all_plates.append(p_entry)

                # ── Tracking ───────────────────────────────────────────────
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
                        vehicle_counter[det["label"]] = vehicle_counter.get(det["label"], 0) + 1
                        next_key += 1

                expired = [
                    k for k, t in active_tracks.items()
                    if proc_frame - t["last_seen_frame"] > MAX_FRAMES_MISSING
                ]
                for k in expired:
                    finished_tracks.append(active_tracks.pop(k))

                last_v_dets = frame_v_dets
                last_p_dets = frame_p_dets

            # ── Frame anotado ──────────────────────────────────────────────
            annotated = _draw_frame(frame, last_v_dets, last_p_dets, vehicle_counter, model_name)
            progress  = round((frame_count / max(total_frames, 1)) * 100)

            await websocket.send_json({
                "type":            "frame",
                "frame":           _frame_to_b64(annotated),
                "progress":        progress,
                "vehicle_counter": vehicle_counter,
                "plates_count":    len(all_plates),
                "inference_ms":    round(sum(inference_times) / len(inference_times), 2) if inference_times else 0.0,
                "frame_num":       frame_count,
            })

            await asyncio.sleep(0.01)
            frame_count += 1

        cap.release()

        duration_s = time.time() - t0
        metrics    = _build_video_metrics(
            model_name, vehicle_counter, finished_tracks, active_tracks,
            all_plates, processed_count, total_frames, fps, duration_s,
        )
        # Añadir tiempo promedio de inferencia de placa
        metrics["avg_inference_ms"] = round(
            sum(inference_times) / len(inference_times), 2
        ) if inference_times else 0.0

        await websocket.send_json({"type": "done", "metrics": metrics})

    except WebSocketDisconnect:
        print(f"[compare/{model}] Cliente desconectado")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
