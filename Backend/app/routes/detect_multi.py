# app/routes/detect_multi.py
# Rutas de detección con soporte para YOLO, RT-DETR y modo comparación/ensemble

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum
import numpy as np
import cv2
import time
import base64

# Imports de módulos AI (ajusta según tu estructura de proyecto)
from app.ai.plate_detector import detect_plate as detect_plate_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr, detect_plate_ensemble
from app.ai.plate_reader import read_plate
from app.ai.plate_classifier import classify_plate
from app.ai.vehicle_detector import detect_vehicles
from app.ai.plate_detector_config import DetectorType, DetectorConfig, DetectorFactory

router = APIRouter()


# ── Modelos de Respuesta ───────────────────────────────────────────────────────
class DetectorTypeEnum(str, Enum):
    YOLO = "yolo"
    RTDETR = "rtdetr"
    ENSEMBLE = "ensemble"


class PlateDetection(BaseModel):
    plate: str
    ocr_confidence: float
    detector_confidence: float
    detector: str
    bbox: List[int]
    labels: Optional[dict] = None   # ← was List[str]
    vehicle: Optional[dict] = None


class DetectionResponse(BaseModel):
    success: bool
    total: int
    vehicles: int = 0          # ← add this
    detector_used: str
    processing_time_ms: int
    plates: List[PlateDetection]


class CompareResponse(BaseModel):
    success: bool
    yolo: DetectionResponse
    rtdetr: DetectionResponse
    ensemble: Optional[DetectionResponse] = None
    comparison: dict


# ── Endpoint: Detección con selector de modelo ─────────────────────────────────
@router.post("/detect/full", response_model=DetectionResponse, tags=["Detección Multi-Modelo"])
async def detect_full(
    file: UploadFile = File(...),
    detector: DetectorTypeEnum = Query(
        default=DetectorTypeEnum.ENSEMBLE,
        description="Modelo a usar: yolo (rápido), rtdetr (preciso), ensemble (ambos)"
    ),
    include_vehicle: bool = Query(
        default=True,
        description="Incluir detección de vehículo asociado"
    ),
    include_labels: bool = Query(
        default=True,
        description="Incluir clasificación de calidad de placa"
    )
):
    """
    Pipeline completo de detección con selector de modelo.
    
    **Detectores disponibles:**
    - `yolo`: YOLOv11n - Mayor velocidad, ideal para tiempo real
    - `rtdetr`: RT-DETR - Mayor precisión, mejor en objetos pequeños
    - `ensemble`: Combina ambos con NMS para máxima robustez
    
    **Pipeline:**
    1. Detecta vehículos en la imagen (opcional)
    2. Detecta placas usando el modelo seleccionado
    3. Lee texto con OCR (EasyOCR)
    4. Clasifica calidad de placa (opcional, Gemini)
    """
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")
    
    t0 = time.time()
    
    # Leer imagen
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")
    
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
    
    plates_found = []
    
    # Pipeline con o sin detección de vehículos
    if include_vehicle:
        vehicles = detect_vehicles(image)
        
        if vehicles:
            for vehicle in vehicles:
                plates_in_vehicle = detect_fn(vehicle["image"])
                
                for plate in plates_in_vehicle:
                    ocr = read_plate(plate["image"])
                    ocr_text = ocr["plate"] if ocr else "No detectado"
                    ocr_confidence = ocr["confidence"] if ocr else 0.0
                    
                    # Ajustar bbox al sistema de coordenadas original
                    vx1, vy1 = vehicle["bbox"][0], vehicle["bbox"][1]
                    abs_bbox = [
                        plate["bbox"][0] + vx1,
                        plate["bbox"][1] + vy1,
                        plate["bbox"][2] + vx1,
                        plate["bbox"][3] + vy1,
                    ]
                    
                    labels = classify_plate(plate["image"], ocr_confidence) if include_labels else None
                    
                    plates_found.append(PlateDetection(
                        plate=ocr_text,
                        ocr_confidence=ocr_confidence,
                        detector_confidence=plate["confidence"],
                        detector=plate.get("detector", detector_name),
                        bbox=abs_bbox,
                        labels=labels,
                        vehicle={
                            "type": vehicle["type"],
                            "type_es": vehicle["type_es"],
                            "bbox": vehicle["bbox"],
                            "confidence": vehicle["confidence"],
                        }
                    ))
        else:
            # Sin vehículos detectados, buscar placas en imagen completa
            plates_in_image = detect_fn(image)
            
            for plate in plates_in_image:
                ocr = read_plate(plate["image"])
                ocr_text = ocr["plate"] if ocr else "No detectado"
                ocr_confidence = ocr["confidence"] if ocr else 0.0
                labels = classify_plate(plate["image"], ocr_confidence) if include_labels else None
                
                plates_found.append(PlateDetection(
                    plate=ocr_text,
                    ocr_confidence=ocr_confidence,
                    detector_confidence=plate["confidence"],
                    detector=plate.get("detector", detector_name),
                    bbox=plate["bbox"],
                    labels=labels,
                    vehicle=None
                ))
    else:
        # Sin detección de vehículos
        plates_in_image = detect_fn(image)
        
        for plate in plates_in_image:
            ocr = read_plate(plate["image"])
            ocr_text = ocr["plate"] if ocr else "No detectado"
            ocr_confidence = ocr["confidence"] if ocr else 0.0
            labels = classify_plate(plate["image"], ocr_confidence) if include_labels else None
            
            plates_found.append(PlateDetection(
                plate=ocr_text,
                ocr_confidence=ocr_confidence,
                detector_confidence=plate["confidence"],
                detector=plate.get("detector", detector_name),
                bbox=plate["bbox"],
                labels=labels,
                vehicle=None
            ))
    
    processing_time = int((time.time() - t0) * 1000)
    
    return DetectionResponse(
        success=True,
        total=len(plates_found),
        vehicles=len(vehicles) if include_vehicle else 0,  # ← add this
        detector_used=detector_name,
        processing_time_ms=processing_time,
        plates=plates_found
    )


# ── Endpoint: Modo Comparación YOLO vs RT-DETR ─────────────────────────────────
@router.post("/detect/compare", response_model=CompareResponse, tags=["Detección Multi-Modelo"])
async def detect_compare(
    file: UploadFile = File(...),
    include_ensemble: bool = Query(
        default=True,
        description="Incluir resultado ensemble (NMS de ambos detectores)"
    )
):
    """
    Ejecuta YOLO y RT-DETR en paralelo para comparar resultados.
    
    Útil para:
    - Benchmarking de modelos
    - Validación cruzada de detecciones
    - Análisis de fortalezas/debilidades de cada modelo
    
    Retorna métricas comparativas:
    - Tiempo de procesamiento de cada modelo
    - Diferencias en detecciones
    - IoU entre detecciones coincidentes
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")
    
    # Leer imagen
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")
    
    # ── Ejecutar YOLO ──────────────────────────────────────────────────────────
    t0_yolo = time.time()
    yolo_plates_raw = detect_plate_yolo(image)
    yolo_time = int((time.time() - t0_yolo) * 1000)
    
    yolo_plates = []
    for plate in yolo_plates_raw:
        ocr = read_plate(plate["image"])
        yolo_plates.append(PlateDetection(
            plate=ocr["plate"] if ocr else "No detectado",
            ocr_confidence=ocr["confidence"] if ocr else 0.0,
            detector_confidence=plate["confidence"],
            detector="yolo",
            bbox=plate["bbox"],
            labels=None,
            vehicle=None
        ))
    
    yolo_response = DetectionResponse(
        success=True,
        total=len(yolo_plates),
        detector_used="yolo",
        processing_time_ms=yolo_time,
        plates=yolo_plates
    )
    
    # ── Ejecutar RT-DETR ───────────────────────────────────────────────────────
    t0_rtdetr = time.time()
    rtdetr_plates_raw = detect_plate_rtdetr(image)
    rtdetr_time = int((time.time() - t0_rtdetr) * 1000)
    
    rtdetr_plates = []
    for plate in rtdetr_plates_raw:
        ocr = read_plate(plate["image"])
        rtdetr_plates.append(PlateDetection(
            plate=ocr["plate"] if ocr else "No detectado",
            ocr_confidence=ocr["confidence"] if ocr else 0.0,
            detector_confidence=plate["confidence"],
            detector="rtdetr",
            bbox=plate["bbox"],
            labels=None,
            vehicle=None
        ))
    
    rtdetr_response = DetectionResponse(
        success=True,
        total=len(rtdetr_plates),
        detector_used="rtdetr",
        processing_time_ms=rtdetr_time,
        plates=rtdetr_plates
    )
    
    # ── Ejecutar Ensemble (opcional) ───────────────────────────────────────────
    ensemble_response = None
    if include_ensemble:
        t0_ensemble = time.time()
        ensemble_plates_raw = detect_plate_ensemble(image)
        ensemble_time = int((time.time() - t0_ensemble) * 1000)
        
        ensemble_plates = []
        for plate in ensemble_plates_raw:
            ocr = read_plate(plate["image"])
            ensemble_plates.append(PlateDetection(
                plate=ocr["plate"] if ocr else "No detectado",
                ocr_confidence=ocr["confidence"] if ocr else 0.0,
                detector_confidence=plate["confidence"],
                detector=plate.get("detector", "ensemble"),
                bbox=plate["bbox"],
                labels=None,
                vehicle=None
            ))
        
        ensemble_response = DetectionResponse(
            success=True,
            total=len(ensemble_plates),
            detector_used="ensemble",
            processing_time_ms=ensemble_time,
            plates=ensemble_plates
        )
    
    # ── Calcular métricas comparativas ─────────────────────────────────────────
    comparison = {
        "yolo_faster": yolo_time < rtdetr_time,
        "time_difference_ms": abs(yolo_time - rtdetr_time),
        "yolo_detections": len(yolo_plates),
        "rtdetr_detections": len(rtdetr_plates),
        "detection_difference": len(yolo_plates) - len(rtdetr_plates),
        "matching_plates": _count_matching_plates(yolo_plates, rtdetr_plates),
        "avg_yolo_confidence": _avg_confidence(yolo_plates),
        "avg_rtdetr_confidence": _avg_confidence(rtdetr_plates),
    }
    
    if include_ensemble and ensemble_response:
        comparison["ensemble_detections"] = len(ensemble_plates)
        comparison["avg_ensemble_confidence"] = _avg_confidence(ensemble_plates)
    
    return CompareResponse(
        success=True,
        yolo=yolo_response,
        rtdetr=rtdetr_response,
        ensemble=ensemble_response,
        comparison=comparison
    )


# ── Endpoint: Solo detección rápida (sin OCR) ──────────────────────────────────
@router.post("/detect/quick", tags=["Detección Multi-Modelo"])
async def detect_quick(
    file: UploadFile = File(...),
    detector: DetectorTypeEnum = Query(default=DetectorTypeEnum.YOLO)
):
    """
    Detección rápida de placas sin OCR ni clasificación.
    Retorna solo bounding boxes y confianza del detector.
    
    Ideal para:
    - Previsualización en tiempo real
    - Conteo de placas
    - Validación de encuadre de imagen
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")
    
    t0 = time.time()
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")
    
    # Seleccionar detector
    if detector == DetectorTypeEnum.YOLO:
        plates = detect_plate_yolo(image)
        detector_name = "yolo"
    elif detector == DetectorTypeEnum.RTDETR:
        plates = detect_plate_rtdetr(image)
        detector_name = "rtdetr"
    else:
        plates = detect_plate_ensemble(image)
        detector_name = "ensemble"
    
    processing_time = int((time.time() - t0) * 1000)
    
    return {
        "success": True,
        "total": len(plates),
        "detector": detector_name,
        "processing_time_ms": processing_time,
        "detections": [
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "detector": p.get("detector", detector_name)
            }
            for p in plates
        ]
    }


# ── Endpoint: Información de detectores disponibles ────────────────────────────
@router.get("/detect/models", tags=["Detección Multi-Modelo"])
async def get_available_models():
    """
    Retorna información sobre los modelos de detección disponibles.
    """
    return {
        "available_detectors": DetectorConfig.list_available_detectors(),
        "default_detector": DetectorConfig.DEFAULT_DETECTOR.value,
        "models": {
            "yolo": {
                "name": "YOLOv11n",
                "description": "Detector rápido basado en arquitectura YOLO",
                "available": DetectorConfig.is_model_available(DetectorType.YOLO),
                "confidence_threshold": DetectorConfig.YOLO_CONFIDENCE_THRESHOLD,
                "strengths": ["Velocidad", "Tiempo real", "Bajo consumo de memoria"],
                "weaknesses": ["Menor precisión en objetos pequeños"]
            },
            "rtdetr": {
                "name": "RT-DETR",
                "description": "Detector transformer end-to-end",
                "available": DetectorConfig.is_model_available(DetectorType.RTDETR),
                "confidence_threshold": DetectorConfig.RTDETR_CONFIDENCE_THRESHOLD,
                "strengths": ["Alta precisión", "Sin anchor boxes", "Mejor en oclusiones"],
                "weaknesses": ["Mayor tiempo de inferencia", "Mayor uso de memoria"]
            },
            "ensemble": {
                "name": "Ensemble (YOLO + RT-DETR)",
                "description": "Combina ambos detectores con NMS",
                "available": DetectorConfig.is_model_available(DetectorType.ENSEMBLE),
                "nms_threshold": DetectorConfig.ENSEMBLE_NMS_THRESHOLD,
                "strengths": ["Máxima robustez", "Menor tasa de falsos negativos"],
                "weaknesses": ["Mayor tiempo de procesamiento"]
            }
        }
    }


# ── Endpoint: Imagen anotada con detecciones ───────────────────────────────────
@router.post("/detect/annotate", tags=["Detección Multi-Modelo"])
async def detect_and_annotate(
    file: UploadFile = File(...),
    detector: DetectorTypeEnum = Query(default=DetectorTypeEnum.ENSEMBLE),
    show_confidence: bool = Query(default=True),
    show_plate_text: bool = Query(default=True)
):
    """
    Detecta placas y retorna la imagen anotada con bounding boxes.
    
    Retorna imagen en base64 con las detecciones dibujadas.
    Útil para visualización directa en el frontend.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")
    
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")
    
    # Colores por detector
    COLORS = {
        "yolo": (0, 255, 136),      # Verde neón
        "rtdetr": (238, 211, 34),   # Cyan
        "ensemble": (246, 130, 59), # Azul
    }
    
    # Seleccionar detector
    if detector == DetectorTypeEnum.YOLO:
        plates = detect_plate_yolo(image)
        detector_name = "yolo"
    elif detector == DetectorTypeEnum.RTDETR:
        plates = detect_plate_rtdetr(image)
        detector_name = "rtdetr"
    else:
        plates = detect_plate_ensemble(image)
        detector_name = "ensemble"
    
    annotated = image.copy()
    results = []
    
    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        det_type = plate.get("detector", detector_name)
        color = COLORS.get(det_type, (255, 255, 255))
        
        # Dibujar bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Texto a mostrar
        label_parts = []
        
        if show_confidence:
            label_parts.append(f"{plate['confidence']*100:.0f}%")
        
        ocr_result = None
        if show_plate_text:
            ocr_result = read_plate(plate["image"])
            if ocr_result:
                label_parts.append(ocr_result["plate"])
        
        if label_parts:
            label = " | ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
            cv2.putText(annotated, label, (x1+4, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        results.append({
            "bbox": plate["bbox"],
            "confidence": plate["confidence"],
            "detector": det_type,
            "plate": ocr_result["plate"] if ocr_result else None,
            "ocr_confidence": ocr_result["confidence"] if ocr_result else None
        })
    
    # Codificar imagen a base64
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "success": True,
        "total": len(results),
        "detector": detector_name,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
        "detections": results
    }


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def _count_matching_plates(yolo_plates: List[PlateDetection], rtdetr_plates: List[PlateDetection]) -> int:
    """Cuenta placas con texto coincidente entre ambos detectores."""
    if not yolo_plates or not rtdetr_plates:
        return 0
    
    yolo_texts = {p.plate for p in yolo_plates if p.plate != "No detectado"}
    rtdetr_texts = {p.plate for p in rtdetr_plates if p.plate != "No detectado"}
    
    return len(yolo_texts.intersection(rtdetr_texts))


def _avg_confidence(plates: List[PlateDetection]) -> float:
    """Calcula la confianza promedio de las detecciones."""
    if not plates:
        return 0.0
    return round(sum(p.detector_confidence for p in plates) / len(plates), 4)
