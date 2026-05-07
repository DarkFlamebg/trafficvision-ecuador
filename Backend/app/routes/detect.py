# Ruta /detect — recibe imagen y retorna placas detectadas con texto OCR
# Usa ambos detectores: YOLOv11n y RT-DETR
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import base64

from app.ai.plate_detector import detect_plate as detect_plate_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr
from app.ai.plate_reader import read_plate

router = APIRouter()

# Colores para anotación
PLATE_COLOR = (0, 255, 255)  # Cian
PLATE_THICKNESS = 2


def _frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convierte un frame OpenCV a base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def _draw_plates_on_image(image: np.ndarray, detections: list, model_name: str) -> np.ndarray:
    """
    Anota las placas detectadas en la imagen.
    
    Args:
        image: array NumPy BGR
        detections: lista de dicts con bbox, confidence, plate
        model_name: nombre del modelo para el label
        
    Returns:
        Imagen anotada
    """
    output = image.copy()
    h, w = image.shape[:2]
    
    # Panel superior con info del modelo
    panel_text = f"Modelo: {model_name} | Placas: {len(detections)}"
    (text_width, text_height), _ = cv2.getTextSize(
        panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(output, (5, 5), (15 + text_width, 30 + text_height), (0, 0, 0), -1)
    cv2.putText(output, panel_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Dibujar cada placa
    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]
        
        # Caja alrededor de la placa
        cv2.rectangle(output, (x1, y1), (x2, y2), PLATE_COLOR, PLATE_THICKNESS)
        
        # Texto con placa detectada y confianza
        plate_text = det.get("plate", "No detectado")
        conf_text = f"{det.get('confidence', 0):.1%}"
        label_text = f"#{idx}: {plate_text} ({conf_text})"
        
        (label_width, label_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Fondo para el texto
        cv2.rectangle(
            output,
            (x1, y1 - label_height - 12),
            (x1 + label_width + 8, y1),
            PLATE_COLOR,
            -1
        )
        
        # Texto
        cv2.putText(
            output,
            label_text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return output


def _process_detections(image: np.ndarray, plates_raw: list) -> list:
    """
    Procesa detecciones raw: ejecuta OCR y formatea resultados.
    
    Args:
        image: imagen original (para contexto)
        plates_raw: lista de dicts con "image", "bbox", "confidence"
        
    Returns:
        Lista de dicts con plate, ocr_confidence, detector_confidence, bbox
    """
    results = []
    
    for plate in plates_raw:
        # OCR sobre el recorte
        ocr = read_plate(plate["image"])
        
        if ocr is None:
            continue  # Descartar detecciones sin texto legible
        
        results.append({
            "plate":                ocr["plate"],
            "ocr_confidence":       ocr["confidence"],
            "detector_confidence":  plate["confidence"],
            "bbox":                 plate["bbox"]
        })
    
    return results


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detección con ambos modelos: YOLOv11n y RT-DETR.
    
    Retorna:
        {
          "yolo": {
            "total": int,
            "detections": [...],
            "image_base64": str
          },
          "rtdetr": {
            "total": int,
            "detections": [...],
            "image_base64": str
          }
        }
    """
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    contents = await file.read()

    # Decodificar imagen
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")

    # ──────────────────────────────────────────────────────────────────────────
    # DETECTOR 1: YOLOv11n
    # ──────────────────────────────────────────────────────────────────────────
    plates_yolo_raw = detect_plate_yolo(image)
    yolo_detections = _process_detections(image, plates_yolo_raw)
    image_yolo_annotated = _draw_plates_on_image(image, yolo_detections, "YOLOv11n")
    image_yolo_b64 = _frame_to_base64(image_yolo_annotated)
    
    # ──────────────────────────────────────────────────────────────────────────
    # DETECTOR 2: RT-DETR
    # ──────────────────────────────────────────────────────────────────────────
    plates_rtdetr_raw = detect_plate_rtdetr(image)
    rtdetr_detections = _process_detections(image, plates_rtdetr_raw)
    image_rtdetr_annotated = _draw_plates_on_image(image, rtdetr_detections, "RT-DETR")
    image_rtdetr_b64 = _frame_to_base64(image_rtdetr_annotated)

    return {
        "yolo": {
            "model": "YOLOv11n",
            "total": len(yolo_detections),
            "detections": yolo_detections,
            "image_base64": image_yolo_b64
        },
        "rtdetr": {
            "model": "RT-DETR",
            "total": len(rtdetr_detections),
            "detections": rtdetr_detections,
            "image_base64": image_rtdetr_b64
        },
        "summary": {
            "yolo_plates": len(yolo_detections),
            "rtdetr_plates": len(rtdetr_detections),
            "total_unique": len(set(
                d["plate"] for d in yolo_detections + rtdetr_detections
                if d.get("plate")
            ))
        }
    }