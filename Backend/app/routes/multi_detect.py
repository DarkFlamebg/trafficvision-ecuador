# app/routes/detect.py
# Ruta /detect — recibe imagen y retorna placas detectadas con texto OCR
# Compara los tres detectores: YOLOv11n, RT-DETR y EfficientDet-D2

from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import base64

from app.ai.detectors.yolo        import detect_plate      as detect_plate_yolo
from app.ai.detectors.rtdetr      import detect_plate_rtdetr
from app.ai.detectors.vision_mamba import detect_plate_vision_mamba
from app.ai.plate_reader                import read_plate

router = APIRouter()

# Colores para anotación
PLATE_COLOR     = (0, 255, 255)   # Cian
PLATE_THICKNESS = 2


def _frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convierte un frame OpenCV a base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def _draw_plates_on_image(
    image: np.ndarray, detections: list, model_name: str
) -> np.ndarray:
    """
    Anota las placas detectadas en la imagen.

    Args:
        image:      array NumPy BGR
        detections: lista de dicts con bbox, detector_confidence, plate
        model_name: nombre del modelo para el panel superior

    Returns:
        Imagen anotada
    """
    output = image.copy()

    # Panel superior con info del modelo
    panel_text = f"Modelo: {model_name} | Placas: {len(detections)}"
    (tw, th), _ = cv2.getTextSize(panel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(output, (5, 5), (15 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(output, panel_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Dibujar cada placa
    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]

        # Caja alrededor de la placa
        cv2.rectangle(output, (x1, y1), (x2, y2), PLATE_COLOR, PLATE_THICKNESS)

        # ✅ usar detector_confidence (campo correcto del dict)
        plate_text = det.get("plate", "No detectado")
        conf       = det.get("detector_confidence", 0.0)
        label_text = f"#{idx}: {plate_text} ({conf:.1%})"

        # Fondo para el texto
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output,
                      (x1, y1 - lh - 12),
                      (x1 + lw + 8, y1),
                      PLATE_COLOR, -1)
        cv2.putText(output, label_text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return output


def _process_detections(plates_raw: list) -> list:
    """
    Procesa detecciones raw: ejecuta OCR y formatea resultados.

    Args:
        plates_raw: lista de dicts con "image", "bbox", "confidence"

    Returns:
        Lista de dicts con plate, ocr_confidence, detector_confidence, bbox
    """
    results = []
    for plate in plates_raw:
        ocr = read_plate(plate["image"])
        if ocr is None:
            continue
        results.append({
            "plate":               ocr["plate"],
            "ocr_confidence":      round(float(ocr["confidence"]), 4),
            "detector_confidence": round(float(plate["confidence"]), 4),
            "bbox":                plate["bbox"],
        })
    return results


def _run_detector(image: np.ndarray, detector_fn, model_name: str) -> dict:
    """
    Ejecuta un detector completo y devuelve su bloque de respuesta.
    Si el detector falla, retorna resultado vacío sin romper el endpoint.
    """
    try:
        plates_raw = detector_fn(image)
        detections = _process_detections(plates_raw)
        annotated  = _draw_plates_on_image(image, detections, model_name)
        image_b64  = _frame_to_base64(annotated)
    except Exception as e:
        print(f"[detect] Error en {model_name}: {e}")
        detections = []
        image_b64  = _frame_to_base64(image)   # imagen original sin anotar

    return {
        "model":        model_name,
        "total":        len(detections),
        "detections":   detections,
        "image_base64": image_b64,
    }


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detección con los tres modelos: YOLOv11n, RT-DETR y EfficientDet-D2.

    Pipeline por cada modelo:
      1. Detectar placa en la imagen completa
      2. OCR con EasyOCR sobre cada recorte
      3. Retornar imagen anotada + métricas

    Response:
        {
        return { "yolo": { model, total, detections, image_base64 },
                 "rtdetr": { model, total, detections, image_base64 },
                 "vision_mamba": { model, total, detections, image_base64 },
                 "summary": { yolo_plates, rtdetr_plates,
                            vision_mamba_plates, total_unique }
        }
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes JPG o PNG")

    contents = await file.read()
    npimg    = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen")

    # ── Ejecutar los tres detectores ──────────────────────────────────────────
    yolo_result   = _run_detector(image, detect_plate_yolo,         "YOLOv11n")
    rtdetr_result = _run_detector(image, detect_plate_rtdetr,       "RT-DETR")
    vm_result     = _run_detector(image, detect_plate_vision_mamba, "Vision Mamba")

    # Placas únicas detectadas entre los tres modelos
    all_plates = set(
        d["plate"]
        for block in [yolo_result, rtdetr_result, vm_result]
        for d in block["detections"]
        if d.get("plate")
    )

    return {
        "yolo":         yolo_result,
        "rtdetr":       rtdetr_result,
        "vision_mamba": vm_result,
        "summary": {
            "yolo_plates":         yolo_result["total"],
            "rtdetr_plates":       rtdetr_result["total"],
            "vision_mamba_plates": vm_result["total"],
            "total_unique":        len(all_plates),
        },
    }