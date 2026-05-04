# app/ai/plate_detector_rtdetr.py
# Detecta placas usando RT-DETR entrenado localmente con dataset ecuatoriano

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import RTDETR

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_PATH            = os.path.join(_ROOT_DIR, "ml", "models", "trained", "rtdetr_combined_all", "best.pt")
CONFIDENCE_THRESHOLD  = 0.45

# Proporción ancho/alto válida para una placa vehicular
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 6.0

# Modelo cargado una sola vez al importar
_model = None

def _get_model() -> RTDETR:
    """Carga el modelo RT-DETR una sola vez (singleton)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo RT-DETR no encontrado: {MODEL_PATH}")
        _model = RTDETR(MODEL_PATH)
    return _model


def _load_image(input_image) -> np.ndarray:
    """
    Carga la imagen respetando la orientación EXIF del celular.
    Retorna array NumPy BGR compatible con OpenCV.
    
    Args:
        input_image: ruta (str) o array NumPy BGR
        
    Returns:
        Array NumPy BGR
    """
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    # Corregir orientación EXIF (importante para fotos de celular)
    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_plate_rtdetr(input_image) -> list:
    """
    Detecta placas vehiculares usando RT-DETR local.

    Ventajas de RT-DETR sobre YOLO:
    - Mayor precisión en detección de objetos pequeños
    - Mejor manejo de oclusiones parciales
    - Arquitectura transformer sin anchor boxes
    - Inferencia end-to-end sin NMS post-procesamiento

    Filtros aplicados:
    - Confianza mínima: 0.45
    - Proporción ancho/alto: entre 1.5 y 6.0 (forma de placa real)

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa
          - "bbox":       [x1, y1, x2, y2] en píxeles (int)
          - "confidence": float 0.0 – 1.0
          - "detector":   str "rtdetr" (para identificar el modelo usado)
    """
    image  = _load_image(input_image)
    model  = _get_model()
    ih, iw = image.shape[:2]

    # Inferencia RT-DETR
    results = model(image, verbose=False)[0]
    plates  = []

    for box in results.boxes:
        conf = float(box.conf[0])
        
        # Filtro 1: Confianza mínima
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # Obtener coordenadas del bbox
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0,  int(x1))
        y1 = max(0,  int(y1))
        x2 = min(iw, int(x2))
        y2 = min(ih, int(y2))

        w_box = x2 - x1
        h_box = y2 - y1

        # Filtro 2: Evitar división por cero
        if h_box == 0:
            continue

        # Filtro 3: Proporción de aspecto (placas son horizontales)
        aspect_ratio = w_box / h_box
        if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
            print(f"[rtdetr] Bbox descartado por proporción: {w_box}x{h_box} = {aspect_ratio:.2f}")
            continue

        # Extraer recorte
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plates.append({
            "image":      crop,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "detector":   "rtdetr",  # Identificador del modelo
        })

    return plates


def detect_plate_ensemble(input_image, use_yolo: bool = True, use_rtdetr: bool = True) -> list:
    """
    Detector ensemble que combina YOLO y RT-DETR para mayor robustez.
    
    Estrategia:
    - Ejecuta ambos modelos en paralelo
    - Combina resultados usando NMS (Non-Maximum Suppression)
    - Elimina detecciones duplicadas
    - Retorna placas ordenadas por confianza
    
    Args:
        input_image: ruta (str) o array NumPy BGR
        use_yolo: activar detector YOLOv11n
        use_rtdetr: activar detector RT-DETR
        
    Returns:
        Lista de dicts con las mejores detecciones combinadas
    """
    all_plates = []
    
    # Detector YOLO
    if use_yolo:
        try:
            from app.ai.plate_detector import detect_plate
            yolo_plates = detect_plate(input_image)
            for p in yolo_plates:
                p["detector"] = "yolo"
                all_plates.append(p)
        except Exception as e:
            print(f"[ensemble] Error en YOLO: {e}")
    
    # Detector RT-DETR
    if use_rtdetr:
        try:
            rtdetr_plates = detect_plate_rtdetr(input_image)
            all_plates.extend(rtdetr_plates)
        except Exception as e:
            print(f"[ensemble] Error en RT-DETR: {e}")
    
    if not all_plates:
        return []
    
    # Aplicar NMS para eliminar duplicados
    plates_nms = _apply_nms(all_plates, iou_threshold=0.5)
    
    # Ordenar por confianza descendente
    plates_nms.sort(key=lambda x: x["confidence"], reverse=True)
    
    return plates_nms


def _apply_nms(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Aplica Non-Maximum Suppression para eliminar detecciones duplicadas.
    
    Args:
        detections: lista de dicts con bbox y confidence
        iou_threshold: umbral de IoU para considerar duplicados
        
    Returns:
        Lista filtrada de detecciones
    """
    if len(detections) <= 1:
        return detections
    
    # Ordenar por confianza descendente
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    
    while detections:
        # Tomar la detección con mayor confianza
        best = detections.pop(0)
        keep.append(best)
        
        # Eliminar detecciones que se solapan mucho con la mejor
        detections = [
            det for det in detections
            if _calculate_iou(best["bbox"], det["bbox"]) < iou_threshold
        ]
    
    return keep


def _calculate_iou(box1: list, box2: list) -> float:
    """
    Calcula Intersection over Union entre dos bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score entre 0.0 y 1.0
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Área de intersección
    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h
    
    # Área de cada bbox
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Área de unión
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


# ── Testing ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Script de prueba para verificar el funcionamiento del detector.
    
    Uso:
        python plate_detector_rtdetr.py <ruta_imagen>
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python plate_detector_rtdetr.py <ruta_imagen>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"🔍 Probando RT-DETR Plate Detector")
    print(f"{'='*60}\n")
    
    # Prueba 1: Solo RT-DETR
    print("📍 Test 1: RT-DETR")
    plates_rtdetr = detect_plate_rtdetr(image_path)
    print(f"   Placas detectadas: {len(plates_rtdetr)}")
    for i, p in enumerate(plates_rtdetr, 1):
        print(f"   {i}. Confianza: {p['confidence']:.2%} | Bbox: {p['bbox']}")
    
    # Prueba 2: Ensemble (YOLO + RT-DETR)
    print("\n📍 Test 2: Ensemble (YOLO + RT-DETR)")
    try:
        plates_ensemble = detect_plate_ensemble(image_path)
        print(f"   Placas detectadas: {len(plates_ensemble)}")
        for i, p in enumerate(plates_ensemble, 1):
            detector = p.get('detector', 'unknown')
            print(f"   {i}. Detector: {detector} | Confianza: {p['confidence']:.2%} | Bbox: {p['bbox']}")
    except Exception as e:
        print(f"   Error en ensemble: {e}")
    
    print(f"\n{'='*60}\n")
