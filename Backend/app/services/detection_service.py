# app/services/detection_service.py
# Pipeline completo de detección de placas y vehículos.
# Extrae la lógica de negocio de main.py para que los routers
# sean solo capas de transporte HTTP/WS.

from app.ai.vehicle_detector      import detect_vehicles
from app.ai.detectors.yolo        import detect_plate
from app.ai.plate_reader          import read_plate
from app.ai.plate_classifier      import classify_plate


def run_plate_pipeline(image_path: str) -> tuple[list[dict], int]:
    """
    Ejecuta el pipeline completo:
      1. Detecta vehículos (YOLOv8n COCO)
      2. En cada vehículo, detecta placas (YOLOv11n entrenado)
      3. OCR sobre cada placa (EasyOCR)
      4. Clasifica calidad (Gemini Flash Vision)

    Si no hay vehículos detectados, busca placas en la imagen completa.

    Args:
        image_path: Ruta al archivo de imagen (temporal).

    Returns:
        Tuple (plates_found, num_vehicles_detected)
    """
    vehicles     = detect_vehicles(image_path)
    plates_found = []

    if vehicles:
        for vehicle in vehicles:
            plates_in_vehicle = detect_plate(vehicle["image"])

            for plate in plates_in_vehicle:
                ocr            = read_plate(plate["image"])
                ocr_text       = ocr["plate"]      if ocr else "No detectado"
                ocr_confidence = ocr["confidence"] if ocr else 0.0
                labels         = classify_plate(plate["image"], ocr_confidence)

                # Convertir coordenadas de placa relativas al vehículo → absolutas
                vx1, vy1 = vehicle["bbox"][0], vehicle["bbox"][1]
                abs_bbox  = [
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
                    },
                })
    else:
        print("[detection_service] No se detectaron vehículos — buscando placa en imagen completa")
        plates_in_image = detect_plate(image_path)

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
                "vehicle":         None,
            })

    return plates_found, len(vehicles)


def run_vehicle_detection(image_path: str) -> list[dict]:
    """
    Detecta únicamente vehículos sin procesar placas.

    Args:
        image_path: Ruta al archivo de imagen (temporal).

    Returns:
        Lista de dicts con type, type_es, bbox, confidence.
    """
    vehicles = detect_vehicles(image_path)
    return [
        {
            "type":       v["type"],
            "type_es":    v["type_es"],
            "bbox":       v["bbox"],
            "confidence": v["confidence"],
        }
        for v in vehicles
    ]
