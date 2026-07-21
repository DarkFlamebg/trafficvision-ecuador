# app/ai/detectors/__init__.py
# Sub-paquete de detectores de placas vehiculares.
# Exporta la interfaz unificada de todos los detectores disponibles.

from app.ai.detectors.yolo        import detect_plate          # YOLOv11n
from app.ai.detectors.rtdetr      import detect_plate_rtdetr   # RT-DETR L
from app.ai.detectors.efficientdet import detect_plate_efficientdet  # EfficientDet-D2
from app.ai.detectors.vision_mamba import detect_plate_vision_mamba  # Vision Mamba

__all__ = [
    "detect_plate",
    "detect_plate_rtdetr",
    "detect_plate_efficientdet",
    "detect_plate_vision_mamba",
]
