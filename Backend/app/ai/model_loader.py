# app/ai/model_loader.py
# Carga y precalienta los 3 modelos en paralelo durante el startup de FastAPI.
# Usar load_all_models() desde el lifespan de main.py

import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ── Estado global de modelos 
_models_ready = False
_load_time_ms: float = 0.0

# ── Precalentamiento con imagen dummy 
_DUMMY_BGR = np.zeros((64, 64, 3), dtype=np.uint8)  # 64x64 negra


def _warmup_yolo(model) -> None:
    """Una inferencia dummy para que YOLO JIT-compile el grafo."""
    try:
        model(_DUMMY_BGR, verbose=False)
    except Exception:
        pass


def _warmup_easyocr(reader) -> None:
    """Inferencia dummy para que EasyOCR inicialice sus buffers internos."""
    try:
        reader.readtext(_DUMMY_BGR, detail=0)
    except Exception:
        pass


# ── Cargadores individuales
def _load_vehicle_model():
    from app.ai.vehicle_detector import _get_model as _get_vehicle
    model = _get_vehicle()
    _warmup_yolo(model)
    print("[model_loader] vehicle_detector listo")
    return model


def _load_plate_model():
    from app.ai.plate_detector import _get_model as _get_plate
    model = _get_plate()
    _warmup_yolo(model)
    print("[model_loader] plate_detector listo")
    return model


def _load_ocr_model():
    # Fuerza la inicialización del reader y el modelo SR
    from app.ai import plate_reader as pr
    reader = pr._get_reader()    # inicializa EasyOCR
    _warmup_easyocr(reader)
    pr._get_sr()                 # precarga SR si existe
    print("[model_loader] plate_reader (EasyOCR + SR) listo")


# ── Carga en paralelo 
def load_all_models() -> None:
    """
    Carga los 3 modelos en paralelo usando un ThreadPoolExecutor.
    Llamar una sola vez desde el lifespan de FastAPI.
    """
    global _models_ready, _load_time_ms

    t0 = time.perf_counter()
    print("[model_loader] Iniciando carga paralela de modelos...")

    loaders = [_load_vehicle_model, _load_plate_model, _load_ocr_model]

    with ThreadPoolExecutor(max_workers=len(loaders), thread_name_prefix="model_") as pool:
        futures = [pool.submit(fn) for fn in loaders]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"[model_loader] ERROR cargando modelo: {e}")

    _load_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    _models_ready = True
    print(f"[model_loader] Todos los modelos listos en {_load_time_ms} ms")


def get_status() -> dict:
    """Retorna el estado de carga para el endpoint /health."""
    return {
        "models_ready": _models_ready,
        "load_time_ms": _load_time_ms,
    }
