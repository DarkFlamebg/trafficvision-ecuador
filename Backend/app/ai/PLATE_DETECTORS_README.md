# 🚗 Detectores de Placas - Documentación

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Modelos Disponibles](#modelos-disponibles)
3. [Instalación](#instalación)
4. [Uso Básico](#uso-básico)
5. [Comparación de Modelos](#comparación-de-modelos)
6. [Benchmarking](#benchmarking)
7. [Integración con API](#integración-con-api)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Introducción

Este módulo proporciona **3 opciones** para detectar placas vehiculares ecuatorianas:

1. **YOLOv11n** - Rápido y ligero
2. **RT-DETR** - Mayor precisión, especialmente en casos difíciles
3. **Ensemble** - Combina ambos modelos para máxima robustez

---

## 🤖 Modelos Disponibles

### 1️⃣ YOLOv11n (`plate_detector.py`)

**Características:**
- ✅ Velocidad de inferencia muy rápida (~20-40ms)
- ✅ Modelo ligero (<10MB)
- ✅ Bueno para detección en tiempo real
- ⚠️ Puede tener problemas con placas muy pequeñas u ocluidas

**Cuándo usar:**
- Aplicaciones en tiempo real (video streaming)
- Dispositivos con recursos limitados
- Cuando la velocidad es prioritaria

**Ejemplo:**
```python
from plate_detector import detect_plate

plates = detect_plate("imagen.jpg")
for plate in plates:
    print(f"Bbox: {plate['bbox']}, Confianza: {plate['confidence']}")
```

---

### 2️⃣ RT-DETR (`plate_detector_rtdetr.py`)

**Características:**
- ✅ Mayor precisión en detección de objetos pequeños
- ✅ Mejor manejo de oclusiones parciales
- ✅ No requiere NMS (Non-Maximum Suppression)
- ✅ Arquitectura transformer end-to-end
- ⚠️ Más lento que YOLO (~50-100ms)

**Cuándo usar:**
- Imágenes estáticas de alta resolución
- Cuando la precisión es más importante que la velocidad
- Placas parcialmente ocluidas o con reflejos
- Dataset de validación/testing

**Ejemplo:**
```python
from plate_detector_rtdetr import detect_plate_rtdetr

plates = detect_plate_rtdetr("imagen.jpg")
for plate in plates:
    print(f"Bbox: {plate['bbox']}, Confianza: {plate['confidence']}")
    print(f"Detector usado: {plate['detector']}")  # 'rtdetr'
```

---

### 3️⃣ Ensemble (`plate_detector_rtdetr.py`)

**Características:**
- ✅ Combina YOLO + RT-DETR
- ✅ Máxima tasa de detección
- ✅ Elimina duplicados con NMS
- ✅ Retorna las mejores detecciones de ambos modelos
- ⚠️ Tiempo de inferencia = suma de ambos modelos

**Cuándo usar:**
- Validación crítica (peajes, parqueaderos)
- Cuando NO puedes permitirte perder ninguna placa
- Procesamiento batch (no tiempo real)

**Ejemplo:**
```python
from plate_detector_rtdetr import detect_plate_ensemble

# Usar ambos modelos
plates = detect_plate_ensemble("imagen.jpg")

# Usar solo RT-DETR
plates = detect_plate_ensemble("imagen.jpg", use_yolo=False, use_rtdetr=True)

# Usar solo YOLO
plates = detect_plate_ensemble("imagen.jpg", use_yolo=True, use_rtdetr=False)
```

---

## 📦 Instalación

### Prerrequisitos

```bash
pip install ultralytics opencv-python pillow numpy
```

### Estructura de Archivos

```
backend/
├── app/
│   ├── ai/
│   │   ├── plate_detector.py          # YOLOv11n
│   │   ├── plate_detector_rtdetr.py   # RT-DETR + Ensemble
│   │   ├── plate_detector_benchmark.py # Comparación
│   │   ├── plate_reader.py             # OCR
│   │   └── plate_classifier.py         # Gemini
│   └── routes/
│       └── detect.py
└── ml/
    └── models/
        └── trained/
            ├── yolo11n_combined_all/
            │   └── best.pt
            └── rtdetr_combined_all/
                └── best.pt
```

---

## 🚀 Uso Básico

### Opción 1: Importar directamente

```python
# YOLOv11n
from app.ai.plate_detector import detect_plate
plates = detect_plate("path/to/image.jpg")

# RT-DETR
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr
plates = detect_plate_rtdetr("path/to/image.jpg")

# Ensemble
from app.ai.plate_detector_rtdetr import detect_plate_ensemble
plates = detect_plate_ensemble("path/to/image.jpg")
```

### Opción 2: Usar desde línea de comandos

```bash
# Probar RT-DETR
python app/ai/plate_detector_rtdetr.py test_image.jpg

# Comparar modelos
python app/ai/plate_detector_benchmark.py test_images/
```

### Opción 3: Integración con FastAPI

```python
from fastapi import APIRouter, UploadFile
from app.ai.plate_detector_rtdetr import detect_plate_ensemble
import cv2
import numpy as np

router = APIRouter()

@router.post("/detect-plate")
async def detect_plate_endpoint(file: UploadFile):
    # Leer imagen
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detectar con ensemble
    plates = detect_plate_ensemble(image)
    
    return {
        "plates_detected": len(plates),
        "results": [
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "detector": p["detector"]
            }
            for p in plates
        ]
    }
```

---

## ⚖️ Comparación de Modelos

| Métrica | YOLOv11n | RT-DETR | Ensemble |
|---------|----------|---------|----------|
| **Velocidad** | 🟢 Muy rápida | 🟡 Media | 🔴 Lenta |
| **Precisión** | 🟢 Buena | 🟢 Excelente | 🟢 Excelente |
| **Placas pequeñas** | 🟡 Aceptable | 🟢 Excelente | 🟢 Excelente |
| **Oclusiones** | 🟡 Aceptable | 🟢 Muy bueno | 🟢 Muy bueno |
| **Consumo RAM** | 🟢 Bajo (~500MB) | 🟡 Medio (~800MB) | 🔴 Alto (~1.2GB) |
| **Tamaño modelo** | 🟢 Pequeño (~6MB) | 🟡 Medio (~15MB) | 🔴 Ambos (~21MB) |
| **Tiempo real** | ✅ Sí | ⚠️ Limitado | ❌ No |

### 💡 Recomendaciones por Caso de Uso

| Caso de Uso | Modelo Recomendado | Razón |
|-------------|-------------------|-------|
| Video streaming en vivo | YOLOv11n | Velocidad crítica |
| App móvil | YOLOv11n | Recursos limitados |
| Peaje automático | Ensemble | No perder ninguna placa |
| Parqueadero | RT-DETR o Ensemble | Precisión importante |
| Procesamiento batch | RT-DETR | Balance precisión/velocidad |
| Dataset de entrenamiento | Ensemble | Máxima cobertura |

---

## 📊 Benchmarking

### Ejecutar Benchmark Completo

```bash
python app/ai/plate_detector_benchmark.py test_images/
```

**Salida esperada:**
```
🔬 BENCHMARK: Comparación de Detectores de Placas
──────────────────────────────────────────────────

📁 Dataset: test_images/

🟦 Test 1/3: YOLOv11n
──────────────────────────────────────────────────
   Imágenes procesadas: 50
   Errores: 0
   
   🕐 Tiempo de Inferencia:
      Promedio: 28.45 ms
      Mínimo:   22.10 ms
      Máximo:   45.30 ms
   
   🎯 Detecciones:
      Total: 48
      Promedio por imagen: 0.96
   
   📈 Confianza:
      Promedio: 78.50%
      Mínima:   45.20%
      Máxima:   94.30%

🟩 Test 2/3: RT-DETR
──────────────────────────────────────────────────
   [Similar output...]

🟪 Test 3/3: Ensemble
──────────────────────────────────────────────────
   [Similar output...]

📊 TABLA COMPARATIVA
──────────────────────────────────────────────────

Métrica                        |    YOLOv11n |     RT-DETR |    Ensemble
───────────────────────────────┼─────────────┼─────────────┼─────────────
Tiempo inferencia (ms)         |       28.45 |       62.30 |       90.75
Detecciones totales            |          48 |          51 |          52
Detecciones/imagen             |        0.96 |        1.02 |        1.04
Confianza promedio             |      78.50% |      82.10% |      83.20%
Errores                        |           0 |           0 |           0

💡 RECOMENDACIÓN:
   🏃 Más rápido: YOLOv11n (28.45 ms)
   🎯 Más confiable: Ensemble (83.20%)
   📈 Más detecciones: Ensemble (52 placas)
```

---

## 🔌 Integración con API

### Modificar `routes/detect.py`

```python
from fastapi import APIRouter, UploadFile, Query
from app.ai.plate_detector import detect_plate as detect_yolo
from app.ai.plate_detector_rtdetr import detect_plate_rtdetr, detect_plate_ensemble
from app.ai.plate_reader import read_plate
from app.ai.plate_classifier import classify_plate

router = APIRouter()

@router.post("/detect/full")
async def detect_full_pipeline(
    file: UploadFile,
    detector: str = Query("ensemble", enum=["yolo", "rtdetr", "ensemble"])
):
    """
    Pipeline completo: Detectar → Leer → Clasificar
    
    Args:
        detector: 'yolo', 'rtdetr', o 'ensemble'
    """
    # 1. Cargar imagen
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Seleccionar detector
    if detector == "yolo":
        plates = detect_yolo(image)
    elif detector == "rtdetr":
        plates = detect_plate_rtdetr(image)
    else:  # ensemble
        plates = detect_plate_ensemble(image)
    
    # 3. Procesar cada placa
    results = []
    for plate_data in plates:
        crop = plate_data["image"]
        
        # Leer texto (OCR)
        ocr_result = read_plate(crop)
        
        # Clasificar calidad
        classification = classify_plate(
            crop, 
            ocr_confidence=ocr_result["confidence"] if ocr_result else 0.0
        )
        
        results.append({
            "bbox": plate_data["bbox"],
            "detector": plate_data.get("detector", detector),
            "detection_confidence": plate_data["confidence"],
            "plate_text": ocr_result["plate"] if ocr_result else None,
            "ocr_confidence": ocr_result["confidence"] if ocr_result else 0.0,
            "quality": classification
        })
    
    return {
        "detector_used": detector,
        "plates_found": len(results),
        "results": results
    }
```

### Probar con curl

```bash
# Usar YOLOv11n
curl -X POST "http://localhost:8000/detect/full?detector=yolo" \
  -F "file=@test.jpg"

# Usar RT-DETR
curl -X POST "http://localhost:8000/detect/full?detector=rtdetr" \
  -F "file=@test.jpg"

# Usar Ensemble
curl -X POST "http://localhost:8000/detect/full?detector=ensemble" \
  -F "file=@test.jpg"
```

---

## 🔧 Troubleshooting

### Error: "Modelo RT-DETR no encontrado"

**Solución:**
```bash
# Verificar que existe el archivo
ls -la ml/models/trained/rtdetr_combined_all/best.pt

# Si no existe, entrenar el modelo RT-DETR
# O ajustar la ruta en plate_detector_rtdetr.py
```

### Error: "cannot import name 'RTDETR'"

**Solución:**
```bash
# Actualizar ultralytics a versión que soporte RT-DETR
pip install --upgrade ultralytics

# Verificar versión
python -c "from ultralytics import RTDETR; print('OK')"
```

### Performance degradado en RT-DETR

**Solución:**
```python
# Activar GPU si está disponible
import torch
print(torch.cuda.is_available())  # Debe ser True

# Si es False, instalar CUDA toolkit
# O usar CPU con batch pequeños
```

### Ensemble detecta duplicados

**Solución:**
```python
# Ajustar umbral de NMS en plate_detector_rtdetr.py
plates_nms = _apply_nms(all_plates, iou_threshold=0.3)  # Más estricto
# O
plates_nms = _apply_nms(all_plates, iou_threshold=0.7)  # Más permisivo
```

---

## 📚 Referencias

- [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Model Ensemble Techniques](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)

---

## 📝 Changelog

### v1.0.0 (2025-01-XX)
- ✅ Implementación inicial de RT-DETR
- ✅ Sistema de ensemble YOLO + RT-DETR
- ✅ Script de benchmarking
- ✅ Documentación completa

---

## 👥 Contribuciones

Para agregar nuevos detectores:

1. Crear archivo `plate_detector_MODELO.py`
2. Seguir la misma estructura de retorno:
   ```python
   return [{
       "image": crop,
       "bbox": [x1, y1, x2, y2],
       "confidence": float,
       "detector": "nombre_modelo"
   }]
   ```
3. Agregar al benchmark
4. Actualizar documentación

---

**Mantenido por:** TrafficVision Team  
**Última actualización:** Abril 2026
