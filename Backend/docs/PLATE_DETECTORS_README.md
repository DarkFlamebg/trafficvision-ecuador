# 🚗 Detectores de Placas - Documentación

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Estructura del Directorio](#estructura-del-directorio)
3. [Modelos Disponibles](#modelos-disponibles)
4. [Uso Básico](#uso-básico)
5. [Comparación de Modelos](#comparación-de-modelos)
6. [Benchmarking](#benchmarking)
7. [Integración con API](#integración-con-api)

---

## 🎯 Introducción

El subsistema de detección cuenta con **3 modelos activos** y **1 inactivo** especializados en la localización de placas vehiculares ecuatorianas (autos, camiones y motos). Estos detectores operan como primera etapa de la inferencia, previo a la lectura OCR.

---

## 📂 Estructura del Directorio

Todos los detectores han sido refactorizados y residen en `app/ai/detectors/`:

- `yolo.py`: Implementación de YOLOv11n.
- `rtdetr.py`: Implementación de RT-DETR.
- `vision_mamba.py`: Implementación de Vision Mamba (mmdetection).
- `efficientdet.py`: Implementación de EfficientDet (inactivo / legacy).
- `crop_utils.py`: Utilidades compartidas (rotación, padding).
- `config.py`: Configuraciones comunes.

---

## 🤖 Modelos Disponibles

### 1️⃣ YOLOv11n (`yolo.py`)

**Características:**
- ✅ Velocidad de inferencia ultra rápida (~20ms en CPU).
- ✅ Modelo muy ligero.
- ✅ Bueno para detección en tiempo real (video streaming).
- ⚠️ Puede tener problemas con placas muy pequeñas u ocluidas.

**Cuándo usar:** Aplicaciones en tiempo real (video), recursos limitados, procesamiento masivo rápido.

---

### 2️⃣ RT-DETR (`rtdetr.py`)

**Características:**
- ✅ Mayor precisión en detección de objetos pequeños (placas lejanas).
- ✅ Mejor manejo de oclusiones parciales y condiciones adversas.
- ✅ Arquitectura transformer end-to-end (sin NMS clásico).
- ⚠️ Más lento que YOLO.

**Cuándo usar:** Imágenes estáticas, controles de seguridad como peajes, cuando la precisión importa más que el tiempo real.

---

### 3️⃣ Vision Mamba (`vision_mamba.py`)

**Características:**
- ✅ Novedosa arquitectura State-Space Models (SSM).
- ✅ Basado en el framework MMDetection (utiliza pesos `swin_r4`).
- ✅ Excelente capacidad para extraer contexto global de la imagen con costo computacional casi lineal.
- ⚠️ Requiere MMDetection y dependencias específicas.

**Cuándo usar:** Comparativas de State-of-the-Art (SOTA), evaluaciones analíticas, y ambientes de investigación.

---

## 💻 Uso Básico

Todos los detectores comparten la misma interfaz y devuelven una lista de recortes (bounding boxes).

```python
# Importar el detector deseado
from app.ai.detectors.yolo import detect_plate

# La entrada puede ser una ruta (str) o un numpy array BGR
detecciones = detect_plate("imagen_auto.jpg")

for d in detecciones:
    print(f"Confianza: {d['detector_confidence']:.2f}")
    print(f"Bounding Box: {d['bbox']}")
    # d['plate'] contiene la imagen numpy recortada y procesada
```

### Preprocesamiento Automático
Internamente, los modelos aplican lógicas específicas a través de `crop_utils.py`:
1. **Ratio de aspecto:** Si es placa de moto (cuadrada/vertical), se rota automáticamente 90°.
2. **Padding:** Agregan márgenes (15-20%) para ayudar al OCR.
3. **Escalado:** Aplican Super Resolución (SR) vía FSRCNN a los parches muy pequeños.

---

## 📊 Comparación de Modelos

Puedes evaluar los tres modelos desde el frontend utilizando la interfaz gráfica.

**Rutas de API:**
- `POST /api/v1/detect`: Ejecuta detección con los 3 modelos simultáneamente y compara resultados.
- `POST /api/v1/compare/image?model=yolo|rtdetr|mamba`: Detección individual para el comparador web.
- `WS /api/v1/compare/video`: Comparación vía websockets para video.

---

## 📈 Benchmarking

Para pruebas cuantitativas con datasets masivos (validación de mAP, latencia y uso de memoria), se utiliza el script interno:

```bash
# Ejecutar benchmark de los detectores
python app/ai/plate_detector_benchmark.py
```

Esto consumirá las imágenes de evaluación y probará el desempeño de YOLO, RT-DETR y Vision Mamba.

---

## 🔌 Integración con API Principal

El endpoint principal del sistema para consumo final utiliza **YOLOv11n** por su velocidad como estándar de producción.

- **Detección Simple:** `POST /api/v1/detection/plate`
- **Pipeline:** Vehículo -> Placa -> OCR -> SR -> JSON Final.
