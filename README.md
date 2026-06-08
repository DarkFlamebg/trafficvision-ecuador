# 🚗 TrafficVision
**Sistema de Identificación y Clasificación de Placas Vehiculares**  
Control Anticorrupción · Seguridad Vial · Ecuador 2026

---

## 📋 Estado del Sistema

| Componente | Estado | Métrica |
|---|---|---|
| Detección (Modelos) | ✅ Activo | YOLOv11n, RT-DETR, EfficientDet |
| Lectura OCR | ✅ Activo | EasyOCR + preprocesamiento por color |
| Clasificación IA | ✅ Activo | Google Gemini Flash (gratuito) |
| API Backend | ✅ Activo | FastAPI + Python 3.11 |
| Frontend | ✅ Activo | React + TS + MUI + Chart.js |

---

## 📌 Descripción

TrafficVision es un sistema de inteligencia artificial para la **detección, lectura y clasificación de placas vehiculares ecuatorianas** en tiempo real. Está orientado al control anticorrupción y la seguridad vial, permitiendo identificar vehículos en imágenes tomadas desde cámaras estáticas o en movimiento, incluyendo condiciones nocturnas y lluvia.

Desarrollado como proyecto de tesis con énfasis en:
- Comparación de **arquitecturas de detección** (YOLOv11, RT-DETR, EfficientDet-D2)
- Sistema OCR optimizado (EasyOCR)
- Clasificación multi-atributo de calidad de imagen usando visión artificial
- Entrenamiento con datasets ecuatorianos propios y globales

---

## 🔄 ¿Qué es un Pipeline?

Un **pipeline** (cadena de procesamiento) es una serie de pasos conectados en secuencia, donde la salida de cada etapa se convierte en la entrada de la siguiente. Es como una línea de ensamblaje: cada estación hace una tarea específica y pasa el resultado a la siguiente.

En TrafficVision el pipeline funciona así:

```
Imagen JPG/PNG
      ↓
[1] Detección de vehículo    → YOLOv8n (clases: car, truck, motorcycle, bus)
      ↓
[2] Detección de placa    → YOLOv8n (97.4% mAP@50)
      ↓
[3] Filtrado              → Confianza >45% + proporción 1.5-6.0
      ↓
[4] Lectura OCR           → EasyOCR + preprocesamiento por color
      ↓
[5] Clasificación calidad → Google Gemini Flash Vision
      ↓
JSON: vehículo + placa + confianzas + etiquetas de calidad
```

---

## 🧠 Pipeline Técnico Detallado

### Etapa 1 — Detección (`plate_detector.py`)

YOLOv8n entrenado con **10,734 imágenes** (global + ecuatorianas). Red neuronal convolucional one-stage que localiza y clasifica la placa en una sola pasada.

Filtros post-detección:
- Confianza mínima: **45%**
- Proporción ancho/alto: **1.5 a 6.0** (forma real de placa)
- Corrección EXIF automática para fotos de celular

### Etapa 2 — Lectura OCR (`plate_reader.py`)

Preprocesamiento inteligente antes de EasyOCR:
- Recorte del **15-22% superior** — elimina etiquetas `ECUA`, `PLACA PROVISIONAL`
- Escalado mínimo a **300px** de ancho
- Detección de color de fondo (HSV) — procesamiento diferente para placas verdes, naranjas, azules y blancas
- Corrección de errores por posición — letras pos. 0-2, números pos. 3-6
- Formato automático `AAA-0000`

### Etapa 3 — Clasificación de Calidad (`plate_classifier.py`)

Google Gemini Flash Vision clasifica 4 atributos:

| Atributo | Valores | Cómo se determina |
|---|---|---|
| Legibilidad | Legible / Ilegible | OCR confidence ≥ 10% → Legible |
| Oclusión | No / Parcial / Severa | Gemini analiza visualmente |
| Reflejo | No / Sí | Gemini analiza visualmente |
| Suciedad | No / Sí | Gemini analiza visualmente |

---

## 📊 Modelos Entrenados y Comparados

| Modelo | Descripción | Estado |
|---|---|---|
| **YOLOv11n** | Single-shot CNN - Máxima velocidad | ✅ Principal |
| **RT-DETR** | Transformer + CNN - Alta precisión | ✅ Comparativa |
| **EfficientDet-D2** | BiFPN + EfficientNet - Balance eficiente | ✅ Comparativa |

El entrenamiento se gestiona comparando mAP, Precisión, Recall y F1-Score usando un dashboard interactivo en React + Chart.js leyendo directamente los `results.csv`.

---

## 🗂️ Datasets

| Dataset | Imágenes | Descripción |
|---|---|---|
| license-plates (global) | 10,125 | Roboflow Universe — placas mundiales |
| license-plates-ec-1 | 144 | Placas ecuatorianas — Roboflow |
| license-plates-ec-2 | 90 | Placas ecuatorianas — dataset adicional |
| license-plates-ec-3 | — | OCR de caracteres (36 clases) — pendiente |
| license-plates-ec-4 | 375 | **Dataset personal** — fotos propias |

---

## 🏗️ Estructura del Proyecto

```
TrafficVision Datasets/
│
├── Backend/                    ← API principal (FastAPI)
│   ├── app/
│   │   ├── ai/                 ← plate_detector, plate_reader, plate_classifier
│   │   ├── core/
│   │   ├── database/
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── services/
│   ├── temp/
│   ├── main.py
│   ├── requirements.txt
│   └── .env                    ← ROBOFLOW_API_KEY + GEMINI_API_KEY
│
├── ml/                         ← todo lo de IA
│   ├── datasets/raw/           ← datasets originales
│   ├── models/
│   │   ├── base/               ← yolov8n.pt
│   │   └── trained/            ← best.pt de cada experimento
│   ├── runs/                   ← métricas y gráficas
│   ├── training/               ← train_yolov8.py
│   └── notebooks/              ← Google Colab (.ipynb)
│
├── dataset_service/            ← microservicio datasets (en desarrollo)
├── storage/                    ← imágenes y detecciones en producción
├── frontend/                   ← React + TypeScript + MUI
└── database/                   ← esquemas de BD
```

---

## 🚀 Instalación

### Backend

```powershell
# 1. Crear entorno virtual con Python 3.11
py -3.11 -m venv venv311
venv311\Scripts\activate

# 2. Instalar dependencias
py -m pip install -r requirements.txt

# 3. Configurar .env
ROBOFLOW_API_KEY
GEMINI_API_KEY

# 4. Iniciar servidor
py -m uvicorn app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Ejemplo de respuesta

```json
{
  "total": 1,
  "plates": [{
    "bbox": [833, 1164, 1027, 1248],
    "yolo_confidence": 0.8649,
    "plate": "PFJ-2048",
    "ocr_confidence": 0.9910,
    "labels": {
      "legible": "Legible",
      "oclusion": "No",
      "reflejo": "No",
      "sucia": "No"
    }
  }]
}
```

---

## 🛠️ Tecnologías

| Categoría | Tecnología | Uso |
|---|---|---|
| Detección | YOLOv8n (Ultralytics) | Localización de placas |
| OCR | EasyOCR | Lectura de caracteres |
| IA Vision | Google Gemini Flash | Clasificación de calidad |
| Backend | FastAPI + Python 3.11 | API REST |
| Frontend | React + TypeScript + MUI | Interfaz de usuario |
| Entrenamiento | Google Colab (Tesla T4) | Entrenamiento GPU |
| Dataset | Roboflow Universe | Datasets anotados YOLO |
| Imagen | OpenCV + Pillow | Procesamiento de imágenes |

---

## 📈 Próximos Pasos / Progreso

- [x] Implementar RT-DETR y EfficientDet-D2 para comparativa de detección
- [x] Crear Dashboard Dinámico de Métricas de Entrenamiento con `Chart.js`
- [x] Completar redacción del Capítulo 3 de la Tesis (Metodología)
- [ ] Implementar dataset_service como microservicio independiente
- [ ] Agregar base de datos para logs de detecciones

---

*TrafficVision · Tesis de Grado · Ecuador 2026 · Backend: `localhost:8000` · Frontend: `localhost:5173`*