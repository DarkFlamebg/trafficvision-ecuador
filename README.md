# TrafficVision 🚦
<img width="1685" height="531" alt="Captura de pantalla 2026-07-17 132111" src="https://github.com/user-attachments/assets/5c229527-3435-4171-9d79-e967be7e47cf" />

**TrafficVision** es una plataforma integral e inteligente diseñada para la monitorización, análisis y gestión avanzada del tráfico vehicular. Utilizando un pipeline multimodal de Inteligencia Artificial, el sistema no solo detecta vehículos, sino que extrae la información de sus patentes (placas) y evalúa el estado físico de las mismas mediante el uso de Modelos Fundacionales (LLMs).

## ✨ Características Principales

- **Detección Vehicular en Tiempo Real:** Detección de múltiples clases de vehículos utilizando modelos optimizados.
- **Lectura Automática de Placas (ALPR/OCR):** Extracción precisa de los caracteres de la patente del vehículo.
- **Análisis de Estado Físico (IA Generativa):** Clasificación del estado de la placa (nivel de oclusión, suciedad, legibilidad general) mediante integración con Gemini 2.5 Flash.
- **Panel de Control Interactivo:** Visualización de métricas, flujos de tráfico y estadísticas en tiempo real.
- **Arquitectura Escalable:** Diseño modular basado en microservicios lógicos (Frontend React, Backend FastAPI).
<img width="1881" height="907" alt="Captura de pantalla 2026-07-17 131142" src="https://github.com/user-attachments/assets/6f987144-d04a-4743-b017-d4ffd20e4f15" />
---

## 📋 Estado del Sistema

| Componente | Estado | Métrica |
|---|---|---|
| Detección (Modelos) | ✅ Activo | YOLOv11n, RT-DETR, Vision Mamba |
| Lectura OCR | ✅ Activo | EasyOCR + preprocesamiento por color |
| Clasificación IA | ✅ Activo | Google Gemini Flash (gratuito) |
| API Backend | ✅ Activo | FastAPI + Python 3.11 |
| Frontend | ✅ Activo | React + TS + MUI + Chart.js |

---

## 🏗️ Arquitectura y Pipeline de Inteligencia Artificial

El núcleo de **TrafficVision** reside en su pipeline secuencial de procesamiento de imágenes y video, el cual se compone de tres etapas principales:

1. **Detección de Objetos (YOLOv11n & RT DETR & Vision Mamba):** 
   - Se utiliza **YOLOv11n** para la localización rápida y precisa de vehículos y sus respectivas placas en el frame.
   - Opcionalmente, se integra la arquitectura **Vision Mamba (Vim)** combinada con backbones Swin (`swin_r4`) para tareas específicas de detección avanzada.
2. **Reconocimiento Óptico de Caracteres (EasyOCR):** 
   - Una vez recortada la región de la placa, el motor **EasyOCR** extrae el texto alfanumérico.
3. **Clasificación y Análisis Semántico (Gemini 2.5 Flash):** 
   - La imagen de la placa junto con el texto extraído se envían a la API de **Google Gemini**. El modelo evalúa el estado físico de la patente, indicando si presenta daños, suciedad u obstrucciones que dificulten su legibilidad.


### Stack Tecnológico

- **Frontend:** React 19, Vite, TypeScript, Material UI (MUI), Chart.js, XYFlow.
- **Backend:** Python 3.11+, FastAPI, Uvicorn, WebSockets (`python-socketio`).
- **Base de Datos:** PostgreSQL, gestionada a través de SQLAlchemy (ORM) y Alembic (migraciones).
- **Inteligencia Artificial:** Ultralytics (YOLOv11), EasyOCR, Google GenAI SDK (`google-genai`).

---

## 📁 Estructura del Proyecto

El proyecto está organizado como un monorepositorio para facilitar el desarrollo "Full-Stack AI":

```text
TrafficVision/
├── backend/            # Lógica central, API REST y servicios de IA (FastAPI)
│   ├── app/            # Código fuente de la API (routers, models, schemas)
│   ├── temp/           # Almacenamiento temporal de procesamiento
│   └── requirements.txt
├── frontend/           # Interfaz de usuario interactiva (React/Vite)
│   ├── src/            # Componentes, vistas y hooks
│   └── package.json
├── database/           # Scripts y recursos para PostgreSQL
├── ml/                 # Notebooks, pruebas de concepto y pesos de modelos
├── storage/            # Volúmenes persistentes (si se usa Docker)
├── docker-compose.yml  # Orquestador de servicios
└── README.md           # Este archivo
```

## 🧠 Pipeline Técnico Detallado

### Etapa 1 — Detección (`plate_detector.py`)

YOLOv11 entrenado con **10,734 imágenes** (global + ecuatorianas). Red neuronal convolucional one-stage que localiza y clasifica la placa en una sola pasada.

Filtros post-detección:
- Confianza mínima: **45%**
- Proporción ancho/alto: **1.5 a 6.0** (forma real de placa)
- Corrección EXIF automática para fotos de celular
<img width="1216" height="507" alt="Captura de pantalla 2026-07-17 131406" src="https://github.com/user-attachments/assets/71c8c087-2ec1-4694-8332-1669c24f886f" />

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

<img width="370" height="147" alt="Captura de pantalla 2026-07-17 131525" src="https://github.com/user-attachments/assets/79a38cf9-b971-4f66-8142-f91804d274b0" />

En TrafficVision el pipeline funciona así:

```
Imagen JPG/PNG
      ↓
[1] Detección de vehículo    → YOLOv8n (clases: car, truck, motorcycle, bus)
      ↓
[2] Detección de placa    → YOLOv11n - RT DETR - Vision Mamba (97.4% mAP@50)
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

## 📊 Modelos Entrenados y Comparados

| Modelo | Descripción | Estado |
|---|---|---|
| **YOLOv11n** | Single-shot CNN - Máxima velocidad | ✅ Principal |
| **RT-DETR** | Transformer + CNN - Alta precisión | ✅ Comparativa |
| **Vision Mamba** | State Space Model backbone - Efficient vision transformer alternative | ✅ Comparativa |

El entrenamiento se gestiona comparando mAP, Precisión, Recall y F1-Score usando un dashboard interactivo en React + Chart.js leyendo directamente los `results.csv`.
<img width="1320" height="430" alt="Captura de pantalla 2026-07-17 131853" src="https://github.com/user-attachments/assets/7df7fc81-af3c-483e-89ad-4f19b172f322" />

<img width="1288" height="421" alt="Captura de pantalla 2026-07-17 131905" src="https://github.com/user-attachments/assets/b42fc731-2992-4b01-b804-0ecd80a0dd42" />

<img width="1286" height="427" alt="Captura de pantalla 2026-07-17 131911" src="https://github.com/user-attachments/assets/1e25804c-c5ce-4fbb-bdd3-a6c93c28ac1d" />

---

## 🗂️ Datasets

| Dataset | Imágenes | Descripción |
|---|---|---|
| license-plates (global) | 10,125 | Roboflow Universe — placas mundiales |
| Plates Ecuadorian - v4 | 2,431 | Roboflow Universe — Ecuador (2100 train / 216 valid / 115 test) |
| Global Ecuador Combined | ~10,000 | Archivo combinado de Ecuador — Google Drive (referencia remota) |
| license-plates-ec-1 | 144 | Placas ecuatorianas — Roboflow |
| license-plates-ec-2 | 90 | Placas ecuatorianas — dataset adicional |
| license-plates-ec-3 | — | OCR de caracteres (36 clases) — pendiente |
| license-plates-ec-4 | 375 | **Dataset personal** — fotos propias |

<img width="1185" height="697" alt="Captura de pantalla 2026-07-17 131719" src="https://github.com/user-attachments/assets/a4f5f438-62bd-4181-b3bc-faee59b7bf8d" />

---

## 🚀 Guía de Instalación y Despliegue

### Requisitos Previos
- **Node.js** (v18 o superior)
- **Python** (v3.11)
- **PostgreSQL** (v14 o superior)
- (Opcional) **Docker** y **Docker Compose**

### Variables de Entorno (`.env`)
El sistema requiere de configuración externa. Debes crear archivos `.env` tanto en `backend/` como en `frontend/`.

**Ejemplo de `backend/.env`:**
```env
DATABASE_URL=postgresql://usuario:password@localhost:5432/trafficvision
GEMINI_API_KEY=tu_clave_api_de_google_ai_studio
```

**Ejemplo de `frontend/.env`:**
```env
VITE_API_URL=http://localhost:8000
```

---

## 🛠️ Tecnologías

| Categoría | Tecnología | Uso |
|---|---|---|
| Detección | YOLOv11n-RT DETR-VisiomMamba | Localización de placas |
| OCR | EasyOCR | Lectura de caracteres |
| IA Vision | Google Gemini Flash | Clasificación de calidad |
| Backend | FastAPI + Python 3.11 | API REST |
| Frontend | React + TypeScript + MUI | Interfaz de usuario |
| Entrenamiento | Google Colab (Tesla T4) | Entrenamiento GPU |
| Dataset | Roboflow Universe | Datasets anotados YOLO |
| Imagen | OpenCV + Pillow | Procesamiento de imágenes |

---

## 🧪 Validación y Pruebas

Para asegurar el correcto funcionamiento del pipeline multimodal, asegúrate de que el backend tenga acceso al modelo base de YOLO.
- El peso de YOLO (`yolov8n.pt`) debe encontrarse descargado en el backend o será descargado automáticamente en la primera ejecución.
- Las credenciales de Gemini deben estar activas para la etapa de clasificación. Si la API Key falla, el sistema proveerá una lectura OCR básica sin el análisis de estado.

<img width="1202" height="621" alt="Captura de pantalla 2026-07-17 131804" src="https://github.com/user-attachments/assets/87664a69-409f-40c6-92ef-bc6a41d96304" />


## 🤝 Contribución

1. Haz un Fork del repositorio.
2. Crea una rama para tu característica (`git checkout -b feature/NuevaCaracteristica`).
3. Haz commit de tus cambios (`git commit -m 'Añade NuevaCaracteristica'`).
4. Haz Push a la rama (`git push origin feature/NuevaCaracteristica`).
5. Abre un Pull Request.


---

*TrafficVision · Tesis de Grado · Ecuador 2026 `*
