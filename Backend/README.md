# TrafficVision - Backend 🧠🚗

Este subproyecto contiene la lógica del servidor, la integración de modelos de Inteligencia Artificial (YOLO, EasyOCR) y la API RESTful para el sistema de análisis de tráfico vehicular **TrafficVision**.

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.11+
- **Framework API:** FastAPI (con Uvicorn)
- **Base de Datos:** PostgreSQL (con SQLAlchemy y psycopg2)
- **Modelos e IA:** Ultralytics (YOLOv8), EasyOCR, Google GenAI
- **Tiempo Real:** python-socketio

## 📂 Estructura del Proyecto

```text
backend/
├── app/
│   ├── routers/       # Endpoints y rutas de la API
│   ├── models/        # Modelos de base de datos (SQLAlchemy)
│   ├── schemas/       # Esquemas de validación de datos (Pydantic)
│   ├── services/      # Lógica de negocio (procesamiento de video, IA, OCR)
│   ├── core/          # Configuraciones y utilidades generales
│   ├── database.py    # Configuración de conexión a la BD
│   └── main.py        # Punto de entrada de la aplicación FastAPI
├── temp/              # Directorio para archivos temporales (imágenes/videos)
├── requirements.txt   # Dependencias del proyecto
└── .env               # Variables de entorno
```

## 🚀 Instalación y Configuración Local

1. **Clonar el repositorio y entrar al backend:**
   ```bash
   git clone <url-del-repo>
   cd TrafficVision/backend
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv311
   # En Windows:
   venv311\Scripts\activate
   # En macOS/Linux:
   source venv311/bin/activate
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar las variables de entorno:**
   Crea un archivo `.env` en la raíz de `backend/` basándote en un posible `.env.example`:
   ```env
   DATABASE_URL=postgresql://usuario:password@localhost:5432/trafficvision
   GEMINI_API_KEY=tu_api_key_aqui
   ```

5. **Levantar el servidor de desarrollo:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *La documentación interactiva de la API estará disponible en `http://localhost:8000/docs`.*

## 🔌 Documentación de la API / Endpoints (Ejemplo)

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Verifica el estado y conexión del servidor. |
| `POST` | `/api/v1/analyze/video` | Sube un video para procesar la detección de vehículos. |
| `POST` | `/api/v1/analyze/plate` | Extrae la patente de un vehículo usando OCR. |
| `GET` | `/api/v1/statistics` | Retorna los datos históricos del tráfico. |
