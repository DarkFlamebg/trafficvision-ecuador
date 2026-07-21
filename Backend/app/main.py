# app/main.py
# Punto de entrada de la API — TrafficVision Backend v1.4.0
# Responsabilidades: configurar la app, middleware CORS, ciclo de vida y routers.
# La lógica de negocio vive en app/services/ y app/routes/.

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_VERSION, TEMP_DIR, CORS_ORIGINS
from app.ai.model_loader import load_all_models, get_status as model_status

# ── Routers ─────────────────────────────────────────────────────────────────────
from app.routes.detection        import router as detection_router
from app.routes.video            import router as video_router
from app.routes.multi_detect       import router as multi_detect_router
from app.routes.compare          import router as compare_router
from app.routes.benchmark        import router as benchmark_router
from app.routes.datasets         import router as datasets_router
from app.routes.anti_corruption  import router as anti_corruption_router


# ── Ciclo de vida ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: crear carpeta temporal y precargar modelos en paralelo
    os.makedirs(TEMP_DIR, exist_ok=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_models)
    yield
    # Shutdown: limpiar archivos temporales
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except Exception:
            pass


# ── Aplicación ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "TrafficVision API",
    description = (
        "Detección de vehículos y placas vehiculares con "
        "YOLOv11n, RT-DETR, Vision Mamba + EasyOCR + Gemini"
    ),
    version     = APP_VERSION,
    lifespan    = lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = CORS_ORIGINS,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    expose_headers = ["X-Metrics", "X-Detections"],
)

# ── Registro de routers ────────────────────────────────────────────────────────
# Detección (v1) — endpoints principales con versionado explícito
app.include_router(detection_router,       prefix="/api/v1", tags=["Detección v1"])
app.include_router(video_router,           prefix="/api/v1")

# Módulos funcionales
app.include_router(multi_detect_router,    prefix="/api/v1", tags=["Detección comparativa"])
app.include_router(compare_router,         prefix="/api/v1", tags=["Comparativa de modelos"])
app.include_router(benchmark_router,       prefix="/api/v1", tags=["Benchmark"])
app.include_router(datasets_router,        prefix="/api/v1", tags=["Datasets"])
app.include_router(anti_corruption_router, prefix="/api/v1", tags=["Control anticorrupción"])


# ── Endpoints base ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Estado"])
def root():
    return {
        "message": "TrafficVision API activa",
        "docs":    "/docs",
        "version": APP_VERSION,
    }


@app.get("/health", tags=["Estado"])
def health():
    """Estado de la API y tiempo de carga de modelos."""
    status = model_status()
    return {
        "status":       "ok" if status["models_ready"] else "loading",
        "models_ready": status["models_ready"],
        "load_time_ms": status["load_time_ms"],
        "version":      APP_VERSION,
    }
