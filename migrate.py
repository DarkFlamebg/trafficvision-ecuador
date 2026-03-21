# migrate.py
# Reorganiza el proyecto TrafficVision a la nueva estructura escalable
# SEGURO: solo COPIA archivos, nunca borra nada
# Ejecutar desde la raíz del proyecto: py migrate.py

import os
import shutil
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────
DRY_RUN = True   # ← Cambia a False cuando quieras ejecutar de verdad
                  #   Con True solo muestra qué haría sin mover nada

BASE     = Path(__file__).parent
BACKEND  = BASE / "Backend"
NEW_BASE = BASE  # la nueva estructura va en la misma raíz

# ── Colores para la consola ────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
RESET  = "\033[0m"

def log(action: str, src: str, dst: str = ""):
    if action == "COPY":
        print(f"{GREEN}  [COPY]{RESET} {src}")
        if dst: print(f"         → {dst}")
    elif action == "MKDIR":
        print(f"{CYAN}  [DIR] {RESET} {dst}")
    elif action == "SKIP":
        print(f"{YELLOW}  [SKIP]{RESET} {src} (no existe)")
    elif action == "INFO":
        print(f"{CYAN}  {src}{RESET}")

def mkdir(path: Path):
    """Crea directorio si no existe."""
    log("MKDIR", "", str(path.relative_to(BASE)))
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    """Copia un archivo a su nuevo destino."""
    if not src.exists():
        log("SKIP", str(src.relative_to(BASE)))
        return
    log("COPY", str(src.relative_to(BASE)), str(dst.relative_to(BASE)))
    if not DRY_RUN:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def copy_dir(src: Path, dst: Path, ignore_patterns=None):
    """Copia un directorio completo."""
    if not src.exists():
        log("SKIP", str(src.relative_to(BASE)))
        return
    log("COPY", str(src.relative_to(BASE)), str(dst.relative_to(BASE)))
    if not DRY_RUN:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if ignore_patterns:
            shutil.copytree(src, dst, dirs_exist_ok=True,
                          ignore=shutil.ignore_patterns(*ignore_patterns))
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
def migrate():
    mode = "DRY RUN — solo simulación" if DRY_RUN else "EJECUCIÓN REAL"
    print(f"\n{'='*60}")
    print(f"  TrafficVision — Migración de estructura")
    print(f"  Modo: {mode}")
    print(f"{'='*60}\n")

    # ── 1. BACKEND ─────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── BACKEND ──────────────────────────────────────────{RESET}")

    # app/ai
    copy_dir(BACKEND / "app/ai",       NEW_BASE / "backend/app/ai")
    # app/core
    copy_dir(BACKEND / "app/core",     NEW_BASE / "backend/app/core")
    # app/database
    copy_dir(BACKEND / "app/database", NEW_BASE / "backend/app/database/connection")
    # app/routes
    copy_dir(BACKEND / "app/routes",   NEW_BASE / "backend/app/routes")
    # app/schemas
    copy_dir(BACKEND / "app/schemas",  NEW_BASE / "backend/app/schemas")
    # app/services
    copy_dir(BACKEND / "app/services", NEW_BASE / "backend/app/services")

    # Crear carpetas nuevas vacías
    mkdir(NEW_BASE / "backend/app/controllers")
    mkdir(NEW_BASE / "backend/app/database/logs")
    mkdir(NEW_BASE / "backend/temp")

    # Archivos raíz del backend
    copy_file(BACKEND / "main.py",          NEW_BASE / "backend/main.py")
    copy_file(BACKEND / "requirements.txt", NEW_BASE / "backend/requirements.txt")
    copy_file(BACKEND / ".env",             NEW_BASE / "backend/.env")

    # Archivos de info — no se pierden
    for f in ["Dependencias.txt", "imporante.me", "check_models.py"]:
        copy_file(BACKEND / f, NEW_BASE / "backend/docs" / f)

    # ── 2. ML ──────────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── ML ───────────────────────────────────────────────{RESET}")

    # Datasets
    copy_dir(
        BACKEND / "datasets/license-plates",
        NEW_BASE / "ml/datasets/raw/license-plates",
        ignore_patterns=["*.cache"]
    )
    copy_dir(
        BACKEND / "datasets/license-plates-ec-combined",
        NEW_BASE / "ml/datasets/raw/license-plates-ec-combined",
        ignore_patterns=["*.cache"]
    )

    # Modelos base
    copy_file(BACKEND / "yolov8n.pt",        NEW_BASE / "ml/models/base/yolov8n.pt")
    copy_file(BACKEND / "models/yolov5s.pt", NEW_BASE / "ml/models/base/yolov5s.pt")

    # Modelos entrenados — solo el mejor
    best_model = BACKEND / "runs/detect/yolov8n_plates_combined_all/weights/best.pt"
    copy_file(best_model, NEW_BASE / "ml/models/trained/yolov8n_combined_all/best.pt")

    last_model = BACKEND / "runs/detect/yolov8n_plates_combined_all/weights/last.pt"
    copy_file(last_model, NEW_BASE / "ml/models/trained/yolov8n_combined_all/last.pt")

    # Ecuador model
    ec_best = BACKEND / "runs/detect/yolov8n_plates_ecuador43/weights/best.pt"
    copy_file(ec_best, NEW_BASE / "ml/models/trained/yolov8n_ecuador/best.pt")

    # Script de entrenamiento
    copy_file(BACKEND / "train_yolov8.py", NEW_BASE / "ml/training/train_yolov8.py")

    # Notebook de Colab
    mkdir(NEW_BASE / "ml/notebooks")

    # Runs completos (métricas, gráficas)
    copy_dir(
        BACKEND / "runs",
        NEW_BASE / "ml/runs",
        ignore_patterns=["*.pt"]   # no copiar .pt aquí, ya están en models/trained
    )

    # Configs yaml
    mkdir(NEW_BASE / "ml/configs")
    for yaml_file in (BACKEND / "datasets/license-plates-ec-combined").glob("*.yaml"):
        copy_file(yaml_file, NEW_BASE / "ml/configs" / yaml_file.name)
    for yaml_file in (BACKEND / "datasets/license-plates").glob("*.yaml"):
        copy_file(yaml_file, NEW_BASE / "ml/configs" / yaml_file.name)

    # ── 3. DATASET SERVICE ─────────────────────────────────────────────────────
    print(f"\n{CYAN}── DATASET SERVICE ──────────────────────────────────{RESET}")
    mkdir(NEW_BASE / "dataset_service/app/routes")
    mkdir(NEW_BASE / "dataset_service/app/services")
    mkdir(NEW_BASE / "dataset_service/app/schemas")
    mkdir(NEW_BASE / "dataset_service/app/storage")

    # ── 4. STORAGE ─────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── STORAGE ──────────────────────────────────────────{RESET}")
    mkdir(NEW_BASE / "storage/images")
    mkdir(NEW_BASE / "storage/detections")
    mkdir(NEW_BASE / "storage/datasets")

    # ── 5. DATABASE ────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── DATABASE ─────────────────────────────────────────{RESET}")
    copy_dir(BASE / "database", NEW_BASE / "database")

    # ── 6. FRONTEND ────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── FRONTEND ─────────────────────────────────────────{RESET}")
    # El frontend ya está en la raíz correcta, no necesita moverse
    log("INFO", "frontend/ ya está en la ubicación correcta — no se mueve")

    # ── 7. RAÍZ ────────────────────────────────────────────────────────────────
    print(f"\n{CYAN}── RAÍZ ─────────────────────────────────────────────{RESET}")
    copy_file(BASE / ".gitignore",        NEW_BASE / ".gitignore")
    copy_file(BASE / "docker-compose.yml", NEW_BASE / "docker-compose.yml")
    copy_file(BASE / "README.md",         NEW_BASE / "README.md")

    # ── Resumen ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if DRY_RUN:
        print(f"  {YELLOW}Simulación completada.{RESET}")
        print(f"  Para ejecutar de verdad: cambia DRY_RUN = False")
        print(f"  {RED}⚠️  Los archivos originales NO serán borrados{RESET}")
        print(f"  Verifica que todo esté correcto antes de proceder")
    else:
        print(f"  {GREEN}✅ Migración completada.{RESET}")
        print(f"  Los archivos originales siguen en Backend/ sin cambios")
        print(f"  Cuando estés seguro puedes borrar Backend/ manualmente")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    migrate()
