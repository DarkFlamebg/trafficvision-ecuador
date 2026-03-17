# train_yolov8.py
# Entrena YOLOv8n para detección de placas vehiculares
# Ejecutar desde: Backend/ con venv311 activo
#
# Uso:
#   py train_yolov8.py              → entrena todos los datasets
#   py train_yolov8.py global       → solo dataset global
#   py train_yolov8.py ecuador      → solo dataset Ecuador ec-1
#   py train_yolov8.py ecuador2     → solo dataset Ecuador ec-2
#   py train_yolov8.py combined     → ec-1 + ec-2 combinados
#
# NOTA: ec-3 es dataset OCR (36 clases de caracteres) — no compatible con detección
#       Se usará más adelante para mejorar el módulo de lectura OCR

import sys
import os
import glob
import yaml
from ultralytics import YOLO

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_BASE  = "yolov8n.pt"
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16

DATASETS = {
    "global": {
        "yaml": "datasets/license-plates/data.yaml",
        "name": "yolov8n_plates_global",
    },
    "ecuador": {
        "yaml": "datasets/license-plates-ec-combined/license-plates-ec-1/data.yaml",
        "name": "yolov8n_plates_ecuador",
    },
    "ecuador2": {
        "yaml": "datasets/license-plates-ec-combined/license-plates-ec-2/data.yaml",
        "name": "yolov8n_plates_ecuador2",
    },
    "combined": {
        "yaml": "datasets/license-plates-ec-combined/data_combined.yaml",
        "name": "yolov8n_plates_ec_combined",
    },
}

# ── Buscar el best.pt real (Ultralytics puede anidar rutas) ───────────────────
def find_best_pt(model_name: str) -> str:
    """Busca best.pt en cualquier subdirectorio que contenga el nombre del modelo."""
    patterns = [
        f"models/runs/{model_name}/weights/best.pt",
        f"runs/detect/models/runs/{model_name}/weights/best.pt",
        f"runs/detect/{model_name}/weights/best.pt",
    ]
    for p in patterns:
        if os.path.exists(p):
            return p

    # Búsqueda recursiva como fallback
    matches = glob.glob(f"**/{model_name}/weights/best.pt", recursive=True)
    if matches:
        return matches[0]

    return f"models/runs/{model_name}/weights/best.pt"  # default para mensaje

# ── Crear data_combined.yaml (ec-1 + ec-2) ────────────────────────────────────
def create_combined_yaml() -> str:
    base = os.path.abspath("datasets/license-plates-ec-combined")

    ec1_train = os.path.join(base, "license-plates-ec-1", "train", "images").replace("\\", "/")
    ec1_valid = os.path.join(base, "license-plates-ec-1", "valid", "images").replace("\\", "/")
    ec1_test  = os.path.join(base, "license-plates-ec-1", "test",  "images").replace("\\", "/")
    ec2_train = os.path.join(base, "license-plates-ec-2", "train", "images").replace("\\", "/")

    data = {
        "train": [ec1_train, ec2_train],
        "val":   ec1_valid,
        "test":  ec1_test,
        "nc":    1,
        "names": ["car plate"],
    }

    path = os.path.join(base, "data_combined.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  [combined] data_combined.yaml generado")
    print(f"  [combined] train → {ec1_train}")
    print(f"  [combined]         {ec2_train}")
    print(f"  [combined] val   → {ec1_valid}")
    return path

# ── Corregir rutas en data.yaml ────────────────────────────────────────────────
def fix_yaml_paths(yaml_path: str) -> str:
    base_dir = os.path.abspath(os.path.dirname(yaml_path))

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = os.path.join(base_dir, "train", "images").replace("\\", "/")
    data["val"]   = os.path.join(base_dir, "valid", "images").replace("\\", "/")
    data["test"]  = os.path.join(base_dir, "test",  "images").replace("\\", "/")

    fixed_path = os.path.join(base_dir, "data_fixed.yaml")
    with open(fixed_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"  [yaml] train → {data['train']}")
    print(f"  [yaml] val   → {data['val']}")
    return fixed_path

# ── Entrenamiento ──────────────────────────────────────────────────────────────
def train(dataset_key: str):
    cfg = DATASETS[dataset_key]
    print("\n" + "=" * 60)
    print(f"  Entrenando: {dataset_key.upper()}")
    print(f"  Modelo:     {cfg['name']}")
    print("=" * 60)

    if dataset_key == "combined":
        fixed_yaml = create_combined_yaml()
    else:
        fixed_yaml = fix_yaml_paths(cfg["yaml"])

    model = YOLO(MODEL_BASE)
    model.train(
        data      = fixed_yaml,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        name      = cfg["name"],
        patience  = 10,
        save      = True,
        plots     = True,
        verbose   = True,
    )

    best = find_best_pt(cfg["name"])
    print(f"\n  ✅ Mejor modelo: {best}")
    return best, cfg["name"]

# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(model_path: str, dataset_key: str):
    if dataset_key == "combined":
        fixed_yaml = os.path.abspath(
            "datasets/license-plates-ec-combined/data_combined.yaml"
        )
    else:
        fixed_yaml = os.path.join(
            os.path.dirname(DATASETS[dataset_key]["yaml"]), "data_fixed.yaml"
        )

    if not os.path.exists(model_path):
        print(f"[eval] ❌ Modelo no encontrado: {model_path}")
        return

    print(f"\n[eval] {dataset_key.upper()} — {model_path}")
    model   = YOLO(model_path)
    metrics = model.val(data=fixed_yaml, imgsz=IMG_SIZE)

    print("\n── Métricas ──────────────────────────────────────────")
    print(f"  Dataset:   {dataset_key.upper()}")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precisión: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("─────────────────────────────────────────────────────")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if arg == "all":
        keys = ["global", "ecuador", "ecuador2", "combined"]
    elif arg in DATASETS:
        keys = [arg]
    else:
        print(f"[error] Dataset '{arg}' no reconocido.")
        print(f"  Opciones: {' | '.join(list(DATASETS.keys()) + ['all'])}")
        sys.exit(1)

    results = {}
    for key in keys:
        best, name = train(key)
        results[key] = best

    for key, best in results.items():
        evaluate(best, key)

    print("\n" + "=" * 60)
    print("  Resumen final")
    for key, best in results.items():
        status = "✅" if os.path.exists(best) else "❌"
        print(f"  {status} {key.upper():12} → {best}")
    print("=" * 60)