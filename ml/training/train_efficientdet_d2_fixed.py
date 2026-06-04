# Uso:
#   py train_efficientdet_d2_fixed.py              → combined_all (recomendado)
#   py train_efficientdet_d2_fixed.py combined     → ec-1 + ec-2 + ec-4
#   py train_efficientdet_d2_fixed.py ecuador4     → solo ec-4 (más rápido para debug)
#   py train_efficientdet_d2_fixed.py export       → solo exportar ONNX desde best.pt
#
# Requisitos sin esto no jala el entrenaiento ekizde :C:
#   pip install effdet timm albumentations opencv-python tqdm pyyaml onnx onnxruntime


import sys
import os
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Dispositivo
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[device] NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    print(f"[device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = "cpu"
    print("[device] ⚠️  Usando CPU — entrenamiento será lento")

# Hiperparámetros
MODEL_NAME   = "tf_efficientdet_d2"
IMG_SIZE     = 768          
EPOCHS       = 80
BATCH_SIZE   = 4            # 4 para GPU con 6GB+ VRAM; 2 para CPU o GPU menor
LR           = 5e-4         
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
NUM_CLASSES  = 1
PATIENCE     = 15

# EfficientDet-D2 preentrenado en ImageNet usa estos valores
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

EC_BASE = "../datasets/raw/license-plates-ec-combined"

DATASETS = {
    "global": {
        "yaml": "../datasets/raw/license-plates/data.yaml",
        "name": "efficientdet_d2_combined_all",   
    },
    "ecuador": {
        "yaml": f"{EC_BASE}/license-plates-ec-1/data.yaml",
        "name": "efficientdet_d2_ecuador",
    },
    "ecuador2": {
        "yaml": f"{EC_BASE}/license-plates-ec-2/data.yaml",
        "name": "efficientdet_d2_ecuador2",
    },
    "ecuador4": {
        "yaml": f"{EC_BASE}/license-plates-ec-4/data.yaml",
        "name": "efficientdet_d2_ecuador4",
    },
    "combined": {
        "yaml": f"{EC_BASE}/data_combined.yaml",
        "name": "efficientdet_d2_combined",
    },
    "combined_all": {
        "yaml": f"{EC_BASE}/data_combined_all.yaml",
        "name": "efficientdet_d2_combined_all",
    },
}


#Dataset
class PlateDataset(Dataset):
    """
    Carga imágenes YOLO y las convierte al formato que espera EfficientDet.

    Formato de bbox EfficientDet: [y_min, x_min, y_max, x_max] en píxeles
    absolutos sobre la imagen ya resizeada a IMG_SIZE.

    Albumentations con format='pascal_voc' trabaja en [x_min, y_min, x_max, y_max].
    La conversión al orden y-first se hace DESPUÉS del transform, ya con las
    coordenadas escaladas correctamente a IMG_SIZE.
    """

    def __init__(self, image_dirs, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform   = transform

        dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        for img_dir in dirs:
            img_dir   = Path(img_dir)
            label_dir = img_dir.parent / "labels"
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in img_dir.glob(ext):
                    label_path = label_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        self.image_paths.append(str(img_path))
                        self.label_paths.append(str(label_path))

        print(f"  [dataset] {len(self.image_paths)} imágenes encontradas")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]

        boxes, labels = [], []
        with open(self.label_paths[idx]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x_min = (xc - bw / 2) * w
                y_min = (yc - bh / 2) * h
                x_max = (xc + bw / 2) * w
                y_max = (yc + bh / 2) * h
                x_min = max(0.0, min(x_min, w - 1))
                y_min = max(0.0, min(y_min, h - 1))
                x_max = max(x_min + 1, min(x_max, w))
                y_max = max(y_min + 1, min(y_max, h))
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls))

        boxes  = np.array(boxes,  dtype=np.float32) if boxes  else np.zeros((0, 4), np.float32)
        labels = np.array(labels, dtype=np.int64)   if labels else np.zeros((0,),   np.int64)

        # Albumentations (resize + aug + normalización)
        if self.transform:
            result = self.transform(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image  = result["image"]                                 
            boxes  = np.array(result["bboxes"],  dtype=np.float32)   
            labels = np.array(result["labels"],  dtype=np.int64)

        # DESPUÉS del transform para que las coords ya estén en escala IMG_SIZE
        if len(boxes) > 0:
            boxes = boxes[:, [1, 0, 3, 2]]  

        _, img_h, img_w = image.shape

        target = {
            "bbox":      torch.tensor(boxes,  dtype=torch.float32),
            "cls":       torch.tensor(labels, dtype=torch.int64),
            "img_scale": torch.tensor([1.0],          dtype=torch.float32),
            "img_size":  torch.tensor([img_h, img_w], dtype=torch.int64),   # [H, W]
        }
        return image, target


#Augmentations
def get_train_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, fill=0),
        # Augmentaciones útiles para placas de autos
        A.HorizontalFlip(p=0.3),                              # placas simétricas horizontalmente
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),                                       # mejora visibilidad en sombras
        A.RandomShadow(p=0.2),                                # simula sombras en calle

        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,   # descartar boxes que queden <30% visibles tras aug
        clip=True,
    ))


def get_val_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, fill=0),

        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        clip=True,
    ))


# Collate
def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def collate_targets(targets, device):
    max_boxes = max(max(t["bbox"].shape[0] for t in targets), 1)
    bboxes, clses = [], []
    for t in targets:
        n   = t["bbox"].shape[0]
        pad = max_boxes - n
        bboxes.append(torch.cat([t["bbox"], torch.zeros(pad, 4)], dim=0))
        clses.append( torch.cat([t["cls"],  torch.zeros(pad, dtype=torch.int64)], dim=0))

    return {
        "bbox":      torch.stack(bboxes).to(device),
        "cls":       torch.stack(clses).to(device),
        "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
        "img_size":  torch.stack([t["img_size"]  for t in targets]).to(device),
    }


#  Modelo 
def create_model(num_classes=NUM_CLASSES, pretrained_backbone=True):
    config             = get_efficientdet_config(MODEL_NAME)
    config.num_classes = num_classes
    config.image_size  = (IMG_SIZE, IMG_SIZE)

    net           = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    model = DetBenchTrain(net, config)
    return model


#YAML helpers
def create_combined_yaml() -> str:
    base = os.path.abspath(EC_BASE)
    paths = {
        "ec1_train": os.path.join(base, "license-plates-ec-1", "train", "images"),
        "ec1_valid": os.path.join(base, "license-plates-ec-1", "valid", "images"),
        "ec1_test":  os.path.join(base, "license-plates-ec-1", "test",  "images"),
        "ec2_train": os.path.join(base, "license-plates-ec-2", "train", "images"),
        "ec4_train": os.path.join(base, "license-plates-ec-4", "train", "images"),
    }
    data = {
        "train": [paths["ec1_train"], paths["ec2_train"], paths["ec4_train"]],
        "val":   paths["ec1_valid"],
        "test":  paths["ec1_test"],
        "nc": 1, "names": ["license plate"],
    }
    path = os.path.join(base, "data_combined.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


def create_combined_all_yaml() -> str:
    base    = os.path.abspath(EC_BASE)
    global_ = os.path.abspath("../datasets/raw/license-plates")
    data = {
        "train": [
            os.path.join(global_, "train", "images"),
            os.path.join(base, "license-plates-ec-1", "train", "images"),
            os.path.join(base, "license-plates-ec-2", "train", "images"),
            os.path.join(base, "license-plates-ec-4", "train", "images"),
        ],
        "val":  os.path.join(global_, "valid", "images"),
        "test": os.path.join(global_, "test",  "images"),
        "nc": 1, "names": ["license plate"],
    }
    path = os.path.join(base, "data_combined_all.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return path


def load_yaml_paths(dataset_key: str) -> Tuple[List[str], str, str]:
    if dataset_key == "combined":
        yaml_path = create_combined_yaml()
    elif dataset_key == "combined_all":
        yaml_path = create_combined_all_yaml()
    else:
        yaml_path = DATASETS[dataset_key]["yaml"]

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    train = data["train"] if isinstance(data["train"], list) else [data["train"]]
    val   = data["val"]
    test  = data.get("test", val)
    return train, val, test


#Training loop
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train {epoch+1}/{EPOCHS}")
    for images, targets in pbar:
        images        = images.to(device)
        batch_targets = collate_targets(targets, device)

        optimizer.zero_grad()
        loss_dict = model(images, batch_targets)
        loss      = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    for images, targets in tqdm(dataloader, desc="  Val  "):
        images        = images.to(device)
        batch_targets = collate_targets(targets, device)
        loss_dict     = model(images, batch_targets)
        running_loss += loss_dict["loss"].item()
    return running_loss / len(dataloader)


# Exportación ONNX
def export_onnx(checkpoint_path: str, output_path: str = None):
    """
    Exporta best.pt a ONNX optimizado para producción.

    El modelo exportado usa DetBenchPredict (incluye decoder de boxes)
    y recibe tensores normalizados [B, 3, 768, 768] como entrada.

    Salida ONNX: tensor [B, max_det, 6] donde 6 = [y1, x1, y2, x2, score, class]
    Usar ONNXRuntime para inferencia, ~3-5x más rápido que PyTorch CPU.
    """
    import onnx
    import onnxruntime as ort

    if output_path is None:
        output_path = checkpoint_path.replace(".pt", ".onnx")

    print(f"\n[export] Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Para inferencia necesitamos el EfficientDet neto
    config             = get_efficientdet_config(MODEL_NAME)
    config.num_classes = NUM_CLASSES
    config.image_size  = (IMG_SIZE, IMG_SIZE)   # FIX #2

    net           = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=NUM_CLASSES)

    # Limpiar prefijo 'model.' del state_dict guardado por DetBenchTrain
    raw_sd = checkpoint["model_state_dict"]
    clean_sd = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in raw_sd.items()
    }
    net.load_state_dict(clean_sd, strict=False)
    net.eval()

    # Envolver en DetBenchPredict para exportar con decoder incluido
    bench = DetBenchPredict(net)
    bench.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"[export] Exportando a ONNX: {output_path}")
    torch.onnx.export(
        bench,
        dummy,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={
            "images":     {0: "batch_size"},
            "detections": {0: "batch_size"},
        },
    )

    # Verificar modelo ONNX
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f"[export] ✅ ONNX válido guardado en: {output_path}")

    # Benchmark rápido
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    dummy_np = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, {"images": dummy_np})
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"[export] Latencia ONNX promedio (CPU, 5 runs): {ms:.1f} ms")

    return output_path


# ── Inferencia ONNX (para plate_detector_efficientdet.py) ─────────────────────
def create_onnx_detector(onnx_path: str):
    """
    Retorna una función detect(image_bgr) compatible con el pipeline existente.
    Usar esto en lugar de plate_detector_efficientdet.py para producción ONNX.

    Ejemplo de uso en compare.py:
        from train_efficientdet_d2_fixed import create_onnx_detector
        detect_plate_efficientdet = create_onnx_detector("path/to/best.onnx")
    """
    import onnxruntime as ort
    import cv2

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Normalización consistente con el entrenamiento
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(NORM_STD,  dtype=np.float32).reshape(3, 1, 1)

    CONF_THRESHOLD   = 0.25
    ASPECT_RATIO_MIN = 0.3
    ASPECT_RATIO_MAX = 6.0

    def _preprocess(image_bgr):
        h, w = image_bgr.shape[:2]
        rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scale = min(IMG_SIZE / w, IMG_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (IMG_SIZE - new_h) // 2
        pad_w = (IMG_SIZE - new_w) // 2
        padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        tensor = padded.astype(np.float32) / 255.0
        tensor = (tensor.transpose(2, 0, 1) - mean) / std   # [C,H,W]
        return tensor[np.newaxis], scale, pad_w, pad_h       # [1,C,H,W]

    def detect(image_bgr):
        from app.ai.crop_utils import extract_plate_crop

        iw, ih = image_bgr.shape[1], image_bgr.shape[0]
        input_tensor, scale, pad_w, pad_h = _preprocess(image_bgr)

        detections = sess.run(None, {"images": input_tensor})[0]  # [1, N, 6]

        plates = []
        if detections is None or len(detections) == 0:
            return plates

        for det in detections[0]:
            
            y1, x1, y2, x2, score, cls = det

            if score < CONF_THRESHOLD:
                continue

            # Reescalar a imagen original
            x1 = int((x1 - pad_w) / scale)
            y1 = int((y1 - pad_h) / scale)
            x2 = int((x2 - pad_w) / scale)
            y2 = int((y2 - pad_h) / scale)

            x1 = max(0, min(x1, iw)); x2 = max(0, min(x2, iw))
            y1 = max(0, min(y1, ih)); y2 = max(0, min(y2, ih))

            bw, bh = x2 - x1, y2 - y1
            if bw == 0 or bh == 0:
                continue

            ar = bw / bh
            if not (ASPECT_RATIO_MIN <= ar <= ASPECT_RATIO_MAX):
                continue

            crop = extract_plate_crop(image_bgr, x1, y1, x2, y2)
            if crop.size == 0:
                continue

            plates.append({
                "image":      crop,
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(float(score), 4),
                "detector":   "efficientdet_onnx",
            })

        return plates

    return detect


# Entrenamiento principal
def train(dataset_key: str):
    cfg        = DATASETS[dataset_key]
    model_name = cfg["name"]

    print("\n" + "=" * 60)
    print(f"  Dataset:     {dataset_key.upper()}")
    print(f"  Modelo:      {model_name}")
    print(f"  Arquitectura:{MODEL_NAME}")
    print(f"  Dispositivo: {DEVICE}")
    print(f"  Épocas:      {EPOCHS}  |  Batch: {BATCH_SIZE}  |  IMG: {IMG_SIZE}")
    print("=" * 60)

    train_dirs, val_dir, _ = load_yaml_paths(dataset_key)

    train_ds = PlateDataset(train_dirs, transform=get_train_transforms())
    val_ds   = PlateDataset(val_dir,   transform=get_val_transforms())

    pin = (DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=pin)

    model = create_model(pretrained_backbone=True).to(DEVICE)

    # Save directories and paths
    save_dir    = Path(f"runs/detect/{model_name}/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path   = save_dir / "last.pt"
    best_path   = save_dir / "best.pt"

    start_epoch  = 0
    best_val     = float("inf")
    pat_counter  = 0
    unfreeze_done = False

    checkpoint_exists = last_path.exists()
    ckpt = None
    if checkpoint_exists:
        print(f"🔄 Encontrado checkpoint de sesión anterior: {last_path}")
        ckpt = torch.load(last_path, map_location=DEVICE)
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("val_loss", float("inf"))

    # Configurar congelamiento/descongelamiento de backbone y optimizador según la época de inicio
    if start_epoch >= 5:
        print("   [resume] Fase de warmup completada en sesión anterior. Backbone descongelado.")
        for p in model.model.backbone.parameters():
            p.requires_grad = True
        unfreeze_done = True
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - 5, eta_min=LR_MIN,
        )
    else:
        print("   [warmup] Entrenando únicamente la cabeza de detección (backbone congelado)...")
        for p in model.model.backbone.parameters():
            p.requires_grad = False
            
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR_MIN,
        )

    # Cargar los estados del checkpoint
    if checkpoint_exists and ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Sincronizar el scheduler
        steps_to_take = start_epoch if start_epoch < 5 else (start_epoch - 5)
        for _ in range(steps_to_take):
            scheduler.step()
        print(f"✅ Reanudación exitosa desde Época {start_epoch + 1}")

    for epoch in range(start_epoch, EPOCHS):
        # Descongelar backbone después de 5 épocas de warmup (si se inició desde 0 a 4)
        if epoch == 5 and not unfreeze_done:
            print("\n[warmup] Descongelando backbone completo...")
            for p in model.model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - 5, eta_min=LR_MIN,
            )
            unfreeze_done = True

        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss   = validate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"  [E{epoch+1:03d}] train={train_loss:.4f}  val={val_loss:.4f}"
              f"  lr={optimizer.param_groups[0]['lr']:.2e}")

        ckpt_data = {
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":         train_loss,
            "val_loss":           val_loss,
        }
        torch.save(ckpt_data, last_path)

        if val_loss < best_val:
            best_val    = val_loss
            pat_counter = 0
            torch.save(ckpt_data, best_path)
            print(f"  ✅ Mejor modelo guardado (val_loss={val_loss:.4f})")
        else:
            pat_counter += 1
            if pat_counter >= PATIENCE:
                print(f"\n  Early stopping en época {epoch+1}")
                break

    print(f"\n✅ Entrenamiento completo → {best_path}")
    return str(best_path)


# Main
if __name__ == "__main__":
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "combined_all"

    if arg == "export":
        # Solo exportar ONNX desde el best.pt existente
        model_name = DATASETS["combined_all"]["name"]
        best_pt    = f"runs/detect/{model_name}/weights/best.pt"
        if not os.path.exists(best_pt):
            print(f"[error] No se encontró best.pt en: {best_pt}")
            sys.exit(1)
        export_onnx(best_pt)
        sys.exit(0)

    if arg == "all":
        keys = ["ecuador4", "combined", "combined_all"]
    elif arg in DATASETS:
        keys = [arg]
    else:
        print(f"[error] Opción '{arg}' no válida.")
        print(f"  Opciones: {' | '.join(list(DATASETS.keys()) + ['all', 'export'])}")
        sys.exit(1)

    for key in keys:
        best = train(key)
        # Exportar ONNX automáticamente al terminar
        try:
            export_onnx(best)
        except Exception as e:
            print(f"[export] ⚠️ Falló exportación ONNX: {e}")
            print("  Puedes exportar manualmente con: py train_efficientdet_d2_fixed.py export")
