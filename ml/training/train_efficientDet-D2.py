# train_efficientdet.py
# Entrena EfficientDet-D2 para detección de placas vehiculares
# Ejecutar desde: Backend/ con venv311 activo
#
# Uso:
#   py train_efficientdet.py              → entrena todos los datasets
#   py train_efficientdet.py global       → dataset global (10,125 imágenes)
#   py train_efficientdet.py ecuador      → ec-1 (144 imágenes)
#   py train_efficientdet.py ecuador2     → ec-2 (90 imágenes)
#   py train_efficientdet.py ecuador4     → ec-4 personal (375 imágenes)
#   py train_efficientdet.py combined     → ec-1 + ec-2 + ec-4 combinados
#   py train_efficientdet.py combined_all → global + ec-1 + ec-2 + ec-4
#
# Requisitos previos:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install effdet albumentations opencv-python pycocotools tqdm pyyaml

import sys
import os
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Detectar dispositivo ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[device] ✅ NVIDIA GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"[device] CUDA version: {torch.version.cuda}")
    print(f"[device] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    DEVICE = "cpu"
    print("[device] ⚠️  CUDA no disponible, usando CPU")

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_NAME   = "tf_efficientdet_d2"
EPOCHS       = 100
IMG_SIZE     = 768  # EfficientDet-D2 usa 768x768
BATCH_SIZE   = 4    # Ajustar según VRAM disponible (4-8 para D2)
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4    # Optimizado para Ryzen 5 5600
NUM_CLASSES  = 1    # Solo 'license plate'

EC_BASE = "datasets/license-plates-ec-combined"

DATASETS = {
    "global": {
        "yaml": "datasets/license-plates/data.yaml",
        "name": "efficientdet_d2_plates_global",
    },
    "ecuador": {
        "yaml": f"{EC_BASE}/license-plates-ec-1/data.yaml",
        "name": "efficientdet_d2_plates_ecuador",
    },
    "ecuador2": {
        "yaml": f"{EC_BASE}/license-plates-ec-2/data.yaml",
        "name": "efficientdet_d2_plates_ecuador2",
    },
    "ecuador4": {
        "yaml": f"{EC_BASE}/license-plates-ec-4/data.yaml",
        "name": "efficientdet_d2_plates_ecuador4",
    },
    "combined": {
        "yaml": f"{EC_BASE}/data_combined.yaml",
        "name": "efficientdet_d2_plates_ec_combined",
    },
    "combined_all": {
        "yaml": f"{EC_BASE}/data_combined_all.yaml",
        "name": "efficientdet_d2_plates_combined_all",
    },
}

# ── Dataset YOLO → COCO ────────────────────────────────────────────────────────
class YOLODataset(Dataset):
    """Convierte dataset YOLO a formato EfficientDet (COCO-style)"""
    
    def __init__(self, image_dirs: List[str], transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform
        
        # Recopilar todas las imágenes y labels
        for img_dir in (image_dirs if isinstance(image_dirs, list) else [image_dirs]):
            img_dir = Path(img_dir)
            label_dir = img_dir.parent / "labels"
            
            for img_path in img_dir.glob("*.jpg"):
                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.image_paths.append(str(img_path))
                    self.label_paths.append(str(label_path))
        
        print(f"  [dataset] Cargadas {len(self.image_paths)} imágenes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Cargar labels YOLO (class x_center y_center width height)
        boxes = []
        labels = []
        
        with open(self.label_paths[idx], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = map(float, parts)
                    
                    # YOLO → COCO (x_min, y_min, x_max, y_max)
                    x_min = (xc - bw / 2) * w
                    y_min = (yc - bh / 2) * h
                    x_max = (xc + bw / 2) * w
                    y_max = (yc + bh / 2) * h
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Validar boxes
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        target = {
            'bbox': torch.tensor(boxes, dtype=torch.float32),
            'cls': torch.tensor(labels, dtype=torch.int64),
            'img_scale': torch.tensor([1.0], dtype=torch.float32),
            'img_size': torch.tensor([image.shape[1], image.shape[2]], dtype=torch.int64)
        }
        
        return image, target

# ── Augmentations ──────────────────────────────────────────────────────────────
def get_train_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# ── Collate function ───────────────────────────────────────────────────────────
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets

# ── Crear YAMLs combinados ─────────────────────────────────────────────────────
def create_combined_yaml() -> str:
    base = os.path.abspath(EC_BASE)
    ec1_train = os.path.join(base, "license-plates-ec-1", "train", "images").replace("\\", "/")
    ec1_valid = os.path.join(base, "license-plates-ec-1", "valid", "images").replace("\\", "/")
    ec1_test  = os.path.join(base, "license-plates-ec-1", "test",  "images").replace("\\", "/")
    ec2_train = os.path.join(base, "license-plates-ec-2", "train", "images").replace("\\", "/")
    ec4_train = os.path.join(base, "license-plates-ec-4", "train", "images").replace("\\", "/")

    data = {
        "train": [ec1_train, ec2_train, ec4_train],
        "val":   ec1_valid,
        "test":  ec1_test,
        "nc":    1,
        "names": ["license plate"],
    }

    path = os.path.join(base, "data_combined.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path

def create_combined_all_yaml() -> str:
    base    = os.path.abspath(EC_BASE)
    global_ = os.path.abspath("datasets/license-plates")
    
    gl_train  = os.path.join(global_, "train", "images").replace("\\", "/")
    gl_valid  = os.path.join(global_, "valid", "images").replace("\\", "/")
    ec1_train = os.path.join(base, "license-plates-ec-1", "train", "images").replace("\\", "/")
    ec2_train = os.path.join(base, "license-plates-ec-2", "train", "images").replace("\\", "/")
    ec4_train = os.path.join(base, "license-plates-ec-4", "train", "images").replace("\\", "/")

    data = {
        "train": [gl_train, ec1_train, ec2_train, ec4_train],
        "val":   gl_valid,
        "test":  os.path.join(global_, "test", "images").replace("\\", "/"),
        "nc":    1,
        "names": ["license plate"],
    }

    path = os.path.join(base, "data_combined_all.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path

def load_yaml_paths(dataset_key: str) -> Tuple[List[str], str, str]:
    """Retorna (train_dirs, val_dir, test_dir)"""
    if dataset_key == "combined":
        yaml_path = create_combined_yaml()
    elif dataset_key == "combined_all":
        yaml_path = create_combined_all_yaml()
    else:
        yaml_path = DATASETS[dataset_key]["yaml"]
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    train = data['train'] if isinstance(data['train'], list) else [data['train']]
    val   = data['val']
    test  = data.get('test', val)
    
    return train, val, test

# ── Modelo EfficientDet ────────────────────────────────────────────────────────
def create_model(num_classes=NUM_CLASSES, pretrained=True):
    config = get_efficientdet_config(MODEL_NAME)
    config.num_classes = num_classes
    config.image_size = IMG_SIZE
    
    model = EfficientDet(config, pretrained_backbone=pretrained)
    model.class_net = HeadNet(
        config,
        num_outputs=num_classes,
    )
    
    # Envolver en DetBenchTrain para training
    model = DetBenchTrain(model, config)
    return model

# ── Training loop ──────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, targets in pbar:
        images = images.to(device)
        
        # Preparar targets para EfficientDet
        batch_targets = {}
        for key in ['bbox', 'cls', 'img_scale', 'img_size']:
            batch_targets[key] = torch.stack([t[key] for t in targets]).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, batch_targets)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader)

# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        
        batch_targets = {}
        for key in ['bbox', 'cls', 'img_scale', 'img_size']:
            batch_targets[key] = torch.stack([t[key] for t in targets]).to(device)
        
        loss_dict = model(images, batch_targets)
        loss = loss_dict['loss']
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# ── Entrenamiento principal ────────────────────────────────────────────────────
def train(dataset_key: str):
    cfg = DATASETS[dataset_key]
    model_name = cfg["name"]
    
    print("\n" + "=" * 60)
    print(f"  Entrenando: {dataset_key.upper()}")
    print(f"  Modelo:     {model_name}")
    print(f"  Arquitectura: {MODEL_NAME}")
    print(f"  Dispositivo: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}")
    print("=" * 60)
    
    # Cargar rutas
    train_dirs, val_dir, test_dir = load_yaml_paths(dataset_key)
    
    # Crear datasets
    train_dataset = YOLODataset(train_dirs, transform=get_train_transforms())
    val_dataset   = YOLODataset(val_dir, transform=get_val_transforms())
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    # Crear modelo
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Directorios de guardado
    save_dir = Path(f"runs/detect/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Guardar último modelo
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, weights_dir / "last.pt")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, weights_dir / "best.pt")
            print(f"  ✅ Mejor modelo guardado (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping después de {epoch+1} épocas")
            break
    
    best_path = str(weights_dir / "best.pt")
    print(f"\n  ✅ Entrenamiento completado")
    print(f"  ✅ Mejor modelo: {best_path}")
    return best_path

# ── Evaluación ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model_path: str, dataset_key: str):
    print(f"\n[eval] {dataset_key.upper()} — {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[eval] ❌ Modelo no encontrado: {model_path}")
        return
    
    # Cargar modelo
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Cargar dataset de validación
    _, val_dir, _ = load_yaml_paths(dataset_key)
    val_dataset = YOLODataset(val_dir, transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loss = validate(model, val_loader, DEVICE)
    
    print("\n── Métricas ──────────────────────────────────────────")
    print(f"  Dataset:     {dataset_key.upper()}")
    print(f"  Val Loss:    {val_loss:.4f}")
    print(f"  Best Epoch:  {checkpoint.get('epoch', 'N/A')}")
    print("─────────────────────────────────────────────────────")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Optimizaciones CUDA
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if arg == "all":
        keys = ["global", "ecuador", "ecuador2", "ecuador4", "combined", "combined_all"]
    elif arg in DATASETS:
        keys = [arg]
    else:
        print(f"[error] Dataset '{arg}' no reconocido.")
        print(f"  Opciones: {' | '.join(list(DATASETS.keys()) + ['all'])}")
        sys.exit(1)
    
    results = {}
    for key in keys:
        best = train(key)
        results[key] = best
    
    for key, best in results.items():
        evaluate(best, key)
    
    print("\n" + "=" * 60)
    print("  Resumen final")
    for key, best in results.items():
        status = "✅" if os.path.exists(best) else "❌"
        print(f"  {status} {key.upper():15} → {best}")
    print("=" * 60)