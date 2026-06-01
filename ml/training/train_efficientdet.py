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
# Requisitos:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   pip install effdet timm albumentations opencv-python tqdm pyyaml

import sys
import os
import glob
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from torch.utils.data import Dataset, DataLoader
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time

# ── Detectar dispositivo ───────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] ✅ NVIDIA GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"[device]    CUDA Version: {torch.version.cuda}")
        print(f"[device]    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        device = torch.device("cpu")
        print("[device] ⚠️  CUDA no disponible, usando CPU")
        return device

DEVICE = get_device()

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_NAME   = "tf_efficientdet_d2"
EPOCHS       = 50
IMG_SIZE     = 768  # EfficientDet-D2 usa 768x768
BATCH_SIZE   = 4    # Ajustar según memoria GPU (D2 usa más memoria que YOLOv8n)
LEARNING_RATE = 1e-4
PATIENCE     = 10
NUM_WORKERS  = 4    # Optimizado para Ryzen 5 5600
NUM_CLASSES  = 1    # Solo "license plate"

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

# ── Dataset personalizado ──────────────────────────────────────────────────────
class LicensePlateDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform=None, img_size=IMG_SIZE):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Obtener lista de imágenes
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + 
                                 list(self.image_dir.glob("*.png")))
        
        print(f"  [dataset] Encontradas {len(self.image_files)} imágenes en {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Cargar etiquetas YOLO format
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # Convertir de YOLO (x_center, y_center, w, h) a (x1, y1, x2, y2)
                        x1 = (x_center - width / 2) * w
                        y1 = (y_center - height / 2) * h
                        x2 = (x_center + width / 2) * w
                        y2 = (y_center + height / 2) * h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(class_id))
        
        # Si no hay boxes, agregar uno dummy para evitar errores
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Convertir a formato EfficientDet
        target = {
            'bbox': torch.as_tensor(boxes, dtype=torch.float32),
            'cls': torch.as_tensor(labels, dtype=torch.int64),
            'img_scale': torch.tensor([1.0]),
            'img_size': torch.tensor([self.img_size, self.img_size])
        }
        
        return image, target

# ── Augmentaciones ─────────────────────────────────────────────────────────────
def get_train_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))

def get_valid_transforms(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

# ── Collate function ───────────────────────────────────────────────────────────
def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    images = torch.stack(images)
    return images, targets

# ── Crear modelo ───────────────────────────────────────────────────────────────
def create_model(num_classes=NUM_CLASSES, pretrained=True):
    config = get_efficientdet_config(MODEL_NAME)
    config.num_classes = num_classes
    config.image_size = IMG_SIZE
    
    model = EfficientDet(config, pretrained_backbone=pretrained)
    model.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    
    model = DetBenchTrain(model, config)
    return model.to(DEVICE)

# ── Funciones auxiliares ───────────────────────────────────────────────────────
def load_yaml_config(yaml_path: str) -> dict:
    """Carga configuración YAML del dataset"""
    base_dir = os.path.abspath(os.path.dirname(yaml_path))
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convertir rutas relativas a absolutas
    if not os.path.isabs(str(data.get('train', ''))):
        data['train'] = os.path.join(base_dir, "train", "images")
    if not os.path.isabs(str(data.get('val', ''))):
        data['val'] = os.path.join(base_dir, "valid", "images")
    if not os.path.isabs(str(data.get('test', ''))):
        data['test'] = os.path.join(base_dir, "test", "images")
    
    return data

def get_label_dir(image_dir: str) -> str:
    """Obtiene directorio de labels correspondiente"""
    return str(Path(image_dir).parent / "labels")

# ── Crear data_combined.yaml ───────────────────────────────────────────────────
def create_combined_yaml() -> str:
    base = os.path.abspath(EC_BASE)
    
    ec1_train = os.path.join(base, "license-plates-ec-1", "train", "images")
    ec1_valid = os.path.join(base, "license-plates-ec-1", "valid", "images")
    ec1_test  = os.path.join(base, "license-plates-ec-1", "test", "images")
    ec2_train = os.path.join(base, "license-plates-ec-2", "train", "images")
    ec4_train = os.path.join(base, "license-plates-ec-4", "train", "images")
    
    data = {
        "train": [ec1_train, ec2_train, ec4_train],
        "val": ec1_valid,
        "test": ec1_test,
        "nc": 1,
        "names": ["license plate"],
    }
    
    path = os.path.join(base, "data_combined.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  [combined] ec-1 train  → {ec1_train}")
    print(f"  [combined] ec-2 train  → {ec2_train}")
    print(f"  [combined] ec-4 train  → {ec4_train}")
    print(f"  [combined] val         → {ec1_valid}")
    return path

def create_combined_all_yaml() -> str:
    base = os.path.abspath(EC_BASE)
    global_ = os.path.abspath("datasets/license-plates")
    
    gl_train  = os.path.join(global_, "train", "images")
    gl_valid  = os.path.join(global_, "valid", "images")
    ec1_train = os.path.join(base, "license-plates-ec-1", "train", "images")
    ec2_train = os.path.join(base, "license-plates-ec-2", "train", "images")
    ec4_train = os.path.join(base, "license-plates-ec-4", "train", "images")
    
    data = {
        "train": [gl_train, ec1_train, ec2_train, ec4_train],
        "val": gl_valid,
        "test": os.path.join(global_, "test", "images"),
        "nc": 1,
        "names": ["license plate"],
    }
    
    path = os.path.join(base, "data_combined_all.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  [combined_all] global  → {gl_train}")
    print(f"  [combined_all] ec-1    → {ec1_train}")
    print(f"  [combined_all] ec-2    → {ec2_train}")
    print(f"  [combined_all] ec-4    → {ec4_train}")
    return path

# ── Entrenamiento ──────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:
        images = images.to(device)
        
        # Mover targets a device
        for i, target in enumerate(targets):
            targets[i] = {k: v.to(device) for k, v in target.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        loss, _, _ = model(images, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            
            for i, target in enumerate(targets):
                targets[i] = {k: v.to(device) for k, v in target.items()}
            
            loss, _, _ = model(images, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(dataset_key: str):
    cfg = DATASETS[dataset_key]
    print("\n" + "=" * 60)
    print(f"  Entrenando: {dataset_key.upper()}")
    print(f"  Modelo:     {cfg['name']}")
    print(f"  Dispositivo: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print("=" * 60)
    
    # Preparar YAML
    if dataset_key == "combined":
        yaml_path = create_combined_yaml()
    elif dataset_key == "combined_all":
        yaml_path = create_combined_all_yaml()
    else:
        yaml_path = cfg["yaml"]
    
    config = load_yaml_config(yaml_path)
    
    # Crear datasets
    train_dirs = config['train'] if isinstance(config['train'], list) else [config['train']]
    
    # Dataset de entrenamiento (combinar múltiples si es necesario)
    train_datasets = []
    for train_dir in train_dirs:
        label_dir = get_label_dir(train_dir)
        ds = LicensePlateDataset(
            train_dir, 
            label_dir, 
            transform=get_train_transforms(),
            img_size=IMG_SIZE
        )
        train_datasets.append(ds)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    
    # Dataset de validación
    val_dir = config['val']
    val_label_dir = get_label_dir(val_dir)
    val_dataset = LicensePlateDataset(
        val_dir,
        val_label_dir,
        transform=get_valid_transforms(),
        img_size=IMG_SIZE
    )
    
    print(f"  [data] Train: {len(train_dataset)} imágenes")
    print(f"  [data] Val:   {len(val_dataset)} imágenes")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Crear modelo
    model = create_model(num_classes=NUM_CLASSES)
    
    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Directorios de guardado
    save_dir = Path(f"runs/efficientdet/{cfg['name']}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        torch.save(checkpoint, save_dir / 'last.pt')
        
        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"  ✅ Nuevo mejor modelo guardado (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping activado después de {epoch+1} epochs")
            break
    
    best_path = str(save_dir / 'best.pt')
    print(f"\n  ✅ Entrenamiento completado")
    print(f"  ✅ Mejor modelo: {best_path}")
    return best_path

# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(model_path: str, dataset_key: str):
    if not os.path.exists(model_path):
        print(f"[eval] ❌ Modelo no encontrado: {model_path}")
        return
    
    print(f"\n[eval] {dataset_key.upper()} — {model_path}")
    
    # Cargar configuración
    cfg = DATASETS[dataset_key]
    if dataset_key == "combined":
        yaml_path = f"{EC_BASE}/data_combined.yaml"
    elif dataset_key == "combined_all":
        yaml_path = f"{EC_BASE}/data_combined_all.yaml"
    else:
        yaml_path = cfg["yaml"]
    
    config = load_yaml_config(yaml_path)
    
    # Crear dataset de test
    test_dir = config.get('test', config['val'])
    test_label_dir = get_label_dir(test_dir)
    test_dataset = LicensePlateDataset(
        test_dir,
        test_label_dir,
        transform=get_valid_transforms(),
        img_size=IMG_SIZE
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Cargar modelo
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluar
    test_loss = validate(model, test_loader, DEVICE)
    
    print("\n── Métricas ──────────────────────────────────────────")
    print(f"  Dataset:   {dataset_key.upper()}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Imágenes:  {len(test_dataset)}")
    print("─────────────────────────────────────────────────────")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
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
    
    # Evaluar todos los modelos
    for key, best in results.items():
        evaluate(best, key)
    
    print("\n" + "=" * 60)
    print("  Resumen final")
    for key, best in results.items():
        status = "✅" if os.path.exists(best) else "❌"
        print(f"  {status} {key.upper():15} → {best}")
    print("=" * 60)