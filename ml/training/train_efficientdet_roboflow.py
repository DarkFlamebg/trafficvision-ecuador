#!/usr/bin/env python
"""
EfficientDet-D2 — CPU optimizado + RESUME + Anti-overfitting
Fixes vs versión anterior:
  - WEIGHT_DECAY 1e-5 → 1e-4 (penaliza memorización)
  - PATIENCE 30 → 20 (detener antes de overfittear más)
  - Augmentaciones más agresivas:
      · RandomResizedCrop (variedad de escala/posición)
      · HueSaturationValue (variedad de color)
      · CLAHE (contraste adaptativo — mejora placas)
      · CoarseDropout 3 holes → 5 holes
  - Dropout dinámico desactivado (incompatible con effdet)
  - Mixup suave en collate_fn (p=0.15) para regularizar
  - Guardar best.pt solo cuando val_loss baja en validaciones REALES
  - Logging de gap val-train para detectar overfitting en tiempo real
  - Prevención de suspensión del sistema en Windows durante entrenamiento
"""

import os, sys, math, torch, warnings, json, ctypes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Dict
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict

warnings.filterwarnings("ignore", category=UserWarning)

# ════════════════════ PREVENIR SUSPENSIÓN ════════════════════════════════════
def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            0x80000000 | 0x00000001 | 0x00000002
        )
        print("[sleep] Suspensión del sistema desactivada durante el entrenamiento")
    except Exception as e:
        print(f"[sleep] No se pudo desactivar suspensión: {e}")

def restore_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[sleep] Suspensión del sistema restaurada")
    except Exception:
        pass

# ════════════════════ CONFIGURACIÓN ══════════════════════════════════════════
MODEL_NAME    = "tf_efficientdet_d2"
IMG_SIZE      = 512
EPOCHS        = 75
BATCH_SIZE    = 4
LR            = 5e-4
LR_MIN        = 1e-7
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0
NUM_CLASSES   = 1
PATIENCE      = 15
WARMUP_EPOCHS = 10
GRAD_CLIP     = 1.0
VAL_EVERY     = 5
CACHE_IMAGES  = True
MIXUP_PROB    = 0.15
DROPOUT_P     = 0.3  # referencia — no se usa (desactivado)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DATASET_BASE = "ml/license-plates-ec-combined"
OUTPUT_DIR   = "ml/runs/detect/efficientdet_d2_v2"

NUM_THREADS = os.cpu_count() or 4
torch.set_num_threads(NUM_THREADS)
print(f"[cpu] Usando {NUM_THREADS} threads")

# ════════════════════ DISPOSITIVO ════════════════════════════════════════════
def get_device():
    print("[device] Ejecutando en CPU de forma nativa para estabilidad de effdet")
    return torch.device("cpu")

DEVICE = get_device()

# ════════════════════ DATASET CON CACHE ══════════════════════════════════════
class RoboflowDataset(Dataset):

    def __init__(self, base_dir: str, split: str = "train", transform=None, cache: bool = False):
        self.transform = transform
        self.cache     = cache
        self.samples   = []
        self._cache: Dict[int, dict] = {}

        img_dir = Path(base_dir) / split / "images"
        lbl_dir = Path(base_dir) / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {img_dir}")

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in img_dir.glob(ext):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))

        print(f"  [{split}] {len(self.samples)} imágenes encontradas")

        if cache:
            self._preload()

    def _preload(self):
        print(f"  [cache] Pre-cargando {len(self.samples)} imágenes en RAM...")
        for idx in tqdm(range(len(self.samples)), desc="  Cacheando", leave=False):
            img_path, lbl_path = self.samples[idx]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            self._cache[idx] = {
                "img": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "lbl": lbl_path,
            }
        print(f"  [cache] {len(self._cache)} imágenes en memoria")

    def __len__(self):
        return len(self.samples)

    def _load_raw(self, idx):
        if self.cache and idx in self._cache:
            entry = self._cache[idx]
            return entry["img"], entry["lbl"]
        img_path, lbl_path = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None, lbl_path
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), lbl_path

    def __getitem__(self, idx):
        img_rgb, lbl_path = self._load_raw(idx)
        if img_rgb is None:
            return self._dummy()

        h, w = img_rgb.shape[:2]

        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                if int(cls) != 0:
                    continue
                x1 = max(0.0, (cx - bw / 2) * w)
                y1 = max(0.0, (cy - bh / 2) * h)
                x2 = min(float(w), (cx + bw / 2) * w)
                y2 = min(float(h), (cy + bh / 2) * h)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(0)

        if not boxes:
            return self._dummy()

        if self.transform:
            sample = self.transform(image=img_rgb, bboxes=boxes, labels=labels)
            img_rgb = sample["image"]
            boxes   = list(sample["bboxes"])
            labels  = list(sample["labels"])
        else:
            img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        if not boxes:
            return self._dummy()

        return {
            "image":    img_rgb,
            "bboxes":   torch.tensor(boxes,  dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(idx),
        }

    def _dummy(self):
        return {
            "image":    torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32),
            "bboxes":   torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.zeros(0, dtype=torch.int64),
            "image_id": torch.tensor(0),
        }

# ════════════════════ AUGMENTACIONES ═════════════════════════════════════════
def get_transforms(split: str = "train"):
    if split == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomResizedCrop(
                size=(IMG_SIZE, IMG_SIZE),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                p=0.4
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.3
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                p=0.35
            ),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.3
        ))
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.3
        ))

# ════════════════════ MODELO ═════════════════════════════════════════════════
class EfficientDetModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        config = get_efficientdet_config(MODEL_NAME)
        config.num_classes = NUM_CLASSES
        config.image_size  = (IMG_SIZE, IMG_SIZE)

        self.model = EfficientDet(config)

        if pretrained and getattr(config, 'url', None):
            weights = torch.hub.load_state_dict_from_url(
                config.url, progress=True, map_location="cpu"
            )
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in weights.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print("  [model] Pesos preentrenados (COCO) cargados")

        self.bench_train   = DetBenchTrain(self.model)
        self.bench_predict = DetBenchPredict(self.model)

    def forward(self, x):
        return self.model(x)

# ════════════════════ COLLATE con Mixup suave ════════════════════════════════
def collate_fn(batch):
    images    = torch.stack([b["image"]    for b in batch])
    image_ids = torch.stack([b["image_id"] for b in batch])
    bs = len(batch)

    if np.random.random() < MIXUP_PROB and bs > 1:
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(bs)
        images = lam * images + (1 - lam) * images[idx]

    max_boxes = max(max(b["bboxes"].shape[0] for b in batch), 1)
    bboxes_padded = torch.zeros((bs, max_boxes, 4), dtype=torch.float32)
    cls_padded    = torch.full((bs, max_boxes), -1, dtype=torch.int64)

    for i, b in enumerate(batch):
        n = b["bboxes"].shape[0]
        if n > 0:
            boxes = b["bboxes"]
            bboxes_padded[i, :n, 0] = boxes[:, 1]  # y1
            bboxes_padded[i, :n, 1] = boxes[:, 0]  # x1
            bboxes_padded[i, :n, 2] = boxes[:, 3]  # y2
            bboxes_padded[i, :n, 3] = boxes[:, 2]  # x2
            cls_padded[i, :n] = b["labels"]

    return {
        "image":    images,
        "bboxes":   bboxes_padded,
        "cls":      cls_padded,
        "image_id": image_ids,
    }

# ════════════════════ TRAIN / VAL ════════════════════════════════════════════
def run_epoch(model, loader, optimizer, epoch, total, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    tag = "TRAIN" if training else "VAL"

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total} [{tag}]")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            bs = images.shape[0]
            targets = {
                "bbox":      batch["bboxes"].cpu(),
                "cls":       batch["cls"].cpu().long(),
                "img_scale": torch.ones(bs),
                "img_size":  torch.tensor([[IMG_SIZE, IMG_SIZE]] * bs,
                                          dtype=torch.float32),
            }

            if training:
                optimizer.zero_grad()
                outputs = model.bench_train(images, targets)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            else:
                outputs = model.bench_train(images, targets)
                loss = outputs["loss"]

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)

# ════════════════════ CHECKPOINT ═════════════════════════════════════════════
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, path):
    torch.save({
        "model_state_dict":     model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "val_loss":             val_loss,
        "best_val_loss":        best_val_loss,
        "img_size":             IMG_SIZE,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
    print(f"[resume] Checkpoint cargado:")
    print(f"         época guardada  : {ckpt['epoch']}")
    print(f"         reanudando desde: época {start_epoch}")
    print(f"         mejor val_loss  : {best_val_loss:.4f}")
    return start_epoch, best_val_loss

# ════════════════════ MAIN ═══════════════════════════════════════════════════
def train():
    prevent_sleep()

    print("\n" + "="*80)
    print(f"EfficientDet-D2 — Anti-overfitting | IMG={IMG_SIZE} | BS={BATCH_SIZE} | WD={WEIGHT_DECAY}")
    print("="*80 + "\n")

    os.makedirs(f"{OUTPUT_DIR}/weights", exist_ok=True)

    if not Path(DATASET_BASE).exists():
        print(f"[ERROR] Dataset no encontrado: {DATASET_BASE}")
        restore_sleep()
        return

    print("[loading] Cargando dataset...")
    train_dataset = RoboflowDataset(DATASET_BASE, "train",
                                    get_transforms("train"), cache=CACHE_IMAGES)
    val_dataset   = RoboflowDataset(DATASET_BASE, "valid",
                                    get_transforms("val"),   cache=CACHE_IMAGES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=False)

    print("[model] Cargando EfficientDet-D2...")
    model = EfficientDetModel(pretrained=True).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (LR_MIN + (LR - LR_MIN) * cosine) / LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    history          = {"train_loss": [], "val_loss": [], "lr": [], "gap": []}
    best_val_loss    = float("inf")
    start_epoch      = 0
    last_val_loss    = float("inf")
    patience_counter = 0

    last_ckpt    = Path(f"{OUTPUT_DIR}/weights/last.pt")
    history_path = Path(f"{OUTPUT_DIR}/training_history.json")

    if last_ckpt.exists():
        print(f"\n[resume] Checkpoint encontrado: {last_ckpt}")
        start_epoch, best_val_loss = load_checkpoint(
            last_ckpt, model, optimizer, scheduler
        )
        last_val_loss = best_val_loss
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            if "gap" not in history:
                history["gap"] = [0.0] * len(history["train_loss"])
            print(f"[resume] Historial cargado: {len(history['train_loss'])} épocas previas")
    else:
        print("[resume] Sin checkpoint previo — entrenando desde cero")

    for epoch in range(start_epoch, EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, epoch, EPOCHS, training=True)
        scheduler.step()

        is_real_val = ((epoch + 1) % VAL_EVERY == 0 or epoch == start_epoch)

        if is_real_val:
            val_loss      = run_epoch(model, val_loader, None, epoch, EPOCHS, training=False)
            last_val_loss = val_loss
        else:
            val_loss = last_val_loss

        gap = val_loss - train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["gap"].append(round(gap, 4))

        cached_tag = "" if is_real_val else " [val cached]"
        gap_color  = "⚠️ " if gap > 0.15 else "✓ "
        print(f"  epoch {epoch+1:>3}/{EPOCHS} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"gap: {gap_color}{gap:+.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}{cached_tag}")

        if is_real_val and val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_loss, best_val_loss,
                            f"{OUTPUT_DIR}/weights/best.pt")
            print(f"  [✓] Mejor modelo guardado (val_loss: {val_loss:.4f})")
        elif is_real_val:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, epoch,
                        val_loss, best_val_loss,
                        f"{OUTPUT_DIR}/weights/last.pt")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= PATIENCE:
            print(f"\n[early_stopping] Sin mejora en val real por {PATIENCE} validaciones")
            break

    restore_sleep()
    print(f"\n[success] Completado!")
    print(f"  Mejor modelo : {OUTPUT_DIR}/weights/best.pt")
    print(f"  Historial    : {OUTPUT_DIR}/training_history.json")
    print(f"\n  Regularización aplicada:")
    print(f"    weight_decay : 1e-4")
    print(f"    patience     : {PATIENCE} validaciones reales")
    print(f"    augmentations: RandomResizedCrop, HueSaturationValue, CLAHE, CoarseDropout")
    print(f"    mixup        : p={MIXUP_PROB} en collate_fn")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n[interrumpido] Entrenamiento cancelado por el usuario")
        restore_sleep()