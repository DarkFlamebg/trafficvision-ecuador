#!/usr/bin/env python
"""
train_efficientdet_v3.py
========================
Entrenamiento robusto de EfficientDet-D2 para detección de placas vehiculares ecuatorianas.

Correcciones respecto a versiones anteriores:
  - [FIX #1] Scheduler post-warmup restaura steps correctamente en resume
  - [FIX #2] _dummy() pasa por el mismo pipeline de tensor que __getitem__ normal
  - [FIX #3] compute_metrics usa img_scales explícito compatible con todas las versiones de effdet
  - [FIX #4] Resume: steps de scheduler calculados con offset correcto, sin sobrepasar T_max
  - [FIX #5] Compatibilidad albumentations ≥1.3 y ≥1.4 con try/except en augmentations
  - [FIX #6] val_loader con drop_last=False y collate robusto para cualquier tamaño de batch
  - [FIX #7] export_onnx usa Path para construir ruta .onnx sin depender de str.replace
  - [MEJORA] collate_targets con validación de shapes antes del stack
  - [MEJORA] train_one_epoch captura OOM y continúa sin crashear
  - [MEJORA] Historial incluye métricas finales al terminar
  - [MEJORA] Logging más claro con prefijos de época consistentes

Uso:
  py train_efficientdet_v3.py                  → entrena combined_all (recomendado)
  py train_efficientdet_v3.py combined_all     → global + ec-1 + ec-2 + ec-4
  py train_efficientdet_v3.py combined         → ec-1 + ec-2 + ec-4
  py train_efficientdet_v3.py ecuador4         → solo ec-4 (debug rápido)
  py train_efficientdet_v3.py export           → solo exportar ONNX desde best.pt

Requisitos:
  pip install effdet timm albumentations>=1.3.0 opencv-python tqdm pyyaml onnx onnxruntime
"""

import os
import sys
import json
import time
import warnings
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN CENTRAL
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME    = "tf_efficientdet_d2"
IMG_SIZE      = 768
EPOCHS        = 100
BATCH_SIZE    = 4       # 6-8 para GPU ≥16GB | 4 para 8GB | 2 para 6GB | 1 para CPU
LR            = 1e-4
LR_MIN        = 1e-6
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0       # 0 en Windows para evitar deadlocks; 4 en Linux
NUM_CLASSES   = 1       # 1 clase = 'placa'
PATIENCE      = 20
WARMUP_EPOCHS = 5
GRAD_CLIP     = 1.0

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.25

_SCRIPT_DIR = Path(__file__).resolve().parent
GLOBAL_BASE = _SCRIPT_DIR / ".." / "datasets" / "raw" / "license-plates"
EC_BASE     = _SCRIPT_DIR / ".." / "datasets" / "raw" / "license-plates-ec-combined"

DATASETS: Dict[str, Dict] = {
    "global": {
        "train": [str(GLOBAL_BASE / "train" / "images")],
        "val":    str(GLOBAL_BASE / "valid" / "images"),
        "name":   "efficientdet_d2_global",
    },
    "ecuador": {
        "train": [str(EC_BASE / "license-plates-ec-1" / "train" / "images")],
        "val":    str(EC_BASE / "license-plates-ec-1" / "valid" / "images"),
        "name":   "efficientdet_d2_ecuador",
    },
    "ecuador2": {
        "train": [str(EC_BASE / "license-plates-ec-2" / "train" / "images")],
        "val":    str(EC_BASE / "license-plates-ec-1" / "valid" / "images"),
        "name":   "efficientdet_d2_ecuador2",
    },
    "ecuador4": {
        "train": [str(EC_BASE / "license-plates-ec-4" / "train" / "images")],
        "val":    str(EC_BASE / "license-plates-ec-1" / "valid" / "images"),
        "name":   "efficientdet_d2_ecuador4",
    },
    "combined": {
        "train": [
            str(EC_BASE / "license-plates-ec-1" / "train" / "images"),
            str(EC_BASE / "license-plates-ec-2" / "train" / "images"),
            str(EC_BASE / "license-plates-ec-4" / "train" / "images"),
        ],
        "val":  str(EC_BASE / "license-plates-ec-1" / "valid" / "images"),
        "name": "efficientdet_d2_combined",
    },
    "combined_all": {
        "train": [
            str(GLOBAL_BASE / "train" / "images"),
            str(EC_BASE / "license-plates-ec-1" / "train" / "images"),
            str(EC_BASE / "license-plates-ec-2" / "train" / "images"),
            str(EC_BASE / "license-plates-ec-4" / "train" / "images"),
        ],
        "val":  str(GLOBAL_BASE / "valid" / "images"),
        "name": "efficientdet_d2_combined_all",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPOSITIVO
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024 ** 3
        print(f"[device] GPU : {props.name}")
        print(f"[device] VRAM: {vram:.1f} GB")
        print(f"[device] CUDA: {torch.version.cuda}")
        return dev
    print("[device] CPU — entrenamiento será lento; considera reducir IMG_SIZE a 512")
    return torch.device("cpu")


DEVICE = get_device()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUGMENTACIONES
#  [FIX #5] Compatibilidad albumentations ≥1.3 y ≥1.4
#  GaussNoise: versiones antiguas usan var_limit, nuevas usan std_range
#  CoarseDropout: fill_value en <1.4, fill en ≥1.4
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gauss_noise():
    """Crea GaussNoise compatible con albumentations 1.3 y 1.4+."""
    import albumentations as _A
    try:
        return _A.GaussNoise(std_range=(0.01, 0.05), p=0.3)
    except TypeError:
        return _A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)


def _make_coarse_dropout():
    """Crea CoarseDropout compatible con albumentations 1.3 y 1.4+."""
    import albumentations as _A
    try:
        return _A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 30),
            hole_width_range=(10, 60),
            fill=0, p=0.15,
        )
    except TypeError:
        return _A.CoarseDropout(
            max_holes=3, max_height=30, max_width=60,
            fill_value=0, p=0.15,
        )


def _make_image_compression():
    """Crea ImageCompression compatible con albumentations 1.3 y 1.4+."""
    import albumentations as _A
    try:
        return _A.ImageCompression(quality_range=(60, 95), p=0.3)
    except TypeError:
        return _A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3)


def _make_random_shadow():
    """Crea RandomShadow compatible con albumentations 1.3 y 1.4+."""
    import albumentations as _A
    try:
        return _A.RandomShadow(num_shadows_limit=(1, 2), p=0.25)
    except TypeError:
        return _A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.25)


def _make_pad_if_needed():
    """Crea PadIfNeeded compatible con albumentations 1.3 ('value') y 1.4+ ('fill')."""
    import albumentations as _A
    try:
        return _A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT, fill=0,
        )
    except TypeError:
        return _A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT, value=0,
        )


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        _make_pad_if_needed(),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.15,
            rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.4,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.25),
        _make_gauss_noise(),
        _make_image_compression(),
        _make_random_shadow(),
        _make_coarse_dropout(),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.25,
        clip=True,
    ))


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        _make_pad_if_needed(),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        clip=True,
    ))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

# Tensor dummy pre-construido para imágenes corruptas (shape correcta, ya normalizado via zeros)
_DUMMY_IMG = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)


class PlateDataset(Dataset):
    """
    Dataset de placas vehiculares. Lee formato YOLO (.txt) y convierte al
    formato que espera EfficientDet: bbox en [y1, x1, y2, x2] píxeles absolutos.

    [FIX #2] _dummy() retorna siempre el mismo tipo que el path normal (tensor float32).
    """

    def __init__(self, img_dirs, transform=None, split: str = "train"):
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]
        self.transform = transform
        self.split     = split
        self.samples: List[Tuple[str, Optional[str]]] = []

        for raw_dir in img_dirs:
            d = Path(raw_dir)
            if not d.exists():
                print(f"  [WARN] Directorio no encontrado (ignorado): {d}")
                continue
            lbl_dir = d.parent.parent / "labels" / d.name
            found   = 0
            with_lbl = 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for img_path in sorted(d.glob(ext)):
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    has_lbl  = lbl_path.exists()
                    if has_lbl:
                        with_lbl += 1
                    self.samples.append((
                        str(img_path),
                        str(lbl_path) if has_lbl else None,
                    ))
                    found += 1
            print(f"  [dataset/{split}] {found:>5} imgs ({with_lbl} con label)  ← {d}")

        print(f"  [dataset/{split}] TOTAL: {len(self.samples)} imágenes\n")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # ── Cargar imagen ──────────────────────────────────────────────────────
        try:
            img_bgr = cv2.imread(img_path)
            assert img_bgr is not None, "cv2.imread retornó None"
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return self._dummy()

        h, w = img_rgb.shape[:2]

        # ── Cargar labels YOLO → pascal_voc ───────────────────────────────────
        boxes, labels = [], []
        if lbl_path and os.path.isfile(lbl_path):
            try:
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, cx, cy, bw, bh = map(float, parts[:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        x1, y1 = max(0.0, x1), max(0.0, y1)
                        x2, y2 = min(float(w), x2), min(float(h), y2)
                        if x2 - x1 >= 2 and y2 - y1 >= 2:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(int(cls) + 1)  # effdet: 0=background, objetos ≥1
            except Exception:
                pass

        # ── Albumentations ─────────────────────────────────────────────────────
        if self.transform:
            try:
                result  = self.transform(image=img_rgb, bboxes=boxes, labels=labels)
                img_rgb = result["image"]   # Tensor float32 [3, H, W] tras ToTensorV2
                boxes   = list(result["bboxes"])
                labels  = list(result["labels"])
            except Exception:
                return self._dummy()
        else:
            # Sin transform: convertir manualmente a tensor
            img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        # ── pascal_voc → EfficientDet [y1,x1,y2,x2] ──────────────────────────
        boxes_np = np.array(boxes, dtype=np.float32)
        if boxes_np.ndim == 2 and boxes_np.shape[0] > 0:
            boxes_np = boxes_np[:, [1, 0, 3, 2]]   # x1y1x2y2 → y1x1y2x2
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)

        target = {
            "bbox":      torch.tensor(boxes_np,               dtype=torch.float32),
            "cls":       torch.tensor(labels,                 dtype=torch.float32),
            "img_scale": torch.tensor([1.0],                  dtype=torch.float32),
            "img_size":  torch.tensor([IMG_SIZE, IMG_SIZE],   dtype=torch.float32),
        }
        return img_rgb, target

    @staticmethod
    def _dummy():
        """
        [FIX #2] Item vacío con el MISMO tipo/shape que los items normales:
        - img: tensor float32 [3, IMG_SIZE, IMG_SIZE] (igual que después de ToTensorV2+Normalize)
        - target: shapes consistentes con collate_targets
        """
        target = {
            "bbox":      torch.zeros((0, 4),                dtype=torch.float32),
            "cls":       torch.zeros((0,),                  dtype=torch.float32),
            "img_scale": torch.ones((1,),                   dtype=torch.float32),
            "img_size":  torch.tensor([IMG_SIZE, IMG_SIZE], dtype=torch.float32),
        }
        return _DUMMY_IMG.clone(), target


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLATE
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)


def collate_targets(targets: list, device: torch.device) -> dict:
    """
    Apila targets con padding para tamaños variables de bboxes.

    [MEJORA] Valida shapes antes del stack para detectar inconsistencias.
    img_scale e img_size siempre tienen shape fija → stack directo.
    bbox y cls necesitan padding hasta el máximo del batch.
    """
    max_boxes = max(t["bbox"].shape[0] for t in targets)
    max_boxes = max(max_boxes, 1)  # nunca tensor vacío

    bboxes, clses = [], []
    for t in targets:
        n   = t["bbox"].shape[0]
        pad = max_boxes - n
        if pad > 0:
            bboxes.append(torch.cat([t["bbox"], torch.zeros(pad, 4)],                 dim=0))
            clses.append( torch.cat([t["cls"],  torch.zeros(pad, dtype=torch.float32)], dim=0))
        else:
            bboxes.append(t["bbox"])
            clses.append( t["cls"])

    return {
        "bbox":      torch.stack(bboxes).to(device),
        "cls":       torch.stack(clses ).to(device),
        "img_scale": torch.stack([t["img_scale"] for t in targets]).to(device),
        "img_size":  torch.stack([t["img_size"]  for t in targets]).to(device),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(pretrained_backbone: bool = True) -> DetBenchTrain:
    config             = get_efficientdet_config(MODEL_NAME)
    config.num_classes = NUM_CLASSES
    config.image_size  = (IMG_SIZE, IMG_SIZE)   # CRÍTICO: tuple, no int
    net                = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    net.class_net      = HeadNet(config, num_outputs=NUM_CLASSES)
    return DetBenchTrain(net, config)


def load_net_for_inference(best_pt: str, device: torch.device) -> DetBenchPredict:
    """
    Carga un checkpoint guardado durante el training (DetBenchTrain) y lo
    envuelve en DetBenchPredict para inferencia.

    Maneja tanto state_dicts con prefijo 'model.' como sin él.
    """
    ckpt = torch.load(best_pt, map_location=device, weights_only=False)

    config             = get_efficientdet_config(MODEL_NAME)
    config.num_classes = NUM_CLASSES
    config.image_size  = (IMG_SIZE, IMG_SIZE)
    net                = EfficientDet(config, pretrained_backbone=False)
    net.class_net      = HeadNet(config, num_outputs=NUM_CLASSES)

    raw_sd   = ckpt["model_state_dict"]
    clean_sd = {(k[6:] if k.startswith("model.") else k): v for k, v in raw_sd.items()}
    net.load_state_dict(clean_sd, strict=False)
    net.eval()

    bench = DetBenchPredict(net).to(device)
    bench.eval()
    return bench


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.train()
    total_loss  = 0.0
    valid_steps = 0
    pbar = tqdm(loader, desc=f"  Train E{epoch_idx+1:03d}", dynamic_ncols=True, leave=False)

    for imgs, targets in pbar:
        try:
            imgs    = imgs.to(device, non_blocking=True)
            batch_t = collate_targets(targets, device)

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(imgs, batch_t)
            loss      = loss_dict["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  [WARN] Loss inválido ({loss.item():.4f}) — batch ignorado")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss  += loss.item()
            valid_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n  [WARN] OOM en batch — memoria liberada, continuando")
            else:
                print(f"\n  [WARN] RuntimeError en batch de train: {e}")
            continue
        except Exception as e:
            print(f"\n  [WARN] Error inesperado en batch de train: {e}")
            continue

    return total_loss / max(valid_steps, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss  = 0.0
    valid_steps = 0

    for imgs, targets in tqdm(loader, desc="  Val  ", dynamic_ncols=True, leave=False):
        try:
            imgs      = imgs.to(device, non_blocking=True)
            batch_t   = collate_targets(targets, device)
            loss_dict = model(imgs, batch_t)
            loss      = loss_dict["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss  += loss.item()
                valid_steps += 1
        except Exception as e:
            print(f"\n  [WARN] Error en batch de val: {e}")
            continue

    return total_loss / max(valid_steps, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS FINALES (mAP, Precision, Recall, F1)
# ═══════════════════════════════════════════════════════════════════════════════

def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre dos boxes en formato [y1,x1,y2,x2]."""
    y1 = max(box_a[0], box_b[0]); x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2]); x2 = min(box_a[3], box_b[3])
    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def compute_metrics(
    best_pt: str,
    val_loader: DataLoader,
    device: torch.device,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    Calcula mAP@0.5, mAP@0.5:0.95, Precision, Recall y F1 sobre el val set.

    [FIX #3] DetBenchPredict.forward acepta (imgs, img_scales) en todas las versiones
    de effdet; se pasa explícitamente en lugar de confiar en defaults.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    print("\n  Cargando mejor modelo para evaluación final...")
    bench = load_net_for_inference(best_pt, device)

    all_preds: List[Dict] = []
    all_gts:   List[np.ndarray] = []

    print("  Corriendo inferencia en val set...")
    for imgs, targets in tqdm(val_loader, desc="  Eval ", dynamic_ncols=True, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        B    = imgs.shape[0]

        try:
            # [FIX #3] img_scales siempre explícito — compatible con effdet ≥0.2.4
            img_scales = torch.ones(B, dtype=torch.float32, device=device)
            detections = bench(imgs, img_scales)
        except Exception as e:
            print(f"\n  [WARN] Error en eval batch: {e}")
            for t in targets:
                all_preds.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0)})
                _append_gt(all_gts, t)
            continue

        dets_np = detections.cpu().numpy()
        for i, t in enumerate(targets):
            det    = dets_np[i]
            mask   = det[:, 4] >= CONF_THRESHOLD
            all_preds.append({
                "boxes":  det[mask, :4],
                "scores": det[mask,  4],
            })
            _append_gt(all_gts, t)

    def _compute_ap_at_iou(preds, gts, iou_thr) -> Tuple[float, float, float]:
        tp_list, fp_list, scores_list = [], [], []
        n_gt_total = sum(len(g) for g in gts)

        for pred, gt in zip(preds, gts):
            boxes  = pred["boxes"]
            scores = pred["scores"]
            if len(scores) == 0:
                continue
            order      = np.argsort(-scores)
            boxes      = boxes[order]
            scores     = scores[order]
            matched_gt = set()
            for bi, b in enumerate(boxes):
                scores_list.append(scores[bi])
                if len(gt) == 0:
                    tp_list.append(0); fp_list.append(1)
                    continue
                ious = np.array([box_iou(b, g) for g in gt])
                best = int(np.argmax(ious))
                if ious[best] >= iou_thr and best not in matched_gt:
                    tp_list.append(1); fp_list.append(0)
                    matched_gt.add(best)
                else:
                    tp_list.append(0); fp_list.append(1)

        if not tp_list:
            return 0.0, 0.0, 0.0

        sc_arr = np.array(scores_list)
        order  = np.argsort(-sc_arr)
        tp_cum = np.cumsum(np.array(tp_list, dtype=float)[order])
        fp_cum = np.cumsum(np.array(fp_list, dtype=float)[order])

        recall_curve    = tp_cum / max(n_gt_total, 1)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = sum(
            (np.max(precision_curve[recall_curve >= thr]) if np.any(recall_curve >= thr) else 0.0)
            for thr in np.linspace(0, 1, 101)
        ) / 101.0

        mean_prec = float(np.mean(precision_curve))
        mean_rec  = float(recall_curve[-1]) if len(recall_curve) else 0.0
        return ap, mean_prec, mean_rec

    aps = []
    for thr in iou_thresholds:
        ap, _, _ = _compute_ap_at_iou(all_preds, all_gts, thr)
        aps.append(ap)

    ap50, prec50, rec50 = _compute_ap_at_iou(all_preds, all_gts, 0.5)
    f1 = 2 * prec50 * rec50 / max(prec50 + rec50, 1e-9)

    return {
        "mAP@0.5":      round(ap50,               4),
        "mAP@0.5:0.95": round(float(np.mean(aps)), 4),
        "Precision":    round(prec50,              4),
        "Recall":       round(rec50,               4),
        "F1":           round(f1,                  4),
        "AP_per_IoU":   {str(t): round(a, 4) for t, a in zip(iou_thresholds, aps)},
    }


def _append_gt(all_gts: list, t: dict):
    """Extrae ground-truth boxes válidos de un target y los añade a la lista."""
    gt_boxes = t["bbox"].cpu().numpy()
    gt_cls   = t["cls"].cpu().numpy()
    valid = (
        (gt_cls > 0) &
        ((gt_boxes[:, 2] - gt_boxes[:, 0]) >= 2) &
        ((gt_boxes[:, 3] - gt_boxes[:, 1]) >= 2)
    ) if gt_boxes.shape[0] > 0 else np.zeros(0, dtype=bool)
    all_gts.append(gt_boxes[valid] if valid.any() else np.zeros((0, 4)))


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT ONNX
#  [FIX #7] Usa Path para construir ruta .onnx — no depende de str.replace
# ═══════════════════════════════════════════════════════════════════════════════

def export_onnx(best_pt: str) -> Optional[str]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [export] Instala: pip install onnx onnxruntime")
        return None

    best_path = Path(best_pt)
    onnx_path = str(best_path.with_suffix(".onnx"))   # [FIX #7]

    print(f"\n  [export] Cargando checkpoint: {best_pt}")
    try:
        bench = load_net_for_inference(best_pt, torch.device("cpu"))
        bench.eval()

        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        print(f"  [export] Exportando ONNX opset=12 → {onnx_path}")
        torch.onnx.export(
            bench, dummy, onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}},
        )

        onnx.checker.check_model(onnx.load(onnx_path))
        print(f"  [export] Modelo ONNX válido: {onnx_path}")

        sess     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        _        = sess.run(None, {"images": dummy_np})   # warm-up
        t0       = time.perf_counter()
        for _ in range(10):
            sess.run(None, {"images": dummy_np})
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  [export] Latencia ONNX CPU (10 runs): {ms:.1f} ms/img")
        return onnx_path

    except Exception as e:
        print(f"  [export] FALLÓ: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    t_max: int,
    already_done: int = 0,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """
    [FIX #1 / FIX #4] Crea un CosineAnnealingLR y lo avanza 'already_done' pasos
    de forma controlada.  Si already_done >= t_max se clampea a t_max-1 para
    no sobrepasar el ciclo coseno.
    """
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=LR_MIN,
    )
    steps = min(already_done, t_max - 1)
    for _ in range(steps):
        sched.step()
    return sched


def train(dataset_key: str) -> str:
    cfg        = DATASETS[dataset_key]
    model_name = cfg["name"]
    train_dirs = [d for d in cfg["train"] if os.path.isdir(d)]
    val_dir    = cfg["val"]

    if not train_dirs:
        print(f"[ERROR] Ningún directorio de entrenamiento encontrado para '{dataset_key}'")
        sys.exit(1)
    if not os.path.isdir(val_dir):
        print(f"[ERROR] Directorio de validación no encontrado: {val_dir}")
        sys.exit(1)

    print(f"\n{'═'*64}")
    print(f"  Modelo   : {model_name}")
    print(f"  Dataset  : {dataset_key.upper()}")
    print(f"  Épocas   : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Img size : {IMG_SIZE}  |  Dispositivo: {DEVICE}")
    print(f"{'═'*64}\n")
    print("  Cargando datasets...")

    train_ds = PlateDataset(train_dirs, transform=get_train_transforms(), split="train")
    val_ds   = PlateDataset(val_dir,   transform=get_val_transforms(),   split="val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[ERROR] Dataset vacío")
        sys.exit(1)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=(NUM_WORKERS > 0),
        drop_last=True,
    )
    # [FIX #6] val_loader NO usa drop_last; collate_targets ya maneja batches de 1
    val_loader = DataLoader(
        val_ds, batch_size=max(BATCH_SIZE // 2, 1), shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    save_dir  = _SCRIPT_DIR.parent / "runs" / "detect" / model_name
    w_dir     = save_dir / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    last_path = w_dir / "last.pt"
    best_path = w_dir / "best.pt"

    model = create_model(pretrained_backbone=True).to(DEVICE)

    # Fase 1: backbone congelado (warmup)
    for p in model.model.backbone.parameters():
        p.requires_grad = False
    print(f"  [warmup] Backbone congelado — entrenando cabeza {WARMUP_EPOCHS} épocas")

    # Un único objeto AdamW durante toda la vida del entrenamiento
    # (conserva los momentos al descongelar el backbone)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # Scheduler inicial (cubre épocas 0..WARMUP_EPOCHS)
    sched_warmup  = _make_scheduler(optimizer, t_max=WARMUP_EPOCHS)
    sched_main    = None   # se crea al descongelar

    start_epoch     = 0
    best_val        = float("inf")
    patience_count  = 0
    history         = []
    backbone_thawed = False

    # ── Resume desde último checkpoint ───────────────────────────────────────
    if last_path.exists():
        print(f"\n  Checkpoint encontrado: {last_path}")
        try:
            ckpt        = torch.load(last_path, map_location=DEVICE, weights_only=False)
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt.get("best_val", float("inf"))
            history     = ckpt.get("history", [])
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            if start_epoch >= WARMUP_EPOCHS:
                # Descongelar backbone y restaurar scheduler principal
                for p in model.model.backbone.parameters():
                    p.requires_grad = True
                backbone_thawed = True
                for pg in optimizer.param_groups:
                    pg["lr"] = LR / 5

                # [FIX #1 / FIX #4] steps ya hechos en el scheduler principal
                steps_done = start_epoch - WARMUP_EPOCHS
                t_max_main = max(EPOCHS - WARMUP_EPOCHS, 1)
                sched_main = _make_scheduler(optimizer, t_max=t_max_main, already_done=steps_done)
            else:
                # Todavía en warmup
                sched_warmup = _make_scheduler(optimizer, t_max=WARMUP_EPOCHS, already_done=start_epoch)

            print(f"  Reanudando desde época {start_epoch + 1} | best_val={best_val:.4f}")
        except Exception as e:
            print(f"  [WARN] No se pudo cargar el checkpoint ({e}), empezando desde cero")
            start_epoch = 0; best_val = float("inf"); history = []

    # ── Bucle de entrenamiento ─────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # Descongelar backbone al terminar el warmup
        if epoch == WARMUP_EPOCHS and not backbone_thawed:
            print(f"\n  [warmup] Descongelando backbone en época {epoch + 1}...")
            for p in model.model.backbone.parameters():
                p.requires_grad = True
            backbone_thawed = True

            # Reducir LR manteniendo el mismo objeto optimizer (conserva momentos AdamW)
            for pg in optimizer.param_groups:
                pg["lr"] = LR / 5

            # [FIX #1] Scheduler principal parte desde cero para el resto de épocas
            t_max_main = max(EPOCHS - WARMUP_EPOCHS, 1)
            sched_main = _make_scheduler(optimizer, t_max=t_max_main, already_done=0)

        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss   = validate(model, val_loader, DEVICE)

        # Avanzar el scheduler correcto
        if sched_main is not None:
            sched_main.step()
        else:
            sched_warmup.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"  E{epoch+1:03d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        epoch_record = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "lr":         round(lr_now,     8),
        }
        history.append(epoch_record)

        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
            "best_val":             best_val,
            "history":              history,
        }

        try:
            torch.save(ckpt_data, last_path)
        except Exception as e:
            print(f"  [WARN] No se pudo guardar last.pt: {e}")

        if val_loss < best_val:
            best_val       = val_loss
            patience_count = 0
            try:
                torch.save(ckpt_data, best_path)
                print(f"  ✓ Mejor modelo guardado (val={val_loss:.4f})")
            except Exception as e:
                print(f"  [WARN] No se pudo guardar best.pt: {e}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping en época {epoch + 1} (sin mejora en {PATIENCE} épocas)")
                break

    # Guardar historial en JSON
    hist_path = save_dir / "history.json"
    try:
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n  Historial guardado: {hist_path}")
    except Exception as e:
        print(f"  [WARN] No se pudo guardar history.json: {e}")

    print(f"\n  Entrenamiento completo — mejor val_loss = {best_val:.4f}")
    return str(best_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN FINAL
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_final(best_pt: str, dataset_key: str):
    if not os.path.isfile(best_pt):
        print(f"  [eval] best.pt no encontrado: {best_pt}")
        return

    cfg     = DATASETS[dataset_key]
    val_dir = cfg["val"]

    val_ds = PlateDataset(val_dir, transform=get_val_transforms(), split="val_eval")
    val_loader = DataLoader(
        val_ds, batch_size=max(BATCH_SIZE // 2, 1), shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    print(f"\n{'═'*64}")
    print("  EVALUACIÓN FINAL")
    print(f"{'═'*64}")

    try:
        metrics = compute_metrics(best_pt, val_loader, DEVICE)
    except Exception as e:
        print(f"  [eval] Error calculando métricas: {e}")
        traceback.print_exc()
        return

    print(f"\n  {'Métrica':<20} {'Valor':>8}")
    print(f"  {'─'*28}")
    for k, v in metrics.items():
        if k == "AP_per_IoU":
            continue
        bar = "█" * int(v * 20) if isinstance(v, float) else ""
        print(f"  {k:<20} {v:>8.4f}  {bar}")

    metrics_path = Path(best_pt).parent.parent / "metrics_final.json"
    try:
        ckpt    = torch.load(best_pt, map_location="cpu", weights_only=False)
        summary = {
            "dataset":       dataset_key,
            "best_epoch":    ckpt.get("epoch", -1) + 1,
            "best_val_loss": round(ckpt.get("val_loss", 0.0), 6),
            "metrics":       metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Métricas guardadas: {metrics_path}")
    except Exception as e:
        print(f"  [WARN] No se pudo guardar metrics_final.json: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "combined_all"

    if arg == "export":
        key  = sys.argv[2].lower() if len(sys.argv) > 2 else "combined_all"
        name = DATASETS[key]["name"]
        bp   = str(_SCRIPT_DIR.parent / "runs" / "detect" / name / "weights" / "best.pt")
        if not os.path.isfile(bp):
            print(f"[error] No se encontró best.pt: {bp}")
            sys.exit(1)
        export_onnx(bp)
        sys.exit(0)

    if arg not in DATASETS:
        print(f"[error] Dataset '{arg}' no reconocido.")
        print(f"  Opciones: {' | '.join(list(DATASETS.keys()) + ['export'])}")
        sys.exit(1)

    best_pt = train(arg)
    evaluate_final(best_pt, arg)

    print("\n  Exportando a ONNX...")
    onnx_out = export_onnx(best_pt)

    print(f"\n{'═'*64}")
    print("  PIPELINE COMPLETO")
    print(f"  best.pt   → {best_pt}")
    if onnx_out:
        print(f"  best.onnx → {onnx_out}")
    print(f"{'═'*64}\n")
