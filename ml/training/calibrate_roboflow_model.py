#!/usr/bin/env python
"""
Calibración de Confianzas Post-Entrenamiento
Soluciona el problema de confianzas desinfladas usando el dataset de validación
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import json
from typing import List, Tuple
import matplotlib.pyplot as plt

from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ════════════════════ CONFIG ════════════════════════════════════════════════════
MODEL_PATH = "ml/runs/detect/efficientdet_d2_roboflow/weights/best.pt"
VAL_IMAGES_DIR = "ml/datasets/roboflow/valid/images"
VAL_LABELS_DIR = "ml/datasets/roboflow/valid/labels"
IMG_SIZE = 768
OUTPUT_DIR = "ml/runs/detect/efficientdet_d2_roboflow"

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ CARGAR MODELO ════════════════════════════════════════════
def load_model(model_path):
    """Carga EfficientDet-D2 entrenado"""
    config = get_efficientdet_config("tf_efficientdet_d2")
    config.num_classes = 1
    
    model = EfficientDet(config)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model

# ════════════════════ INFERENCIA ════════════════════════════════════════════════
def infer_image(model, img_path, img_size=768):
    """Realiza inferencia en una imagen"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    
    # Resize
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])
    
    sample = transform(image=img_rgb)
    img_tensor = sample["image"].unsqueeze(0).to(DEVICE)
    
    # Inferencia
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Formato: [detections, class_scores]
    # detections: [batch, num_detections, 4 (xywh)]
    # class_scores: [batch, num_detections, num_classes]
    
    detections = outputs[0][0].cpu().numpy()  # [num_detections, 4]
    scores = outputs[1][0].cpu().numpy()      # [num_detections, 1]
    
    # Convertir xywh -> x1y1x2y2
    boxes_pred = []
    confs_pred = []
    
    for i, (det, conf) in enumerate(zip(detections, scores)):
        if conf[0] < 1e-6:  # Filtro mínimo
            continue
        
        x, y, w, h = det
        x1 = (x - w/2) * w_orig / img_size
        y1 = (y - h/2) * h_orig / img_size
        x2 = (x + w/2) * w_orig / img_size
        y2 = (y + h/2) * h_orig / img_size
        
        boxes_pred.append([x1, y1, x2, y2])
        confs_pred.append(float(conf[0]))
    
    return np.array(boxes_pred) if boxes_pred else np.array([]), np.array(confs_pred)

# ════════════════════ IOU ════════════════════════════════════════════════════════
def compute_iou(box1, box2):
    """Calcula IoU entre dos boxes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# ════════════════════ CALIBRACIÓN ════════════════════════════════════════════════
def calibrate_confidences(model, val_dir_imgs, val_dir_labels, iou_threshold=0.5):
    """
    Calibra las confianzas usando el dataset de validación
    Retorna: confidence_bins, correction_factors
    """
    
    print("[calibration] Recolectando predicciones en validación...")
    
    img_files = list(Path(val_dir_imgs).glob("*.jpg")) + list(Path(val_dir_imgs).glob("*.png"))
    
    confidence_vs_iou = []  # [(conf_pred, max_iou_con_gt), ...]
    
    for img_path in img_files:
        # Obtener predicciones
        boxes_pred, confs_pred = infer_image(model, str(img_path))
        
        if len(boxes_pred) == 0:
            continue
        
        # Obtener ground truth
        lbl_path = Path(val_dir_labels) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        
        boxes_gt = []
        h_orig, w_orig = cv2.imread(str(img_path)).shape[:2]
        
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * w_orig
                y1 = (cy - bh / 2) * h_orig
                x2 = (cx + bw / 2) * w_orig
                y2 = (cy + bh / 2) * h_orig
                boxes_gt.append([x1, y1, x2, y2])
        
        if not boxes_gt:
            continue
        
        # Matchear predicciones con GT
        for box_pred, conf_pred in zip(boxes_pred, confs_pred):
            max_iou = max([compute_iou(box_pred, box_gt) for box_gt in boxes_gt])
            confidence_vs_iou.append((conf_pred, max_iou))
    
    if not confidence_vs_iou:
        print("[WARNING] No se encontraron predicciones en validación")
        return None
    
    # Crear bins de confianza
    conf_array = np.array([x[0] for x in confidence_vs_iou])
    iou_array = np.array([x[1] for x in confidence_vs_iou])
    
    num_bins = 10
    bins = np.linspace(conf_array.min(), conf_array.max(), num_bins + 1)
    
    calibration = {}
    for i in range(num_bins):
        mask = (conf_array >= bins[i]) & (conf_array < bins[i + 1])
        if mask.sum() > 0:
            avg_conf = conf_array[mask].mean()
            avg_iou = iou_array[mask].mean()
            calibration[f"bin_{i}"] = {
                "conf_range": [float(bins[i]), float(bins[i + 1])],
                "avg_pred_conf": float(avg_conf),
                "avg_iou": float(avg_iou),
                "count": int(mask.sum())
            }
    
    # Guardar calibración
    output_path = f"{OUTPUT_DIR}/confidence_calibration.json"
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)
    
    print(f"[success] Calibración guardada: {output_path}")
    
    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(conf_array, iou_array, alpha=0.5)
    plt.xlabel("Predicted Confidence")
    plt.ylabel("IoU with Ground Truth")
    plt.title("Confidence Calibration Analysis")
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/confidence_calibration.png", dpi=100)
    print(f"[success] Gráfico guardado: {OUTPUT_DIR}/confidence_calibration.png")
    
    return calibration

def main():
    print("\n" + "="*80)
    print("Calibración de Confianzas - EfficientDet-D2 Roboflow")
    print("="*80 + "\n")
    
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Modelo no encontrado: {MODEL_PATH}")
        return
    
    if not Path(VAL_IMAGES_DIR).exists():
        print(f"[ERROR] Directorio de validación no encontrado: {VAL_IMAGES_DIR}")
        return
    
    print("[loading] Cargando modelo...")
    model = load_model(MODEL_PATH)
    
    print("[calibrating] Calibrando confianzas...")
    calibration = calibrate_confidences(model, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    
    if calibration:
        print("\n[results] Calibración completada:")
        for bin_name, bin_data in calibration.items():
            print(f"  {bin_name}: conf_range={bin_data['conf_range']}, "
                  f"avg_iou={bin_data['avg_iou']:.3f}, n={bin_data['count']}")

if __name__ == "__main__":
    main()
