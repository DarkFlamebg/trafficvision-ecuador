#!/usr/bin/env python
"""
Genera gráficas de entrenamiento:
  - Curva de pérdida (train vs val)
  - Curva PR (Precision-Recall)
  - Matriz de confusión
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # sin GUI — guarda directo a archivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import cv2
from tqdm import tqdm

# ════════════════════ CONFIGURACIÓN ══════════════════════════════════════════
MODEL_PATH     = "ml/runs/detect/efficientdet_d2_v2/weights/best.pt"
HISTORY_PATH   = "ml/runs/detect/efficientdet_d2_v2/training_history.json"
VAL_IMAGES_DIR = "ml/license-plates-ec-combined/valid/images"
VAL_LABELS_DIR = "ml/license-plates-ec-combined/valid/labels"
OUTPUT_DIR     = "ml/runs/detect/efficientdet_d2_v2/plots"
IMG_SIZE       = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════ 1. CURVA DE PÉRDIDA ════════════════════════════════════
def plot_loss_curve():
    print("[plot] Generando curva de pérdida...")

    if not Path(HISTORY_PATH).exists():
        print(f"  No se encontró historial: {HISTORY_PATH}")
        return

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    epochs     = list(range(1, len(train_loss) + 1))

    # Filtrar val cached (son repetidos) para la línea de val real
    val_real_epochs = []
    val_real_loss   = []
    seen = set()
    for i, v in enumerate(val_loss):
        if v not in seen or i == 0:
            val_real_epochs.append(i + 1)
            val_real_loss.append(v)
            seen.add(v)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Curvas de Pérdida — EfficientDet-D2", fontsize=14, fontweight='bold')

    # — Gráfica izquierda: escala normal
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-', linewidth=1.5, alpha=0.8, label='Train Loss')
    ax.plot(epochs, val_loss,   'r-', linewidth=1.0, alpha=0.4, label='Val Loss (cached)')
    ax.scatter(val_real_epochs, val_real_loss, color='red', s=40, zorder=5, label='Val Loss (real)')
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Pérdida por época")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — Gráfica derecha: escala log (para ver convergencia)
    ax2 = axes[1]
    ax2.semilogy(epochs, train_loss, 'b-', linewidth=1.5, alpha=0.8, label='Train Loss')
    ax2.semilogy(epochs, val_loss,   'r-', linewidth=1.0, alpha=0.4, label='Val Loss (cached)')
    ax2.scatter(val_real_epochs, val_real_loss, color='red', s=40, zorder=5, label='Val Loss (real)')
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss (log)")
    ax2.set_title("Pérdida por época (escala log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Anotar mejor val
    best_val   = min(val_real_loss)
    best_epoch = val_real_epochs[val_real_loss.index(best_val)]
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6)
    axes[0].annotate(f'Best: {best_val:.4f}\n(época {best_epoch})',
                     xy=(best_epoch, best_val),
                     xytext=(best_epoch + 2, best_val * 1.5),
                     fontsize=9, color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/loss_curve.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {out}")

    # — Gráfica del gap (overfitting monitor)
    if "gap" in history:
        fig, ax = plt.subplots(figsize=(10, 4))
        gaps = history["gap"]
        colors = ['red' if g > 0.15 else 'green' for g in gaps]
        ax.bar(epochs, gaps, color=colors, alpha=0.7)
        ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, label='Umbral overfitting (0.15)')
        ax.axhline(y=0,    color='gray',   linestyle='-',  linewidth=0.8)
        ax.set_xlabel("Época")
        ax.set_ylabel("Gap (val - train)")
        ax.set_title("Monitor de Overfitting — Gap val-train")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        verde  = mpatches.Patch(color='green', alpha=0.7, label='Sin overfitting (gap ≤ 0.15)')
        rojo   = mpatches.Patch(color='red',   alpha=0.7, label='Overfitting detectado (gap > 0.15)')
        naranja = mpatches.Patch(color='orange', label='Umbral (0.15)')
        ax.legend(handles=[verde, rojo, naranja])

        plt.tight_layout()
        out2 = f"{OUTPUT_DIR}/overfitting_gap.png"
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {out2}")


# ════════════════════ INFERENCIA PARA PR Y CONFUSIÓN ═════════════════════════
def load_model():
    from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict

    device = torch.device("cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes       = 1
    config.image_size        = (IMG_SIZE, IMG_SIZE)
    config.score_thresh      = 0.0001
    config.max_det_per_image = 100

    net = EfficientDet(config, pretrained_backbone=False)

    state_dict = ckpt.get('model_state_dict', ckpt)
    cleaned    = {(k[6:] if k.startswith("model.") else k): v
                  for k, v in state_dict.items()}
    net.load_state_dict(cleaned, strict=False)
    net.eval()

    return DetBenchPredict(net).to(device), device


def infer_image(model, device, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], img

    h, w   = img.shape[:2]
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale  = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(rgb, (new_w, new_h))

    pad_h = (IMG_SIZE - new_h) // 2
    pad_w = (IMG_SIZE - new_w) // 2
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (padded.astype(np.float32) / 255.0 - mean) / std

    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        dets = model(tensor)

    if isinstance(dets, (list, tuple)):
        dets = dets[0]
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().numpy()
    if dets.ndim == 3:
        dets = dets[0]
    if dets.shape[0] == 0 or dets.shape[1] < 6:
        return [], img

    preds = []
    for det in dets:
        score = float(det[4])
        if score < 1e-5:
            continue
        x1 = int((float(det[0]) - pad_w) / scale)
        y1 = int((float(det[1]) - pad_h) / scale)
        x2 = int((float(det[2]) - pad_w) / scale)
        y2 = int((float(det[3]) - pad_h) / scale)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        if (x2 - x1) < 10 or (y2 - y1) < 5:
            continue
        preds.append([x1, y1, x2, y2, score])

    return preds, img


def load_gt(label_path, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def collect_predictions(model, device):
    """Recolecta todas las predicciones y GT del set de validación."""
    images_path = Path(VAL_IMAGES_DIR)
    labels_path = Path(VAL_LABELS_DIR)
    all_images  = list(images_path.glob("*.jpg"))

    all_scores, all_tp_flags = [], []
    total_gt = 0

    print(f"[eval] Procesando {len(all_images)} imágenes de validación...")
    for img_path in tqdm(all_images, desc="  Evaluando"):
        label_file = labels_path / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        gts   = load_gt(label_file, img_path)
        preds, _ = infer_image(model, device, img_path)
        total_gt += len(gts)

        if not preds:
            continue

        preds_sorted = sorted(preds, key=lambda x: x[4], reverse=True)
        matched_gt   = set()

        for pred in preds_sorted:
            score   = pred[4]
            box     = pred[:4]
            best_iou, best_idx = 0.0, -1

            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                iou = compute_iou(box, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi

            if best_iou >= 0.5 and best_idx >= 0:
                matched_gt.add(best_idx)
                all_scores.append(score)
                all_tp_flags.append(1)
            else:
                all_scores.append(score)
                all_tp_flags.append(0)

    return np.array(all_scores), np.array(all_tp_flags), total_gt


# ════════════════════ 2. CURVA PR ════════════════════════════════════════════
def plot_pr_curve(scores, tp_flags, total_gt):
    print("[plot] Generando curva PR...")

    if len(scores) == 0:
        print("  Sin predicciones — no se puede generar curva PR")
        return

    sorted_idx = np.argsort(-scores)
    tp_sorted  = tp_flags[sorted_idx]

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1 - tp_sorted)

    precision = tp_cum / (tp_cum + fp_cum + 1e-10)
    recall    = tp_cum / (total_gt + 1e-10)

    # AP usando interpolación de 11 puntos
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap  += precision[mask].max() if mask.any() else 0.0
    ap /= 11.0

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AP={ap:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color='blue')
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Curva Precision-Recall — EfficientDet-D2\nAP@0.5 = {ap:.3f}", fontsize=13)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Punto F1 máximo
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1)
    ax.scatter(recall[best_f1_idx], precision[best_f1_idx],
               color='red', s=100, zorder=5,
               label=f'Best F1={f1[best_f1_idx]:.3f} '
                     f'(P={precision[best_f1_idx]:.2f}, R={recall[best_f1_idx]:.2f})')
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/pr_curve.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {out}")
    print(f"  AP@0.5 = {ap:.4f} | Best F1 = {f1[best_f1_idx]:.4f}")
    return ap


# ════════════════════ 3. MATRIZ DE CONFUSIÓN ═════════════════════════════════
def plot_confusion_matrix(scores, tp_flags, total_gt, threshold=None):
    print("[plot] Generando matriz de confusión...")

    if len(scores) == 0:
        print("    Sin predicciones")
        return

    # Threshold automático: el que maximiza F1
    if threshold is None:
        sorted_idx = np.argsort(-scores)
        tp_sorted  = tp_flags[sorted_idx]
        sc_sorted  = scores[sorted_idx]
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(1 - tp_sorted)
        fn_cum = total_gt - tp_cum
        prec   = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec    = tp_cum / (total_gt + 1e-10)
        f1     = 2 * prec * rec / (prec + rec + 1e-10)
        best   = np.argmax(f1)
        threshold = float(sc_sorted[best])
        print(f"  Threshold automático (max F1): {threshold:.5f}")

    mask = scores >= threshold
    tp   = int(tp_flags[mask].sum())
    fp   = int((1 - tp_flags[mask]).sum())
    fn   = total_gt - tp
    tn   = 0  # detección de objetos: no hay TN significativo

    cm = np.array([[tp, fn],
                   [fp, tn]])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes   = ['Placa (positivo)', 'Fondo (negativo)']
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(['Predicho: Placa', 'Predicho: Fondo'], fontsize=10)
    ax.set_yticks(tick_marks); ax.set_yticklabels(['Real: Placa', 'Real: Fondo'],         fontsize=10)

    labels = [['TP', 'FN'], ['FP', 'TN*']]
    thresh_color = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight='bold',
                    color="white" if cm[i, j] > thresh_color else "black")

    total_preds = tp + fp
    prec  = tp / (tp + fp + 1e-10)
    rec   = tp / (tp + fn + 1e-10)
    f1    = 2 * prec * rec / (prec + rec + 1e-10)

    ax.set_title(
        f"Matriz de Confusión — EfficientDet-D2\n"
        f"Threshold: {threshold:.5f} | "
        f"P: {prec:.2%} | R: {rec:.2%} | F1: {f1:.3f}",
        fontsize=11
    )
    ax.set_ylabel('Etiqueta Real',     fontsize=11)
    ax.set_xlabel('Etiqueta Predicha', fontsize=11)

    note = "* TN no aplicable en detección de objetos"
    fig.text(0.5, 0.01, note, ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {out}")
    print(f"  TP={tp} | FP={fp} | FN={fn} | P={prec:.2%} | R={rec:.2%} | F1={f1:.3f}")


# ════════════════════ MAIN ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  GENERANDO GRÁFICAS DE ENTRENAMIENTO")
    print("=" * 60 + "\n")

    # 1. Curva de pérdida (no necesita modelo)
    plot_loss_curve()

    # 2. PR + Confusión (necesita modelo)
    if not Path(MODEL_PATH).exists():
        print(f"\n  Modelo no encontrado: {MODEL_PATH}")
        print("   Solo se generó la curva de pérdida.")
    else:
        print("\n[model] Cargando modelo para evaluación...")
        model, device = load_model()
        print(" Modelo listo\n")

        scores, tp_flags, total_gt = collect_predictions(model, device)
        print(f"\n  Total GT       : {total_gt}")
        print(f"  Total predicciones evaluadas: {len(scores)}\n")

        plot_pr_curve(scores, tp_flags, total_gt)
        plot_confusion_matrix(scores, tp_flags, total_gt)

    print(f"\n Todas las gráficas guardadas en: {OUTPUT_DIR}")