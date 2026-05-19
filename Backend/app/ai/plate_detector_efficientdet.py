# app/ai/plate_detector_efficientdet.py
# Detecta placas usando EfficientDet-D2 entrenado localmente con dataset ecuatoriano

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

# ── Rutas ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.abspath(os.path.join(_BASE_DIR, "../../.."))

MODEL_PATH            = os.path.join(_ROOT_DIR, "ml", "models", "trained", "efficientdet_d2_combined_all", "best.pt")
CONFIDENCE_THRESHOLD  = 0.25

# Autos/camiones: ratio > 1.5 (placa horizontal)
# Motos EC:       ratio < 1.0 (placa vertical ~10x15cm)
# Rango combinado: 0.3 – 6.0, el giro se maneja en crop_utils._rotate_if_moto
ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 6.0

_model = None
_device = None


def _get_device():
    """Detecta si hay GPU disponible."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[efficientdet] Usando dispositivo: {_device}")
    return _device


def _get_model():
    """
    Carga el modelo EfficientDet-D2 preentrenado.
    
    Returns:
        DetBenchEval: Modelo listo para inferencia
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo EfficientDet no encontrado: {MODEL_PATH}")
        
        device = _get_device()
        
        # Configuración del modelo D2
        config = get_efficientdet_config('tf_efficientdet_d2')
        config.num_classes = 1  # Solo clase "placa"
        config.image_size = (768, 768)  # Tamaño de entrada estándar para D2
        
        # Crear modelo
        net = EfficientDet(config, pretrained_backbone=False)
        net.class_net = HeadNet(config, num_outputs=1)
        
        # Cargar pesos entrenados
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Manejar diferentes formatos de checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Limpiar prefijo 'model.' si está presente
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_state_dict[k[6:]] = v
            else:
                cleaned_state_dict[k] = v
                
        # Cargar pesos de forma flexible (strict=False) para evitar fallas por buffers dinámicos (e.g., anchors.boxes)
        net.load_state_dict(cleaned_state_dict, strict=False)
        
        # Modo evaluación
        net.eval()
        
        # Wrapper para inferencia
        _model = DetBenchPredict(net)
        _model = _model.to(device)
        
        print(f"[efficientdet] Modelo cargado: {MODEL_PATH}")
    
    return _model


def _load_image(input_image) -> np.ndarray:
    """
    Carga la imagen respetando la orientación EXIF del celular.
    Retorna array NumPy BGR compatible con OpenCV.
    """
    if isinstance(input_image, str):
        pil_img = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("input_image debe ser str o np.ndarray")

    pil_img = ImageOps.exif_transpose(pil_img)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _preprocess_image(image: np.ndarray, target_size: tuple = (768, 768)) -> tuple:
    """
    Preprocesa la imagen para EfficientDet con padding centrado.
    
    Args:
        image: Imagen BGR de OpenCV
        target_size: Tamaño objetivo (ancho, alto)
        
    Returns:
        (tensor, scale, pad_w, pad_h): Tensor, factor de escala, y paddings en px
    """
    h, w = image.shape[:2]
    
    # Convertir a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar manteniendo aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Padding centrado para llegar al tamaño objetivo
    pad_h = (target_size[1] - new_h) // 2
    pad_w = (target_size[0] - new_w) // 2
    
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # El modelo espera rango [0.0, 255.0] sin normalizar
    normalized = padded.astype(np.float32)
    
    # Convertir a tensor [C, H, W]
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    
    return tensor.unsqueeze(0), scale, pad_w, pad_h


def detect_plate_efficientdet(input_image) -> list:
    """
    Detecta placas vehiculares usando EfficientDet-D2 local.

    Filtros aplicados:
    - Confianza mínima: 0.45
    - Proporción ancho/alto: entre 0.3 y 6.0 (forma de placa real)

    Args:
        input_image: ruta (str) o array NumPy BGR

    Returns:
        Lista de dicts:
          - "image":      recorte NumPy BGR de la placa (con padding + deskew)
          - "bbox":       [x1, y1, x2, y2] en píxeles originales (int)
          - "confidence": float 0.0 – 1.0
          - "detector":   str "efficientdet"
    """
    from app.ai.crop_utils import extract_plate_crop
    
    image  = _load_image(input_image)
    model  = _get_model()
    device = _get_device()
    ih, iw = image.shape[:2]

    # Preprocesar imagen
    input_tensor, scale, pad_w, pad_h = _preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Inferencia
    with torch.no_grad():
        detections = model(input_tensor)
    
    plates = []
    
    # Procesar detecciones
    # detections shape: [batch, max_det, 6] donde 6 = [x1, y1, x2, y2, score, class]
    print(f"[efficientdet] Raw detections shape: {detections.shape}")
    if len(detections) > 0 and detections[0] is not None:
        sorted_dets = sorted(detections[0].cpu().numpy(), key=lambda x: x[4], reverse=True)
        print(f"[efficientdet] Top 5 detections:")
        for idx, det in enumerate(sorted_dets[:5]):
            print(f"  {idx+1}: score={det[4]:.4f}, box={det[0:4]}, class={det[5]}")
            
        for det in detections[0]:
            # Convertir a CPU numpy
            det = det.cpu().numpy()
            
            y1, x1, y2, x2, score, cls = det
            
            # Filtrar por confianza
            if score < CONFIDENCE_THRESHOLD:
                continue
            
            # Reescalar coordenadas a imagen original
            x1 = int((x1 - pad_w) / scale)
            y1 = int((y1 - pad_h) / scale)
            x2 = int((x2 - pad_w) / scale)
            y2 = int((y2 - pad_h) / scale)
            
            # Clamp a dimensiones de imagen
            x1 = max(0, min(x1, iw))
            y1 = max(0, min(y1, ih))
            x2 = max(0, min(x2, iw))
            y2 = max(0, min(y2, ih))
            
            w_box = x2 - x1
            h_box = y2 - y1
            
            if h_box == 0 or w_box == 0:
                continue
            
            aspect_ratio = w_box / h_box
            if not (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
                print(f"[efficientdet] Bbox descartado por proporción: {w_box}x{h_box} = {aspect_ratio:.2f}")
                continue
            
            # Extraer crop con padding adaptativo + deskew
            crop = extract_plate_crop(image, x1, y1, x2, y2)
            if crop.size == 0:
                continue
            
            plates.append({
                "image":      crop,
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(float(score), 4),
                "detector":   "efficientdet",
            })
    
    return plates


# ── Testing ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python plate_detector_efficientdet.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Probando EfficientDet-D2 Plate Detector")
    print(f"{'='*60}\n")

    plates = detect_plate_efficientdet(image_path)
    print(f"Placas detectadas: {len(plates)}")
    
    for i, p in enumerate(plates, 1):
        print(f"\nPlaca #{i}:")
        print(f"  Confianza: {p['confidence']:.2%}")
        print(f"  Bbox: {p['bbox']}")
        print(f"  Detector: {p['detector']}")
        print(f"  Tamaño crop: {p['image'].shape}")

    print(f"\n{'='*60}\n")
