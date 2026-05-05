# Ejecutar desde el backend:
# python test_ocr_params.py
#
# Requiere: crop_original_*_98x45.jpg en el mismo directorio
# (el que generaste con el debug anterior)

import cv2
import glob
import easyocr
import numpy as np

# Buscar el crop original automáticamente
crops = glob.glob("crop_original_*_98x45.jpg")
if not crops:
    print("ERROR: no se encontró crop_original_*_98x45.jpg")
    print("Asegúrate de correr este script desde el directorio del backend")
    exit(1)

img_orig = cv2.imread(crops[0])
print(f"Usando: {crops[0]}  shape: {img_orig.shape}")

# ── Pipeline v4 ────────────────────────────────────────────────────────────────
h, w = img_orig.shape[:2]
upscaled  = cv2.resize(img_orig, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
hu, wu    = upscaled.shape[:2]
crop_top  = int(hu * 0.15)
cropped   = upscaled[crop_top:, :]
gray      = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
clahe     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
gray      = clahe.apply(gray)
blur      = cv2.GaussianBlur(gray, (0,0), 2.0)
gray      = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
print(f"Procesada:  {processed.shape}")

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

configs = [
    ("default",          {}),
    ("contrast_low",     {"contrast_ths": 0.05, "adjust_contrast": 0.7}),
    ("mag_ratio_2",      {"mag_ratio": 2}),
    ("mag_ratio_3",      {"mag_ratio": 3}),
    ("text_thr_low",     {"text_threshold": 0.3, "low_text": 0.2}),
    ("width_ths_09",     {"width_ths": 0.9}),
    ("combined",         {"contrast_ths": 0.05, "adjust_contrast": 0.7,
                          "text_threshold": 0.3, "low_text": 0.2, "width_ths": 0.9}),
    ("paragraph_true",   {"paragraph": True}),
    ("decoder_beamsearch", {"decoder": "beamsearch"}),
    ("no_allowlist",     {"allowlist": None}),
]

print("\n── Resultados ────────────────────────────────────────────────────────")
for name, extra in configs:
    kwargs = {
        "allowlist": 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        "detail":    1,
        "paragraph": False,
        **extra,
    }
    if extra.get("allowlist") is None:
        kwargs.pop("allowlist")
    if extra.get("paragraph") is True:
        kwargs["paragraph"] = True

    raw     = reader.readtext(processed, **kwargs)
    results = [(t, round(float(c), 3)) for _, t, c in raw]
    print(f"  {name:22s}: {results}")
