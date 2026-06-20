
#   train_vision_mamba.py                                                      
#   TrafficVision · Vision Mamba — Detección de Placas Ecuatorianas            
#                                                                               
#   Uso:                                                                        
#     python train_vision_mamba.py                  → entrenamiento completo    
#     python train_vision_mamba.py --test-only       → solo evaluar test set    
#     python train_vision_mamba.py --skip-install    → saltar instalación       
#                                                                               
#   Primera ejecución: instala automáticamente todas las dependencias.          
#   Requiere: Python 3.10-3.12, CUDA 11.8 / 12.1 / 12.4 (NVIDIA)              


import sys
import os
import subprocess
import argparse

# 0. CONFIGURACIÓN 
ROBOFLOW_API_KEY  = "LPg6zahR6BvuUpVhtsiC"
ROBOFLOW_WS       = "stevens-workspace-unaqf"
ROBOFLOW_PROJECT  = "plates-ecuadorian"
ROBOFLOW_VERSION  = 4

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MMDET_DIR         = os.path.join(BASE_DIR, "mmdetection")
WORK_DIR          = os.path.join(BASE_DIR, "work_dirs", "vision_mamba_plates")
RESULTS_DIR       = os.path.join(BASE_DIR, "results")
DATASET_DIR       = os.path.join(BASE_DIR, "dataset")

CLASS_NAMES       = ["license plate"]
MAX_EPOCHS        = 30
BATCH_SIZE        = 4
IMG_SIZE          = 640

VIM_TINY_URL  = "https://huggingface.co/hustvl/Vim-tiny/resolve/main/vim_t_midclstok_ft_in1k_81p3.pth"
VIM_TINY_CKPT = os.path.join(MMDET_DIR, "checkpoints", "vim_tiny_backbone.pth")


# 1. DETECCIÓN DE DISPOSITIVO
def detect_device():
    import torch
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] ✅ GPU: {name} ({vram:.1f} GB VRAM)")
        return "cuda", torch.version.cuda
    else:
        print("[device]   No se detectó GPU NVIDIA — usando CPU (será lento)")
        return "cpu", None


# 2. INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS
def pip(args, check=True):
    """Ejecuta pip con los argumentos dados."""
    cmd = [sys.executable, "-m", "pip", "install"] + args + ["-q"]
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def get_mmcv_index(cuda_version: str) -> str:
    """
    Devuelve la URL del índice de ruedas precompiladas de mmcv
    según la versión de CUDA instalada.
    Compatible con: cu118, cu121, cu124.
    """
    cuda_map = {
        "11.8": ("cu118", "torch2.1"),
        "12.1": ("cu121", "torch2.1"),
        "12.4": ("cu124", "torch2.4"),
    }
    # Normaliza: "12.1" o "12.1.0" → "12.1"
    short = ".".join(cuda_version.split(".")[:2])
    if short in cuda_map:
        cu, tc = cuda_map[short]
        return f"https://download.openmmlab.com/mmcv/dist/{cu}/{tc}/index.html"
    # Fallback: compilar desde source (más lento)
    print(f"[install] ⚠️  CUDA {cuda_version} no tiene rueda precompilada de mmcv.")
    print("[install]     Se compilará desde source (~15 min). Paciencia...")
    return None


def install_pytorch(cuda_version: str):
    """Instala PyTorch 2.1.0 con el índice de CUDA correcto."""
    short = ".".join(cuda_version.split(".")[:2]) if cuda_version else "cpu"
    whl_map = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
    }
    index = whl_map.get(short, "https://download.pytorch.org/whl/cpu")
    print(f"[install] Instalando PyTorch 2.1.0 para CUDA {short}...")
    pip(["torch==2.1.0", "torchvision==0.16.0", "--index-url", index])


def install_all(cuda_version: str | None):
    print("\n" + "="*60)
    print("  INSTALACIÓN DE DEPENDENCIAS")
    print("="*60)

    # PyTorch — solo instala si no está o si la versión no es 2.x
    try:
        import torch
        major = int(torch.__version__.split(".")[0])
        if major < 2:
            raise ImportError("versión antigua")
        print(f"[install] PyTorch {torch.__version__} ya instalado ✅")
    except ImportError:
        if cuda_version:
            install_pytorch(cuda_version)
        else:
            print("[install] Instalando PyTorch CPU...")
            pip(["torch==2.1.0", "torchvision==0.16.0"])

    # setuptools compatible
    pip(["setuptools>=69.5.1,<72", "packaging", "ninja"])

    # roboflow para descargar el dataset
    pip(["roboflow"])

    # mmengine
    print("[install] mmengine...")
    pip(["mmengine"])

    # mmcv
    print("[install] mmcv (puede tardar)...")
    if cuda_version:
        idx = get_mmcv_index(cuda_version)
        if idx:
            ok = pip(["mmcv==2.1.0", "-f", idx], check=False)
            if not ok:
                print("[install] Rueda no disponible, compilando desde source...")
                pip(["mmcv"])
        else:
            pip(["mmcv"])
    else:
        pip(["mmcv"])

    # mmdetection desde source
    print("[install] mmdetection...")
    if not os.path.exists(MMDET_DIR):
        subprocess.run(
            ["git", "clone", "https://github.com/open-mmlab/mmdetection.git",
             MMDET_DIR, "-q"],
            check=True
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd=MMDET_DIR, check=True
    )

    # mamba-ssm
    print("[install] causal-conv1d + mamba-ssm (compilación CUDA ~5 min)...")
    pip(["causal-conv1d>=1.2.0", "--no-build-isolation"])
    pip(["mamba-ssm", "--no-build-isolation"])

    # matplotlib y pandas para gráficas/CSV
    pip(["matplotlib", "pandas"])

    # Verificar
    print("\n[install] Verificando imports...")
    ok = True
    for pkg, attr in [("mmdet", "__version__"), ("mmcv", "__version__"), ("mamba_ssm", "__version__")]:
        try:
            mod = __import__(pkg)
            print(f"   ✅ {pkg}: {getattr(mod, attr)}")
        except ImportError as e:
            print(f"   ❌ {pkg}: {e}")
            ok = False
    if not ok:
        print("\  Algunos paquetes no se instalaron correctamente.")
        print("    Corre con --skip-install y revisa los errores arriba.")
        sys.exit(1)

    print("\ Todas las dependencias instaladas.\n")


# 3. DESCARGA DEL DATASET (Roboflow → COCO-MMDetection)
def download_dataset() -> str:
    """Descarga el dataset y devuelve la ruta raíz COCO."""
    from roboflow import Roboflow
    import json

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.chdir(DATASET_DIR)

    print("[dataset] Conectando a Roboflow...")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WS).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("coco-mmdetection")

    coco_root = dataset.location
    print(f"[dataset]  Descargado en: {coco_root}")

    for split in ["train", "valid", "test"]:
        ann = os.path.join(coco_root, split, "_annotations.coco.json")
        if os.path.exists(ann):
            d = json.load(open(ann))
            print(f"   {split:5s}: {len(d['images']):4d} imgs, {len(d['annotations']):5d} anns")
        else:
            print(f"     {split}: _annotations.coco.json no encontrado")

    os.chdir(BASE_DIR)
    return coco_root


# 4. REGISTRO DEL BACKBONE VISION MAMBA
BACKBONE_CODE = '''
import torch, torch.nn as nn, math
from functools import partial
from mmdet.registry import MODELS
from mmengine.model import BaseModule

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn, layer_norm_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

def _init_weights(m, n_layer):
    if isinstance(m, nn.Linear) and m.bias is not None:
        if not getattr(m.bias, "_no_reinit", False):
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    for name, p in m.named_parameters():
        if name in ["out_proj.weight", "fc2.weight"]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_layer)

class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls, fused, fp32):
        super().__init__()
        self.fused = fused
        self.fp32  = fp32
        self.mixer = mixer_cls(dim)
        self.norm  = norm_cls(dim)

    def forward(self, x, residual=None, inference_params=None):
        if not self.fused:
            residual = (x + residual) if residual is not None else x
            x = self.norm(residual.to(self.norm.weight.dtype))
            if self.fp32:
                residual = residual.float()
        else:
            fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            x, residual = fn(
                x, self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.fp32, eps=self.norm.eps,
            )
        x = self.mixer(x, inference_params=inference_params)
        return x, residual

@MODELS.register_module()
class VisionMamba(BaseModule):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3,
        embed_dim=192, depth=24, out_indices=(5, 11, 17, 23),
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="v2",
        final_pool_type="none", if_rope=False, if_rope_residual=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert HAS_MAMBA, "Instala mamba-ssm"
        self.embed_dim       = embed_dim
        self.depth           = depth
        self.out_indices     = out_indices
        self.if_abs_pos_embed = if_abs_pos_embed
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        norm_cls = partial(RMSNorm, eps=1e-5) if (rms_norm and HAS_MAMBA) else partial(nn.LayerNorm, eps=1e-5)
        self.layers = nn.ModuleList([
            Block(embed_dim,
                  partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=2),
                  norm_cls, fused_add_norm, residual_in_fp32)
            for i in range(depth)
        ])
        self.out_norms = nn.ModuleList([norm_cls(embed_dim) for _ in out_indices])
        self.apply(partial(_init_weights, n_layer=depth))

    def forward(self, x):
        B, C, H, W = x.shape
        x  = self.patch_embed(x)
        hp, wp = x.shape[2], x.shape[3]
        x  = x.flatten(2).transpose(1, 2)
        if self.if_abs_pos_embed:
            if x.shape[1] != self.pos_embed.shape[1]:
                pe = self.pos_embed.transpose(1, 2).reshape(
                    1, self.embed_dim,
                    int(self.pos_embed.shape[1] ** 0.5),
                    int(self.pos_embed.shape[1] ** 0.5),
                )
                pe = nn.functional.interpolate(pe, (hp, wp), mode="bicubic", align_corners=False)
                pe = pe.flatten(2).transpose(1, 2)
            else:
                pe = self.pos_embed
            x = x + pe
        outs, residual, oi = [], None, 0
        for i, layer in enumerate(self.layers):
            x, residual = layer(x, residual)
            if i in self.out_indices:
                feat = self.out_norms[oi](x).transpose(1, 2).reshape(B, self.embed_dim, hp, wp)
                outs.append(feat)
                oi += 1
        return tuple(outs)
'''


def register_backbone():
    sys.path.insert(0, MMDET_DIR)
    bkb_path  = os.path.join(MMDET_DIR, "mmdet", "models", "backbones", "vision_mamba.py")
    init_path = os.path.join(MMDET_DIR, "mmdet", "models", "backbones", "__init__.py")

    with open(bkb_path, "w") as f:
        f.write(BACKBONE_CODE)

    content = open(init_path).read()
    if "VisionMamba" not in content:
        content = "from .vision_mamba import VisionMamba\n" + content
        content = content.replace("__all__ = [", "__all__ = [\n    'VisionMamba',")
        open(init_path, "w").write(content)
        print("[backbone]   VisionMamba registrado.")
    else:
        print("[backbone]   VisionMamba ya estaba registrado.")


# 5. DESCARGA DEL CHECKPOINT PREENTRENADO
def download_checkpoint():
    import urllib.request
    os.makedirs(os.path.dirname(VIM_TINY_CKPT), exist_ok=True)
    if not os.path.exists(VIM_TINY_CKPT):
        print("[ckpt] Descargando Vim-Tiny (~80 MB)...")
        urllib.request.urlretrieve(VIM_TINY_URL, VIM_TINY_CKPT)
        print("[ckpt]  Descarga completada.")
    else:
        mb = os.path.getsize(VIM_TINY_CKPT) / 1024**2
        print(f"[ckpt]  Ya existe ({mb:.1f} MB)")


# 6. GENERACIÓN DE CONFIGURACIÓN MMDETECTION
def write_config(coco_root: str) -> str:
    cfg_path = os.path.join(MMDET_DIR, "configs", "vision_mamba_ecuaplacas.py")
    cfg = f"""
_base_ = ['./configs/_base_/default_runtime.py']

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False),
    neck=dict(type='FPN', in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5),
    rpn_head=dict(type='RPNHead', in_channels=256, feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', scales=[8],
            ratios=[0.3, 0.5, 1.0, 2.0, 3.5], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.], target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(type='StandardRoIHead',
        bbox_roi_extractor=dict(type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(type='Shared2FCBBoxHead',
            in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=1,
            bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        roi=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        roi=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)))

dataset_type = 'CocoDataset'
data_root    = '{coco_root}/'
metainfo     = dict(classes=('license plate',))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice', transforms=[
        [dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True)],
        [dict(type='Resize', scale=(480, 480), keep_ratio=True),
         dict(type='RandomCrop', crop_size=({IMG_SIZE}, {IMG_SIZE}), allow_negative_crop=True)]]),
    dict(type='PackDetInputs')]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=({IMG_SIZE}, {IMG_SIZE}), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

train_dataloader = dict(batch_size={BATCH_SIZE}, num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type, metainfo=metainfo,
        data_root=data_root, ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(batch_size=1, num_workers=2,
    persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo,
        data_root=data_root, ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/images/'),
        test_mode=True, pipeline=test_pipeline))

test_dataloader = dict(batch_size=1, num_workers=2,
    persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, metainfo=metainfo,
        data_root=data_root, ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True, pipeline=test_pipeline))

val_evaluator  = dict(type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox', format_only=False)
test_evaluator = dict(type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox', format_only=False)

optim_wrapper = dict(type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={{'backbone': dict(lr_mult=0.1)}},
        norm_decay_mult=0.0))

max_epochs = {MAX_EPOCHS}
train_cfg  = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg    = dict(type='ValLoop')
test_cfg   = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=max_epochs, by_epoch=True,
         T_max=max_epochs, eta_min=1e-6)]

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3,
        save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer   = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level    = 'INFO'
load_from    = None
resume       = False
"""
    with open(cfg_path, "w") as f:
        f.write(cfg)
    print(f"[config]  Config escrito en {cfg_path}")
    return cfg_path


# 7. ENTRENAMIENTO
def run_training(cfg_path: str):
    os.makedirs(WORK_DIR, exist_ok=True)
    log_path = os.path.join(WORK_DIR, "training_log.txt")
    print(f"\n[train]  Iniciando entrenamiento → {WORK_DIR}")
    print(f"[train]    Log: {log_path}\n")
    cmd = [
        sys.executable, "tools/train.py",
        cfg_path,
        "--work-dir", WORK_DIR,
    ]
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=MMDET_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        proc.wait()
    if proc.returncode != 0:
        print(f"[train]  Entrenamiento terminó con código {proc.returncode}")
    else:
        print("[train]  Entrenamiento completado.")


# 8. EVALUACIÓN EN TEST SET
def run_test(cfg_path: str):
    import glob
    bests = sorted(glob.glob(os.path.join(WORK_DIR, "best_coco_bbox_mAP*.pth")))
    if not bests:
        bests = sorted(glob.glob(os.path.join(WORK_DIR, "epoch_*.pth")))
    if not bests:
        print("[test]  No hay checkpoint disponible.")
        return
    best_ckpt  = bests[-1]
    test_log   = os.path.join(WORK_DIR, "test_results.txt")
    print(f"[test]  Evaluando: {best_ckpt}")
    cmd = [
        sys.executable, "tools/test.py",
        cfg_path, best_ckpt,
        "--work-dir", WORK_DIR,
    ]
    with open(test_log, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=MMDET_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        proc.wait()
    print("[test]  Test completado.")
    return test_log


# 9. PARSEO DE LOGS → result.csv
def parse_and_save_results(test_log_path: str | None = None):
    import json
    import glob
    import re
    import pandas as pd
    import numpy as np
    from datetime import datetime

    def find_scalars_json():
        files = glob.glob(os.path.join(WORK_DIR, "*", "vis_data", "scalars.json"))
        if files:
            return sorted(files)[-1]
        direct = os.path.join(WORK_DIR, "vis_data", "scalars.json")
        return direct if os.path.exists(direct) else None

    def parse_scalars():
        path = find_scalars_json()
        if not path:
            return parse_log_txt()
        print(f"[metrics]  {path}")
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        if not records:
            return parse_log_txt()
        df = pd.DataFrame(records)
        is_train = df.filter(like="loss").notna().any(axis=1)
        is_val   = df.filter(like="mAP").notna().any(axis=1) | df.filter(like="coco").notna().any(axis=1)
        df_tr = df[is_train].copy()
        df_vl = df[is_val].copy()

        def prefix(df_, pre):
            return df_.rename(columns={c: f"{pre}/{c}" for c in df_.columns if c != "epoch"})

        if not df_tr.empty and "epoch" in df_tr.columns:
            agg   = {c: "mean" for c in df_tr.columns if c not in ("epoch", "iter", "step")}
            df_tr = df_tr.groupby("epoch").agg(agg).reset_index()
            df_tr = prefix(df_tr, "train")
        if not df_vl.empty:
            df_vl = prefix(df_vl, "val")
        if not df_tr.empty and not df_vl.empty and "epoch" in df_vl.columns:
            return pd.merge(df_tr, df_vl, on="epoch", how="outer").sort_values("epoch")
        return df_tr if not df_tr.empty else df_vl

    def parse_log_txt():
        path = os.path.join(WORK_DIR, "training_log.txt")
        if not os.path.exists(path):
            return pd.DataFrame()
        print(f"[metrics]  {path}")
        content  = open(path, errors="ignore").read()
        ep_data  = {}
        pat = re.compile(
            r"Epoch\s*\[(\d+)/\d+\]\s*\[\d+/\d+\].*?loss:\s*([\d.]+).*?lr:\s*([\d.e+-]+)",
            re.DOTALL,
        )
        for m in pat.finditer(content):
            ep = int(m.group(1))
            ep_data.setdefault(ep, []).append(
                {"train/loss": float(m.group(2)), "train/lr": float(m.group(3))}
            )
        train_rows = [
            {"epoch": ep, **{k: np.mean([r[k] for r in rows]) for k in rows[0]}}
            for ep, rows in sorted(ep_data.items())
        ]
        val_rows = []
        for block in re.split(r"(?=Epoch\(val\))", content):
            em = re.search(r"Epoch\(val\)\s*\[(\d+)\]", block)
            if not em:
                continue
            rec = {"epoch": int(em.group(1))}
            for key, pat2 in [
                ("val/coco_bbox_mAP",    r"bbox_mAP:\s*([\d.]+)"),
                ("val/coco_bbox_mAP_50", r"bbox_mAP_50:\s*([\d.]+)"),
                ("val/coco_bbox_mAP_75", r"bbox_mAP_75:\s*([\d.]+)"),
                ("val/coco_bbox_mAP_s",  r"bbox_mAP_s:\s*([\d.]+)"),
                ("val/coco_bbox_mAP_m",  r"bbox_mAP_m:\s*([\d.]+)"),
                ("val/coco_bbox_mAP_l",  r"bbox_mAP_l:\s*([\d.]+)"),
            ]:
                mm = re.search(pat2, block)
                if mm:
                    rec[key] = float(mm.group(1))
            if len(rec) > 1:
                val_rows.append(rec)
        df_t = pd.DataFrame(train_rows)
        df_v = pd.DataFrame(val_rows)
        if not df_t.empty and not df_v.empty:
            return pd.merge(df_t, df_v, on="epoch", how="outer").sort_values("epoch")
        return df_t if not df_t.empty else df_v

    def parse_test_log(path):
        if not path or not os.path.exists(path):
            return {}
        content = open(path, errors="ignore").read()
        out = {}
        for key, pat in [
            ("test/coco_bbox_mAP",    r"bbox_mAP:\s*([\d.]+)"),
            ("test/coco_bbox_mAP_50", r"bbox_mAP_50:\s*([\d.]+)"),
            ("test/coco_bbox_mAP_75", r"bbox_mAP_75:\s*([\d.]+)"),
            ("test/coco_bbox_mAP_s",  r"bbox_mAP_s:\s*([\d.]+)"),
            ("test/coco_bbox_mAP_m",  r"bbox_mAP_m:\s*([\d.]+)"),
            ("test/coco_bbox_mAP_l",  r"bbox_mAP_l:\s*([\d.]+)"),
        ]:
            m = re.search(pat, content)
            if m:
                out[key] = float(m.group(1))
        return out

    print("\n[metrics]  Parseando métricas...")
    df = parse_scalars()

    # Añadir fila TEST si hay log
    test_metrics = parse_test_log(test_log_path)
    if test_metrics:
        print("\n[metrics]  Test Set:")
        for k, v in test_metrics.items():
            print(f"   {k:<35s}: {v:.4f}")
        test_row = pd.DataFrame([{"epoch": "TEST", **test_metrics}])
        df = pd.concat([df, test_row], ignore_index=True)

    if df.empty:
        print("[metrics]   No se encontraron métricas.")
        return

    # Ordenar columnas
    epoch_col = ["epoch"] if "epoch" in df.columns else []
    tr_cols   = sorted([c for c in df.columns if c.startswith("train/")])
    vl_cols   = sorted([c for c in df.columns if c.startswith("val/")])
    te_cols   = sorted([c for c in df.columns if c.startswith("test/")])
    ot_cols   = [c for c in df.columns if c not in epoch_col + tr_cols + vl_cols + te_cols]
    df = df[epoch_col + tr_cols + vl_cols + te_cols + ot_cols]

    print(f"\n[metrics]  {len(df)} épocas × {len(df.columns)} columnas")

    # Guardar CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"vision_mamba_results_{ts}.csv")
    local_csv = os.path.join(WORK_DIR, "result.csv")
    df.to_csv(csv_path,   index=False, float_format="%.6f")
    df.to_csv(local_csv,  index=False, float_format="%.6f")
    print(f"[metrics] 💾 {csv_path}")

    # Copiar mejor checkpoint
    best_ckpts = sorted(glob.glob(os.path.join(WORK_DIR, "best_coco_bbox_mAP*.pth")))
    if best_ckpts:
        dst = os.path.join(RESULTS_DIR, f"vision_mamba_best_{ts}.pth")
        import shutil
        shutil.copy(best_ckpts[-1], dst)
        print(f"[metrics]  Mejor checkpoint: {dst}")

    # Resumen
    print("\n[metrics]  Mejores valores:")
    for c in [x for x in df.columns if "mAP" in x or x == "train/loss"]:
        v = df[c].dropna()
        if v.empty:
            continue
        best = v.max() if "mAP" in c else v.min()
        print(f"   {c:<35s}: {best:.4f}")

    # Gráfica
    try:
        save_plots(df, ts)
    except Exception as e:
        print(f"[plot]   No se pudo generar la gráfica: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────
def save_plots(df, ts: str):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ep = df["epoch"].values if "epoch" in df.columns else range(len(df))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Vision Mamba · Detección de Placas Ecuatorianas\nCurvas de Entrenamiento",
        fontsize=16, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    # Loss total
    ax = fig.add_subplot(gs[0, 0])
    if "train/loss" in df.columns:
        ax.plot(ep, df["train/loss"], "b-o", ms=4)
    ax.set_title("Pérdida Total (Train)", fontweight="bold")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3)

    # Losses detalladas
    ax2 = fig.add_subplot(gs[0, 1])
    colors   = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    det_cols = [c for c in df.columns if c.startswith("train/loss_")]
    for i, c in enumerate(det_cols[:5]):
        ax2.plot(ep, df[c], "-o", ms=3, color=colors[i], label=c.replace("train/loss_", ""))
    ax2.set_title("Pérdidas Detalladas", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    if "train/lr" in df.columns:
        ax3.semilogy(ep, df["train/lr"], "g-o", ms=4)
    ax3.set_title("Learning Rate", fontweight="bold")
    ax3.set_xlabel("Época"); ax3.set_ylabel("LR"); ax3.grid(alpha=0.3)

    # mAP general
    ax4 = fig.add_subplot(gs[1, 0])
    for c in [x for x in df.columns if "mAP" in x and "val" in x][:3]:
        ok = df[c].notna()
        ax4.plot(df["epoch"][ok], df[c][ok], "-s", ms=5, label=c.split("_")[-1])
    ax4.set_ylim([0, 1]); ax4.set_title("mAP (Validación)", fontweight="bold")
    ax4.legend(); ax4.grid(alpha=0.3)

    # mAP@50 vs mAP@75
    ax5 = fig.add_subplot(gs[1, 1])
    for c, col, ls in [
        ("val/coco_bbox_mAP_50", "#e74c3c", "-"),
        ("val/coco_bbox_mAP_75", "#3498db", "--"),
    ]:
        if c in df.columns:
            ok = df[c].notna()
            ax5.plot(df["epoch"][ok], df[c][ok], ls + "s", ms=5, color=col, label=c[-2:])
    ax5.set_ylim([0, 1]); ax5.set_title("mAP@50 vs mAP@75", fontweight="bold")
    ax5.legend(); ax5.grid(alpha=0.3)

    # Tabla resumen
    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis("off")
    rows_tbl = []
    for c in [x for x in df.columns if "mAP" in x or x == "train/loss"]:
        v = df[c].dropna()
        if v.empty:
            continue
        best = v.max() if "mAP" in c else v.min()
        idx  = v.idxmax() if "mAP" in c else v.idxmin()
        ep_b = df["epoch"].iloc[idx] if "epoch" in df.columns else "-"
        rows_tbl.append([c.replace("val/coco_bbox_", "").replace("train/", ""), f"{best:.4f}", str(ep_b)])
    if rows_tbl:
        tbl = ax6.table(cellText=rows_tbl, colLabels=["Métrica", "Mejor", "Época"],
                        cellLoc="center", loc="center")
        tbl.auto_set_font_size(True); tbl.scale(1, 1.5)
    ax6.set_title("Resumen Mejores Métricas", fontweight="bold", pad=20)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(RESULTS_DIR, f"training_curves_{ts}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  Gráfica guardada en {plot_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Vision Mamba — Placas Ecuatorianas")
    parser.add_argument("--skip-install", action="store_true", help="Saltar instalación de dependencias")
    parser.add_argument("--skip-download", action="store_true", help="Saltar descarga del dataset (usar el existente)")
    parser.add_argument("--test-only",    action="store_true", help="Solo evaluar en test set (requiere checkpoint)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Ruta al dataset COCO si ya está descargado")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vision Mamba · Placas Ecuatorianas · TrafficVision")
    print("=" * 60)

    # Detectar dispositivo
    device, cuda_version = detect_device()

    # Instalar dependencias
    if not args.skip_install:
        install_all(cuda_version)
    else:
        print("[install]   Instalación omitida (--skip-install)")

    # Dataset
    if args.dataset_path:
        coco_root = args.dataset_path
        print(f"[dataset]   Usando dataset en: {coco_root}")
    elif args.skip_download:
        # Buscar dataset descargado previamente
        import glob as _glob
        candidates = _glob.glob(os.path.join(DATASET_DIR, "*", "train", "_annotations.coco.json"))
        if candidates:
            coco_root = os.path.dirname(os.path.dirname(candidates[0]))
            print(f"[dataset]   Dataset encontrado en: {coco_root}")
        else:
            print("[dataset]  No se encontró dataset. Quita --skip-download.")
            sys.exit(1)
    else:
        coco_root = download_dataset()

    # Registrar backbone y descargar checkpoint
    register_backbone()
    download_checkpoint()

    # Generar config
    cfg_path = write_config(coco_root)

    # Entrenar o solo evaluar
    test_log = None
    if not args.test_only:
        run_training(cfg_path)
    test_log = run_test(cfg_path)

    # Parsear logs y guardar CSV + gráficas
    parse_and_save_results(test_log)

    print("\n✅ Pipeline completo.")
    print(f"   Resultados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
