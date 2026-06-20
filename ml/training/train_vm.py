# ==============================================================================
# 1. INSTALACIÓN DE DEPENDENCIAS (ENTORNO MMCONSTRUCT)
# ==============================================================================
# MMDetection requiere versiones específicas de CUDA y PyTorch para compilar extensiones de C++
!pip install -U openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"

# Clonar el repositorio de MMDetection para tener acceso a los frameworks de detección
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -v -e .

# Instalar dependencias requeridas por Vision Mamba (Mamba-ssm y causal-conv1d)
!pip install causal-conv1d>=1.1.0
!pip install mamba-ssm

# ==============================================================================
# 2. DESCARGA DEL BACKBONE PREENTRENADO (VISION MAMBA)
# ==============================================================================
import os
import urllib.request

# Creamos un directorio para los checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Descargar los pesos oficiales de Vim-Tiny preentrenado en ImageNet-1k desde HuggingFace/GitHub
vim_tiny_url = "https://huggingface.co/hustvl/Vim-tiny/resolve/main/vim_t_midclstok_ft_in1k_81p3.pth"
checkpoint_path = "checkpoints/vim_tiny_backbone.pth"

if not os.path.exists(checkpoint_path):
    print("Descargando pesos preentrenados de Vision Mamba (Vim-Tiny)...")
    urllib.request.urlretrieve(vim_tiny_url, checkpoint_path)
    print("Descarga completada.")

# ==============================================================================
# 3. CONFIGURACIÓN DEL PIPELINE Y ARQUITECTURA (CONFIG FILE)
# ==============================================================================
# Nota: Configura tu dataset en formato COCO JSON (Roboflow te lo da listo)
# Sube tu dataset a Colab de forma que la estructura sea:
# /content/dataset/train/_annotations.coco.json, /content/dataset/train/images...

config_text = """
# Configuración base usando una cabeza Mask R-CNN o Cascade R-CNN adaptada
_base_ = ['./configs/_base_/default_runtime.py']

# Definición del Modelo fusionando Vision Mamba con una Cabeza de Detección
model = dict(
    type='FasterRCNN', # Usamos la estructura Faster R-CNN como meta-detector
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='VisionMamba', # Registrado en el ecosistema mmdet
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='none',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type='v2',
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/vim_tiny_backbone.pth')),
    neck=dict(
        type='FPN', # Feature Pyramid Network para manejar escalas de placas
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            feat_channels=256),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1, # IMPORTANTE: 1 sola clase (Placa Ecuatoriana)
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.0))),
    # Parámetros de entrenamiento y prueba (proposiciones de bounding boxes)
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False), allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        roi=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True), pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        roi=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)) # NMS final para limpiar detecciones de placas
)

# Configuración de Datasets (Rutas de carpetas en Colab)
dataset_type = 'CocoDataset'
data_root = '/content/dataset/' # Cambia esto a la ruta de tu carpeta descomprimida

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4, # Ajusta según la memoria de la GPU asignada en Colab
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json', # Nombre de tu JSON de entrenamiento
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json', # Nombre de tu JSON de validación
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluadores para obtener el mAP (Mean Average Precision) requerido en la tesis
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'valid/_annotations.coco.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator

# Estrategia de Optimización (AdamW suele ser el mejor para modelos tipo Transformer/Mamba)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})) # Menor Learning Rate en el backbone para conservar lo preentrenado

# Calendario de Entrenamiento (Epochs)
max_epochs = 30
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Configuración de políticas de Learning Rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[24, 28], gamma=0.1)
]

# Configuración de Checkpoints e Historiales
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3))
"""

with open('configs/vision_mamba_ecuaplacas.py', 'w') as f:
    f.write(config_text)
print("Archivo de configuración generado en configs/vision_mamba_ecuaplacas.py")

# ==============================================================================
# 4. LANZAMIENTO DEL ENTRENAMIENTO
# ==============================================================================
# Ejecutamos el script de entrenamiento nativo de MMDetection con nuestra configuración
!python tools/train.py configs/vision_mamba_ecuaplacas.py --work-dir ./work_dirs/vision_mamba_plates