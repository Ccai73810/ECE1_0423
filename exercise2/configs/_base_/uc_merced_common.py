import os

classes = (
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
data_root = os.environ.get(
    "ECE1_DATA_ROOT",
    os.path.join(project_root, "UCMerced_LandUse", "uc_merced_dataset"),
)

dataset_type = "CustomDataset"
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=len(classes),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="RandomResizedCrop",
        scale=224,
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="RandAugment",
        policies="timm_increasing",
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(interpolation="bicubic", pad_val=[104, 116, 124]),
    ),
    dict(
        type="RandomErasing",
        erase_prob=0.25,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std,
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="ResizeEdge",
        scale=256,
        edge="short",
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="train.txt",
        data_prefix="train",
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="val.txt",
        data_prefix="val",
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="val.txt",
        data_prefix="val",
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)

model = dict(
    head=dict(
        num_classes=len(classes),
        init_cfg=None,
        loss=dict(type="LabelSmoothLoss", label_smooth_val=0.1, mode="original"),
    ),
    train_cfg=dict(
        augments=[
            dict(type="Mixup", alpha=0.8),
            dict(type="CutMix", alpha=1.0),
        ]
    ),
)

val_evaluator = dict(type="Accuracy", topk=(1,))
test_evaluator = val_evaluator

optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            ".absolute_pos_embed": dict(decay_mult=0.0),
            ".relative_position_bias_table": dict(decay_mult=0.0),
        },
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="accuracy/top1",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)

custom_hooks = [dict(type="EMAHook", momentum=0.0001, priority="ABOVE_NORMAL")]

auto_scale_lr = dict(base_batch_size=16)
randomness = dict(seed=42, deterministic=False)
default_scope = "mmpretrain"
launcher = "none"
log_level = "INFO"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
resume = False
load_from = None
