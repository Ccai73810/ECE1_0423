
_base_ = ['/root/mmpretrain/configs/convnext/convnext-tiny_32xb128_in1k.py']

classes = (
    'agricultural',
    'airplane',
    'baseballdiamond',
    'beach',
    'buildings',
    'chaparral',
    'denseresidential',
    'forest',
    'freeway',
    'golfcourse',
    'harbor',
    'intersection',
    'mediumresidential',
    'mobilehomepark',
    'overpass',
    'parkinglot',
    'river',
    'runway',
    'sparseresidential',
    'storagetanks',
    'tenniscourt',
)

project_root = '/root/ECE1_0423'
data_root = project_root + '/UCMerced_LandUse/uc_merced_dataset'
work_dir = project_root + '/exercise1/work_dirs/convnext_tiny_ucmerced'

pretrained_checkpoint = (
    'https://download.openmmlab.com/mmclassification/v0/convnext/'
    'convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth'
)

data_preprocessor = dict(num_classes=len(classes))

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint,
            prefix='backbone',
        )),
    head=dict(
        num_classes=len(classes),
        init_cfg=None,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(interpolation='bicubic', pad_val=[104, 116, 124])),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CustomDataset',
        data_root=data_root,
        ann_file='train.txt',
        data_prefix='train',
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',
        data_prefix='val',
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',
        data_prefix='val',
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)



val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='accuracy/top1',
        rule='greater',
    ),
)

auto_scale_lr = dict(base_batch_size=16)

randomness = dict(seed=42, deterministic=False)
