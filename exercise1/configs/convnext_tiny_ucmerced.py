from pathlib import Path

_base_ = ['mmpretrain::convnext/convnext-tiny_32xb128_in1k.py']

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

project_root = Path(__file__).resolve().parents[2]
data_root = str(project_root / 'UCMerced_LandUse' / 'uc_merced_dataset')
work_dir = str(project_root / 'exercise1' / 'work_dirs' / 'convnext_tiny_ucmerced')

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

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='train.txt',
        data_prefix='train',
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',
        data_prefix='val',
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader

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
        end=30,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)

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
