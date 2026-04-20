import os

_mmpretrain_root = os.environ.get("MMPRETRAIN_ROOT", "/root/mmpretrain")
_base_ = [
    os.path.join(
        _mmpretrain_root,
        "configs",
        "convnext",
        "convnext-tiny_32xb128_in1k.py",
    ),
    "./_base_/uc_merced_common.py",
]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
work_dir = os.path.join(
    project_root,
    "exercise2",
    "results",
    "models",
    "convnext_tiny",
    "work_dir",
)

model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint=(
                "https://download.openmmlab.com/mmclassification/v0/convnext/"
                "convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth"
            ),
            prefix="backbone",
        ),
    ),
    head=dict(num_classes=21, init_cfg=None),
)
