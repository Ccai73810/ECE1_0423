import os

_mmpretrain_root = os.environ.get("MMPRETRAIN_ROOT", "/root/mmpretrain")
_base_ = [
    os.path.join(
        _mmpretrain_root,
        "configs",
        "mobilenet_v3",
        "mobilenet-v3-large_8xb128_in1k.py",
    ),
    "./_base_/uc_merced_common.py",
]

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
work_dir = os.path.join(
    project_root,
    "exercise2",
    "results",
    "models",
    "mobilenet_v3_large",
    "work_dir",
)

model = dict(
    head=dict(num_classes=21, init_cfg=None),
)
