_base_ = ["/root/mmpretrain/configs/convnext/convnext-tiny_32xb128_in1k.py"]

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

project_root = "/root/ECE1_0423"
data_root = project_root + "/UCMerced_LandUse/uc_merced_dataset"
work_dir = project_root + "/exercise1/work_dirs/convnext_tiny_ucmerced"

pretrained_checkpoint = (
    "https://download.openmmlab.com/mmclassification/v0/convnext/"
    "convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth"
)

data_preprocessor = dict(num_classes=len(classes))
# data_preprocessor是一个字典，包含了数据预处理的相关配置。
# 在这个配置中，我们指定了num_classes参数，表示分类任务中的类别数量。
# 这个参数对于模型的输出层非常重要，因为它决定了模型需要输出多少个类别的概率分布。
# 在这个例子中，我们将num_classes设置为len(classes)，即我们定义的classes元组中的类别数量。

model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint=pretrained_checkpoint,
            prefix="backbone",
        )
    ),
    head=dict(
        num_classes=len(classes),
        init_cfg=None,
    ),
)  # 这里我们定义了模型的配置，特别是骨干网络（backbone）和头部（head）的初始化配置。
# 在骨干网络的配置中，我们使用了init_cfg来指定预训练权重的加载方式。我们指定了type为'Pretrained'，表示我们要加载预训练模型的权重。
# checkpoint参数指定了预训练模型的URL地址，prefix参数指定了我们要加载权重的模块，这里是'backbone'，表示我们只加载骨干网络的权重。
# 在头部的配置中，我们设置了num_classes为len(classes)，表示头部需要输出的类别数量与我们定义的classes元组中的类别数量相同。
# 我们还将init_cfg设置为None，表示我们不使用预训练权重来初始化头部，这样头部的权重将会在训练过程中随机初始化。

train_pipeline = [
    dict(type="LoadImageFromFile"),  # 载入图像
    dict(
        type="RandomResizedCrop",
        scale=224,
        backend="pillow",  # 这里我们指定了图像处理的后端库为Pillow。Pillow是一个流行的Python图像处理库，提供了丰富的图像操作功能。
        interpolation="bicubic",
    ),  # 图像随机剪裁到224x224的大小，使用双三次插值进行缩放。
    dict(
        type="RandomFlip", prob=0.5, direction="horizontal"
    ),  # 图像以50%的概率进行水平翻转。
    dict(
        type="RandAugment",
        policies="timm_increasing",
        # RandAugment是一种数据增强方法，它通过随机选择和应用一系列预定义的图像变换来增强训练数据。
        # 这里我们指定了policies参数为'timm_increasing'，这是一套预定义的增强策略，包含了多种图像变换，如旋转、剪裁、颜色调整等。
        # 这些变换将被随机应用于训练图像，以增加数据的多样性，从而提高模型的泛化能力。
        num_policies=2,  # num_policies参数指定了每个图像上应用的随机增强策略的数量。
        # 在这个例子中，我们设置num_policies为2，表示每个图像将随机应用两种不同的增强策略。
        total_level=10,  # total_level参数指定了增强策略的总强度级别。增强策略的强度级别通常是一个整数，表示变换的程度或幅度。
        magnitude_level=9,  # magnitude_level参数指定了增强策略的强度级别。这个参数通常是一个整数，表示变换的程度或幅度。
        # 在这个例子中，我们设置magnitude_level为9，表示我们希望应用较强的增强变换。
        magnitude_std=0.5,
        hparams=dict(interpolation="bicubic", pad_val=[104, 116, 124]),
        # hparams参数是一个字典，用于指定增强策略的超参数。
        # 在这个例子中，我们指定了插值方法为双三次插值（bicubic），以及填充颜色的值为[104, 116, 124]，这通常是图像预处理中的常用填充颜色。
    ),
    dict(
        type="RandomErasing",  # 随机擦除
        erase_prob=0.25,
        mode="rand",
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395],
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),  # 载入图像
    dict(
        type="ResizeEdge",
        scale=256,
        edge="short",
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type="CenterCrop", crop_size=224),  # 中心裁剪图像到224x224的大小。
    dict(type="PackInputs"),  # 将处理后的图像和标签打包成模型输入所需的格式。
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,  # 数据加载时使用的工作线程数。增加num_workers可以加速数据加载过程，特别是在处理大型数据集时。
    dataset=dict(
        _delete_=True,
        type="CustomDataset",  # 我们指定了数据集的类型为CustomDataset，这意味着我们将使用自定义的数据集类来加载和处理数据。
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
    dataset=dict(
        _delete_=True,
        type="CustomDataset",
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
    dataset=dict(
        _delete_=True,
        type="CustomDataset",
        data_root=data_root,
        ann_file="val.txt",
        data_prefix="val",
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)


val_evaluator = dict(type="Accuracy", topk=(1,))
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    # 学习率（learning rate）是训练神经网络时的一个重要超参数，它控制着模型权重更新的步长。
    # 较高的学习率可能导致训练过程不稳定，甚至发散，而较低的学习率可能导致训练过程过慢，难以收敛。
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),  # LinearLR是一种学习率调度器，它通过线性插值的方式在训练过程中调整学习率。
    # 在这个例子中，我们设置了start_factor为1e-3，表示学习率将在训练开始时从初始学习率的0.001倍逐渐增加到初始学习率。
    # by_epoch=True表示学习率调度器将按照训练的 epoch进行调整，begin=0和end=5表示学习率将在前5个epoch内逐渐增加。
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
    ),  # CosineAnnealingLR是一种学习率调度器，它通过余弦函数的方式在训练过程中调整学习率。
    # 在这个例子中，我们设置了eta_min为1e-6，表示
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="accuracy/top1",
        rule="greater",
    ),
)

auto_scale_lr = dict(base_batch_size=16)

randomness = dict(seed=42, deterministic=False)
