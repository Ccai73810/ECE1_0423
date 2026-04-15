# ECE1_0423 远程训练说明（中文版）

本仓库用于完成 ECE371 Assignment 1 的 Exercise 1，模型选择为 `convnext-tiny_32xb128_in1k`，训练框架为 **MMPretrain 1.x**。

仓库中已经提供：

- `exercise1/configs/convnext_tiny_ucmerced.py`
- `exercise1/scripts/check_dataset.py`
- 训练前需要的配置与数据检查脚本

仓库中**不包含数据集本体**。由于 `UCMerced_LandUse` 数据集体积较大，GitHub 仓库只上传了代码，训练前需要你把数据集单独传到远程平台。

## 1. 适用场景

本说明适用于以下场景：

- AutoDL / Linux GPU 云服务器
- 学校或实验室远程算力平台
- 自己租用的 Ubuntu GPU 服务器

默认假设：

- 远程系统为 Linux
- 你有一个可用的 GPU
- 你可以使用 `bash`
- 你可以使用 `conda`

## 2. 最终目录结构

远程平台上的目录建议严格保持为下面这样：

```text
ECE1_0423/
├── README.md
├── exercise1/
│   ├── configs/
│   │   └── convnext_tiny_ucmerced.py
│   └── scripts/
│       └── check_dataset.py
└── UCMerced_LandUse/
    └── uc_merced_dataset/
        ├── classes.txt
        ├── train.txt
        ├── val.txt
        ├── train/
        └── val/
```

请注意：

- 配置文件会自动把仓库根目录识别为项目根目录。
- 因此数据集必须放在 `ECE1_0423/UCMerced_LandUse/uc_merced_dataset` 下。
- 如果你把数据放到别的位置，训练前就需要手动修改配置文件。

## 3. 第一步：在远程平台克隆仓库

登录远程平台后，执行：

```bash
cd ~
git clone https://github.com/Ccai73810/ECE1_0423.git
cd ECE1_0423
```

如果你后续更新了本仓库代码，可以执行：

```bash
cd ~/ECE1_0423
git pull
```

## 4. 第二步：把数据集上传到远程平台

由于仓库里没有数据集，你必须单独上传 `UCMerced_LandUse` 文件夹。

### 方法 A：从你本地电脑上传到服务器

在你自己的电脑上执行：

```bash
scp -r UCMerced_LandUse <你的用户名>@<你的服务器地址>:~/ECE1_0423/
```

例如：

```bash
scp -r UCMerced_LandUse user@123.123.123.123:~/ECE1_0423/
```

如果你希望断点续传或更稳一点，可以用：

```bash
rsync -avh UCMerced_LandUse <你的用户名>@<你的服务器地址>:~/ECE1_0423/
```

### 方法 B：如果平台支持网页上传

直接把本地的 `UCMerced_LandUse` 文件夹上传到远程平台的：

```text
~/ECE1_0423/
```

上传完成后，确保路径是：

```text
~/ECE1_0423/UCMerced_LandUse/uc_merced_dataset
```

### 方法 C：如果你已经把数据打包成 zip

先把压缩包上传到服务器，然后在服务器上解压：

```bash
cd ~/ECE1_0423
unzip UCMerced_LandUse.zip
```

如果你上传的是 `tar.gz`：

```bash
cd ~/ECE1_0423
tar -xzf UCMerced_LandUse.tar.gz
```

## 5. 第三步：创建训练环境

根据 MMPretrain 官方文档，推荐使用 conda 环境，并从源码安装 MMPretrain。

### 5.1 创建 conda 环境

```bash
conda create -n openmmlab python=3.10 -y
conda activate openmmlab
```

### 5.2 安装 PyTorch

先查看 GPU 是否正常：

```bash
nvidia-smi
```

如果你的平台已经自带 PyTorch，可以跳过这一节。  
如果没有，则先安装 PyTorch。一个通用写法是：

```bash
conda install pytorch torchvision -c pytorch -y
```

说明：

- 这是 MMPretrain 官方文档中的通用安装方式。
- 远程平台如果对 CUDA 版本有固定要求，请以平台文档为准。
- 如果平台已经提供预装 CUDA/PyTorch 环境，优先使用平台建议方案。

### 5.3 克隆并安装 MMPretrain

```bash
cd ~
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim
mim install -e .
```

## 6. 第四步：验证仓库和数据是否就位

回到你的作业仓库目录，先检查数据集：

```bash
cd ~/ECE1_0423
python exercise1/scripts/check_dataset.py
```

正常情况下，你会看到类似输出：

```text
Dataset summary for report
- Data root: ...
- Number of classes: 21
- Training images: 1680 (80 per class)
- Validation images: 420 (20 per class)
- Split ratio: 8:2
- Metadata files: train.txt and val.txt verified
```

如果你的数据不在默认位置，也可以显式指定路径：

```bash
python exercise1/scripts/check_dataset.py --data-root ~/ECE1_0423/UCMerced_LandUse/uc_merced_dataset
```

## 7. 第五步：检查配置文件能否被 MMPretrain 正确解析

进入 MMPretrain 仓库目录：

```bash
cd ~/mmpretrain
python tools/misc/print_config.py ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py
```

如果想把展开后的完整配置保存到文件：

```bash
python tools/misc/print_config.py ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py > ~/ECE1_0423/final_config.py
```

这一步通过，说明：

- `_base_` 继承正确
- MMPretrain 能识别 `convnext-tiny_32xb128_in1k`
- 你的自定义数据集配置没有明显语法问题

## 8. 第六步：开始训练

### 8.1 单卡训练命令

```bash
cd ~/mmpretrain
CUDA_VISIBLE_DEVICES=0 python tools/train.py ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py --amp
```

说明：

- `CUDA_VISIBLE_DEVICES=0` 表示使用第 0 张卡
- `--amp` 表示开启混合精度训练，通常更省显存

### 8.2 指定输出目录训练

如果你想明确指定日志和 checkpoint 保存路径：

```bash
cd ~/mmpretrain
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    --work-dir ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced \
    --amp
```

### 8.3 后台训练

如果你是通过 SSH 远程连接服务器，建议用 `nohup`：

```bash
cd ~/mmpretrain
nohup bash -lc "CUDA_VISIBLE_DEVICES=0 python tools/train.py ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py --work-dir ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced --amp" > ~/ECE1_0423/train.log 2>&1 &
```

查看日志：

```bash
tail -f ~/ECE1_0423/train.log
```

## 9. 第七步：恢复训练

如果训练中断，可以恢复：

```bash
cd ~/mmpretrain
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    --work-dir ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced \
    --resume auto \
    --amp
```

如果你想从指定 checkpoint 恢复：

```bash
cd ~/mmpretrain
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    --work-dir ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced \
    --resume ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced/epoch_10.pth \
    --amp
```

## 10. 第八步：测试最佳模型

训练完成后，工作目录一般在：

```text
~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced
```

由于配置中启用了：

```text
save_best='accuracy/top1'
```

所以你通常会得到一个最佳 checkpoint，文件名类似：

```text
best_accuracy_top1_epoch_*.pth
```

测试命令如下：

```bash
cd ~/mmpretrain
python tools/test.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced/best_accuracy_top1_epoch_*.pth
```

如果你想用某个指定 checkpoint 测试：

```bash
cd ~/mmpretrain
python tools/test.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced/epoch_30.pth
```

## 11. 当前配置的关键设置

当前 `exercise1/configs/convnext_tiny_ucmerced.py` 已经固定了以下关键参数：

- 基模型：`convnext-tiny_32xb128_in1k`
- 数据集类型：`CustomDataset`
- 类别数：`21`
- 训练 batch size：`16`
- `num_workers=2`
- 训练轮数：`30`
- 验证指标：`top-1 accuracy`
- 学习率：`1e-4`
- checkpoint 策略：按 `accuracy/top1` 保存最佳模型

## 12. 常用排错命令

### 查看 GPU

```bash
nvidia-smi
```

### 查看当前 Python

```bash
which python
python --version
```

### 查看 MMPretrain 是否安装成功

```bash
python -c "import mmpretrain; print(mmpretrain.__version__)"
python -c "import mmengine; print(mmengine.__version__)"
```

### 查看训练日志

```bash
tail -f ~/ECE1_0423/train.log
```

### 查看 work_dir 里的文件

```bash
ls -lah ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced
```

## 13. 训练完成后建议保留的内容

建议最终保留以下材料用于作业提交：

- 训练配置文件
- 最佳模型 checkpoint
- 训练日志
- 最终验证结果
- 报告中使用的关键命令和实验设置

## 14. 一套最短可运行流程

如果你只想看最短命令链，可以直接按下面执行：

```bash
cd ~
git clone https://github.com/Ccai73810/ECE1_0423.git
git clone https://github.com/open-mmlab/mmpretrain.git

conda create -n openmmlab python=3.10 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch -y
pip install -U openmim

cd ~/mmpretrain
mim install -e .

cd ~/ECE1_0423
python exercise1/scripts/check_dataset.py

cd ~/mmpretrain
python tools/misc/print_config.py ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    --work-dir ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced \
    --amp
```

训练结束后测试：

```bash
cd ~/mmpretrain
python tools/test.py \
    ~/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py \
    ~/ECE1_0423/exercise1/work_dirs/convnext_tiny_ucmerced/best_accuracy_top1_epoch_*.pth
```

## 15. 参考资料

- MMPretrain 安装文档：[Prerequisites / Get Started](https://mmpretrain.readthedocs.io/en/stable/get_started.html)
- MMPretrain 训练文档：[Train](https://mmpretrain.readthedocs.io/en/stable/user_guides/train.html)
- MMPretrain 配置展开工具：[Print Config](https://mmpretrain.readthedocs.io/en/dev/useful_tools/print_config.html)
- MMPretrain 自定义数据集微调：[How to Fine-tune with Custom Dataset](https://mmpretrain.readthedocs.io/en/dev/notes/finetune_custom_dataset.html)

如果远程平台有自己的 CUDA、PyTorch 或 conda 约束，优先遵循平台要求；本 README 主要负责把当前仓库这套 Exercise 1 代码完整跑起来。
