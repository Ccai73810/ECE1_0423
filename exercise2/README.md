# Exercise 2 目录说明

本目录用于完成 `Exercise 2: Model Comparison and Error Analysis`。

核心内容包括：

- `configs/`：4 个模型的 MMPretrain 配置文件
- `scripts/run_exercise2.py`：统一训练、收集、评估、绘图、分析入口
- `results/`：标准化实验产物输出目录

推荐工作流：

```bash
python exercise2/scripts/run_exercise2.py train --models resnet50 swin_tiny mobilenet_v3_large --mmpretrain-root /root/mmpretrain
python exercise2/scripts/run_exercise2.py collect-convnext --source-work-dir exercise1/work_dirs/convnext_tiny_ucmerced
python exercise2/scripts/run_exercise2.py evaluate --models all --mmpretrain-root /root/mmpretrain
python exercise2/scripts/run_exercise2.py plot --models all
python exercise2/scripts/run_exercise2.py analyze --models all
```

默认约定：

- 数据集根目录：`UCMerced_LandUse/uc_merced_dataset`
- 测试样本：当前仓库中的验证集 `val.txt`
- 最差样本定义：按单样本交叉熵损失从高到低排序

脚本会在 `exercise2/results/` 下生成：

- 每个模型的标准化训练指标
- 每个模型的逐样本预测
- 每个模型最差的 3 个样本
- 每个模型的混淆矩阵
- 多模型训练曲线对比图
- 模型比较表
- 全模型最难样本汇总
