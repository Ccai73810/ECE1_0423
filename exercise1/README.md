# Exercise 1: ConvNeXt-Tiny Fine-Tuning Package

This folder contains a submission-ready starting point for Exercise 1 of the ECE371 assignment.
It does not run training by default and does not include any trained checkpoint.

## Included files

- `configs/convnext_tiny_ucmerced.py`: MMPretrain fine-tuning config for `convnext-tiny_32xb128_in1k`
- `scripts/check_dataset.py`: static dataset validation script

## What is already prepared

- The dataset split already matches the assignment requirement:
  - `21` classes
  - `1680` training images
  - `420` validation images
  - `80/20` images per class
- The config already overrides the official MMPretrain ConvNeXt-Tiny config for:
  - `21` output classes
  - UC Merced dataset paths
  - top-1 validation accuracy
  - single-GPU / Colab-oriented batch size and workers
  - `30` training epochs
  - AdamW fine-tuning learning rate
  - best-checkpoint selection by `accuracy/top1`

## Environment note

The current local machine uses Python `3.14`, which is suitable for file preparation here but is not the target runtime for MMPretrain.
Use one of the following instead:

- Google Colab
- a local conda environment with Python `3.8` to `3.11`

## Recommended Colab setup

Clone or upload this assignment folder into Colab first, then install MMPretrain and its toolchain.

```bash
python --version
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.0.0"
pip install "mmpretrain>=1.0.0"
git clone https://github.com/open-mmlab/mmpretrain.git
```

## Validate the dataset split

Run this from the assignment root:

```bash
python exercise1/scripts/check_dataset.py
```

If the dataset is in a different location, pass it explicitly:

```bash
python exercise1/scripts/check_dataset.py --data-root /path/to/uc_merced_dataset
```

## Inspect the final config

From the cloned `mmpretrain` repository:

```bash
python tools/misc/print_config.py /path/to/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py
```

This verifies that MMEngine can resolve the inherited base config and your overrides.

## Future training command

Training is intentionally not executed in this package, but the prepared command is:

```bash
python tools/train.py /path/to/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py --amp
```

The config writes outputs to:

```text
exercise1/work_dirs/convnext_tiny_ucmerced
```

## Future validation / test command

After training produces a checkpoint, evaluate it with:

```bash
python tools/test.py /path/to/ECE1_0423/exercise1/configs/convnext_tiny_ucmerced.py /path/to/checkpoint.pth
```

## Implementation notes

- The config uses MMPretrain package inheritance:
  - `_base_ = ['mmpretrain::convnext/convnext-tiny_32xb128_in1k.py']`
- The pre-trained ConvNeXt-Tiny ImageNet-1k checkpoint is configured as the initialization source.
- Class names are written directly into the config through `metainfo`, so the run does not depend on loading `classes.txt` dynamically.
- The dataset checker performs validation only. It never rewrites files or resplits the dataset.
