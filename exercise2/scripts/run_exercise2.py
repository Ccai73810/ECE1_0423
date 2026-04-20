from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSES = (
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
DEFAULT_MAIN_SETTINGS = "input=224, batch=16, lr=1e-4, epochs=100, optimizer=AdamW"
JSON_INDENT = 2
EPS = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[2]
EXERCISE2_ROOT = REPO_ROOT / "exercise2"
RESULTS_ROOT = EXERCISE2_ROOT / "results"
MODELS_ROOT = RESULTS_ROOT / "models"
COMPARISON_ROOT = RESULTS_ROOT / "comparison"
DEFAULT_DATA_ROOT = REPO_ROOT / "UCMerced_LandUse" / "uc_merced_dataset"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    config_path: Path
    result_dir: Path
    notes: str
    trainable: bool = True

    @property
    def work_dir(self) -> Path:
        return self.result_dir / "work_dir"


MODEL_SPECS = {
    "resnet50": ModelSpec(
        model_id="resnet50",
        display_name="ResNet-50",
        config_path=EXERCISE2_ROOT / "configs" / "resnet50_uc_merced.py",
        result_dir=MODELS_ROOT / "resnet50",
        notes="Industry-standard convolutional baseline.",
    ),
    "swin_tiny": ModelSpec(
        model_id="swin_tiny",
        display_name="Swin-Transformer-Tiny",
        config_path=EXERCISE2_ROOT / "configs" / "swin_tiny_uc_merced.py",
        result_dir=MODELS_ROOT / "swin_tiny",
        notes="Transformer-based hierarchical backbone.",
    ),
    "mobilenet_v3_large": ModelSpec(
        model_id="mobilenet_v3_large",
        display_name="MobileNet-V3-Large",
        config_path=EXERCISE2_ROOT / "configs" / "mobilenet_v3_large_uc_merced.py",
        result_dir=MODELS_ROOT / "mobilenet_v3_large",
        notes="Edge-oriented lightweight baseline.",
    ),
    "convnext_tiny": ModelSpec(
        model_id="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        config_path=EXERCISE2_ROOT / "configs" / "convnext_tiny_uc_merced.py",
        result_dir=MODELS_ROOT / "convnext_tiny",
        notes="Reused Exercise 1 result for an extra comparison data point.",
        trainable=False,
    ),
}

MODEL_ALIASES = {
    "resnet50": "resnet50",
    "resnet-50": "resnet50",
    "swin_tiny": "swin_tiny",
    "swin-transformer-tiny": "swin_tiny",
    "swin_transformer_tiny": "swin_tiny",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet-v3-large": "mobilenet_v3_large",
    "convnext_tiny": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise 2 orchestration for multi-model comparison and error analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the selected models.")
    train_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Model IDs or aliases. Use "all" to train all trainable models.',
    )
    train_parser.add_argument(
        "--mmpretrain-root",
        type=Path,
        required=True,
        help="Path to the local MMPretrain repository.",
    )
    train_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to uc_merced_dataset.",
    )
    train_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch MMPretrain training.",
    )
    train_parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP training. AMP is enabled by default.",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint if work_dir already contains one.",
    )

    collect_parser = subparsers.add_parser(
        "collect-convnext", help="Reuse the existing Exercise 1 ConvNeXt results."
    )
    collect_parser.add_argument(
        "--source-work-dir",
        type=Path,
        default=REPO_ROOT / "exercise1" / "work_dirs" / "convnext_tiny_ucmerced",
        help="Existing Exercise 1 ConvNeXt work_dir.",
    )
    collect_parser.add_argument(
        "--source-config",
        type=Path,
        default=REPO_ROOT / "exercise1" / "configs" / "convnext_tiny_ucmerced.py",
        help="Config path associated with the reused ConvNeXt checkpoint.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run validation inference and produce per-model artifacts."
    )
    evaluate_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Model IDs or aliases. Use "all" for all four models.',
    )
    evaluate_parser.add_argument(
        "--mmpretrain-root",
        type=Path,
        required=True,
        help="Path to the local MMPretrain repository.",
    )
    evaluate_parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to uc_merced_dataset.",
    )
    evaluate_parser.add_argument(
        "--device",
        default=None,
        help='Inference device, e.g. "cuda:0" or "cpu". Defaults to CUDA if available.',
    )
    evaluate_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )

    plot_parser = subparsers.add_parser(
        "plot", help="Plot combined multi-model training curves."
    )
    plot_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Model IDs or aliases. Use "all" for all four models.',
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Produce comparison tables and global hardest-sample analysis."
    )
    analyze_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='Model IDs or aliases. Use "all" for all four models.',
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_specs(requested: Iterable[str], *, for_training: bool = False) -> list[ModelSpec]:
    values = list(requested)
    if not values:
        raise ValueError("No models were provided.")

    if "all" in values:
        specs = list(MODEL_SPECS.values())
        if for_training:
            specs = [spec for spec in specs if spec.trainable]
        return specs

    resolved: list[ModelSpec] = []
    seen: set[str] = set()
    for raw in values:
        key = MODEL_ALIASES.get(raw.lower())
        if key is None:
            raise ValueError(f"Unsupported model identifier: {raw}")
        spec = MODEL_SPECS[key]
        if for_training and not spec.trainable:
            continue
        if spec.model_id not in seen:
            resolved.append(spec)
            seen.add(spec.model_id)
    return resolved


def build_env(mmpretrain_root: Path | None = None, data_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if mmpretrain_root is not None:
        env["MMPRETRAIN_ROOT"] = str(mmpretrain_root.resolve())
    if data_root is not None:
        env["ECE1_DATA_ROOT"] = str(data_root.resolve())
    return env


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_scalar_files(work_dir: Path) -> list[Path]:
    scalar_files: list[Path] = []
    if not work_dir.exists():
        return scalar_files
    for run_dir in sorted(path for path in work_dir.iterdir() if path.is_dir()):
        candidate = run_dir / "vis_data" / "scalars.json"
        if candidate.is_file():
            scalar_files.append(candidate)
    return scalar_files


def parse_scalar_files(scalar_files: list[Path]) -> list[dict]:
    rows_by_epoch: dict[int, dict] = {}
    for scalar_file in scalar_files:
        with scalar_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "loss" in record and "epoch" in record:
                    epoch = int(record["epoch"])
                    row = rows_by_epoch.setdefault(epoch, {"epoch": epoch})
                    row["train_loss"] = float(record["loss"])
                    if "lr" in record:
                        row["lr"] = float(record["lr"])
                    if "iter" in record:
                        row["iter"] = int(record["iter"])
                    if "memory" in record:
                        row["memory"] = float(record["memory"])
                if "accuracy/top1" in record:
                    epoch = int(record.get("epoch", record.get("step", 0)))
                    row = rows_by_epoch.setdefault(epoch, {"epoch": epoch})
                    row["val_top1"] = float(record["accuracy/top1"])
    rows = [rows_by_epoch[idx] for idx in sorted(rows_by_epoch)]
    return rows


def find_best_checkpoint(work_dir: Path) -> Path | None:
    candidates = sorted(
        work_dir.glob("best_accuracy_top1*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_summary(
    spec: ModelSpec,
    metrics: list[dict],
    *,
    source_work_dir: Path,
    config_path: Path,
    source_config_path: Path | None,
    notes: str,
    raw_scalar_files: list[Path],
) -> dict:
    best_val = max((float(row.get("val_top1", float("-inf"))) for row in metrics), default=float("-inf"))
    max_epoch = max((int(row["epoch"]) for row in metrics), default=0)
    best_checkpoint = find_best_checkpoint(source_work_dir)
    return {
        "model_id": spec.model_id,
        "model_name": spec.display_name,
        "config_path": str(config_path.resolve()),
        "source_config_path": (
            str(source_config_path.resolve()) if source_config_path is not None else None
        ),
        "source_work_dir": str(source_work_dir.resolve()),
        "result_dir": str(spec.result_dir.resolve()),
        "work_dir": str(spec.work_dir.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint else None,
        "best_val_top1": None if best_val == float("-inf") else round(best_val, 6),
        "num_logged_epochs": max_epoch,
        "main_settings": DEFAULT_MAIN_SETTINGS,
        "notes": notes,
        "raw_scalar_files": [str(path.resolve()) for path in raw_scalar_files],
    }


def standardize_training_artifacts(
    spec: ModelSpec,
    *,
    source_work_dir: Path,
    config_path: Path,
    source_config_path: Path | None = None,
    notes: str,
) -> dict:
    ensure_dir(spec.result_dir)
    ensure_dir(spec.work_dir)

    scalar_files = find_scalar_files(source_work_dir)
    if not scalar_files:
        raise FileNotFoundError(f"No scalars.json files found under {source_work_dir}")

    metrics = parse_scalar_files(scalar_files)
    if not metrics:
        raise RuntimeError(f"No metrics parsed from scalar files under {source_work_dir}")

    metric_rows: list[dict] = []
    for row in metrics:
        metric_rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": row.get("train_loss"),
                "val_top1": row.get("val_top1"),
                "lr": row.get("lr"),
                "iter": row.get("iter"),
                "memory": row.get("memory"),
            }
        )

    best_checkpoint = find_best_checkpoint(source_work_dir)
    summary = build_summary(
        spec,
        metric_rows,
        source_work_dir=source_work_dir,
        config_path=config_path,
        source_config_path=source_config_path,
        notes=notes,
        raw_scalar_files=scalar_files,
    )

    metrics_csv = spec.result_dir / "metrics.csv"
    metrics_jsonl = spec.result_dir / "metrics.jsonl"
    summary_json = spec.result_dir / "summary.json"
    best_txt = spec.result_dir / "best_checkpoint.txt"

    write_csv(
        metrics_csv,
        metric_rows,
        ["epoch", "train_loss", "val_top1", "lr", "iter", "memory"],
    )
    write_jsonl(metrics_jsonl, metric_rows)
    write_json(summary_json, summary)
    if best_checkpoint is not None:
        best_txt.write_text(str(best_checkpoint.resolve()) + "\n", encoding="utf-8")

    return summary


def run_training(args: argparse.Namespace) -> int:
    specs = resolve_model_specs(args.models, for_training=True)
    if not specs:
        raise ValueError("No trainable models were selected.")

    mmpretrain_root = args.mmpretrain_root.resolve()
    train_script = mmpretrain_root / "tools" / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"MMPretrain train.py not found: {train_script}")

    for spec in specs:
        ensure_dir(spec.result_dir)
        ensure_dir(spec.work_dir)
        command = [
            args.python,
            str(train_script),
            str(spec.config_path.resolve()),
            "--work-dir",
            str(spec.work_dir.resolve()),
        ]
        if not args.no_amp:
            command.append("--amp")
        if args.resume and (spec.work_dir / "last_checkpoint").is_file():
            command.extend(["--resume", "auto"])

        print(f"[train] {spec.model_id}: {' '.join(command)}")
        subprocess.run(
            command,
            check=True,
            cwd=str(mmpretrain_root),
            env=build_env(mmpretrain_root=mmpretrain_root, data_root=args.data_root),
        )

        summary = standardize_training_artifacts(
            spec,
            source_work_dir=spec.work_dir,
            config_path=spec.config_path,
            source_config_path=spec.config_path,
            notes=spec.notes,
        )
        print(
            f"[train] standardized {spec.model_id}: best_val_top1={summary['best_val_top1']}, "
            f"checkpoint={summary['best_checkpoint']}"
        )
    return 0


def collect_convnext(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS["convnext_tiny"]
    summary = standardize_training_artifacts(
        spec,
        source_work_dir=args.source_work_dir.resolve(),
        config_path=spec.config_path,
        source_config_path=args.source_config.resolve(),
        notes="Reused from exercise1; no retraining in exercise2.",
    )
    print(
        f"[collect-convnext] best_val_top1={summary['best_val_top1']}, "
        f"checkpoint={summary['best_checkpoint']}"
    )
    return 0


def load_summary(spec: ModelSpec) -> dict:
    summary_path = spec.result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing summary for {spec.model_id}: {summary_path}. "
            "Run train/collect-convnext first."
        )
    return read_json(summary_path)


def load_annotation_rows(data_root: Path, split: str = "val") -> list[dict]:
    ann_path = data_root / f"{split}.txt"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    rows: list[dict] = []
    with ann_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path_text, label_text = line.split()
            label = int(label_text)
            image_path = data_root / split / rel_path_text
            rows.append(
                {
                    "relative_path": rel_path_text,
                    "image_path": str(image_path.resolve()),
                    "gt_label": label,
                    "gt_class": CLASSES[label],
                }
            )
    return rows


def ensure_probabilities(scores: object) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.size != len(CLASSES):
        raise ValueError(
            f"Expected {len(CLASSES)} scores, received shape {array.shape}"
        )
    total = float(array.sum())
    if np.all(array >= 0.0) and math.isfinite(total) and abs(total - 1.0) <= 1e-4:
        return array
    shifted = array - np.max(array)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def get_default_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    fieldnames = ["gt_class"] + list(CLASSES)
    rows: list[dict] = []
    for idx, class_name in enumerate(CLASSES):
        row = {"gt_class": class_name}
        for pred_idx, pred_name in enumerate(CLASSES):
            row[pred_name] = int(matrix[idx, pred_idx])
        rows.append(row)
    write_csv(path, rows, fieldnames)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def copy_hardest_sample_images(hardest_rows: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for stale in output_dir.iterdir():
        if stale.is_file():
            stale.unlink()
    for row in hardest_rows:
        source = Path(row["image_path"])
        if not source.is_file():
            continue
        destination = output_dir / f"{int(row['rank']):02d}_{source.name}"
        shutil.copy2(source, destination)


def evaluate_single_model(
    spec: ModelSpec,
    *,
    summary: dict,
    data_root: Path,
    device: str,
    batch_size: int,
    mmpretrain_root: Path,
) -> None:
    checkpoint = summary.get("best_checkpoint")
    if not checkpoint:
        raise FileNotFoundError(f"No best checkpoint recorded for {spec.model_id}")

    config_path = Path(summary["config_path"])
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found for {spec.model_id}: {config_path}")

    annotation_rows = load_annotation_rows(data_root, split="val")
    image_paths = [row["image_path"] for row in annotation_rows]

    env = build_env(mmpretrain_root=mmpretrain_root, data_root=data_root)
    os.environ.update(env)

    from mmpretrain import ImageClassificationInferencer

    inferencer = ImageClassificationInferencer(
        model=str(config_path.resolve()),
        pretrained=str(Path(checkpoint).resolve()),
        device=device,
    )

    raw_results = inferencer(image_paths, batch_size=batch_size)
    if len(raw_results) != len(annotation_rows):
        raise RuntimeError(
            f"Inference count mismatch for {spec.model_id}: "
            f"{len(raw_results)} results vs {len(annotation_rows)} annotations"
        )

    predictions: list[dict] = []
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for ann_row, result in zip(annotation_rows, raw_results):
        probs = ensure_probabilities(result["pred_scores"])
        gt_label = int(ann_row["gt_label"])
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])
        true_class_probability = float(probs[gt_label])
        sample_cross_entropy = float(-math.log(max(true_class_probability, EPS)))
        matrix[gt_label, pred_label] += 1

        predictions.append(
            {
                "relative_path": ann_row["relative_path"],
                "image_path": ann_row["image_path"],
                "gt_label": gt_label,
                "gt_class": ann_row["gt_class"],
                "pred_label": pred_label,
                "pred_class": CLASSES[pred_label],
                "confidence": round(confidence, 8),
                "true_class_probability": round(true_class_probability, 8),
                "sample_cross_entropy": round(sample_cross_entropy, 8),
                "is_correct": int(pred_label == gt_label),
                "probabilities": [round(float(value), 8) for value in probs.tolist()],
            }
        )

    hardest_rows = sorted(
        predictions,
        key=lambda row: float(row["sample_cross_entropy"]),
        reverse=True,
    )[:3]
    hardest_ranked = []
    for index, row in enumerate(hardest_rows, start=1):
        ranked = dict(row)
        ranked["rank"] = index
        hardest_ranked.append(ranked)

    predictions_csv_rows = []
    for row in predictions:
        predictions_csv_rows.append(
            {
                "relative_path": row["relative_path"],
                "image_path": row["image_path"],
                "gt_label": row["gt_label"],
                "gt_class": row["gt_class"],
                "pred_label": row["pred_label"],
                "pred_class": row["pred_class"],
                "confidence": row["confidence"],
                "true_class_probability": row["true_class_probability"],
                "sample_cross_entropy": row["sample_cross_entropy"],
                "is_correct": row["is_correct"],
            }
        )

    write_csv(
        spec.result_dir / "predictions.csv",
        predictions_csv_rows,
        [
            "relative_path",
            "image_path",
            "gt_label",
            "gt_class",
            "pred_label",
            "pred_class",
            "confidence",
            "true_class_probability",
            "sample_cross_entropy",
            "is_correct",
        ],
    )
    write_jsonl(spec.result_dir / "predictions.jsonl", predictions)
    write_csv(
        spec.result_dir / "hardest_samples.csv",
        hardest_ranked,
        [
            "rank",
            "relative_path",
            "image_path",
            "gt_label",
            "gt_class",
            "pred_label",
            "pred_class",
            "confidence",
            "true_class_probability",
            "sample_cross_entropy",
            "is_correct",
            "probabilities",
        ],
    )
    write_confusion_matrix_csv(spec.result_dir / "confusion_matrix.csv", matrix)
    plot_confusion_matrix(
        matrix,
        spec.result_dir / "confusion_matrix.png",
        f"{spec.display_name} Confusion Matrix",
    )
    copy_hardest_sample_images(hardest_ranked, spec.result_dir / "hardest_samples")

    summary["evaluation"] = {
        "num_samples": len(predictions),
        "device": device,
        "batch_size": batch_size,
        "prediction_file": str((spec.result_dir / "predictions.csv").resolve()),
        "hardest_samples_file": str((spec.result_dir / "hardest_samples.csv").resolve()),
        "confusion_matrix_file": str((spec.result_dir / "confusion_matrix.csv").resolve()),
    }
    write_json(spec.result_dir / "summary.json", summary)


def evaluate_models(args: argparse.Namespace) -> int:
    specs = resolve_model_specs(args.models)
    device = get_default_device(args.device)
    for spec in specs:
        summary = load_summary(spec)
        print(f"[evaluate] {spec.model_id} on {device}")
        evaluate_single_model(
            spec,
            summary=summary,
            data_root=args.data_root.resolve(),
            device=device,
            batch_size=args.batch_size,
            mmpretrain_root=args.mmpretrain_root.resolve(),
        )
    return 0


def plot_combined_curves(args: argparse.Namespace) -> int:
    specs = resolve_model_specs(args.models)
    ensure_dir(COMPARISON_ROOT)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_loss, ax_acc = axes

    plotted = 0
    for spec in specs:
        metrics_path = spec.result_dir / "metrics.csv"
        if not metrics_path.is_file():
            print(f"[plot] skip {spec.model_id}: missing {metrics_path}")
            continue
        rows = read_csv(metrics_path)
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [
            float(row["train_loss"]) for row in rows if row.get("train_loss") not in ("", None)
        ]
        loss_epochs = [
            int(row["epoch"]) for row in rows if row.get("train_loss") not in ("", None)
        ]
        val_top1 = [
            float(row["val_top1"]) for row in rows if row.get("val_top1") not in ("", None)
        ]
        val_epochs = [
            int(row["epoch"]) for row in rows if row.get("val_top1") not in ("", None)
        ]

        if train_loss:
            ax_loss.plot(loss_epochs, train_loss, label=spec.display_name, linewidth=2)
        if val_top1:
            ax_acc.plot(val_epochs, val_top1, label=spec.display_name, linewidth=2)
        if epochs:
            plotted += 1

    if not plotted:
        raise FileNotFoundError("No metrics.csv files were found for the selected models.")

    ax_loss.set_title("Training Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.grid(alpha=0.3)

    ax_acc.set_title("Validation Top-1 Accuracy vs Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Top-1 Accuracy (%)")
    ax_acc.grid(alpha=0.3)

    handles, labels = ax_acc.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = COMPARISON_ROOT / "combined_training_curves.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"[plot] wrote {output_path}")
    return 0


def save_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_models(args: argparse.Namespace) -> int:
    specs = resolve_model_specs(args.models)
    ensure_dir(COMPARISON_ROOT)

    comparison_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for spec in specs:
        summary = load_summary(spec)
        summaries[spec.model_id] = summary
        predictions_path = spec.result_dir / "predictions.csv"
        confusion_png = spec.result_dir / "confusion_matrix.png"
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"Missing predictions for {spec.model_id}: {predictions_path}. Run evaluate first."
            )
        if not confusion_png.is_file():
            raise FileNotFoundError(
                f"Missing confusion matrix image for {spec.model_id}: {confusion_png}"
            )
        comparison_rows.append(
            {
                "model_id": spec.model_id,
                "model_name": spec.display_name,
                "main_settings": summary.get("main_settings", DEFAULT_MAIN_SETTINGS),
                "best_val_top1": summary.get("best_val_top1"),
                "final_checkpoint": summary.get("best_checkpoint"),
                "notes": summary.get("notes", spec.notes),
            }
        )

    comparison_rows.sort(
        key=lambda row: float(row["best_val_top1"]) if row["best_val_top1"] is not None else float("-inf"),
        reverse=True,
    )

    write_csv(
        COMPARISON_ROOT / "model_comparison.csv",
        comparison_rows,
        ["model_id", "model_name", "main_settings", "best_val_top1", "final_checkpoint", "notes"],
    )
    save_markdown_table(
        COMPARISON_ROOT / "model_comparison.md",
        ["Model", "Main Setting(s)", "Best Val Top-1", "Final Checkpoint", "Notes"],
        [
            [
                row["model_name"],
                row["main_settings"],
                "" if row["best_val_top1"] is None else f"{float(row['best_val_top1']):.4f}",
                row["final_checkpoint"] or "",
                row["notes"],
            ]
            for row in comparison_rows
        ],
    )

    best_row = comparison_rows[0]
    worst_row = comparison_rows[-1]
    best_spec = MODEL_SPECS[best_row["model_id"]]
    worst_spec = MODEL_SPECS[worst_row["model_id"]]
    shutil.copy2(
        best_spec.result_dir / "confusion_matrix.png",
        COMPARISON_ROOT / "best_model_confusion_matrix.png",
    )
    shutil.copy2(
        worst_spec.result_dir / "confusion_matrix.png",
        COMPARISON_ROOT / "worst_model_confusion_matrix.png",
    )

    aggregate: dict[str, dict] = {}
    for spec in specs:
        rows = read_csv(spec.result_dir / "predictions.csv")
        for row in rows:
            key = row["relative_path"]
            entry = aggregate.setdefault(
                key,
                {
                    "relative_path": row["relative_path"],
                    "image_path": row["image_path"],
                    "gt_label": int(row["gt_label"]),
                    "gt_class": row["gt_class"],
                    "losses": {},
                },
            )
            entry["losses"][spec.model_id] = float(row["sample_cross_entropy"])

    global_rows: list[dict] = []
    for entry in aggregate.values():
        losses = entry["losses"]
        if len(losses) != len(specs):
            continue
        average_loss = float(sum(losses.values()) / len(losses))
        row = {
            "relative_path": entry["relative_path"],
            "image_path": entry["image_path"],
            "gt_label": entry["gt_label"],
            "gt_class": entry["gt_class"],
            "average_sample_cross_entropy": round(average_loss, 8),
        }
        for spec in specs:
            row[f"{spec.model_id}_loss"] = round(losses[spec.model_id], 8)
        global_rows.append(row)

    global_rows.sort(key=lambda row: float(row["average_sample_cross_entropy"]), reverse=True)
    for index, row in enumerate(global_rows, start=1):
        row["rank"] = index

    fieldnames = [
        "rank",
        "relative_path",
        "image_path",
        "gt_label",
        "gt_class",
        "average_sample_cross_entropy",
    ] + [f"{spec.model_id}_loss" for spec in specs]
    write_csv(COMPARISON_ROOT / "global_hardest_samples.csv", global_rows, fieldnames)

    if global_rows:
        hardest = global_rows[0]
        hardest_dir = ensure_dir(COMPARISON_ROOT / "global_hardest_sample")
        for stale in hardest_dir.iterdir():
            if stale.is_file():
                stale.unlink()
        source = Path(hardest["image_path"])
        if source.is_file():
            shutil.copy2(source, hardest_dir / f"01_{source.name}")
        write_json(COMPARISON_ROOT / "global_hardest_sample.json", hardest)

    analysis_summary = {
        "best_model": best_row,
        "worst_model": worst_row,
        "num_models": len(specs),
        "num_global_ranked_samples": len(global_rows),
    }
    write_json(COMPARISON_ROOT / "analysis_summary.json", analysis_summary)

    print(f"[analyze] best model: {best_row['model_name']}")
    print(f"[analyze] worst model: {worst_row['model_name']}")
    print(f"[analyze] wrote {COMPARISON_ROOT / 'model_comparison.csv'}")
    print(f"[analyze] wrote {COMPARISON_ROOT / 'global_hardest_samples.csv'}")
    return 0


def main() -> int:
    args = parse_args()
    ensure_dir(RESULTS_ROOT)
    ensure_dir(MODELS_ROOT)
    ensure_dir(COMPARISON_ROOT)

    if args.command == "train":
        return run_training(args)
    if args.command == "collect-convnext":
        return collect_convnext(args)
    if args.command == "evaluate":
        return evaluate_models(args)
    if args.command == "plot":
        return plot_combined_curves(args)
    if args.command == "analyze":
        return analyze_models(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
