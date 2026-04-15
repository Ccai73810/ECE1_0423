from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_CLASSES = (
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
EXPECTED_COUNTS = {'train': 80, 'val': 20}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2] / 'UCMerced_LandUse' / 'uc_merced_dataset'
    parser = argparse.ArgumentParser(
        description='Validate the UC Merced split used by Exercise 1.'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=default_root,
        help='Path to uc_merced_dataset (default: %(default)s)',
    )
    return parser.parse_args()


def count_images(class_dir: Path) -> int:
    return sum(1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_split(data_root: Path, split: str) -> tuple[dict[str, int], list[str]]:
    split_dir = data_root / split
    errors: list[str] = []
    counts: dict[str, int] = {}

    if not split_dir.is_dir():
        return counts, [f'Missing split directory: {split_dir}']

    actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    expected_classes = list(EXPECTED_CLASSES)

    missing_classes = sorted(set(expected_classes) - set(actual_classes))
    extra_classes = sorted(set(actual_classes) - set(expected_classes))
    if missing_classes:
        errors.append(f'{split}: missing classes: {missing_classes}')
    if extra_classes:
        errors.append(f'{split}: unexpected classes: {extra_classes}')

    for class_name in expected_classes:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        image_count = count_images(class_dir)
        counts[class_name] = image_count
        expected_count = EXPECTED_COUNTS[split]
        if image_count != expected_count:
            errors.append(
                f'{split}/{class_name}: expected {expected_count} images, found {image_count}'
            )

    return counts, errors


def validate_annotation_file(
    data_root: Path,
    split: str,
    class_counts: dict[str, int],
) -> tuple[int, list[str]]:
    ann_path = data_root / f'{split}.txt'
    errors: list[str] = []

    if not ann_path.is_file():
        return 0, [f'Missing annotation file: {ann_path}']

    lines = [line.strip() for line in ann_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected_total = sum(class_counts.values())
    if len(lines) != expected_total:
        errors.append(
            f'{ann_path.name}: expected {expected_total} entries, found {len(lines)}'
        )

    class_to_idx = {name: idx for idx, name in enumerate(EXPECTED_CLASSES)}
    seen_paths: set[str] = set()
    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2:
            errors.append(f'{ann_path.name}:{index}: expected "<relative_path> <label>", got "{line}"')
            continue

        rel_path_text, label_text = parts
        rel_path = Path(rel_path_text)
        if rel_path_text in seen_paths:
            errors.append(f'{ann_path.name}:{index}: duplicated entry "{rel_path_text}"')
        seen_paths.add(rel_path_text)

        if len(rel_path.parts) < 2:
            errors.append(f'{ann_path.name}:{index}: invalid relative path "{rel_path_text}"')
            continue

        class_name = rel_path.parts[0]
        if class_name not in class_to_idx:
            errors.append(f'{ann_path.name}:{index}: unknown class "{class_name}"')
            continue

        try:
            label = int(label_text)
        except ValueError:
            errors.append(f'{ann_path.name}:{index}: label "{label_text}" is not an integer')
            continue

        expected_label = class_to_idx[class_name]
        if label != expected_label:
            errors.append(
                f'{ann_path.name}:{index}: class "{class_name}" should use label {expected_label}, found {label}'
            )

        image_path = data_root / split / rel_path
        if not image_path.is_file():
            errors.append(f'{ann_path.name}:{index}: missing image "{image_path}"')

    return len(lines), errors


def print_summary(data_root: Path, train_total: int, val_total: int) -> None:
    print('Dataset summary for report')
    print(f'- Data root: {data_root}')
    print(f'- Number of classes: {len(EXPECTED_CLASSES)}')
    print(f'- Training images: {train_total} ({EXPECTED_COUNTS["train"]} per class)')
    print(f'- Validation images: {val_total} ({EXPECTED_COUNTS["val"]} per class)')
    print('- Split ratio: 8:2')
    print('- Metadata files: train.txt and val.txt verified')


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()

    all_errors: list[str] = []
    train_counts, train_errors = validate_split(data_root, 'train')
    val_counts, val_errors = validate_split(data_root, 'val')
    all_errors.extend(train_errors)
    all_errors.extend(val_errors)

    _, train_ann_errors = validate_annotation_file(data_root, 'train', train_counts)
    _, val_ann_errors = validate_annotation_file(data_root, 'val', val_counts)
    all_errors.extend(train_ann_errors)
    all_errors.extend(val_ann_errors)

    if all_errors:
        print('Dataset validation failed:', file=sys.stderr)
        for error in all_errors:
            print(f'- {error}', file=sys.stderr)
        return 1

    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())

    print_summary(data_root, train_total, val_total)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
