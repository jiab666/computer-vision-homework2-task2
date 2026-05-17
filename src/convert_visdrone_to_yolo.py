from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from visdrone_utils import (
    VISDRONE_CLASS_NAMES,
    ensure_dir,
    list_images,
    load_image_size,
    read_visdrone_annotation,
)


def resolve_split_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    if split == "train":
        base = dataset_root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    elif split == "val":
        base = dataset_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"
    elif split == "test":
        base = dataset_root / "VisDrone2019-DET-test-dev"
    else:
        raise ValueError(f"Unsupported split: {split}")
    return base / "images", base / "annotations"


def convert_split(dataset_root: Path, output_root: Path, split: str) -> int:
    images_dir, annotations_dir = resolve_split_paths(dataset_root, split)
    target_images_dir = output_root / "images" / split
    target_labels_dir = output_root / "labels" / split
    ensure_dir(target_images_dir)
    ensure_dir(target_labels_dir)

    converted_count = 0
    for image_path in list_images(images_dir):
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        image_width, image_height = load_image_size(image_path)
        yolo_lines = [
            obj.to_yolo_line(image_width, image_height)
            for obj in read_visdrone_annotation(annotation_path)
            if not obj.is_ignored
        ]

        shutil.copy2(image_path, target_images_dir / image_path.name)
        (target_labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(yolo_lines),
            encoding="utf-8",
        )
        converted_count += 1
    return converted_count


def write_dataset_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "visdrone_local.yaml"
    yaml_content = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(VISDRONE_CLASS_NAMES)}",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(VISDRONE_CLASS_NAMES)],
            "",
        ]
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone detection annotations to YOLO format."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("archive"),
        help="Directory that contains the unpacked VisDrone folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/visdrone_yolo"),
        help="Directory to write the YOLO-formatted dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    summary: dict[str, int] = {}
    for split in ("train", "val", "test"):
        summary[split] = convert_split(args.dataset_root, args.output_root, split)

    yaml_path = write_dataset_yaml(args.output_root)
    print("Conversion finished.")
    for split, count in summary.items():
        print(f"  {split}: {count} images")
    print(f"  dataset_yaml: {yaml_path}")


if __name__ == "__main__":
    main()
