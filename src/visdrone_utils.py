from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# VisDrone category ids 1-10 are the official detection classes.
VALID_CATEGORY_IDS = set(range(1, 11))


@dataclass(frozen=True)
class VisDroneObject:
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    score: int
    category_id: int
    truncation: int
    occlusion: int

    @property
    def x1(self) -> float:
        return self.bbox_left

    @property
    def y1(self) -> float:
        return self.bbox_top

    @property
    def x2(self) -> float:
        return self.bbox_left + self.bbox_width

    @property
    def y2(self) -> float:
        return self.bbox_top + self.bbox_height

    @property
    def yolo_class_id(self) -> int:
        return self.category_id - 1

    @property
    def is_ignored(self) -> bool:
        return self.category_id not in VALID_CATEGORY_IDS or self.score == 0

    def to_yolo_line(self, image_width: int, image_height: int) -> str:
        center_x = (self.bbox_left + self.bbox_width / 2.0) / image_width
        center_y = (self.bbox_top + self.bbox_height / 2.0) / image_height
        width = self.bbox_width / image_width
        height = self.bbox_height / image_height
        return (
            f"{self.yolo_class_id} "
            f"{center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def parse_annotation_line(line: str) -> VisDroneObject:
    # Some VisDrone files in this homework dump have a trailing comma.
    # We drop empty fields so rows like "...,0,0," still parse correctly.
    fields = [item.strip() for item in line.strip().split(",") if item.strip() != ""]
    parts = [int(float(item)) for item in fields]
    if len(parts) != 8:
        raise ValueError(f"Invalid VisDrone annotation row: {line!r}")
    return VisDroneObject(*parts)


def read_visdrone_annotation(txt_path: Path) -> list[VisDroneObject]:
    rows: list[VisDroneObject] = []
    if not txt_path.exists():
        return rows
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            rows.append(parse_annotation_line(raw_line))
    return rows


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def sequence_key(image_path: Path) -> tuple[str, int]:
    stem_parts = image_path.stem.split("_")
    prefix = stem_parts[0]
    frame_idx = int(stem_parts[-1])
    return prefix, frame_idx


def group_images_by_sequence(images: Iterable[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in images:
        prefix, _ = sequence_key(image_path)
        grouped.setdefault(prefix, []).append(image_path)
    for prefix in grouped:
        grouped[prefix].sort(key=sequence_key)
    return grouped


def compute_iou(box_a: VisDroneObject, box_b: VisDroneObject) -> float:
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    area_a = box_a.bbox_width * box_a.bbox_height
    area_b = box_b.bbox_width * box_b.bbox_height
    union_area = area_a + area_b - inter_area
    return inter_area / max(union_area, 1e-6)
