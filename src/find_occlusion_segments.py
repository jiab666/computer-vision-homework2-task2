from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from visdrone_utils import (
    compute_iou,
    group_images_by_sequence,
    list_images,
    read_visdrone_annotation,
)


@dataclass(frozen=True)
class FrameScore:
    image_name: str
    object_count: int
    occluded_count: int
    overlap_pairs: int
    max_iou: float
    score: float


def score_frame(annotation_path: Path) -> FrameScore:
    objects = [obj for obj in read_visdrone_annotation(annotation_path) if not obj.is_ignored]
    overlap_pairs = 0
    max_iou = 0.0
    for idx, obj_a in enumerate(objects):
        for obj_b in objects[idx + 1 :]:
            iou = compute_iou(obj_a, obj_b)
            if iou >= 0.15:
                overlap_pairs += 1
            max_iou = max(max_iou, iou)

    occluded_count = sum(1 for obj in objects if obj.occlusion > 0)
    score = len(objects) + 3.0 * occluded_count + 4.0 * overlap_pairs + 10.0 * max_iou
    return FrameScore(
        image_name=annotation_path.with_suffix(".jpg").name,
        object_count=len(objects),
        occluded_count=occluded_count,
        overlap_pairs=overlap_pairs,
        max_iou=max_iou,
        score=score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank VisDrone test-dev frame sequences by likely occlusion difficulty."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("archive/VisDrone2019-DET-test-dev/images"),
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("archive/VisDrone2019-DET-test-dev/annotations"),
    )
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/occlusion_candidates.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_map = group_images_by_sequence(list_images(args.images_dir))
    ranked_windows: list[dict[str, object]] = []

    for sequence_id, frames in sequence_map.items():
        frame_scores = [
            score_frame(args.annotations_dir / f"{frame_path.stem}.txt")
            for frame_path in frames
        ]
        if len(frame_scores) < args.window_size:
            continue

        for start_idx in range(0, len(frame_scores) - args.window_size + 1):
            window = frame_scores[start_idx : start_idx + args.window_size]
            ranked_windows.append(
                {
                    "sequence_id": sequence_id,
                    "start_index": start_idx,
                    "mean_score": round(sum(item.score for item in window) / len(window), 4),
                    "frames": [asdict(item) for item in window],
                }
            )

    ranked_windows.sort(key=lambda item: item["mean_score"], reverse=True)
    top_windows = ranked_windows[: args.top_k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(top_windows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved top-{len(top_windows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
