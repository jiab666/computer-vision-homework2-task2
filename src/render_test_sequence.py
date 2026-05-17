from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from visdrone_utils import ensure_dir, group_images_by_sequence, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one VisDrone test-dev image sequence into a video."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("archive/VisDrone2019-DET-test-dev/images"),
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default="0000074",
        help="Prefix before the first underscore in the image filename.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/videos/test_sequence_0000074.mp4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_map = group_images_by_sequence(list_images(args.images_dir))
    frames = sequence_map.get(args.sequence_id)
    if not frames:
        raise FileNotFoundError(f"Sequence {args.sequence_id!r} not found in {args.images_dir}")

    ensure_dir(args.output.parent)
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to load {frames[0]}")
    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to load {frame_path}")
            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
