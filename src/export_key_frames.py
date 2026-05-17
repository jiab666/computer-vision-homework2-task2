from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from visdrone_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a consecutive frame window from a tracked video for report figures."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("artifacts/tracking/tracked_counted.mp4"),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="First frame index to export.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="How many consecutive frames to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/key_frames"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    current_idx = -1
    saved_count = 0
    target_range = range(args.start_frame, args.start_frame + args.num_frames)

    while True:
        success, frame = capture.read()
        if not success:
            break
        current_idx += 1
        if current_idx not in target_range:
            continue
        output_path = args.output_dir / f"frame_{current_idx:05d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved_count += 1

    capture.release()
    print(f"Saved {saved_count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
