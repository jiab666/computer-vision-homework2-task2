from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from visdrone_utils import VISDRONE_CLASS_NAMES, ensure_dir


def parse_line_points(line_text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    x1, y1, x2, y2 = [int(value) for value in line_text.split(",")]
    return (x1, y1), (x2, y2)


def side_of_line(point: tuple[float, float], line: tuple[tuple[int, int], tuple[int, int]]) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO tracking and line-crossing counting on a video or image sequence."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/task2_detect/visdrone_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="artifacts/videos/test_sequence_0000074.mp4",
        help="Video path, webcam id, or a directory of frames supported by Ultralytics.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument(
        "--line",
        type=str,
        default="200,420,1160,420",
        help="Line endpoints formatted as x1,y1,x2,y2.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Bounding box line width used in result visualization.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=0.4,
        help="Label font scale used in result visualization.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("artifacts/tracking/tracked_counted.mp4"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/tracking/tracks.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    ensure_dir(args.output_video.parent)
    ensure_dir(args.output_csv.parent)

    line = parse_line_points(args.line)
    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    previous_side: dict[int, float] = {}
    counted_ids: set[int] = set()
    total_crossings = 0
    frame_index = -1

    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame_index",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "crossed_line_total",
            ]
        )

        generator = model.track(
            source=args.source,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            persist=True,
            stream=True,
            verbose=False,
        )

        for result in generator:
            frame_index += 1
            frame = result.plot(line_width=args.line_width, font_size=args.font_size)
            cv2.line(frame, line[0], line[1], (0, 255, 255), 3)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                xyxy_boxes = boxes.xyxy.cpu().tolist()

                for track_id, class_id, confidence, xyxy in zip(
                    track_ids, classes, confidences, xyxy_boxes
                ):
                    x1, y1, x2, y2 = xyxy
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    side = side_of_line(center, line)
                    track_history[track_id].append(center)
                    track_history[track_id] = track_history[track_id][-30:]

                    if track_id in previous_side and track_id not in counted_ids:
                        if previous_side[track_id] == 0:
                            pass
                        elif side == 0 or previous_side[track_id] * side < 0:
                            counted_ids.add(track_id)
                            total_crossings += 1
                    previous_side[track_id] = side

                    class_name = VISDRONE_CLASS_NAMES[class_id]
                    writer.writerow(
                        [
                            frame_index,
                            track_id,
                            class_id,
                            class_name,
                            round(confidence, 5),
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                            round(center[0], 2),
                            round(center[1], 2),
                            total_crossings,
                        ]
                    )

                    trail = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    if len(trail) >= 2:
                        cv2.polylines(frame, [trail], False, (255, 255, 0), 2)

            cv2.putText(
                frame,
                f"Cross Count: {total_crossings}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
            )
            video_writer.write(frame)

    video_writer.release()
    print(f"Saved annotated video to {args.output_video}")
    print(f"Saved track table to {args.output_csv}")
    print(f"Final crossing count: {total_crossings}")


if __name__ == "__main__":
    main()
