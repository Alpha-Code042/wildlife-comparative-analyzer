
import os
import time
from pathlib import Path

import cv2

from algorithms.image_processor import (
    WILDLIFE_CLASSES,
    get_detection_model,
)
from algorithms.tracker import CentroidTracker


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "videos"

# Performance settings
FRAME_SKIP = 5
INFERENCE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.30
PROGRESS_INTERVAL_SECONDS = 5


def draw_tracking_annotations(frame, tracks):
    """Draw bounding boxes and tracking IDs on a video frame."""
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["id"]
        class_name = track["class_name"]
        confidence = track["confidence"]

        label = (
            f"{class_name} "
            f"#{track_id} "
            f"{confidence * 100:.1f}%"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def analyze_video(video_path):
    """
    Analyse a wildlife video using YOLO and the centroid tracker.

    YOLO inference runs every fifth frame at an inference size of
    640 pixels. The output retains the source video's dimensions
    and frame rate.
    """
    video_path = os.path.realpath(str(video_path))

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            "The selected video could not be opened."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if not fps or fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(
            "The video has invalid frame dimensions."
        )

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{Path(video_path).stem}_detected.mp4"
    )

    # Load the model before starting the frame-processing loop.
    model = get_detection_model()

    wildlife_class_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in WILDLIFE_CLASSES
    ]

    if not wildlife_class_ids:
        capture.release()
        raise ValueError(
            "No matching wildlife classes were found in the "
            "detection model. Check WILDLIFE_CLASSES and model.names."
        )

    tracker = CentroidTracker(
        max_distance=max(
            50,
            int(max(width, height) * 0.08),
        ),
        max_missed=3,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError(
            "The processed video could not be created."
        )

    total_detections = 0
    class_counts = {}
    processed_frames = 0
    analyzed_frames = 0
    frame_index = 0
    unique_track_ids = set()
    last_tracks = []

    start_time = time.perf_counter()
    last_progress_time = start_time

    print("\n" + "=" * 55)
    print("WILDLIFE VIDEO ANALYSIS STARTED")
    print(f"Input: {Path(video_path).name}")
    print(f"Video dimensions: {width} x {height}")
    print(f"Source FPS: {fps:.2f}")
    print(f"Source frames: {frame_count}")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"YOLO inference size: {INFERENCE_SIZE}")
    print(f"Analyse every {FRAME_SKIP} frames")
    print("=" * 55)

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_index += 1

            # Run inference on the first frame and every fifth frame.
            should_analyze = (
                (frame_index - 1) % FRAME_SKIP == 0
            )

            if should_analyze:
                results = model.predict(
                    source=frame,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    classes=wildlife_class_ids,
                    verbose=False,
                )

                result = results[0]
                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(
                            box.conf[0].item()
                        )

                        raw_class_name = model.names[class_id]

                        class_name = WILDLIFE_CLASSES.get(
                            raw_class_name,
                            raw_class_name,
                        )

                        x1, y1, x2, y2 = (
                            int(value)
                            for value in box.xyxy[0].tolist()
                        )

                        detections.append({
                            "class_name": class_name,
                            "raw_class_name": raw_class_name,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                        })

                        total_detections += 1

                        class_counts[class_name] = (
                            class_counts.get(class_name, 0) + 1
                        )

                last_tracks = tracker.update(detections)

                for track in last_tracks:
                    unique_track_ids.add(track["id"])

                analyzed_frames += 1

            # Always write the current frame. On skipped frames,
            # reuse the latest tracks rather than an old video frame.
            output_frame = draw_tracking_annotations(
                frame,
                last_tracks,
            )

            writer.write(output_frame)
            processed_frames += 1

            # Report progress in the terminal without printing
            # a message for every frame.
            now = time.perf_counter()

            if (
                now - last_progress_time
                >= PROGRESS_INTERVAL_SECONDS
            ):
                elapsed = now - start_time

                if frame_count > 0:
                    progress = (
                        processed_frames / frame_count
                    ) * 100
                    progress_text = f"{progress:.1f}%"
                else:
                    progress_text = (
                        f"{processed_frames} frames"
                    )

                print(
                    f"[VIDEO] Progress: {progress_text} | "
                    f"Frames: {processed_frames}/{frame_count} | "
                    f"YOLO runs: {analyzed_frames} | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

                last_progress_time = now

    finally:
        capture.release()
        writer.release()

    processing_time = (
        time.perf_counter() - start_time
    )

    processing_fps = (
        processed_frames / processing_time
        if processing_time > 0
        else 0
    )

    if processed_frames == 0:
        if output_path.exists():
            output_path.unlink()

        raise ValueError(
            "No frames could be processed from this video."
        )

    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise OSError(
            "The processed video output is missing or empty."
        )

    file_size_mb = (
        os.path.getsize(output_path)
        / (1024 * 1024)
    )

    print("\n" + "=" * 55)
    print("WILDLIFE VIDEO ANALYSIS COMPLETED")
    print(f"Frames written: {processed_frames}")
    print(f"Frames analysed by YOLO: {analyzed_frames}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing speed: {processing_fps:.2f} frames/second")
    print(f"Unique tracked objects: {len(unique_track_ids)}")
    print(f"Output size: {file_size_mb:.2f} MB")
    print(f"Output file: {output_path}")
    print("=" * 55)

    return {
        "filename": os.path.basename(video_path),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "analyzed_frames": analyzed_frames,
        "frame_skip": FRAME_SKIP,
        "duration_seconds": round(duration, 2),
        "processing_time_seconds": round(
            processing_time, 2
        ),
        "processing_fps": round(processing_fps, 2),
        "total_detections": total_detections,
        "unique_tracked_objects": len(unique_track_ids),
        "class_counts": class_counts,
        "output_path": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
    }
