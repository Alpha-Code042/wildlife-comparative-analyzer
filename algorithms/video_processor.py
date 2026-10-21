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

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "videos"
)

# Analyze every 3rd frame to reduce processing time
FRAME_SKIP = 3

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.30


def draw_tracking_annotations(frame, tracks):
    """
    Draw bounding boxes and tracking IDs on a video frame.
    """

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

        label_y = max(y1 - 10, 20)

        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def analyze_video(video_path):
    """
    Analyze a wildlife video using the Sub-Saharan Africa
    wildlife detection model and the centroid tracker.
    """

    video_path = os.path.realpath(video_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
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

    if fps <= 0:
        fps = 25.0

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        f"{Path(video_path).stem}_detected.mp4"
    )

    output_path = (
        RESULTS_DIR
        / output_filename
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise OSError(
            "The processed video could not be created."
        )

    # Load the new wildlife ONNX model
    model = get_detection_model()

    # Get only the wildlife classes.
    # This automatically excludes Person and Car.
    wildlife_class_ids = [
        class_id
        for class_id, class_name
        in model.names.items()
        if class_name in WILDLIFE_CLASSES
    ]

    # Tracker
    max_distance = max(
        50,
        int(max(width, height) * 0.08)
    )

    tracker = CentroidTracker(
        max_distance=max_distance,
        max_missed=3,
    )

    total_detections = 0
    class_counts = {}

    processed_frames = 0
    analyzed_frames = 0

    frame_index = 0

    unique_track_ids = set()

    start_time = time.perf_counter()

    last_annotated_frame = None

    try:

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_index += 1

            # Analyze every 3rd frame
            should_analyze = (
                frame_index % FRAME_SKIP == 1
            )

            if should_analyze:

                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    classes=wildlife_class_ids,
                    verbose=False,
                )

                result = results[0]

                detections = []

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        raw_class_name = (
                            model.names[class_id]
                        )

                        # Convert scientific model name
                        # to common wildlife name.
                        class_name = WILDLIFE_CLASSES.get(
                            raw_class_name,
                            raw_class_name,
                        )

                        x1, y1, x2, y2 = (
                            int(value)
                            for value in box.xyxy[0].tolist()
                        )

                        detections.append(
                            {
                                "class_name": class_name,
                                "raw_class_name": raw_class_name,
                                "confidence": confidence,
                                "bbox": (
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                ),
                            }
                        )

                        total_detections += 1

                        class_counts[class_name] = (
                            class_counts.get(
                                class_name,
                                0,
                            )
                            + 1
                        )

                # Update tracker
                tracks = tracker.update(
                    detections
                )

                # Record unique tracking IDs
                for track in tracks:
                    unique_track_ids.add(
                        track["id"]
                    )

                # Draw tracking information
                last_annotated_frame = (
                    draw_tracking_annotations(
                        frame,
                        tracks,
                    )
                )

                analyzed_frames += 1

            # Use the latest analyzed frame
            # for skipped frames.
            output_frame = (
                last_annotated_frame
                if last_annotated_frame is not None
                else frame
            )

            writer.write(output_frame)

            processed_frames += 1

    finally:

        capture.release()
        writer.release()

    processing_time = (
        time.perf_counter()
        - start_time
    )

    processing_fps = (
        processed_frames / processing_time
        if processing_time > 0
        else 0
    )

    file_size_mb = (
        os.path.getsize(output_path)
        / (1024 * 1024)
        if output_path.exists()
        else 0
    )

    return {
        "filename": os.path.basename(
            video_path
        ),

        "width": width,

        "height": height,

        "fps": round(fps, 2),

        "frame_count": frame_count,

        "processed_frames": processed_frames,

        "analyzed_frames": analyzed_frames,

        "frame_skip": FRAME_SKIP,

        "duration_seconds": round(
            duration,
            2,
        ),

        "processing_time_seconds": round(
            processing_time,
            2,
        ),

        "processing_fps": round(
            processing_fps,
            2,
        ),

        "total_detections": total_detections,

        "unique_tracked_objects": len(
            unique_track_ids
        ),

        "class_counts": class_counts,

        "output_path": str(
            output_path
        ),

        "file_size_mb": round(
            file_size_mb,
            2,
        ),
    }