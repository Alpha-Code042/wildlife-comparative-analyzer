import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "videos"


def analyze_video_features(video_path):
    """
    Generate edge-detection, SIFT, and HOG-related visualizations
    from an uploaded video.

    Returns metadata compatible with the existing Flask
    video_comparison route and template.
    """

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = PROJECT_DIR / video_path

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file was not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            "The selected video could not be opened."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(
            "The video has invalid frame dimensions."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    outputs = {
        "edges": RESULTS_DIR / f"{stem}_edges.mp4",
        "sift": RESULTS_DIR / f"{stem}_sift_features.mp4",
        "hog": RESULTS_DIR / f"{stem}_hog_features.mp4",
    }

    writers = {}
    processed_frames = 0
    total_keypoints = 0
    total_hog_features = 0
    total_edge_percentage = 0.0

    start_time = time.perf_counter()

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for name, output_path in outputs.items():
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
            )

            if not writer.isOpened():
                writer.release()
                raise OSError(
                    f"Could not create the {name} output video."
                )

            writers[name] = writer

        sift = cv2.SIFT_create()

        hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            grayscale = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            # 1. Edge detection using Canny.
            edges = cv2.Canny(
                grayscale,
                100,
                200,
            )

            total_edge_percentage += (
                cv2.countNonZero(edges) / edges.size
            ) * 100

            edge_frame = cv2.cvtColor(
                edges,
                cv2.COLOR_GRAY2BGR,
            )

            cv2.putText(
                edge_frame,
                "Canny Edge Detection",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2. SIFT feature extraction.
            keypoints, descriptors = sift.detectAndCompute(
                grayscale,
                None,
            )

            sift_frame = cv2.drawKeypoints(
                frame,
                keypoints,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            total_keypoints += len(keypoints)

            cv2.putText(
                sift_frame,
                f"SIFT Keypoints: {len(keypoints)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # 3. HOG descriptor extraction.
            # HOG requires a 64 x 128 grayscale input window.
            resized = cv2.resize(
                grayscale,
                (64, 128),
            )

            hog_features = hog.compute(resized)

            if hog_features is not None:
                total_hog_features += len(hog_features)

            # Create a gradient-magnitude visualization.
            # This visualizes gradients; it is not an animal detector.
            gradient_x = cv2.Sobel(
                grayscale,
                cv2.CV_32F,
                1,
                0,
                ksize=1,
            )

            gradient_y = cv2.Sobel(
                grayscale,
                cv2.CV_32F,
                0,
                1,
                ksize=1,
            )

            magnitude, _ = cv2.cartToPolar(
                gradient_x,
                gradient_y,
            )

            normalized = cv2.normalize(
                magnitude,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype("uint8")

            hog_frame = cv2.cvtColor(
                normalized,
                cv2.COLOR_GRAY2BGR,
            )

            cv2.putText(
                hog_frame,
                "Gradient Magnitude (HOG-related)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Write each processed frame to its output video.
            writers["edges"].write(edge_frame)
            writers["sift"].write(sift_frame)
            writers["hog"].write(hog_frame)

            processed_frames += 1

        processing_time = time.perf_counter() - start_time

    finally:
        capture.release()

        for writer in writers.values():
            writer.release()

    if processed_frames == 0:
        for output_path in outputs.values():
            if output_path.exists():
                output_path.unlink()

        raise ValueError(
            "No frames could be processed from this video."
        )

    # Confirm that OpenCV actually created the output files.
    missing_outputs = [
        str(path)
        for path in outputs.values()
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing_outputs:
        for output_path in outputs.values():
            if output_path.exists():
                output_path.unlink()

        raise OSError(
            "One or more processed video files could not be created."
        )

    return {
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "original_frame_count": original_frame_count,
        "processed_frames": processed_frames,
        "processing_time_seconds": round(
            processing_time,
            4,
        ),
        "processing_fps": round(
            processed_frames / processing_time,
            2,
        ) if processing_time > 0 else 0,
        "average_keypoints_per_frame": round(
            total_keypoints / processed_frames,
            2,
        ),
        "average_hog_features_per_frame": round(
            total_hog_features / processed_frames,
            2,
        ),
        "average_edge_pixel_percentage": round(
            total_edge_percentage / processed_frames,
            2,
        ),
        "edges_filename": outputs["edges"].name,
        "sift_filename": outputs["sift"].name,
        "hog_filename": outputs["hog"].name,
    }