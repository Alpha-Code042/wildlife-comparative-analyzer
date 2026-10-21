
import os
import time
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "sift"


def analyze_sift_image(image_path):
    """
    Extract SIFT keypoints and descriptors from a wildlife image.

    SIFT identifies distinctive visual features. It does not
    independently detect or classify wildlife species.
    """
    start_time = time.perf_counter()

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file was not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "The selected file could not be read as an image."
        )

    height, width = image.shape[:2]
    file_size_kb = round(
        os.path.getsize(image_path) / 1024, 2
    )

    grayscale = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is unavailable in this OpenCV installation."
        )

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        grayscale, None
    )

    # Draw the extracted keypoints on the original image.
    feature_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_filename = (
        f"{image_path.stem}_sift_features.jpg"
    )

    output_path = RESULTS_DIR / output_filename

    if not cv2.imwrite(str(output_path), feature_image):
        raise OSError(
            "Could not save the SIFT feature image."
        )

    descriptor_count = (
        0 if descriptors is None else len(descriptors)
    )

    processing_time = round(
        time.perf_counter() - start_time, 4
    )

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "file_size_kb": file_size_kb,
        "keypoint_count": len(keypoints),
        "descriptor_count": descriptor_count,
        "descriptor_dimensions": (
            0 if descriptors is None
            else descriptors.shape[1]
        ),
        "processing_time_seconds": processing_time,
        "feature_image_filename": output_filename,
        "feature_image_path": str(output_path),
    }
