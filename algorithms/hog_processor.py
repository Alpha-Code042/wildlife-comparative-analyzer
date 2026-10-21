import os
import time
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "hog"


def analyze_hog_image(image_path):
    """
    Extract Histogram of Oriented Gradients (HOG)
    features from an image.

    HOG describes edge and gradient patterns.
    It does not independently detect or classify
    wildlife species.
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

    # HOG requires image dimensions compatible with
    # its window and block configuration.
    window_width = 64
    window_height = 128

    resized = cv2.resize(
        grayscale,
        (window_width, window_height)
    )

    hog = cv2.HOGDescriptor(
        _winSize=(window_width, window_height),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    features = hog.compute(resized)

    # Create a visual representation of image gradients.
    gradient_x = cv2.Sobel(
        resized, cv2.CV_32F, 1, 0, ksize=1
    )

    gradient_y = cv2.Sobel(
        resized, cv2.CV_32F, 0, 1, ksize=1
    )

    magnitude, angle = cv2.cartToPolar(
        gradient_x, gradient_y, angleInDegrees=True
    )

    gradient_image = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype("uint8")

    gradient_image = cv2.cvtColor(
        gradient_image, cv2.COLOR_GRAY2BGR
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_filename = (
        f"{image_path.stem}_hog_features.jpg"
    )

    output_path = RESULTS_DIR / output_filename

    if not cv2.imwrite(
        str(output_path), gradient_image
    ):
        raise OSError(
            "Could not save the HOG feature image."
        )

    processing_time = round(
        time.perf_counter() - start_time, 4
    )

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "file_size_kb": file_size_kb,
        "feature_vector_length": (
            0 if features is None else len(features)
        ),
        "processing_time_seconds": processing_time,
        "feature_image_filename": output_filename,
        "feature_image_path": str(output_path),
    }
