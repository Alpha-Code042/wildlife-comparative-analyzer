
import os
import time
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "classical"


def analyze_classical_image(image_path):
    """
    Apply traditional OpenCV image processing.

    Generates:
    1. A grayscale version of the image.
    2. An edge-detection version using the Canny algorithm.

    This function does not detect or classify wildlife species.
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
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    file_size_kb = round(os.path.getsize(image_path) / 1024, 2)

    # Convert the original image to grayscale.
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using the Canny edge-detection algorithm.
    edges = cv2.Canny(grayscale, 100, 200)

    # Calculate the percentage of pixels classified as edges.
    total_pixels = width * height
    edge_pixels = cv2.countNonZero(edges)
    edge_pixel_percentage = round(
        (edge_pixels / total_pixels) * 100, 2
    )

    # Create the output directory if it does not exist.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use the uploaded image's filename to name the outputs.
    image_stem = image_path.stem

    grayscale_path = RESULTS_DIR / f"{image_stem}_grayscale.jpg"
    edges_path = RESULTS_DIR / f"{image_stem}_edges.jpg"

    # Save both processed images.
    grayscale_saved = cv2.imwrite(
        str(grayscale_path), grayscale
    )
    edges_saved = cv2.imwrite(
        str(edges_path), edges
    )

    if not grayscale_saved or not edges_saved:
        raise OSError(
            "Could not save the processed images."
        )

    processing_time = round(
        time.perf_counter() - start_time, 4
    )

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_kb": file_size_kb,
        "processing_time_seconds": processing_time,
        "grayscale_image_path": str(grayscale_path),
        "edges_image_path": str(edges_path),
        "edge_pixel_percentage": edge_pixel_percentage,
    }
