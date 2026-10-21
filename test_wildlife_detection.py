from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
RESULT_DIR = PROJECT_DIR / "results"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not UPLOAD_DIR.exists():
        print("The uploads folder does not exist.")
        return

    images = [
        path for path in UPLOAD_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print("No supported images found in the uploads folder.")
        return

    # Select the most recently modified image.
    image_path = max(images, key=lambda path: path.stat().st_mtime)

    print(f"Testing image: {image_path.name}")
    print("Loading YOLO model...")

    model = YOLO("yolo26n.pt")

    print("Detecting objects...")

    results = model.predict(
        source=str(image_path),
        conf=0.15,
        save=True,
        project=str(RESULT_DIR),
        name="wildlife_detection",
        exist_ok=True,
        verbose=False,
    )

    result = results[0]

    print("\nDetection results:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item()) * 100

            print(
                f"- {class_name}: "
                f"{confidence:.2f}% confidence"
            )

    print("\nAnnotated image saved in:")
    print(RESULT_DIR / "wildlife_detection")


if __name__ == "__main__":
    main()

