from pathlib import Path

from ultralytics import YOLO


# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Wildlife model
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "wildlife_detector"
    / "model.onnx"
)


def main():
    print("=" * 60)
    print("WILDLIFE MODEL TEST")
    print("=" * 60)

    print(f"\nModel path:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        print("\nERROR: model.onnx was not found.")
        return

    print("\nLoading wildlife model...")

    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as error:
        print("\nERROR: The model could not be loaded.")
        print(error)
        return

    print("\nModel loaded successfully.")

    print("\nAvailable wildlife classes:")
    print("-" * 40)

    names = model.names

    for class_id, class_name in names.items():
        print(f"{class_id}: {class_name}")

    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()