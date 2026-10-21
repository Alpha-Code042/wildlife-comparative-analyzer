import time

import cv2

from algorithms.image_processor import get_detection_model


IMAGE_PATH = r"C:\Users\USER\Desktop\Analyser\zebra.jpg"


print("Loading image...")

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print(
    f"Image loaded: "
    f"{image.shape[1]}x{image.shape[0]}"
)

print()
print("Loading model...")

start = time.perf_counter()

model = get_detection_model()

model_load_time = (
    time.perf_counter() - start
)

print(
    f"Model loading time: "
    f"{model_load_time:.2f} seconds"
)

print()
print("Running inference...")

start = time.perf_counter()

results = model.predict(
    source=image,
    conf=0.30,
    verbose=False,
)

inference_time = (
    time.perf_counter() - start
)

print(
    f"Inference time: "
    f"{inference_time:.2f} seconds"
)

print()
print("Checking detections...")

result = results[0]

detection_count = (
    len(result.boxes)
    if result.boxes is not None
    else 0
)

print(
    f"Detections: {detection_count}"
)

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        print(
            f"- {model.names[class_id]}: "
            f"{confidence * 100:.2f}%"
        )

print()
print("==============================")
print("BENCHMARK COMPLETE")
print("==============================")