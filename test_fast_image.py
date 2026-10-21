import time

from algorithms.image_processor import analyze_image


image_path = r"C:\Users\USER\Desktop\Analyser\zebra.jpg"

start = time.perf_counter()

result = analyze_image(image_path)

elapsed = time.perf_counter() - start

print()
print("================================")
print("WILDLIFE IMAGE TEST")
print("================================")
print(f"Processing time: {elapsed:.2f} seconds")
print(f"Detections: {result['detection_count']}")

for detection in result["detections"]:
    print(
        f"- {detection['class_name']}: "
        f"{detection['confidence']}%"
    )

print()
print(
    "Result:",
    result["annotated_image_path"]
)