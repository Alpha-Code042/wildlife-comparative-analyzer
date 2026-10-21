
import os
from algorithms.image_processor import analyze_image

uploads_folder = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

image_path = None

for filename in os.listdir(uploads_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(
            uploads_folder,
            filename
        )
        break

if image_path is None:
    print("No uploaded image was found.")
else:
    result = analyze_image(image_path)

    print("Image processing successful!")
    print("Filename:", result["filename"])
    print("Width:", result["width"])
    print("Height:", result["height"])
    print("Channels:", result["channels"])
    print("File size (KB):", result["file_size_kb"])
    print("Grayscale conversion successful.")