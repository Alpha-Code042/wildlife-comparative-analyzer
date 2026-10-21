import os
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "wildlife_detector"
    / "model.onnx"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "detections"
)


WILDLIFE_CLASSES = {
    "AcinonyxJubatusJubatus": "Cheetah",
    "ConnochaetesTaurinus": "Wildebeest",
    "EquusQuagga": "Zebra",
    "PapioSp": "Baboon",
    "GiraffaCamelopardalis": "Giraffe",
    "LoxodontaAfricana": "African Elephant",
    "PantheraLeo": "Lion",
    "Rhino": "Rhino",
    "SyncerusCaffer": "African Buffalo",
    "TragelaphusOryx": "Gemsbok",
    "CrocutaCrocuta": "Spotted Hyena",
    "AepycerosMelampus": "Impala",
    "CanisMesomelas": "Black-backed Jackal",
    "HippotragusEquinus": "Roan Antelope",
    "OryctolagusCuniculus": "Rabbit",
    "PhacochoerusAfricanus": "Warthog",
    "PanTroglodytes": "Chimpanzee",
    "SmutsiaGigantea": "Giant Pangolin",
    "HystrixCristata": "Porcupine",
    "OrycteropusAfer": "Aardvark",
    "HippopotamusAmphibious": "Hippopotamus",
    "OryxGazella": "Oryx",
    "StruthioCamelus": "Ostrich",
    "AlcelaphusBuselaphus": "Hartebeest",
    "KobusEllipsiprymnus": "Waterbuck",
    "GorillaSp": "Gorilla",
    "TragelaphusEurycerus": "Bongo",
    "KobusKob": "Kob",
    "NumidaMeleagris": "Guineafowl",
    "SylvicapraGrimmia": "Common Duiker",
    "GenettaSp": "Genet",
    "CricetomysSp": "Giant Rat",
    "ChlorocebusTantalus": "Vervet Monkey",
}


_model = None


def get_detection_model():
    """
    Load the wildlife ONNX model once and keep it
    in memory for subsequent analyses.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Wildlife model not found: {MODEL_PATH}"
            )

        _model = YOLO(
            str(MODEL_PATH)
        )

    return _model


def analyze_image(image_path):
    """
    Analyze an uploaded wildlife image using
    the Sub-Saharan Africa wildlife detector.
    """

    image_path = os.path.realpath(
        image_path
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            "The selected file could not "
            "be read as an image."
        )

    height, width = (
        image.shape[:2]
    )

    channels = (
        1
        if image.ndim == 2
        else image.shape[2]
    )

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    model = get_detection_model()

    # Select only the wildlife classes.
    wildlife_class_ids = [
        class_id
        for class_id, class_name
        in model.names.items()
        if class_name in WILDLIFE_CLASSES
    ]

    results = model.predict(
        source=image,
        conf=0.30,
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

            class_name = (
                WILDLIFE_CLASSES.get(
                    raw_class_name,
                    raw_class_name,
                )
            )

            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0].tolist()
            )

            detections.append(
                {
                    "class_name": class_name,

                    "raw_class_name": (
                        raw_class_name
                    ),

                    "confidence": round(
                        confidence * 100,
                        2,
                    ),

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    # Ultralytics handles the model's
    # output decoding and NMS.
    annotated_image = result.plot()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        f"{Path(image_path).stem}"
        f"_detected.jpg"
    )

    annotated_image_path = (
        RESULTS_DIR
        / output_filename
    )

    saved = cv2.imwrite(
        str(annotated_image_path),
        annotated_image,
    )

    if not saved:
        raise OSError(
            "The annotated image could "
            "not be saved."
        )

    return {
        "filename": os.path.basename(
            image_path
        ),

        "width": width,

        "height": height,

        "channels": channels,

        "file_size_kb": round(
            os.path.getsize(
                image_path
            ) / 1024,
            2,
        ),

        "grayscale": grayscale,

        "image": image,

        "detections": detections,

        "detection_count": len(
            detections
        ),

        "annotated_image_path": str(
            annotated_image_path
        ),
    }