from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "wildlife_detector"
    / "model.onnx"
)

VIDEO_PATH = PROJECT_DIR / "lion.mp4"

OUTPUT_DIR = PROJECT_DIR / "results" / "wildlife_test"
OUTPUT_PATH = OUTPUT_DIR / "lion_test_result.mp4"


# Model classes that we want to display using normal names.
CLASS_NAMES = {
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
    "HippopotamusAmphibius": "Hippopotamus",
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


def main():
    print("=" * 60)
    print("WILDLIFE VIDEO DETECTION TEST")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found:")
        print(MODEL_PATH)
        return

    if not VIDEO_PATH.exists():
        print(f"\nERROR: Video not found:")
        print(VIDEO_PATH)
        return

    print("\nLoading wildlife model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    print("\nOpening video...")

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        print("ERROR: Could not open lion.mp4.")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0
    )

    print(f"\nVideo information:")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        print("\nERROR: Could not create output video.")
        return

    detected_species = {}

    frame_number = 0

    print("\nProcessing video...")
    print("This may take some time on CPU.")
    print()

    try:
        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            results = model.predict(
                source=frame,
                conf=0.30,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    raw_name = model.names[class_id]

                    display_name = CLASS_NAMES.get(
                        raw_name,
                        raw_name
                    )

                    detected_species[
                        display_name
                    ] = (
                        detected_species.get(
                            display_name,
                            0
                        ) + 1
                    )

                    x1, y1, x2, y2 = (
                        int(value)
                        for value in box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{display_name} "
                        f"{confidence * 100:.1f}%"
                    )

                    label_y = max(
                        y1 - 10,
                        20
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

            writer.write(frame)

            if frame_number % 30 == 0:

                percentage = (
                    frame_number
                    / frame_count
                    * 100
                    if frame_count > 0
                    else 0
                )

                print(
                    f"Processed frame "
                    f"{frame_number}/{frame_count} "
                    f"({percentage:.1f}%)"
                )

    finally:

        capture.release()
        writer.release()

    print()
    print("=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    if detected_species:

        for species, count in sorted(
            detected_species.items()
        ):
            print(
                f"{species}: "
                f"{count} detections"
            )

    else:

        print("No wildlife detected.")

    print()
    print(f"Annotated video saved to:")
    print(OUTPUT_PATH)

    print()
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()