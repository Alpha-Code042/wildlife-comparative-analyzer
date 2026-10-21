import math


class CentroidTracker:
    """
    Simple centroid-based object tracker.

    Objects are matched between frames using:
    - Animal class
    - Distance between object centroids

    Each tracked object receives a unique ID.
    """

    def __init__(self, max_distance=100, max_missed=3):
        self.next_id = 1

        self.tracks = {}

        self.max_distance = max_distance

        self.max_missed = max_missed

    def _calculate_centroid(self, bbox):
        """
        Calculate the center point of a bounding box.
        """

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        return center_x, center_y

    def _calculate_distance(
        self,
        point_a,
        point_b
    ):
        """
        Calculate Euclidean distance between two points.
        """

        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2
            + (point_a[1] - point_b[1]) ** 2
        )

    def _create_track(
        self,
        detection
    ):
        """
        Create a new tracked object.
        """

        track_id = self.next_id

        self.next_id += 1

        centroid = self._calculate_centroid(
            detection["bbox"]
        )

        self.tracks[track_id] = {
            "id": track_id,
            "class_name": detection["class_name"],
            "bbox": detection["bbox"],
            "centroid": centroid,
            "confidence": detection["confidence"],
            "missed": 0,
        }

    def update(
        self,
        detections
    ):
        """
        Match current detections with existing tracks.

        Returns the currently active tracked objects.
        """

        if not detections:

            for track in self.tracks.values():
                track["missed"] += 1

            self._remove_lost_tracks()

            return list(
                self.tracks.values()
            )

        detection_centroids = []

        for detection in detections:

            centroid = self._calculate_centroid(
                detection["bbox"]
            )

            detection_centroids.append(
                centroid
            )

        matched_tracks = set()

        matched_detections = set()

        possible_matches = []

        # Compare every existing track with
        # every current detection.
        for track_id, track in self.tracks.items():

            for detection_index, detection in enumerate(
                detections
            ):

                if (
                    detection["class_name"]
                    != track["class_name"]
                ):
                    continue

                distance = self._calculate_distance(
                    track["centroid"],
                    detection_centroids[
                        detection_index
                    ]
                )

                if distance <= self.max_distance:

                    possible_matches.append(
                        (
                            distance,
                            track_id,
                            detection_index
                        )
                    )

        # Match closest objects first.
        possible_matches.sort(
            key=lambda item: item[0]
        )

        for (
            distance,
            track_id,
            detection_index
        ) in possible_matches:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            detection = detections[
                detection_index
            ]

            centroid = detection_centroids[
                detection_index
            ]

            self.tracks[track_id][
                "bbox"
            ] = detection["bbox"]

            self.tracks[track_id][
                "centroid"
            ] = centroid

            self.tracks[track_id][
                "confidence"
            ] = detection["confidence"]

            self.tracks[track_id][
                "missed"
            ] = 0

            matched_tracks.add(track_id)

            matched_detections.add(
                detection_index
            )

        # Increase missed count for tracks
        # that were not matched.
        for track_id in list(
            self.tracks.keys()
        ):

            if track_id not in matched_tracks:

                self.tracks[track_id][
                    "missed"
                ] += 1

        self._remove_lost_tracks()

        # Create new tracks for detections
        # that were not matched.
        for detection_index, detection in enumerate(
            detections
        ):

            if detection_index not in matched_detections:

                self._create_track(
                    detection
                )

        return list(
            self.tracks.values()
        )

    def _remove_lost_tracks(self):
        """
        Remove tracks that have been missing
        for too many consecutive updates.
        """

        lost_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]

        for track_id in lost_ids:

            del self.tracks[track_id]

    def get_track_count(self):
        """
        Return the number of currently active tracks.
        """

        return len(self.tracks) 