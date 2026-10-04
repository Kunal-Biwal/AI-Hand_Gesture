"""Wraps Google MediaPipe's HandLandmarker for the rest of the application.

main.py flips the webcam frame horizontally for a mirror-style preview,
which makes MediaPipe's handedness label anatomically backwards.
``get_hands`` corrects this by swapping Left/Right.
"""

import os
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from . import config


class HandDetection:
    """Plain-Python container for one detected hand (no MediaPipe types leak out)."""

    __slots__ = ("landmarks", "label", "score")

    def __init__(self, landmarks, label, score):
        self.landmarks = landmarks  # 21 (x, y, z) tuples, normalized to [0, 1]
        self.label = label  # "Left" or "Right", corrected for mirroring
        self.score = score  # handedness confidence, 0..1


class HandTracker:
    """Real-time hand detector/tracker built on MediaPipe's HandLandmarker."""

    def __init__(
        self,
        model_path=config.HAND_LANDMARKER_MODEL_PATH,
        model_url=config.HAND_LANDMARKER_MODEL_URL,
        max_hands=config.MAX_NUM_HANDS,
        detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
        tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    ):
        self._ensure_model_downloaded(model_path, model_url)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # MediaPipe requires monotonically increasing timestamps.
        self._start_time = time.time()
        self._last_timestamp_ms = -1

    # ----------------------------------------------------------------
    @staticmethod
    def _ensure_model_downloaded(model_path, model_url):
        """Download the hand landmark model once and cache it on disk."""
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            return

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(
            "[HandTracker] Hand landmark model not found locally.\n"
            f"[HandTracker] Downloading it once from:\n  {model_url}\n"
            f"[HandTracker] Saving to: {model_path}\n"
            "[HandTracker] This requires an internet connection only this "
            "first time; after that it is cached and used fully offline."
        )
        try:
            urllib.request.urlretrieve(model_url, model_path)
        except Exception as exc:  # noqa: BLE001 - we re-raise a clearer error
            raise RuntimeError(
                "Could not download the MediaPipe hand landmark model "
                f"({exc}). Connect to the internet once, or manually place "
                f"a 'hand_landmarker.task' file at: {model_path}"
            ) from exc
        print("[HandTracker] Model downloaded successfully.")

    # ----------------------------------------------------------------
    def find_hands(self, frame_bgr):
        """Run detection on one BGR frame. Returns the raw MediaPipe HandLandmarkerResult."""
        rgb_frame = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    # ----------------------------------------------------------------
    @staticmethod
    def get_hands(result):
        """Convert a HandLandmarkerResult into a list of ``HandDetection``."""
        hands = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            category = handedness[0]
            # Correct for the mirror flip applied before detection.
            raw_label = category.category_name
            corrected_label = "Left" if raw_label == "Right" else "Right"
            hands.append(HandDetection(landmarks, corrected_label, category.score))
        return hands

    # ----------------------------------------------------------------
    @staticmethod
    def draw_landmarks(frame_bgr, result, hands):
        """Draw all 21 landmarks and the hand skeleton for every detected hand."""
        connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for hand_landmarks, hand in zip(result.hand_landmarks, hands):
            color = (
                config.COLOR_LANDMARK_RIGHT
                if hand.label == "Right"
                else config.COLOR_LANDMARK_LEFT
            )
            landmark_spec = vision.drawing_utils.DrawingSpec(
                color=color, thickness=2, circle_radius=3
            )
            connection_spec = vision.drawing_utils.DrawingSpec(
                color=config.COLOR_CONNECTION, thickness=2
            )
            vision.drawing_utils.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                connections,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec,
            )

    # ----------------------------------------------------------------
    def close(self):
        """Release the underlying MediaPipe model. Call on shutdown."""
        try:
            self._landmarker.close()
        except Exception:
            pass
