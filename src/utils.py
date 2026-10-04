"""Small, reusable, dependency-free helpers shared across the project."""

import math
import time
from collections import deque

import cv2


# MediaPipe's 21 hand landmark indices.
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def distance(point_a, point_b):
    """Euclidean distance between two (x, y) or (x, y, z) points."""
    return math.dist(point_a[:2], point_b[:2])


def hand_scale(landmarks):
    """Wrist-to-middle-knuckle distance, used as a per-hand ruler for scale-invariant ratios."""
    return distance(landmarks[WRIST], landmarks[MIDDLE_MCP]) or 1e-6


def clamp(value, lo, hi):
    """Clamp ``value`` into the inclusive [lo, hi] range."""
    return max(lo, min(hi, value))


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t


class FPSCounter:
    """Smoothed frames-per-second counter using a rolling window of timestamps."""

    def __init__(self, window_size=30):
        self._timestamps = deque(maxlen=window_size)

    def tick(self):
        """Call once per processed frame. Returns the current smoothed FPS."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class GestureStabilizer:
    """Debounces a noisy per-frame gesture label into a stable, cooldown-aware trigger.

    ``can_trigger_repeat`` fires repeatedly every ``cooldown`` seconds
    while a gesture is held (e.g. volume up/down). ``can_trigger_once``
    fires once per stable "entry" into a gesture and re-arms only after
    the gesture is released (e.g. click, play/pause).
    """

    def __init__(self, stability_frames, repeat_cooldown, one_shot_cooldown):
        self.stability_frames = stability_frames
        self.repeat_cooldown = repeat_cooldown
        self.one_shot_cooldown = one_shot_cooldown
        self.reset()

    def reset(self):
        """Clear all history -- used on startup and when the user presses R."""
        self._current_candidate = None
        self._consecutive_count = 0
        self._stable_gesture = None
        self._last_repeat_time = {}
        self._last_one_shot_time = {}
        self._last_fired_one_shot_gesture = None

    def update(self, raw_gesture):
        """Feed this frame's raw gesture in. Returns the stable gesture, or None."""
        if raw_gesture == self._current_candidate:
            self._consecutive_count += 1
        else:
            self._current_candidate = raw_gesture
            self._consecutive_count = 1

        if self._consecutive_count >= self.stability_frames:
            self._stable_gesture = raw_gesture
        else:
            self._stable_gesture = None

        if self._stable_gesture is None:
            # Re-arm one-shot actions once nothing is held steadily.
            self._last_fired_one_shot_gesture = None

        return self._stable_gesture

    def can_trigger_repeat(self, gesture):
        """True at most once every ``repeat_cooldown`` seconds per gesture."""
        now = time.time()
        last = self._last_repeat_time.get(gesture, 0.0)
        if now - last < self.repeat_cooldown:
            return False
        self._last_repeat_time[gesture] = now
        return True

    def can_trigger_once(self, gesture):
        """True exactly once per "entry" into this stable gesture."""
        if gesture == self._last_fired_one_shot_gesture:
            return False
        now = time.time()
        last = self._last_one_shot_time.get(gesture, 0.0)
        if now - last < self.one_shot_cooldown:
            return False
        self._last_fired_one_shot_gesture = gesture
        self._last_one_shot_time[gesture] = now
        return True


class ExponentialSmoother:
    """1-D exponential moving average, used for x/y cursor smoothing."""

    def __init__(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor
        self._value = None

    def reset(self):
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = new_value
        else:
            self._value = lerp(self._value, new_value, self.smoothing_factor)
        return self._value


def put_text(frame, text, origin, color, font, scale=0.6, thickness=1):
    """Thin wrapper around cv2.putText with the project's default font."""
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, top_left, bottom_right, color, alpha=0.55):
    """Draw a semi-transparent filled rectangle, used as a text backdrop."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
