"""Centralized configuration. Every tunable constant lives here."""

import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


HAND_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720
CAMERA_FPS_REQUEST = 30
# On Windows, the DirectShow backend opens noticeably faster and more
# reliably than the default MSMF backend for most webcams.
USE_DSHOW_ON_WINDOWS = True


# MediaPipe Hand Detection
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.65
MIN_PRESENCE_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.6

# Gesture geometry thresholds (scale-invariant ratios; see gesture_recognizer.py)

# A finger counts as "extended" when its tip is this many times farther
# from the wrist than its own PIP joint.
FINGER_EXTENSION_RATIO = 1.05

# Thumb uses the same ratio test against its own IP/MCP joints.
THUMB_EXTENSION_RATIO = 1.05

# Pinch = distance(thumb_tip, index_tip) / hand_scale must be below this.
PINCH_DISTANCE_RATIO = 0.35

# distance(thumb_tip, index_mcp) / hand_scale below this = thumb folded into fist.
THUMB_FOLD_RATIO = 0.55

# Consecutive raw pinch frames required before a click/drag can start in
# Virtual Mouse mode. Lighter than GESTURE_STABILITY_FRAMES so clicks stay responsive.
PINCH_CONFIRM_FRAMES = 3

# Gesture stabilization (see utils.GestureStabilizer)

# Consecutive frames a gesture must hold before it counts as "stable".
GESTURE_STABILITY_FRAMES = 8

# Minimum seconds between triggers of a repeatable action (volume up/down)
# while the gesture is held.
REPEAT_ACTION_COOLDOWN = 0.6

# Minimum seconds between triggers of a one-shot action (click, play/pause, mute).
ONE_SHOT_ACTION_COOLDOWN = 0.35

# Virtual mouse

# Fraction of the frame trimmed from each edge to form the active control
# rectangle that maps to the full screen.
MOUSE_ACTIVE_REGION_MARGIN = 0.15

# EMA weight applied to each new cursor sample. 1.0 = no smoothing.
MOUSE_SMOOTHING_FACTOR = 0.35

# Multiplies the mapped screen delta from center; > 1.0 = more sensitive.
MOUSE_SENSITIVITY = 1.3

# Minimum seconds between two left/right clicks.
CLICK_COOLDOWN = 0.4

# Normalized cursor movement while pinching above which it becomes a drag.
DRAG_MOVEMENT_THRESHOLD = 0.012

# Volume control
VOLUME_STEP_PERCENT = 2
VOLUME_KEY_PRESSES_PER_STEP = 1

# UI
WINDOW_NAME = "AI Hand Gesture Control"
FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BG_PANEL = (20, 20, 20)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_ACCENT = (0, 220, 255)
COLOR_TEXT_GOOD = (80, 220, 80)
COLOR_TEXT_WARN = (60, 170, 255)
COLOR_LANDMARK_RIGHT = (0, 220, 255)
COLOR_LANDMARK_LEFT = (255, 150, 0)
COLOR_CONNECTION = (200, 200, 200)

UI_PANEL_WIDTH = 300

# Application modes
MODE_GESTURE_CONTROL = "GESTURE CONTROL"
MODE_VIRTUAL_MOUSE = "VIRTUAL MOUSE"
