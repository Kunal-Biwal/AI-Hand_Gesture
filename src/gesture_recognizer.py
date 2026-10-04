"""Turns 21 hand landmarks into a gesture label using geometry (distances/ratios).

All thresholds are ratios, not raw coordinates, so gestures keep working
regardless of how close the hand is to the camera.
"""

from . import config
from .utils import (
    WRIST,
    THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
    distance, hand_scale,
)


class Gesture:
    """String constants for every gesture this project recognizes."""

    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    VICTORY = "VICTORY"
    INDEX_UP = "INDEX_UP"
    PINCH = "PINCH"
    RIGHT_CLICK = "RIGHT_CLICK"


def _is_extended(landmarks, tip_idx, pip_idx, mcp_idx, ratio):
    """True if the fingertip is farther from the wrist than its PIP/MCP joints by ``ratio``."""
    wrist = landmarks[WRIST]
    d_tip = distance(wrist, landmarks[tip_idx])
    d_pip = distance(wrist, landmarks[pip_idx])
    d_mcp = distance(wrist, landmarks[mcp_idx])
    return d_tip > d_pip * ratio and d_tip > d_mcp


def is_index_up(landmarks):
    """True if the index finger is extended."""
    return _is_extended(
        landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, config.FINGER_EXTENSION_RATIO
    )


def is_middle_up(landmarks):
    """True if the middle finger is extended."""
    return _is_extended(
        landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, config.FINGER_EXTENSION_RATIO
    )


def is_ring_up(landmarks):
    """True if the ring finger is extended."""
    return _is_extended(
        landmarks, RING_TIP, RING_PIP, RING_MCP, config.FINGER_EXTENSION_RATIO
    )


def is_pinky_up(landmarks):
    """True if the pinky finger is extended."""
    return _is_extended(
        landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP, config.FINGER_EXTENSION_RATIO
    )


def is_thumb_extended(landmarks):
    """True if the thumb is held out away from the palm, in any direction."""
    return _is_extended(
        landmarks, THUMB_TIP, THUMB_IP, THUMB_MCP, config.THUMB_EXTENSION_RATIO
    )


def is_thumb_up(landmarks):
    """True if the thumb is extended and pointing upward on screen."""
    if not is_thumb_extended(landmarks):
        return False
    return landmarks[THUMB_TIP][1] < landmarks[THUMB_MCP][1]


def is_thumb_down(landmarks):
    """True if the thumb is extended AND pointing downward on screen."""
    if not is_thumb_extended(landmarks):
        return False
    return landmarks[THUMB_TIP][1] > landmarks[THUMB_MCP][1]


def is_thumb_folded(landmarks):
    """True if the thumb is tucked into a closed fist.

    The thumb's CMC joint lets it rotate across the palm, so a curled
    thumb can stay as far from the wrist as an extended one -- unlike the
    other fingers, distance-from-wrist doesn't work for it. Instead this
    checks proximity to INDEX_MCP, where a folded thumb actually rests.
    """
    raw = distance(landmarks[THUMB_TIP], landmarks[INDEX_MCP])
    return (raw / hand_scale(landmarks)) < config.THUMB_FOLD_RATIO


def get_finger_states(landmarks):
    """Return a dict of {finger_name: bool} describing which fingers are up."""
    return {
        "thumb": is_thumb_extended(landmarks),
        "index": is_index_up(landmarks),
        "middle": is_middle_up(landmarks),
        "ring": is_ring_up(landmarks),
        "pinky": is_pinky_up(landmarks),
    }


def pinch_distance_ratio(landmarks):
    """Thumb-tip-to-index-tip distance, normalized by hand scale."""
    raw = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    return raw / hand_scale(landmarks)


def is_pinch(landmarks, threshold=None):
    """True if the thumb tip and index tip are touching (normalized)."""
    threshold = config.PINCH_DISTANCE_RATIO if threshold is None else threshold
    return pinch_distance_ratio(landmarks) < threshold


class GestureRecognizer:
    """Stateless, rule-based gesture classifier."""

    def recognize(self, landmarks):
        """``landmarks``: list of 21 (x, y, z) normalized tuples. Returns a Gesture string."""
        if landmarks is None or len(landmarks) < 21:
            return Gesture.NONE

        # Pinch: thumb + index touching. Takes priority since it drives
        # mouse clicks/drag and must fire even if other fingers drift up.
        if is_pinch(landmarks):
            return Gesture.PINCH

        fingers = get_finger_states(landmarks)
        index, middle, ring, pinky = (
            fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]
        )

        # Fist: index/middle/ring/pinky curled + thumb folded.
        other_four_curled = not index and not middle and not ring and not pinky
        if other_four_curled and is_thumb_folded(landmarks):
            return Gesture.FIST

        # Open palm: all five fingers extended.
        if index and middle and ring and pinky and fingers["thumb"]:
            return Gesture.OPEN_PALM

        # Thumbs up/down: only the thumb extended.
        thumb_only = fingers["thumb"] and other_four_curled
        if thumb_only:
            if is_thumb_up(landmarks):
                return Gesture.THUMBS_UP
            if is_thumb_down(landmarks):
                return Gesture.THUMBS_DOWN

        # Victory: index + middle up, ring + pinky down.
        if index and middle and not ring and not pinky:
            return Gesture.VICTORY

        # Right click: pinky only.
        if pinky and not index and not middle and not ring:
            return Gesture.RIGHT_CLICK

        # Index only: virtual mouse cursor.
        if index and not middle and not ring and not pinky:
            return Gesture.INDEX_UP

        return Gesture.NONE
