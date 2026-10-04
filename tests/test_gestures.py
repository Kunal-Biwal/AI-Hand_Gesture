"""Unit tests for the rule-based gesture recognition pipeline.

Uses hand-built, synthetic 21-point landmark lists -- no webcam, MediaPipe
model, or mouse/keyboard/volume side effects.

Run with: python -m unittest discover -s tests -v
"""

import time
import unittest

from unittest import mock

from src.gesture_recognizer import (
    Gesture,
    GestureRecognizer,
    get_finger_states,
    is_pinch,
    is_thumb_down,
    is_thumb_folded,
    is_thumb_up,
)
from src.mouse_controller import MouseController
from src.utils import (
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
    GestureStabilizer,
    clamp,
    distance,
    hand_scale,
)


# Synthetic landmark builders. All "standard" hands share wrist (0.5, 0.95),
# with fingers reaching upward (decreasing y) when extended.
WRIST_XY = (0.5, 0.95)


def _extended_chain(x):
    """(mcp, pip, dip, tip) for a STRAIGHT, extended finger at column x."""
    mcp = (x, 0.70)
    pip = (x, 0.55)
    tip = (x, 0.35)
    dip = ((pip[0] + tip[0]) / 2, (pip[1] + tip[1]) / 2)
    return mcp, pip, dip, tip


def _curled_chain(x, tip_x=None):
    """(mcp, pip, dip, tip) for a CURLED (folded back) finger at column x."""
    tip_x = x if tip_x is None else tip_x
    mcp = (x, 0.70)
    pip = (x, 0.60)
    tip = (tip_x, 0.68)
    dip = ((pip[0] + tip[0]) / 2, (pip[1] + tip[1]) / 2)
    return mcp, pip, dip, tip


def _base_hand():
    """21 Nones -- filled in by the helpers below before use."""
    return [(0.0, 0.0, 0.0)] * 21


def _set(landmarks, idx, point):
    landmarks[idx] = (point[0], point[1], 0.0)


def _place_finger(landmarks, mcp_i, pip_i, dip_i, tip_i, chain):
    mcp, pip, dip, tip = chain
    _set(landmarks, mcp_i, mcp)
    _set(landmarks, pip_i, pip)
    _set(landmarks, dip_i, dip)
    _set(landmarks, tip_i, tip)


def make_fist():
    """All five fingers curled -- a closed fist."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.36, 0.80))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.33, 0.70), (0.32, 0.62), (0.32, 0.62), (0.30, 0.75)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _curled_chain(0.40))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _curled_chain(0.50))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _curled_chain(0.60))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _curled_chain(0.70))
    return lm


def make_open_palm():
    """All five fingers extended -- an open hand."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.40, 0.85))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.38, 0.75), (0.30, 0.68), (0.30, 0.68), (0.20, 0.60)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _extended_chain(0.40))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _extended_chain(0.50))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _extended_chain(0.60))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _extended_chain(0.70))
    return lm


def make_thumbs_up():
    """Thumb extended upward; other four fingers curled."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.35, 0.80))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.33, 0.72), (0.31, 0.55), (0.31, 0.55), (0.29, 0.35)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _curled_chain(0.40))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _curled_chain(0.50))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _curled_chain(0.60))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _curled_chain(0.70))
    return lm


def mirror_vertical(landmarks):
    """Flip y (y -> 1 - y). Preserves distances, so THUMBS_UP mirrors into THUMBS_DOWN."""
    return [(x, 1.0 - y, z) for (x, y, z) in landmarks]


def make_thumbs_down():
    return mirror_vertical(make_thumbs_up())


def make_victory():
    """Index + middle extended and spread apart; ring/pinky/thumb curled."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.36, 0.80))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.33, 0.70), (0.32, 0.62), (0.32, 0.62), (0.30, 0.75)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
                  ((0.35, 0.70), (0.32, 0.55), (0.31, 0.45), (0.30, 0.35)))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
                  ((0.65, 0.70), (0.68, 0.55), (0.69, 0.45), (0.70, 0.35)))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _curled_chain(0.60))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _curled_chain(0.72))
    return lm


def make_right_click_gesture():
    """Only the pinky extended; index/middle/ring/thumb curled."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.36, 0.80))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.33, 0.70), (0.32, 0.62), (0.32, 0.62), (0.30, 0.75)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _curled_chain(0.40))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _curled_chain(0.50))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _curled_chain(0.60))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _extended_chain(0.72))
    return lm


def make_index_up():
    """Only the index finger extended."""
    lm = _base_hand()
    _set(lm, WRIST, WRIST_XY)
    _set(lm, THUMB_CMC, (0.36, 0.80))
    _place_finger(lm, THUMB_MCP, THUMB_IP, THUMB_IP, THUMB_TIP,
                  ((0.33, 0.70), (0.32, 0.62), (0.32, 0.62), (0.30, 0.75)))
    _place_finger(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _extended_chain(0.50))
    _place_finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _curled_chain(0.55))
    _place_finger(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, _curled_chain(0.62))
    _place_finger(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _curled_chain(0.70))
    return lm


def make_pinch():
    """Thumb tip and index tip touching; other fingers curled (as in a fist)."""
    lm = make_fist()
    _set(lm, THUMB_TIP, (0.45, 0.45))
    _set(lm, INDEX_TIP, (0.45, 0.45))
    return lm


class TestUtils(unittest.TestCase):
    def test_distance_basic(self):
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0)

    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-5, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_hand_scale_matches_wrist_to_middle_mcp(self):
        lm = make_fist()
        expected = distance(lm[WRIST], lm[MIDDLE_MCP])
        self.assertAlmostEqual(hand_scale(lm), expected)


# --------------------------------------------------------------------------
# Finger-state tests
# --------------------------------------------------------------------------
class TestFingerStates(unittest.TestCase):
    def test_fist_has_all_fingers_down(self):
        states = get_finger_states(make_fist())
        self.assertFalse(any(states.values()), states)

    def test_open_palm_has_all_fingers_up(self):
        states = get_finger_states(make_open_palm())
        self.assertTrue(all(states.values()), states)

    def test_index_up_only_index_true(self):
        states = get_finger_states(make_index_up())
        self.assertTrue(states["index"])
        self.assertFalse(states["middle"])
        self.assertFalse(states["ring"])
        self.assertFalse(states["pinky"])

    def test_thumb_up_direction(self):
        lm = make_thumbs_up()
        self.assertTrue(is_thumb_up(lm))
        self.assertFalse(is_thumb_down(lm))

    def test_thumb_down_direction(self):
        lm = make_thumbs_down()
        self.assertTrue(is_thumb_down(lm))
        self.assertFalse(is_thumb_up(lm))


class TestPinch(unittest.TestCase):
    def test_touching_tips_is_a_pinch(self):
        self.assertTrue(is_pinch(make_pinch()))

    def test_open_palm_is_not_a_pinch(self):
        self.assertFalse(is_pinch(make_open_palm()))

    def test_fist_is_not_a_pinch(self):
        # Regression: closed fist must not be detected as pinch.
        self.assertFalse(is_pinch(make_fist()))


class TestGestureRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = GestureRecognizer()

    def test_fist(self):
        self.assertEqual(self.recognizer.recognize(make_fist()), Gesture.FIST)

    def test_open_palm(self):
        self.assertEqual(self.recognizer.recognize(make_open_palm()), Gesture.OPEN_PALM)

    def test_thumbs_up(self):
        self.assertEqual(self.recognizer.recognize(make_thumbs_up()), Gesture.THUMBS_UP)

    def test_thumbs_down(self):
        self.assertEqual(self.recognizer.recognize(make_thumbs_down()), Gesture.THUMBS_DOWN)

    def test_victory(self):
        self.assertEqual(self.recognizer.recognize(make_victory()), Gesture.VICTORY)

    def test_right_click_gesture(self):
        self.assertEqual(
            self.recognizer.recognize(make_right_click_gesture()), Gesture.RIGHT_CLICK
        )

    def test_index_up(self):
        self.assertEqual(self.recognizer.recognize(make_index_up()), Gesture.INDEX_UP)

    def test_pinch(self):
        self.assertEqual(self.recognizer.recognize(make_pinch()), Gesture.PINCH)

    def test_too_few_landmarks_is_none(self):
        self.assertEqual(self.recognizer.recognize([(0, 0, 0)] * 5), Gesture.NONE)

    # Regression tests for the Victory/Right-Click/Fist bug fixes.

    def test_victory_does_not_classify_as_right_click(self):
        # Regression: a peace sign must not be classified as RIGHT_CLICK.
        result = self.recognizer.recognize(make_victory())
        self.assertEqual(result, Gesture.VICTORY)
        self.assertNotEqual(result, Gesture.RIGHT_CLICK)

    def test_right_click_does_not_classify_as_victory(self):
        result = self.recognizer.recognize(make_right_click_gesture())
        self.assertEqual(result, Gesture.RIGHT_CLICK)
        self.assertNotEqual(result, Gesture.VICTORY)

    def test_fist_does_not_classify_as_open_palm_or_none(self):
        result = self.recognizer.recognize(make_fist())
        self.assertEqual(result, Gesture.FIST)
        self.assertNotIn(result, (Gesture.OPEN_PALM, Gesture.NONE))

    def test_fist_thumb_is_detected_as_folded(self):
        self.assertTrue(is_thumb_folded(make_fist()))

    def test_thumbs_up_thumb_is_not_folded(self):
        # Regression: fold check must not misclassify a real thumbs-up.
        self.assertFalse(is_thumb_folded(make_thumbs_up()))


# Virtual Mouse tests. Every pyautogui call is mocked -- no test touches
# the real mouse.
class TestMouseController(unittest.TestCase):
    def _make_controller(self):
        with mock.patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080)):
            return MouseController()

    def test_mouse_mode_controller_can_be_created(self):
        mc = self._make_controller()
        self.assertEqual(mc.screen_w, 1920)
        self.assertEqual(mc.screen_h, 1080)

    def test_index_up_moves_cursor_every_frame_without_stability_wait(self):
        """Regression: a single INDEX_UP frame must move the cursor immediately,
        without waiting for GESTURE_STABILITY_FRAMES."""
        mc = self._make_controller()
        lm = make_index_up()
        with mock.patch("src.mouse_controller.pyautogui.moveTo") as mock_move:
            gesture, action = mc.process(Gesture.INDEX_UP, lm)

        mock_move.assert_called_once()
        self.assertEqual(action, "Move Cursor")
        x, y = mock_move.call_args[0]
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 1920)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y, 1080)

    def test_pinch_clicks_without_flooding_mouse_down(self):
        """Pinch must call mouseDown() exactly once while held, not every frame."""
        mc = self._make_controller()
        lm = make_pinch()
        with mock.patch("src.mouse_controller.pyautogui.mouseDown") as mock_down, \
             mock.patch("src.mouse_controller.pyautogui.mouseUp") as mock_up, \
             mock.patch("src.mouse_controller.pyautogui.moveTo"):
            for _ in range(10):
                mc.process(Gesture.PINCH, lm)
            mock_down.assert_called_once()
            mock_up.assert_not_called()

            # Releasing the pinch must call mouseUp exactly once.
            mc.process(Gesture.NONE, None)
            mock_up.assert_called_once()

    def test_pinch_requires_a_short_confirmation_before_clicking(self):
        """A single stray PINCH frame should not immediately mouseDown."""
        mc = self._make_controller()
        lm = make_pinch()
        with mock.patch("src.mouse_controller.pyautogui.mouseDown") as mock_down:
            mc.process(Gesture.PINCH, lm)
            self.assertFalse(mock_down.called)  # below PINCH_CONFIRM_FRAMES

    def test_right_click_requires_stability_and_fires_only_once(self):
        mc = self._make_controller()
        lm = make_right_click_gesture()
        with mock.patch("src.mouse_controller.pyautogui.click") as mock_click:
            for _ in range(mc.stabilizer.stability_frames - 1):
                mc.process(Gesture.RIGHT_CLICK, lm)
            mock_click.assert_not_called()

            mc.process(Gesture.RIGHT_CLICK, lm)
            mock_click.assert_called_once_with(button="right")

            # Still held -> must not fire a second time.
            mc.process(Gesture.RIGHT_CLICK, lm)
            mock_click.assert_called_once()

    def test_gesture_control_mode_never_touches_the_mouse(self):
        """Sanity check: GestureController must have no pyautogui mouse calls at all."""
        from src import gesture_controller as gc_module

        self.assertFalse(hasattr(gc_module, "pyautogui"))


class TestGestureStabilizer(unittest.TestCase):
    def test_requires_consecutive_frames_before_stable(self):
        stabilizer = GestureStabilizer(
            stability_frames=5, repeat_cooldown=10.0, one_shot_cooldown=10.0
        )
        for _ in range(4):
            stable = stabilizer.update(Gesture.THUMBS_UP)
            self.assertIsNone(stable)
        stable = stabilizer.update(Gesture.THUMBS_UP)
        self.assertEqual(stable, Gesture.THUMBS_UP)

    def test_interrupted_sequence_resets_the_counter(self):
        stabilizer = GestureStabilizer(
            stability_frames=3, repeat_cooldown=10.0, one_shot_cooldown=10.0
        )
        stabilizer.update(Gesture.THUMBS_UP)
        stabilizer.update(Gesture.THUMBS_UP)
        stabilizer.update(Gesture.NONE)  # interrupts the streak
        stable = stabilizer.update(Gesture.THUMBS_UP)
        self.assertIsNone(stable)  # back to only 1 consecutive frame

    def test_one_shot_trigger_fires_once_then_blocks_until_gesture_releases(self):
        stabilizer = GestureStabilizer(
            stability_frames=2, repeat_cooldown=10.0, one_shot_cooldown=0.0
        )
        stabilizer.update(Gesture.VICTORY)
        stable = stabilizer.update(Gesture.VICTORY)
        self.assertEqual(stable, Gesture.VICTORY)
        self.assertTrue(stabilizer.can_trigger_once(Gesture.VICTORY))
        # Still holding the same gesture -> must NOT fire again.
        self.assertFalse(stabilizer.can_trigger_once(Gesture.VICTORY))
        self.assertFalse(stabilizer.can_trigger_once(Gesture.VICTORY))

        # Gesture goes away...
        stabilizer.update(Gesture.NONE)
        # ...and comes back and becomes stable again -> allowed to fire once more.
        stabilizer.update(Gesture.VICTORY)
        stable = stabilizer.update(Gesture.VICTORY)
        self.assertEqual(stable, Gesture.VICTORY)
        self.assertTrue(stabilizer.can_trigger_once(Gesture.VICTORY))

    def test_repeat_trigger_respects_cooldown(self):
        stabilizer = GestureStabilizer(
            stability_frames=1, repeat_cooldown=0.2, one_shot_cooldown=0.0
        )
        stabilizer.update(Gesture.THUMBS_UP)
        self.assertTrue(stabilizer.can_trigger_repeat(Gesture.THUMBS_UP))
        # Immediately again -> blocked by cooldown.
        self.assertFalse(stabilizer.can_trigger_repeat(Gesture.THUMBS_UP))
        time.sleep(0.25)
        self.assertTrue(stabilizer.can_trigger_repeat(Gesture.THUMBS_UP))

    def test_reset_clears_history(self):
        stabilizer = GestureStabilizer(
            stability_frames=3, repeat_cooldown=10.0, one_shot_cooldown=10.0
        )
        stabilizer.update(Gesture.FIST)
        stabilizer.update(Gesture.FIST)
        stabilizer.reset()
        # After reset, a single frame must not already count toward stability.
        stable = stabilizer.update(Gesture.FIST)
        self.assertIsNone(stable)


if __name__ == "__main__":
    unittest.main()
