"""Mode 2: Virtual Mouse. Maps the index fingertip to the cursor; pinch to click/drag.

Cursor movement and pinch click/drag react to the RAW per-frame gesture,
not the debounced GestureStabilizer: that debounce needs the same
gesture on N consecutive frames, which fights continuous cursor motion
(normal jitter would reset the counter and the cursor would barely
move). Position jitter is smoothed by ExponentialSmoother instead, and a
short ``_pinch_confirm_count`` guards against a stray single-frame pinch.
Only RIGHT_CLICK, a rarer one-shot action, still goes through the full
stabilizer + cooldown.
"""

import math

import pyautogui

from . import config
from .gesture_recognizer import Gesture
from .utils import (
    INDEX_TIP, THUMB_TIP,
    ExponentialSmoother, GestureStabilizer, clamp,
)

pyautogui.PAUSE = 0
# Keep PyAutoGUI's fail-safe enabled.
pyautogui.FAILSAFE = True

SCREEN_EDGE_PADDING = 2  # px kept away from the exact corner/edge


class MouseController:
    """Turns stable virtual-mouse gestures into real cursor movement/clicks."""

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        self.stabilizer = GestureStabilizer(
            stability_frames=config.GESTURE_STABILITY_FRAMES,
            repeat_cooldown=config.CLICK_COOLDOWN,
            one_shot_cooldown=config.CLICK_COOLDOWN,
        )
        self._smoother_x = ExponentialSmoother(config.MOUSE_SMOOTHING_FACTOR)
        self._smoother_y = ExponentialSmoother(config.MOUSE_SMOOTHING_FACTOR)

        self._pinch_active = False
        self._pinch_start_norm = None
        self._is_dragging = False
        self._pinch_confirm_count = 0

    def reset(self):
        """Clear smoothing/click state without restarting the app (key 'R')."""
        self.stabilizer.reset()
        self._smoother_x.reset()
        self._smoother_y.reset()
        self._pinch_confirm_count = 0
        self._release_pinch_if_needed()

    def _map_to_screen(self, x_norm, y_norm):
        margin = config.MOUSE_ACTIVE_REGION_MARGIN
        usable = max(1e-6, 1.0 - 2 * margin)

        x_rel = clamp((x_norm - margin) / usable, 0.0, 1.0)
        y_rel = clamp((y_norm - margin) / usable, 0.0, 1.0)

        screen_x = x_rel * self.screen_w
        screen_y = y_rel * self.screen_h

        # Sensitivity expands movement outward from the screen center.
        cx, cy = self.screen_w / 2, self.screen_h / 2
        screen_x = cx + (screen_x - cx) * config.MOUSE_SENSITIVITY
        screen_y = cy + (screen_y - cy) * config.MOUSE_SENSITIVITY

        screen_x = clamp(screen_x, SCREEN_EDGE_PADDING, self.screen_w - SCREEN_EDGE_PADDING)
        screen_y = clamp(screen_y, SCREEN_EDGE_PADDING, self.screen_h - SCREEN_EDGE_PADDING)
        return screen_x, screen_y

    def _smoothed_screen_point(self, x_norm, y_norm):
        raw_x, raw_y = self._map_to_screen(x_norm, y_norm)
        return self._smoother_x.update(raw_x), self._smoother_y.update(raw_y)

    def _release_pinch_if_needed(self):
        if self._pinch_active:
            try:
                pyautogui.mouseUp()
            except Exception as exc:  # pragma: no cover
                print(f"[MouseController] Failed to release mouse button: {exc}")
        self._pinch_active = False
        self._is_dragging = False
        self._pinch_start_norm = None

    def process(self, raw_gesture, landmarks):
        """Process this frame's raw gesture; returns (displayed_gesture, action_text)."""
        stable_gesture = self.stabilizer.update(raw_gesture)
        action_text = "None"

        if raw_gesture == Gesture.PINCH:
            self._pinch_confirm_count += 1
        else:
            self._pinch_confirm_count = 0
            if self._pinch_active:
                # Pinch just ended -> release the button. A pinch that
                # never moved beyond the drag threshold was a plain click.
                was_dragging = self._is_dragging
                self._release_pinch_if_needed()
                action_text = "Drop (Drag End)" if was_dragging else "Left Click"

        pinch_confirmed = self._pinch_active or (
            self._pinch_confirm_count >= config.PINCH_CONFIRM_FRAMES
        )

        if raw_gesture == Gesture.PINCH and pinch_confirmed:
            thumb = landmarks[THUMB_TIP]
            index = landmarks[INDEX_TIP]
            mid_x = (thumb[0] + index[0]) / 2
            mid_y = (thumb[1] + index[1]) / 2

            if not self._pinch_active:
                self._pinch_active = True
                self._is_dragging = False
                self._pinch_start_norm = (mid_x, mid_y)
                try:
                    pyautogui.mouseDown()
                except Exception as exc:  # pragma: no cover
                    print(f"[MouseController] mouseDown failed: {exc}")
                action_text = "Pinch (Click/Drag)"
            else:
                start_x, start_y = self._pinch_start_norm
                moved = math.hypot(mid_x - start_x, mid_y - start_y)
                if moved > config.DRAG_MOVEMENT_THRESHOLD:
                    self._is_dragging = True
                if self._is_dragging:
                    x, y = self._smoothed_screen_point(mid_x, mid_y)
                    try:
                        pyautogui.moveTo(x, y)
                    except pyautogui.FailSafeException:
                        print("[MouseController] Fail-safe corner hit, drag paused.")
                    action_text = "Dragging"
                else:
                    action_text = "Pinch Hold"

        elif raw_gesture == Gesture.INDEX_UP:
            x_norm, y_norm = landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]
            x, y = self._smoothed_screen_point(x_norm, y_norm)
            try:
                pyautogui.moveTo(x, y)
            except pyautogui.FailSafeException:
                print("[MouseController] Fail-safe corner hit, cursor control paused.")
            action_text = "Move Cursor"

        elif stable_gesture == Gesture.RIGHT_CLICK:
            # One-shot action -> keep the full debounce + cooldown so a
            # single noisy frame cannot fire an accidental right click.
            if self.stabilizer.can_trigger_once(stable_gesture):
                try:
                    pyautogui.click(button="right")
                except Exception as exc:  # pragma: no cover
                    print(f"[MouseController] Right click failed: {exc}")
                action_text = "Right Click"

        # Report the raw gesture for a responsive debug/UI readout,
        # falling back to the debounced one only while raw is NONE (idle).
        displayed_gesture = raw_gesture if raw_gesture != Gesture.NONE else stable_gesture
        return displayed_gesture, action_text
