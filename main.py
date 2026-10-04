#!/usr/bin/env python
"""Entry point: webcam loop, keyboard shortcuts and on-screen UI."""

import sys
import time

import cv2

from src import config
from src.hand_tracker import HandTracker
from src.gesture_recognizer import Gesture, GestureRecognizer, get_finger_states, is_pinch
from src.gesture_controller import GestureController
from src.mouse_controller import MouseController
from src.utils import FPSCounter, put_text, draw_panel


ACTION_LABELS = {
    Gesture.NONE: "-",
    Gesture.OPEN_PALM: "Open Palm",
    Gesture.FIST: "Fist",
    Gesture.THUMBS_UP: "Thumbs Up",
    Gesture.THUMBS_DOWN: "Thumbs Down",
    Gesture.VICTORY: "Victory / Peace",
    Gesture.INDEX_UP: "Index Finger Up",
    Gesture.PINCH: "Pinch",
    Gesture.RIGHT_CLICK: "Right-Click Gesture",
}

HELP_LINES = [
    "M - Toggle Mouse Mode",
    "R - Reset Gesture State",
    "H - Show / Hide Help",
    "Q - Quit",
]

GESTURE_MODE_HELP = [
    "THUMBS UP    -> Volume Up",
    "THUMBS DOWN  -> Volume Down",
    "VICTORY      -> Play / Pause",
    "FIST         -> Mute / Unmute",
]

MOUSE_MODE_HELP = [
    "INDEX UP     -> Move Cursor",
    "PINCH        -> Left Click / Drag",
    "PINKY UP     -> Right Click",
]


def open_camera():
    """Open the webcam, raising a clear error message on failure."""
    backend = cv2.CAP_DSHOW if (sys.platform.startswith("win") and config.USE_DSHOW_ON_WINDOWS) else 0
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)

    if not cap.isOpened():
        # Retry with the default backend in case DirectShow is the issue.
        cap.release()
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {config.CAMERA_INDEX}.\n"
            "Possible causes:\n"
            "  - No webcam is connected.\n"
            "  - Another application (Zoom, Teams, browser tab, etc.) is\n"
            "    currently using the webcam exclusively.\n"
            "  - The wrong CAMERA_INDEX is set in src/config.py.\n"
            "Close other apps that use the camera and try again, or try\n"
            "changing CAMERA_INDEX to 1 in src/config.py if you have more\n"
            "than one camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_REQUEST)
    return cap


def draw_ui(
    frame,
    mode,
    gesture_label,
    action_text,
    hand_label,
    fps,
    volume_percent,
    volume_is_estimated,
    show_help,
    finger_states=None,
    pinch_active=False,
):
    """Draw the sectioned overlay: STATUS / DEBUG / CONTROLS / GESTURE MAP."""
    h, w = frame.shape[:2]
    # Panel width adapts to the frame but never exceeds the configured
    # default, so a smaller camera resolution doesn't eat the webcam view.
    panel_w = min(config.UI_PANEL_WIDTH, max(220, int(w * 0.42)))
    draw_panel(frame, (0, 0), (panel_w, h), config.COLOR_BG_PANEL, alpha=0.55)

    # Scale rows/font to the actual frame height so a shorter webcam
    # fallback (e.g. 480p) doesn't push content past the bottom of the frame.
    k = max(0.55, min(1.0, h / 560.0))

    x = 12
    right = panel_w - 12
    cursor = [int(16 * k)]  # boxed so the nested helpers below can advance it

    def line(text, color, scale=0.44, thickness=1, dy=15):
        put_text(frame, text, (x, cursor[0]), color, config.FONT, max(0.30, scale * k), thickness)
        cursor[0] += max(9, int(dy * k))

    def separator(gap_before=8, gap_after=14):
        # gap_after must clear the glyph height of the next line (cv2.putText's
        # y origin is the text baseline) or the rule cuts through the text.
        cursor[0] += max(4, int(gap_before * k))
        cv2.line(frame, (x, cursor[0]), (right, cursor[0]), config.COLOR_TEXT_PRIMARY, 1)
        cursor[0] += max(10, int(gap_after * k))

    def section(title):
        separator()
        line(title, config.COLOR_TEXT_ACCENT, 0.48, 1, dy=16)

    # ---- Title / Mode (always shown) ----
    line("AI HAND GESTURE CONTROL", config.COLOR_TEXT_ACCENT, 0.56, 2, dy=20)
    separator()
    line(f"Mode: {mode}", config.COLOR_TEXT_PRIMARY, 0.52, 2, dy=20)

    # ---- STATUS ----
    section("STATUS")
    line(f"Gesture: {gesture_label}", config.COLOR_TEXT_GOOD, 0.46, 1, dy=15)
    line(f"Action: {action_text}", config.COLOR_TEXT_WARN, 0.46, 1, dy=15)
    line(f"Hand: {hand_label}", config.COLOR_TEXT_PRIMARY, 0.46, 1, dy=15)
    line(f"FPS: {fps:.1f}", config.COLOR_TEXT_PRIMARY, 0.46, 1, dy=15)
    vol_suffix = " (est.)" if volume_is_estimated else ""
    line(f"Volume: {volume_percent}%{vol_suffix}", config.COLOR_TEXT_PRIMARY, 0.46, 1, dy=15)

    # ---- DEBUG ----
    section("DEBUG")
    fs = finger_states or {}
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        state = fs.get(name)
        text = "UP" if state else ("DOWN" if state is not None else "-")
        color = config.COLOR_TEXT_GOOD if state else config.COLOR_TEXT_PRIMARY
        line(f"{name.capitalize()}: {text}", color, 0.42, 1, dy=14)
    pinch_color = config.COLOR_TEXT_GOOD if pinch_active else config.COLOR_TEXT_PRIMARY
    line(f"Pinch: {'YES' if pinch_active else 'NO'}", pinch_color, 0.42, 1, dy=14)

    if show_help:
        # ---- CONTROLS ----
        section("CONTROLS")
        for text in HELP_LINES:
            line(text, config.COLOR_TEXT_PRIMARY, 0.42, 1, dy=14)

        # ---- GESTURE MAP ----
        section("GESTURE MAP")
        mode_help = GESTURE_MODE_HELP if mode == config.MODE_GESTURE_CONTROL else MOUSE_MODE_HELP
        for text in mode_help:
            line(text, config.COLOR_TEXT_PRIMARY, 0.42, 1, dy=14)
    else:
        separator()
        line("Press H for help", config.COLOR_TEXT_PRIMARY, 0.42, 1, dy=14)


def main():
    print("[main] Starting AI Hand Gesture Controlled PC ...")
    print("[main] Press 'Q' in the video window to quit at any time.")

    try:
        hand_tracker = HandTracker()
    except Exception as exc:
        print(f"[main] FATAL: could not initialize the hand tracker: {exc}")
        return 1

    recognizer = GestureRecognizer()
    gesture_controller = GestureController()
    mouse_controller = MouseController()
    fps_counter = FPSCounter()

    cap = None
    try:
        cap = open_camera()
    except Exception as exc:
        print(f"[main] FATAL: {exc}")
        hand_tracker.close()
        return 1

    mode = config.MODE_GESTURE_CONTROL
    show_help = True

    try:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[main] Warning: failed to read a frame from the webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror the feed so it behaves like a mirror the user looks into.
            frame = cv2.flip(frame, 1)

            result = hand_tracker.find_hands(frame)
            hands = hand_tracker.get_hands(result)
            hand_tracker.draw_landmarks(frame, result, hands)

            if hands:
                # Prefer the right hand if both are visible (deterministic choice).
                primary = next((h for h in hands if h.label == "Right"), hands[0])
                raw_gesture = recognizer.recognize(primary.landmarks)
                hand_label = primary.label
                finger_states = get_finger_states(primary.landmarks)
                pinch_active = is_pinch(primary.landmarks)
            else:
                raw_gesture = Gesture.NONE
                hand_label = "None"
                finger_states = None
                pinch_active = False

            if mode == config.MODE_GESTURE_CONTROL:
                stable_gesture, action_text = gesture_controller.process(raw_gesture)
            else:
                landmarks = primary.landmarks if hands else None
                stable_gesture, action_text = mouse_controller.process(raw_gesture, landmarks)

            gesture_label = ACTION_LABELS.get(stable_gesture, "-") if stable_gesture else "-"
            fps = fps_counter.tick()

            draw_ui(
                frame,
                mode,
                gesture_label,
                action_text,
                hand_label,
                fps,
                gesture_controller.get_volume_percent(),
                gesture_controller.volume_is_estimated(),
                show_help,
                finger_states,
                pinch_active,
            )

            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[main] Quit requested by user.")
                break
            elif key == ord("m"):
                if mode == config.MODE_GESTURE_CONTROL:
                    mode = config.MODE_VIRTUAL_MOUSE
                else:
                    mode = config.MODE_GESTURE_CONTROL
                gesture_controller.reset()
                mouse_controller.reset()
                print(f"[main] Switched mode -> {mode}")
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("r"):
                gesture_controller.reset()
                mouse_controller.reset()
                print("[main] Gesture/mouse state reset.")

    except KeyboardInterrupt:
        print("[main] Interrupted by user (Ctrl+C).")
    except Exception as exc:  # noqa: BLE001 - top level safety net
        print(f"[main] Unexpected error: {exc}")
    finally:
        print("[main] Shutting down: releasing camera and closing windows...")
        if cap is not None:
            cap.release()
        mouse_controller.reset()  # ensures any held mouse button is released
        hand_tracker.close()
        cv2.destroyAllWindows()
        print("[main] Clean shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
