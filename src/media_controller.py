"""Media key control via pyautogui (standard OS multimedia key events only)."""

import pyautogui

pyautogui.PAUSE = 0


class MediaController:
    """Sends safe, standard multimedia key presses."""

    def play_pause(self):
        try:
            pyautogui.press("playpause")
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[MediaController] Failed to send play/pause key: {exc}")
            return False

    def toggle_mute(self):
        try:
            pyautogui.press("volumemute")
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[MediaController] Failed to send mute key: {exc}")
            return False
