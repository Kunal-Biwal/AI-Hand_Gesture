"""Windows system volume control.

Volume up/down are sent as real OS multimedia key presses. Without
``pycaw`` installed, the displayed percentage is an internal estimate
(labelled "(est.)" in the UI); if ``pycaw`` is available, it is used to
read/set the real system volume instead (see ``_try_load_pycaw``).
"""

import pyautogui

from . import config

pyautogui.PAUSE = 0  # we control pacing ourselves via cooldowns


def _try_load_pycaw():
    """Return a pycaw volume interface if available, else None (falls back to the estimate)."""
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume
    except Exception:
        return None


class VolumeController:
    """Increases/decreases the Windows system volume from gesture triggers."""

    def __init__(self, step_percent=config.VOLUME_STEP_PERCENT):
        self.step_percent = step_percent
        self._pycaw_volume = _try_load_pycaw()
        self._using_real_volume = self._pycaw_volume is not None
        self._estimated_percent = 50.0
        if self._using_real_volume:
            try:
                self._estimated_percent = round(
                    self._pycaw_volume.GetMasterVolumeLevelScalar() * 100
                )
            except Exception:
                self._using_real_volume = False

    def increase(self):
        """Raise system volume by one step. Returns the new displayed value."""
        try:
            pyautogui.press("volumeup")
        except Exception as exc:  # pragma: no cover - depends on OS state
            print(f"[VolumeController] Failed to send volume-up key: {exc}")
        self._estimated_percent = min(100, self._estimated_percent + self.step_percent)
        return self.get_volume_percent()

    def decrease(self):
        """Lower system volume by one step. Returns the new displayed value."""
        try:
            pyautogui.press("volumedown")
        except Exception as exc:  # pragma: no cover
            print(f"[VolumeController] Failed to send volume-down key: {exc}")
        self._estimated_percent = max(0, self._estimated_percent - self.step_percent)
        return self.get_volume_percent()

    def get_volume_percent(self):
        """Best available volume percentage: real value via pycaw, else the estimate."""
        if self._using_real_volume:
            try:
                return round(self._pycaw_volume.GetMasterVolumeLevelScalar() * 100)
            except Exception:
                self._using_real_volume = False
        return round(self._estimated_percent)

    def is_estimated(self):
        """True if the displayed percentage is an estimate, not the real OS value."""
        return not self._using_real_volume
