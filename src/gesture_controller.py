"""Mode 1: Gesture Control. Maps a stable gesture to a volume/media action.

THUMBS_UP/DOWN repeat while held (volume); VICTORY/FIST are one-shot
(play/pause, mute).
"""

from . import config
from .gesture_recognizer import Gesture
from .media_controller import MediaController
from .volume_controller import VolumeController
from .utils import GestureStabilizer


class GestureController:
    """Stabilizes gestures and dispatches safe system actions in Mode 1."""

    def __init__(self):
        self.stabilizer = GestureStabilizer(
            stability_frames=config.GESTURE_STABILITY_FRAMES,
            repeat_cooldown=config.REPEAT_ACTION_COOLDOWN,
            one_shot_cooldown=config.ONE_SHOT_ACTION_COOLDOWN,
        )
        self.volume = VolumeController()
        self.media = MediaController()
        self.last_action = "None"

    def reset(self):
        """Clear gesture history/cooldowns without restarting the app (key 'R')."""
        self.stabilizer.reset()
        self.last_action = "None"

    def process(self, raw_gesture):
        """Process this frame's raw gesture; returns (stable_gesture, action_text)."""
        stable_gesture = self.stabilizer.update(raw_gesture)
        action_text = "None"

        if stable_gesture == Gesture.THUMBS_UP:
            if self.stabilizer.can_trigger_repeat(stable_gesture):
                self.volume.increase()
                action_text = "Volume Up"
        elif stable_gesture == Gesture.THUMBS_DOWN:
            if self.stabilizer.can_trigger_repeat(stable_gesture):
                self.volume.decrease()
                action_text = "Volume Down"
        elif stable_gesture == Gesture.VICTORY:
            if self.stabilizer.can_trigger_once(stable_gesture):
                self.media.play_pause()
                action_text = "Play / Pause"
        elif stable_gesture == Gesture.FIST:
            if self.stabilizer.can_trigger_once(stable_gesture):
                self.media.toggle_mute()
                action_text = "Mute / Unmute"

        if action_text != "None":
            self.last_action = action_text
        return stable_gesture, action_text

    def get_volume_percent(self):
        return self.volume.get_volume_percent()

    def volume_is_estimated(self):
        return self.volume.is_estimated()
