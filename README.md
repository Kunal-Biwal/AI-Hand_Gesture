# AI Hand Gesture Controlled PC

A Python project that lets you control some PC functions using hand gestures through a webcam.

The project uses OpenCV and MediaPipe to detect hand landmarks and recognizes gestures using landmark-based rules. PyAutoGUI is used to control the mouse and send media/volume key presses.

## Features

- Hand tracking using MediaPipe
- Gesture recognition using 21 hand landmarks
- Volume up/down
- Play/Pause
- Mute/Unmute
- Virtual mouse control
- Left click and drag using pinch
- Right click using a hand gesture
- Gesture stabilization to reduce accidental actions
- On-screen status and gesture information
- Keyboard controls for switching modes and exiting

## Gesture Controls

### Gesture Control Mode

| Gesture | Action |
|---|---|
| 👍 Thumbs Up | Volume Up |
| 👎 Thumbs Down | Volume Down |
| ✌️ Victory | Play / Pause |
| ✊ Fist | Mute / Unmute |

### Virtual Mouse Mode

Press `M` to switch to Virtual Mouse mode.

| Gesture | Action |
|---|---|
| ☝️ Index Finger | Move Cursor |
| 🤏 Pinch | Left Click / Drag |
| Pinky Up | Right Click |

## Tech Stack

- Python 3.12
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarks
   ↓
Gesture Recognition
   ↓
Gesture Stabilization
   ↓
PC Action