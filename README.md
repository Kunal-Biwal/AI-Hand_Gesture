# AI Hand Gesture Controlled PC

Control your Windows PC's **volume**, **media playback**, and **mouse cursor**
using your webcam and hand gestures -- no extra hardware, no API keys,
fully local.

```
Webcam -> OpenCV -> MediaPipe Hand Detection -> 21 Hand Landmarks
        -> Landmark Processing -> Gesture Recognition
        -> Gesture Stabilization -> Action Controller -> PC Action
```

---

## 1. Overview

This project uses a webcam feed, Google's **MediaPipe** hand-landmark model,
and classic **Computer Vision geometry** (distances, ratios, angles -- no
trained classifier) to recognize 9 hand gestures and map them to **safe**
system actions: changing the volume, toggling play/pause or mute, and
running a full virtual mouse (move / left-click / right-click / drag).

Every stage lives in its own small, readable module, documented inline
and in this README.

---

## 2. Features

- Real-time webcam capture with a mirrored (selfie-style) preview and
  on-screen FPS counter.
- Detects up to **2 hands** simultaneously, draws all **21 landmarks**
  and the hand skeleton, and labels each hand **Left**/**Right**.
- **9 rule-based gestures**: Open Palm, Fist, Thumbs Up, Thumbs Down,
  Victory/Peace, Index Finger Up, Pinch, and a dedicated Right-Click
  gesture.
- **Gesture stabilization**: nothing fires from a single noisy frame --
  gestures must hold steady for N consecutive frames, and then obey a
  cooldown, before any PC action runs.
- **Volume control** (Thumbs Up / Thumbs Down) via real OS multimedia
  keys.
- **Media control** (Victory = Play/Pause, Fist = Mute/Unmute) via real
  OS multimedia keys.
- **Virtual Mouse mode**: move the cursor with your index finger, click
  by pinching (and drag by pinching + moving), right-click with a second
  gesture.
- Two switchable modes, a clean OpenCV overlay UI, and single-key
  Reset/Help/Quit controls.
- Fully modular, documented, unit-tested architecture with **no network
  calls, no API keys, and no destructive system actions**.

## 3. Demo (what you will actually see)

Running `python main.py` opens one window showing your mirrored webcam
feed with:

- Your hand's 21 landmarks and skeleton drawn in real time.
- A left-hand info panel showing the current **Mode**, the **Gesture**
  currently being held, the **Action** it triggered, which **Hand** is
  being tracked, the live **FPS**, and the current **Volume %**.
- A short **Controls** / **Gesture Map** legend (toggle with `H`).

Hold your thumb up for about a quarter of a second -> the panel shows
`Gesture: Thumbs Up`, `Action: Volume Up`, and your actual Windows volume
increases. Press `M` -> the panel switches to `Mode: VIRTUAL MOUSE`,
point your index finger -> the OS cursor follows it.

## 4. Technologies

| Technology | Role |
|---|---|
| Python 3.12 | Language |
| OpenCV (`opencv-python`) | Webcam capture, drawing, the UI window |
| MediaPipe (`mediapipe`) Tasks API, `HandLandmarker` | Pre-trained neural network that finds hands and their 21 landmarks |
| NumPy | Array/image plumbing under OpenCV/MediaPipe |
| PyAutoGUI | Sending safe mouse moves/clicks and OS multimedia key presses |

**Why OpenCV?** It is the standard, battle-tested library for reading
webcam frames, manipulating images (flip, color conversion) and drawing
an interactive UI on top of them in real time.

**Why MediaPipe?** Training a hand-landmark detector from scratch needs a
large labelled dataset and real GPU training time. MediaPipe ships a
model Google already trained for exactly this, that runs in real time on
a normal CPU. It lets this project focus on the interesting part: turning
landmarks into gestures and gestures into actions.

## 5. Computer Vision Pipeline (step by step)

1. **Capture** -- `cv2.VideoCapture` grabs a BGR frame from the webcam at
   a configured resolution (`src/config.py`). The frame is flipped
   horizontally so the preview behaves like a mirror.
2. **Detection** -- `src/hand_tracker.py` converts the frame to RGB, wraps
   it as a `mediapipe.Image`, and runs Google's `HandLandmarker` model
   (`detect_for_video`, since we feed a continuous video stream). The
   model returns, per detected hand, 21 normalized `(x, y, z)` landmarks
   and a Left/Right classification + confidence score.
3. **Mirror correction** -- because the frame was flipped before
   detection, MediaPipe's Left/Right label is anatomically backwards; we
   swap it back in `HandTracker.get_hands`.
4. **Landmark processing** -- `src/utils.py` provides named indices for
   all 21 landmarks and small geometry helpers (`distance`, `hand_scale`)
   used everywhere downstream.
5. **Gesture recognition** -- `src/gesture_recognizer.py` turns the 21
   points into finger states (`up`/`down`) and then into one gesture
   label, using only distances and ratios (see Section 7 below).
6. **Stabilization** -- `src/utils.GestureStabilizer` requires the same
   raw gesture on N consecutive frames before calling it "stable", then
   rate-limits how often it is allowed to actually trigger an action.
7. **Action dispatch** -- `src/gesture_controller.py` (Mode 1) and
   `src/mouse_controller.py` (Mode 2) turn a stable gesture into a call
   into `volume_controller.py` / `media_controller.py` / PyAutoGUI mouse
   calls.
8. **PC action** -- the OS actually changes volume, toggles media,
   moves/clicks the mouse.

## 6. Gesture -> Action Mapping

### Mode 1 -- GESTURE CONTROL (default on launch)

| Gesture | Trigger style | Action |
|---|---|---|
| Thumbs Up | repeats while held (cooldown-limited) | Volume **Up** |
| Thumbs Down | repeats while held (cooldown-limited) | Volume **Down** |
| Victory / Peace ✌ | one-shot | **Play / Pause** |
| Fist | one-shot | **Mute / Unmute** |
| Open Palm, Index Up, Pinch, Right-Click gesture | -- | No Mode-1 action (reserved for Mode 2) |

### Mode 2 -- VIRTUAL MOUSE (press `M` to enter)

| Gesture | Action |
|---|---|
| Index Finger Up | **Move** the system cursor to follow the fingertip |
| Pinch (thumb + index touching) | Press-hold = **Left Click**; pinch + move = **Drag** |
| Pinky finger up (only) | **Right Click** |

## 7. How Each Gesture Is Actually Detected

All checks live in `src/gesture_recognizer.py` and use **ratios**, never
raw pixel coordinates, so they keep working as your hand moves closer to
or farther from the camera.

- **Finger up/down**: a finger is "extended" if its fingertip is farther
  from the **wrist** than its own PIP joint (and MCP joint) by a small
  margin. When a finger curls into the palm, its tip swings back toward
  the wrist, flipping this comparison. Because everything is measured
  from the same wrist point, this stays correct even if the hand is
  rotated in the camera's view.
- **Thumbs Up / Down**: the thumb must be "extended" by the same
  distance test, AND all four other fingers must be curled. The
  direction (up vs down) compares the thumb tip's Y position to its own
  MCP joint.
- **Open Palm**: all five fingers extended. **Fist**: index/middle/ring/
  pinky all curled (the shared wrist-distance test) AND the thumb is
  "folded" (see below) -- NOT just "not extended".
- **Thumb folded (used only by Fist)**: the other four fingers hinge
  straight back toward the wrist when curled, so the wrist-distance test
  works for them. The thumb's CMC joint lets it rotate ACROSS the palm
  instead, so in a real fist it can still sit just as far from the wrist
  as an extended thumb -- reusing the wrist-distance test for the thumb
  was exactly why Fist used to be unreliable. Instead, `is_thumb_folded`
  checks how close the thumb tip is to the base of the index finger
  (`INDEX_MCP`), which is where a folded thumb actually rests.
- **Pinch**: distance between thumb tip and index tip, divided by
  `hand_scale` (the wrist-to-middle-knuckle distance, which acts as a
  built-in ruler for *this* hand at *this* distance from the camera). If
  that ratio is small, the fingertips are touching.
- **Victory**: index AND middle extended, ring AND pinky curled. This is
  the complete condition -- no secondary spread/angle check -- which is
  what guarantees it cannot be confused with any other gesture.
- **Right-Click**: ONLY the pinky extended (index/middle/ring curled). A
  completely different raised finger than Victory (index+middle) or
  Index Up (index only), so there is no finger-state overlap with either.
- **Index Finger Up**: only the index finger extended, used to drive the
  virtual mouse.

Every rule after Pinch checks a different, mutually-exclusive
combination of "which fingers are up" -- so two gestures can never
satisfy the same branch, and the order they are listed in does not
affect the result (see the priority list in Section 6/10 and the
docstring of `GestureRecognizer.recognize`).

## 8. Gesture Stabilization (why gestures don't spam-trigger)

Implemented in `src/utils.GestureStabilizer`, used identically by both
`GestureController` and `MouseController`:

```
raw gesture seen for 8 consecutive frames
        |
        v
   "stable" gesture
        |
        v
  trigger allowed, subject to a cooldown
```

Two trigger styles are used on top of that stability check:

- **Repeatable** (`can_trigger_repeat`) -- for Volume Up/Down. Fires
  again every `REPEAT_ACTION_COOLDOWN` seconds *while the gesture is
  still held*, so holding Thumbs Up steadily raises the volume, like a
  physical button.
- **One-shot** (`can_trigger_once`) -- for Play/Pause, Mute, Click,
  Right-Click. Fires exactly once per "entry" into the gesture; it will
  not fire again until the gesture is released (goes back to no stable
  gesture) and is then re-entered. This is what stops a single 2-second
  peace sign from firing 50 play/pause toggles.

Pressing **R** calls `reset()` on the active controller(s), which clears
the consecutive-frame counter, all cooldown timestamps, the "last fired"
gesture, and the mouse's smoothing history -- without restarting the
app.

## 9. Virtual Mouse Details

- The index fingertip's normalized `(x, y)` is remapped from a smaller
  **active rectangle** in the middle of the frame (controlled by
  `MOUSE_ACTIVE_REGION_MARGIN`) up to the full screen size, so you don't
  have to reach the literal edge of the webcam's view to hit the edge of
  the screen.
- `MOUSE_SENSITIVITY` then expands/contracts that mapped point around the
  screen's center -- an independent "how far does my hand motion travel"
  knob.
- Two `ExponentialSmoother` instances (one for X, one for Y) blend each
  new raw point with the previous smoothed point, removing landmark
  jitter without adding noticeable lag.
- **Click vs drag** uses one simple state machine: a pinch calls
  `pyautogui.mouseDown()` once; while it stays a pinch, if the hand moves
  more than `DRAG_MOVEMENT_THRESHOLD` it becomes a drag (cursor keeps
  following); the moment the pinch ends, `mouseUp()` fires. A quick
  pinch-release with no movement is therefore an ordinary left click.
- `pyautogui.FAILSAFE` is left **ON** on purpose: slamming the cursor
  into the screen's top-left corner immediately aborts all automated
  mouse control -- a deliberate, documented safety net while testing.

## 10. Architecture

```
AI-Hand/
├── venv/                     (your existing virtual environment)
├── models/
│   └── hand_landmarker.task  (auto-downloaded on first run, gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py             # every tunable constant, in one place
│   ├── utils.py              # geometry helpers, FPS counter, GestureStabilizer
│   ├── hand_tracker.py       # MediaPipe HandLandmarker wrapper
│   ├── gesture_recognizer.py # 21 landmarks -> gesture label (rule-based)
│   ├── gesture_controller.py # Mode 1: stable gesture -> volume/media action
│   ├── mouse_controller.py   # Mode 2: stable gesture -> cursor/click/drag
│   ├── volume_controller.py  # safe Windows volume up/down
│   └── media_controller.py   # safe play/pause + mute key presses
├── tests/
│   ├── __init__.py
│   └── test_gestures.py      # unit tests, synthetic landmarks only
├── main.py                   # webcam loop, keyboard shortcuts, UI overlay
├── requirements.txt
├── README.md
└── .gitignore
```

Each module maps to one pipeline stage, so it can be explained (and
tested, and later swapped out) independently of the others.

## 11. Installation

Prerequisites: Python 3.12 on Windows, with a working webcam.

```powershell
# From the project root (AI-Hand/). The venv already exists and is active.
# If it is not active, activate it first:
venv\Scripts\Activate.ps1

# Dependencies are already installed; to (re)install from scratch:
pip install -r requirements.txt
```

The first time you run the app, it downloads MediaPipe's public
`hand_landmarker.task` model file (~8 MB) into `models/` and caches it
there permanently. This is a one-time model download, **not** an API
key -- no account, token, or paid service is involved, and every run
after the first is fully offline.

## 12. Running the Application

```powershell
python main.py
```

The webcam window opens immediately. Press **Q** at any time to quit --
the camera is always released and the window always closed, even if an
error occurs (the shutdown code runs inside a `finally` block).

## 13. Keyboard Controls

| Key | Effect |
|---|---|
| `M` | Toggle between GESTURE CONTROL and VIRTUAL MOUSE modes |
| `H` | Show / hide the on-screen help legend |
| `R` | Reset gesture history, cooldowns and mouse smoothing (no restart) |
| `Q` | Quit the application (always works, always cleans up) |

## 14. Configuration

Every tunable value lives in `src/config.py` -- nothing else in the
project hard-codes these numbers. Highlights:

| Setting | Meaning |
|---|---|
| `CAMERA_INDEX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT` | Which webcam, and at what resolution |
| `MAX_NUM_HANDS` | Detect up to this many hands at once |
| `MIN_DETECTION_CONFIDENCE`, `MIN_TRACKING_CONFIDENCE` | MediaPipe confidence thresholds |
| `GESTURE_STABILITY_FRAMES` | Consecutive frames required before a gesture is "stable" |
| `REPEAT_ACTION_COOLDOWN` / `ONE_SHOT_ACTION_COOLDOWN` | Minimum seconds between repeated / one-shot triggers |
| `PINCH_DISTANCE_RATIO` | How close (relative to hand size) fingertips must be for a pinch |
| `THUMB_FOLD_RATIO` | How close the thumb tip must be to the index knuckle to count as "folded" (used by Fist) |
| `PINCH_CONFIRM_FRAMES` | Short raw-frame confirmation before a pinch starts a click/drag (Virtual Mouse only) |
| `MOUSE_SMOOTHING_FACTOR` | Cursor smoothing (lower = smoother, more lag) |
| `MOUSE_SENSITIVITY` | Cursor movement scaling |
| `MOUSE_ACTIVE_REGION_MARGIN` | Size of the in-frame rectangle mapped to the full screen |

If gestures feel too twitchy, raise `GESTURE_STABILITY_FRAMES`. If the
mouse feels laggy, raise `MOUSE_SMOOTHING_FACTOR` toward `1.0`; if it
feels jittery, lower it.

## 15. Troubleshooting

- **"Could not open webcam"** -- another app (Zoom/Teams/browser) may be
  holding the camera; close it and retry. Try `CAMERA_INDEX = 1` in
  `config.py` if you have more than one camera.
- **Model download fails** -- you need an internet connection for the
  *first* run only, to fetch `hand_landmarker.task`. Behind a restrictive
  firewall, download the file from the URL in `config.py` manually and
  place it at `models/hand_landmarker.task`.
- **Low FPS / laggy** -- lower `CAMERA_WIDTH`/`CAMERA_HEIGHT` in
  `config.py`, close other apps using the CPU, and ensure good, even
  lighting (dark rooms make detection slower and less stable).
- **Gestures feel unreliable** -- make sure your whole hand (including
  the wrist) is in frame, keep lighting even, and avoid a cluttered
  background directly behind your hand.
- **Mouse mode feels jumpy** -- raise `MOUSE_SMOOTHING_FACTOR` slightly,
  or increase `GESTURE_STABILITY_FRAMES`.
- **Volume % looks "stuck" / wrong** -- see Limitations below; without
  `pycaw` installed this is an internal estimate, not the real OS value.

## 16. Limitations

- **Volume percentage is estimated, not read from Windows**, because the
  only installed dependency is PyAutoGUI, which can *send* volume
  key-presses but cannot *read* the system mixer level. The actual
  volume-up/down key presses are 100% real; only the on-screen `%`
  number is a local estimate (clearly labelled "(est.)" in the UI). If
  `pycaw`+`comtypes` are installed, `volume_controller.py`
  **automatically** switches to reporting the real OS volume -- no code
  changes needed.
- **Thumbs Up/Down assume a roughly upright hand/camera** (gravity-based
  gestures are inherently defined relative to "up"). A hand held fully
  sideways may misclassify thumb direction.
- **Depth (toward/away from the camera) is not used** for finger
  extension -- this project reasons in 2D screen-space geometry, which
  is simpler to explain and sufficiently robust for this use case, but a
  finger pointed straight at the camera can be harder to classify.
- **One active hand at a time** drives gestures/mouse even when two
  hands are visible (the right hand is preferred if both are present) --
  this is a deliberate, documented, deterministic choice, not a crash.
- Only safe actions are implemented (see Section 17); this is by design,
  not an oversight.

## 17. Safety

Only these actions are ever performed, all through PyAutoGUI's standard,
reversible APIs:

- Mouse movement, left click, right click, click-drag.
- OS multimedia keys: volume up, volume down, mute, play/pause.
- Clean application exit.

This project **never** deletes files, runs shell/PowerShell commands,
touches the Windows registry, shuts down the PC, launches arbitrary
programs, or performs any file-system operation. `Q` always quits
reliably; the webcam and all native resources are released in a
`finally` block even if an error occurs mid-run.

## 18. Future ML Extension

Today, `gesture_recognizer.py` is 100% rule-based geometry -- no model is
trained. It was deliberately structured so the *only* thing that would
need to change to go further is that one file:

```
Webcam -> MediaPipe -> 21 landmarks -> Feature extraction
        -> ML classifier -> Gesture prediction -> PC action
```

Concretely: the 21 `(x, y, z)` landmarks already ARE a clean, fixed-size
feature vector (63 numbers, or fewer if you engineer ratios/angles like
this project already computes). A future version could:

1. Record many labelled samples of each gesture's landmark vector (across
   many hands, distances, and lighting).
2. Train a small classifier -- a **Random Forest** or **SVM** are both
   strong, fast, interpretable choices for this not-very-large feature
   space; a small **feed-forward neural network** would also work well
   and could support adding brand-new, more subtle gestures.
3. Replace `GestureRecognizer.recognize()`'s rule chain with
   `model.predict(feature_vector)`, keeping every other module
   (`hand_tracker.py`, `gesture_controller.py`, `mouse_controller.py`,
   `main.py`) completely unchanged.

The CV/detection stage is already ML (MediaPipe's trained model); this
upgrade would make the *classification* stage learned instead of
hand-ruled, letting the system learn user-specific gesture variations
and support gestures that are hard to describe geometrically.

---

## 19. Testing

```powershell
python -m unittest discover -s tests -v
```

`tests/test_gestures.py` builds synthetic, hand-crafted 21-point hand
poses (fist, open palm, thumbs up/down, victory, right-click gesture,
index-up, pinch) and asserts the rule-based recognizer classifies each
one correctly, plus dedicated tests for finger-state logic, pinch
detection, and the `GestureStabilizer`'s debouncing/cooldown behavior.
**No test moves the mouse, presses a media key, or opens a camera.**
