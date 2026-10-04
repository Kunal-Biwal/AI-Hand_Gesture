"""AI Hand Gesture Controlled PC: source package.

Pipeline: Webcam -> OpenCV -> MediaPipe HandLandmarker -> 21 landmarks
-> GestureRecognizer -> GestureController (stabilization)
-> VolumeController / MediaController / MouseController -> PC action.
"""
