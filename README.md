# Hand Gesture Controller

This is a Python project that uses a webcam to detect hand gestures for two main features:
- ✋ Finger Counting
- 🔊 Volume Control using hand gestures

## Features

- Real-time hand tracking with MediaPipe
- Finger count recognition
- Volume control using thumb–index distance
- FPS display
- Exit with 'q' or closing the window

## Requirements

- Python 3.11 (⚠️ not 3.13)
- OpenCV
- MediaPipe
- NumPy
- Pycaw (for volume control, Windows only)
- Comtypes

Install all dependencies:

```bash
pip install -r requirements.txt
