import cv2
import mediapipe as mp
import numpy as np
import math
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class HandTracker:
    def __init__(self, detectionCon=0.7, trackCon=0.7, maxHands=1):
        self.mode = False
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, c = image.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id in [4, 8]:
                    cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return lmList

def main():
    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    # Audio setup using pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    volRange = volume.GetVolumeRange()  # (-96.0, 0.0)
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPercent = 0

    # Tracker and FPS
    tracker = HandTracker()
    pTime = 0

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = tracker.findHands(img)
        lmList = tracker.findPosition(img, draw=True)

        if len(lmList) != 0:
            # Get coordinates of thumb tip and index finger tip
            x1, y1 = lmList[4][1], lmList[4][2]     # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]     # Index tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw line and circle
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Convert hand range to volume range
            # vol = np.interp(length, [20, 200], [minVol, maxVol])
            volBar = np.interp(length, [20, 200], [400, 150])
            volPercent = np.interp(length, [20, 200], [0, 100])

            volume.SetMasterVolumeLevelScalar(volPercent / 100, None)

            if length < 25:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{int(volPercent)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (255, 0, 0), 2)

        # FPS counter
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (500, 50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 0), 2)

        # Display image
        cv2.imshow("Hand Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Hand Volume Control", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
