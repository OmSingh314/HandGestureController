import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm

img=np.zeros((480,640,3),np.uint8)
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handTracker(detectionCon=0.5)
while True:
    ret,frame=cap.read()
    frame = detector.handsFinder(frame)
    lmList = detector.positionFinder(frame, draw=False)
    #frame=frame.reshape(640,480)
    #print(frame.shape)
    #print(lmList)
    if len(lmList)!=0:
        cv2.circle(img,(lmList[8][1],lmList[8][2]),5,(0,255,0),5)
        img1=cv2.add(frame,img)
        cv2.imshow("2",img1)
    else:
        img1=cv2.add(frame,img)
        cv2.imshow("2",img1)
        #cv2.imshow("2",frame)
    #cv2.imshow("1",frame)
    cv2.waitKey(1)
