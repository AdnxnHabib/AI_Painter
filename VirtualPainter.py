import cv2
import numpy as np
import time 
import os
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
cap.set(3, 1280) # width
cap.set(4, 720) # height
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
myhandDetector = htm.HandDetector(min_detection_confidence=0.85)

while True:
    # import img
    success, img = cap.read()
    img = cv2.flip(img, 1) # so that when u draw with ur left hand, you will see ur left hand (flips the camera)
    # Find hand landmarks 
    img = myhandDetector.findHands(img)
    lmList = myhandDetector.findPosition(img, draw=False)


    if lmList:
        # index 8 represnts the tip of the index finger (x, y)
        x1, y1 = lmList[8][1:]
        # index 12 represnts the tip of the middle finger (x, y)
        x2, y2 = lmList[12][1:]
    # Find which fingers are up
        fingers = myhandDetector.findFingers(img, lmList)
        

        print(fingers)
    # If Erasing mode -> two fingers are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 25)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

             # draw a black line, black because the image canvas is black, so it effectively erases 
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 50)

            xp, yp = x1, y1
            print("selection mode")

    # If Drawing mode -> Index finger is up
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(imgCanvas, (xp, yp), (x1, y1), (255, 0, 0), 15)

            xp, yp = x1, y1

    # Show and break frame:
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) # combine both the image and image canvas together
    cv2.imshow("Image", img) 
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
   