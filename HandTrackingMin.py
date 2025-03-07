import cv2
import mediapipe as mp
import time 
from mediapipe.python.solutions import hands as mpHands
from mediapipe.python.solutions import drawing_utils as mpDraw

cap = cv2.VideoCapture(0)
hands = mpHands.Hands() # initalizes a mediapipe hand object 

prev_time = 0
curr_time = 0


while True: 
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks: # represents one detected hand -> each hand has 21 landmarks associated with it
            for id, lm in enumerate(handlms.landmark): # index of the landmark and the landmark in (x, y, z) represented as a ratio of the image
                 height, width, channels = img.shape # 720, 1280 = (height, width)
                 center_x, center_y = int(lm.x * width), int(lm.y * height)
                 # now we can detect an indiviual landmark and print a circle on top of it
                 if id == 8: 
                     print(center_x, center_y)
                     cv2.circle(img, (center_x, center_y), 25, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS); 

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0),  3 )
    cv2.imshow("Image", img) 
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break