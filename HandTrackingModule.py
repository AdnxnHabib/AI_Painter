import cv2
import mediapipe as mp
import time 
from mediapipe.python.solutions import hands as mpHands
from mediapipe.python.solutions import drawing_utils as mpDraw


class HandDetector():
    def __init__(self, static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mpHands
        self.hands = self.mpHands.Hands( self.static_image_mode, self.max_num_hands, self.model_complexity, self.min_detection_confidence,  self.min_tracking_confidence)
        self.mpDraw = mpDraw

        self.tipIndices = [4, 8, 12, 16, 20] # thumb, index, middle, ring, pinky
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # find a hand 
        if self.results.multi_hand_landmarks: 
            for handlms in self.results.multi_hand_landmarks: # represents one detected hand -> each hand has 21 landmarks associated with it
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS); 
        return img

    def findPosition(self, img, handNo=0, draw=True): # gets the landmark list 
          lmlist = []
          if self.results.multi_hand_landmarks: 
            myhandlms = self.results.multi_hand_landmarks[handNo] # selects an indiviual hand and represents all 21 of that hands landmarks
            for id, lm in enumerate(myhandlms.landmark): # index of the landmark and the landmark in (x, y, z) represented as a ratio of the image
                            height, width, channels = img.shape # 720, 1280 = (height, width)
                            center_x, center_y = int(lm.x * width), int(lm.y * height)
                            lmlist.append([id, center_x, center_y])
                            # now we can detect an indiviual landmark and print a circle on top of it
                            # if id == 8: 
                            #     cv2.circle(img, (center_x, center_y), 25, (255, 0, 0), cv2.FILLED)
            return lmlist
        
    def findFingers(self, img, lmlist):
         # for the thumb, we measure the x position. (close ur thumb and youll see a closed position has less width compared to open)
        fingers = []
        thumb_tip = self.tipIndices[0] # landmark 4

        if(lmlist[thumb_tip][1] < lmlist[thumb_tip - 2][1]): # the tip of the thumb x position is greater than the thumbs other landmark
            fingers.append(1)
        else:
            fingers.append(0)
        
         # other fingers
        for i in range(1, len(self.tipIndices)):
             finger_tip = self.tipIndices[i]
             if lmlist[finger_tip][2] < lmlist[finger_tip - 2][2]: 
                  # the y axis begins from the top of the screen, thefore the more down you go, the higher the y position is
                  # therefore the tip of the finger y position must be less than the 
                  fingers.append(1)
             else:
                  fingers.append(0)
        
        return fingers
             
         
         

def main():
    # cap = cv2.VideoCapture(0)
    # prev_time = 0
    # curr_time = 0
    # myhandDetector = HandDetector() 
    # while True: 
    #     success, img = cap.read()
    #     img = myhandDetector.findHands(img) # do I need to return img liek this? 
    #     lmlist = myhandDetector.findPosition(img)
    #     if lmlist:
    #          print(lmlist)
    #     curr_time = time.time()
    #     fps = 1 / (curr_time - prev_time)
    #     prev_time = curr_time

    #     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0),  3 )
    #     cv2.imshow("Image", img) 
    #     if cv2.waitKey(1) & 0xFF == ord('q') :
    #         break
    pass

if __name__ == "__main__":
    main()