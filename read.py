import cv2 as cv
import os

image_path = '/Users/adnanhabib/Downloads/photo.JPG'
video_object = cv.VideoCapture(0)
if __name__ == "__main__" :
    try: 
        while True:
            # returns a tuple
            # 1st item is a boolean value
            # 2nd item is the frame of the video
            # returns false when the live video has ended, so an error will be generated
            ret, frame = video_object.read() 
            cv.imshow('Frames',frame)

            if cv.waitKey(10) & 0xFF == ord('q') :
                break
                

    except:
        print("Video has ended")