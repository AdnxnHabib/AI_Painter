import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

# Import your model class
from digitClassifier import MNISTClassifier, predict_digit

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

# Initialize variables
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Black canvas
predictionResult = None

# Initialize hand detector
myhandDetector = htm.HandDetector(min_detection_confidence=0.85)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mnist_cnn_model.pth" 
model = MNISTClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def preprocess_drawing_for_prediction(canvas):
    """
    Process the drawing canvas into a format suitable for the MNIST model
    """
    drawing = canvas.copy()
    drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY) # convert to gray scale
    
    # Find the bounding box of the drawing
    nonzero = cv2.findNonZero(drawing_gray) # fomd all non-zero pixels (the digit)
    if nonzero is None:
        # No drawing found
        return None
    
    x, y, w, h = cv2.boundingRect(nonzero)
    
    # If the drawing is too small, return None
    if w < 20 or h < 20:
        return None
    
    # Crop the drawing
    cropped = drawing_gray[y:y+h, x:x+w]
    
    # Add padding to make it square (for aspect ratio consistency)
    size = max(w, h) + 20  # Add padding
    square = np.zeros((size, size), np.uint8)
    
    # Center the drawing in the square
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    square[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
    
    # Resize to 28x28 (MNIST size)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    display_size = 140  # 5x the MNIST size for visibility
    display_img = cv2.resize(resized, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    # Normalize to match MNIST preprocessing
    normalized = resized / 255.0
    
    return normalized, display_img

def make_prediction(canvas):
    drawing_mask = cv2.inRange(canvas, (255, 255, 255), (255, 255, 255))
    canvas = cv2.bitwise_and(canvas, canvas, mask=drawing_mask)
    processed_img, display_img = preprocess_drawing_for_prediction(canvas)
    if processed_img is not None:
        predicted_digit, confidence_scores = predict_digit(model, processed_img, device)
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence scores: {confidence_scores}")
        return predicted_digit, display_img
    return None

# Add instructions to the canvas
instructions = imgCanvas.copy()
cv2.putText(instructions, "Press 'p' to predict", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
cv2.putText(instructions, "Press 'c' to clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
cv2.putText(instructions, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
imgCanvas = instructions.copy()
preprocessed_display = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip camera 
    
    # Find hand landmarks
    img = myhandDetector.findHands(img)
    lmList = myhandDetector.findPosition(img, draw=False)
    
    displayCanvas = imgCanvas.copy()
    
    current_mode = "Idle"
    
    if lmList:
        # Get index and middle finger positions
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        
        # Find which fingers are up
        fingers = myhandDetector.findFingers(img, lmList)
        
        # Selection mode -> two fingers are up
        if fingers[1] and fingers[2]:
            current_mode = "Selection"
            # You could add interface elements here if needed
        
        # Drawing mode -> Index finger is up
        elif fingers[1] and fingers[2] == 0:
            current_mode = "Drawing"
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            # Draw white line on the black canvas
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (255, 255, 255), 15)
            
            xp, yp = x1, y1
    else:
        xp, yp = 0, 0
    

    cv2.putText(displayCanvas, f"Mode: {current_mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if predictionResult is not None:
        cv2.putText(displayCanvas, f"Prediction: {predictionResult}", (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Show frames
    cv2.imshow("Hand Tracking", img) 
    cv2.imshow("Drawing Canvas", displayCanvas) 
    
    if preprocessed_display is not None:
        cv2.imshow("Preprocessed Image", preprocessed_display)
    
    key = cv2.waitKey(1) & 0xFF # keyboard input

    if key == ord('p'):
        
        predictionResult, preprocessed_display = make_prediction(imgCanvas)
    
    elif key == ord('c'):
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        # clear text and then re-add instructions
        cv2.putText(imgCanvas, "Press 'p' to predict", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(imgCanvas, "Press 'c' to clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.putText(imgCanvas, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        predictionResult = None
        preprocessed_display = None
        print("Canvas Cleared")
        xp, yp = 0, 0
    
    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()