# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load class names
f = open('/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/HandSignDetection/gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Load the gesture recognizer model
model = load_model('/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/HandSignDetection/mp_hand_gesture')


# Initialize the webcam
cap = cv2.VideoCapture(0)
with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/Data/thumbs_down.csv", "a") as text_file:
    for i in range(1,22):
        print("x"+str(i) + ",y"+str(i)+ ",z"+ str(i), file=text_file, end=",")
    print('\n', file=text_file, end="")

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/Data/thumbs_down.csv", "a") as text_file:
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    print(str(lm.x)+","+ str(lm.y)+","+ str(lm.z), file=text_file, end=",")
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
            print('\n', file=text_file, end="")

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()