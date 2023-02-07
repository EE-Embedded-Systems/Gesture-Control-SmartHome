# TechVidvan hand Gesture Recognizer

# import necessary packages
import train_NN_Model
from train_NN_Model import Classifier
from train_NN_Model import NeuralNetwork
import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
import torch

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer models
with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/gesture_recognition_model.pickle", "rb") as target:
    trained_model = pickle.load(target)
print("\nLoaded model in gesture_recongnition_model.pickle\n")


# Load class names
label_map = {0: 'fist', 1: 'thumbs_down', 2: 'thumbs_up',
             3: 'one', 4: 'two', 5: 'three', 6: 'middle finger'}


# Initialize the webcam
cap = cv2.VideoCapture(0)

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
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm
                landmarks.append([lm.x, lm.y, lm.z])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(
                frame, handslms, mpHands.HAND_CONNECTIONS)
            landmarks = np.array(landmarks)
            reshaped_landmarks = landmarks.reshape(1, 63)
            torch_landmarks = torch.from_numpy(reshaped_landmarks)

            # Predict gesture
            y_predicted = trained_model.model(torch_landmarks.float())

            # print(prediction)
            classID = np.argmax(y_predicted.detach().numpy())
            className = label_map[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
