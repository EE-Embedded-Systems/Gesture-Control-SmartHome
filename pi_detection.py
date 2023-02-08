# import necessary packages
import train_NN_Model
from train_NN_Model import Classifier
from train_NN_Model import NeuralNetwork
import cv2
import numpy as np
import mediapipe as mp
import requests


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Load class names
label_map = {0: 'fist', 1: 'thumbs_down', 2: 'thumbs_up',
             3: 'one', 4: 'two', 5: 'three', 6: 'fuck you dumbass'}


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
            print(landmarks)

        # HTTP post request
        data = {'coordinates': landmarks}
        response = requests.post(
            'https://your-server.com/receive_coordinates.php', json=data)
        if response.status_code == 200:
            print('Coordinates sent successfully')
        else:
            print(f'Error sending coordinates: {response.text}')
