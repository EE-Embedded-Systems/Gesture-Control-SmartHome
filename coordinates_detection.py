# TechVidvan hand Gesture Recognizer

# import necessary packages
import train_NN_Model
from train_NN_Model import Classifier
from train_NN_Model import NeuralNetwork
import pandas as pd
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
import torch
import csv

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
             3: 'one', 4: 'two', 5: 'three', 6: 'fuck you dumbass'}

reader = pd.read_csv(
    '/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/data/test_data.csv')
for index, row in reader.iterrows():

    landmarks = np.array(row)
    reshaped_landmarks = landmarks.reshape(1, 63)
    torch_landmarks = torch.from_numpy(reshaped_landmarks)

    # Predict gesture
    y_predicted = trained_model.model(torch_landmarks.float())

    # print(prediction)
    classID = np.argmax(y_predicted.detach().numpy())
    className = label_map[classID]

    print(index, " ", className)
