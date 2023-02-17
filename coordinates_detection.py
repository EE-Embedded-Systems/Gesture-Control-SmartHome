# TechVidvan hand Gesture Recognizer

# import necessary packages
from train_NN_Model import Classifier
from train_NN_Model import NeuralNetwork
import numpy as np
import pickle
import torch
import json
import time 
import fcntl


# Load the gesture recognizer models
with open("/opt/lampp/htdocs/Gesture-Control/gesture_recognition_model.pickle", "rb") as target:
    trained_model = pickle.load(target)
print("\nLoaded model in gesture_recongnition_model.pickle\n")


# Load class names
label_map = {0: 'Fist', 1: 'Thumbs Down', 2: 'Thumbs Up', 3: 'One', 4: 'Two', 5: 'Three', 6: 'Rock', 7: 'Right', 8: 'Left', 9: 'Three', 10: 'Three'}

#Initialize className 
className = ""
while True: 
    className_previous = className
    with open('coordinates.json') as f:
        coordinates = json.load(f)

    if coordinates == []: 
        className = "Not Detected"
    else: 
        landmarks = np.array(coordinates)
        reshaped_landmarks = landmarks.reshape(1, 63)
        torch_landmarks = torch.from_numpy(reshaped_landmarks)

        # Predict gesture
        y_predicted = trained_model.model(torch_landmarks.float())

        # print(prediction)
        classID = np.argmax(y_predicted.detach().numpy())
        className = label_map[classID]

    # Only write on change in label 
    if(className != className_previous): 
        with open("/opt/lampp/htdocs/dash/resources/prediction.txt", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # do some processing here
                f.write(className)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
                print(className)

    # Add artificial slowdown to reduce CPU load (40% improvement)
    time.sleep(0.2)
