# import necessary packages
import cv2
import numpy as np
import mediapipe as mp

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


GESTURE = "five"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# print headers
with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/data/"+GESTURE+".csv", "a") as text_file:
    for i in range(1, 22):
        print("x"+str(i) + ",y"+str(i) + ",z" +
              str(i), file=text_file, end=",")
    print("label", file=text_file, end="\n")

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
        with open("/Users/marcochan/Desktop/Github/Gesture-Control/Gesture-Control/data/"+GESTURE+".csv", "a") as text_file:
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    print(str(lm.x)+"," + str(lm.y)+"," +
                          str(lm.z), file=text_file, end=",")
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(
                    frame, handslms, mpHands.HAND_CONNECTIONS)
            print(GESTURE, file=text_file, end="\n")

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
