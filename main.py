from tracemalloc import start
import numpy as np
import pandas as pdc
import cv2 as cv
import random
import time
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

from tensorflow.keras.models import load_model
model = load_model('modelRaw.h5')

letters = ["R", "U", "I", "N", "G", "T", "S", "A", "F", "O", "H", "M", "C", "D", "V", "Q", "X", "E", "B", "K", "L", "Y", "P", "W"]
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

startTime = time.time()
correctLetter = letters[random.randint(0, len(letters) - 1)]

while True:
    currentTime = time.time()
    if round(currentTime - startTime) % 7 == 0:
        correctLetter = letters[random.randint(0, len(letters) - 1)]
        #correctLetter = "C"

    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    h, w, c = frame.shape
    result = hands.process(frame)
    handLandmarks = result.multi_hand_landmarks

    cropped = frame

    if handLandmarks:
        for handLandmark in handLandmarks:
            xMax = 0
            yMax = 0
            xMin = w
            yMin = h
            for landmark in handLandmark.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x > xMax:
                    xMax = x
                if x < xMin:
                    xMin = x
                if y > yMax:
                    yMax = y
                if y < yMin:
                    yMin = y

            if xMin < 1:
                xMin = 1
            if yMin < 1:
                yMin = 1

            offset = 50
            squareSide = max(xMax - xMin, yMax - yMin)
            if xMin + squareSide + offset > w - 1:
                squareSide = w - xMin - offset
            if yMin + squareSide + offset > h - 1:
                squareSide = h - yMin - offset
            if yMin < offset:
                yMin = offset + 1
            if xMin < offset:
                xMin = offset + 1
            
            cv.rectangle(frame, (xMin - offset, yMin - offset), (xMin + squareSide + offset, yMin + squareSide + offset), (255, 0, 255), 2)

            cropped = frame
            mp_drawing.draw_landmarks(cropped, handLandmark, mphands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=5, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255,0,255), thickness=5, circle_radius=1))
        
        cropped = frame[yMin - offset: yMin + squareSide + offset, xMin - offset: xMin + squareSide + offset]
    else:
        print("Hand not in frame")


    cv.putText(frame, "PLEASE SIGN " + correctLetter, (10, 580), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)

    cropped = cv.resize(cropped, (128, 128)).astype('float32') / 255.0
    prediction = model.predict(np.array([cropped]), verbose=0)
    indices = np.argpartition(prediction[0], -5)[-5:]
    
    predictedLetter1 = letters[indices[0]]
    predictedLetter2 = letters[indices[1]]
    predictedLetter3 = letters[indices[2]]
    predictedLetter4 = letters[indices[3]]
    predictedLetter5 = letters[indices[4]]

    #print(predictedLetter1, predictedLetter2, predictedLetter3, predictedLetter4, predictedLetter5)
    
    if (correctLetter == predictedLetter5) or (correctLetter == predictedLetter4) or (correctLetter == predictedLetter3):
        cv.putText(frame, "RIGHT LETTER", (10, 680), cv.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
    else:
        cv.putText(frame, "WRONG LETTER", (10, 680), cv.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
