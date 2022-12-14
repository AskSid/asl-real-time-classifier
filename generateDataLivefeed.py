import numpy as np
import cv2 as cv
import random
import time

import mediapipe as mp
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

letters = ["R", "U", "I", "N", "G", "Z", "T", "S", "A", "F", "O", "H", "nothing", "M", "J", "C", "D", "V", "Q", "X", "E", "B", "K", "L", "Y", "P", "W"]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

startTime = time.time()
timeLine = 0
correctLetter = letters[random.randint(0, len(letters) - 1)]

while True:
    timeLine += 1
    # setting a different letter every 5 seconds
    currentTime = time.time()
    if round(currentTime - startTime) % 5 == 0:
        correctLetter = letters[random.randint(0, len(letters) - 1)]

    # getting the current frame and its shape
    ret, frame = cap.read()
    h, w, c = frame.shape

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_landmarks = result.multi_hand_landmarks
    
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            if x_min < 1:
                x_min = 1
            if y_min < 1:
                y_min = 1

            squareSide = max(x_max - x_min, y_max - y_min)
            if x_min + squareSide > w - 1:
                squareSide = w - x_min
            if y_min + squareSide > h - 1:
                squareSide = h - y_min
            
            pt1x = int(x_min - squareSide/4)
            pt1y = int(y_min - squareSide/4)
            pt2x = int(x_min + (squareSide) + 50)
            pt2y = int(y_min + (squareSide) + 50)
            cv.rectangle(rgb, (pt1x, pt1y), (pt2x, pt2y), (0, 255, 0), 2)
            
            cropped = rgb[y_min : y_min + squareSide, x_min : x_min + squareSide]
    
        cropped = rgb
        result2 = hands.process(cropped)
        hand_landmarks2 = result2.multi_hand_landmarks
        for handLMs2 in hand_landmarks2:
            mp_drawing.draw_landmarks(cropped, handLMs2, mphands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=5, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=1))


    else:
        print("hand not in frame")
    
    try:
        if timeLine % 2 == 0:
            cropped = rgb[pt1y : pt2y, pt1x : pt2x]
            img_name = f"opencv_frame_{timeLine}.jpg"
            cv.imwrite(f"../../../../Downloads/liveDataset/P/{img_name}", cropped)
            print(timeLine)
    except:
        print("error") 
    
    cv.putText(rgb, correctLetter, (60, 680), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    cv.imshow('frame', rgb)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()