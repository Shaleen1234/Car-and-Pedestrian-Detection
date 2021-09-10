#Green boxes are for Cars and Red boxes are for humans :)
#This is a project for detecting car and pedestrian
#Hold q to exit the program
#Change the directories of the files

import cv2
import numpy as np

classifier_1 = (r"C:\Users\Shale\Downloads\Car and Pedestrian Detector\ai\cars.xml")
classifier_2 = (r"C:\Users\Shale\Downloads\Car and Pedestrian Detector\ai\cars.xml")


video = cv2.VideoCapture(0)

while True:

    bool_, frame = video.read()
    
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_tracker = cv2.CascadeClassifier(classifier_1)

    trained_data = car_tracker.detectMultiScale(grey_image)

    for (x,y,w,h) in trained_data:
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0), 2)
        cv2.putText(frame, 'Cars', (x, y+20+h), fontScale = 0.8,fontFace = cv2.FONT_HERSHEY_SIMPLEX,color = (0,255, 0))


    grey_image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    body_tracker = cv2.CascadeClassifier(classifier_2)

    trained_data1 = body_tracker.detectMultiScale(grey_image1)

    for (x,y,w,h) in trained_data1:
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,0,255), 2)
        cv2.putText(frame, 'Pedestrians', (x, y+20+h), fontScale = 0.8,fontFace = cv2.FONT_HERSHEY_SIMPLEX,color = (0,0, 255))
    

    print(trained_data, trained_data1)

    cv2.imshow('hello car !!!!!!!!!', frame)

    cv2.waitKey(1)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()
