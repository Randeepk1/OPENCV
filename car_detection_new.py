#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np

car = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('cars.avi')
while cap.isOpened():
    
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car.detectMultiScale(gray, 1.4, 2)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

cap.release()
cv2.destroyAllWindows()

