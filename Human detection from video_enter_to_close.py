#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('G:\COMPUTER VISION\haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture(r'C:\Users\dell\Downloads/walking.avi')

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 5, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:



