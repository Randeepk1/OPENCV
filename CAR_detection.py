#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import time


# In[2]:


car = cv2.CascadeClassifier('haarcascade_car.xml')


# In[3]:


video_capture = cv2.VideoCapture('cars.avi')
while True:
    ret,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars = car.detectMultiScale(gray,1.4,2)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('car',frame)
    if cv2.waitKey(1) & 0xff ==27:
        break
video_capture.release()
cv2.destroyAllwindows()


# In[ ]:





# In[ ]:




