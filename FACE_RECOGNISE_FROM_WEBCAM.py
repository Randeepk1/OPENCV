#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 


# In[2]:


face_cascade1 = cv2.CascadeClassifier('G:\COMPUTER VISION\haarcascade_frontalface_default.xml')
eye_cascade1 = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[3]:


# video_capture = cv2.VideoCapture(0)
# while True:
#     ret,frame = video_capture.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     face = face_cascade1.detectMultiScale(gray,scaleFactor =1.1,minNeighbors=5,minSize=(30,30))
#     for (x,y,w,h) in face:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow('video',frame)
#     if cv2.waitKey(1) & 0xff ==27:
# #     if cv2.waitkey(1) & 0xff ==27:
#         break
# video_capture.release()
# cv2.destroyAllwindows()


# In[ ]:





# In[ ]:




