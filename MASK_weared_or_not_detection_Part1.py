#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('Randeepk.jpg')


# In[3]:


img.shape


# In[4]:


img[0]  


# In[5]:


plt.imshow(img)


# In[6]:


# while True:
#     cv2.imshow('Randeep',img)
#     if cv2.waitKey(0) == 10:
#         break
#     cv2.destroyAllWindows()
    


# In[7]:


har_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[8]:


har_data.detectMultiScale(img)


# In[9]:


while True:
    faces = har_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4) # x is xlabel,y is y label,w is width,h is height,255,0,255 is color of rgb,4 is width of label
    cv2.imshow('Randeep',img)
    if cv2.waitKey(2) == 13:
        break
cv2.destroyAllWindows()


# In[17]:


cap = cv2.VideoCapture(0)
data = []
while True:
    rect,frame = cap.read()
    if rect:
        faces = har_data.detectMultiScale(frame)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),5)
            face = frame[y:y+h,x:x+w,:]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<300:
                data.append(face)
        cv2.imshow('Randeep',frame)
        if cv2.waitKey(2) == 13 or len(data)>=200:
            break
cap.release()
cv2.destroyAllWindows()
        
    


# In[18]:


np.save("With_Mask.npy",data)


# In[16]:


np.save("Without_Mask.npy",data)


# In[20]:


plt.imshow(data[2])

