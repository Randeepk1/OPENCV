#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import shutil
import os


# In[2]:


vid = cv2.VideoCapture('street.mp4')
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv2.CAP_PROP_FPS)


# In[3]:


print("width %d ,height,%d"% (width,height))


# In[4]:


print("fps %f" %(fps))


# In[5]:


rect, frame = vid.read()
if os.path.exists('Randeepopencvvido'):
    shutil.rmtree("./Randeepopencvvido")
os.mkdir("./Randeepopencvvido")
forcc = cv2.VideoWriter.fourcc(*"XVID")
vid_writer = cv2.VideoWriter("./Randeepopencvvido5.avi",forcc,fps,(int(width),int(height)))


# In[6]:


img1 = cv2.imread("Randeepk.jpg")
img1 = cv2.resize(img1,(int(width),int(height)))
while rect:
    frame  = cv2.medianBlur(frame,5)
    frame  = cv2.addWeighted(frame,0.4,img1,0.7,0)
    vid_writer.write(frame)
    rect,frame = vid.read()
vid.release()
vid_writer.release()


# In[ ]:




