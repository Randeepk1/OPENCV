#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import os
import shutil


# In[2]:


vid = cv2.VideoCapture("./Data Science.mp4")
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv2.CAP_PROP_FPS)


# In[3]:


print("width %d ,height,%d"% (width,height))


# In[4]:


print("fps %f" %(fps))


# In[5]:


rect,frame = vid.read()
forcc = cv2.VideoWriter.fourcc(*'XVID')
vid_writr = cv2.VideoWriter("./output2.avi",forcc,fps,(int(width),int(height)))


# In[6]:


img2 = cv2.imread("Logo.png")


# In[7]:


img2 = cv2.resize(img2,(int(250),int(250)))


# In[8]:


while rect:
    img1 = frame
    rows,cols,channels = img2.shape
#     img1[0:rows,0:cols]=img2 # logo for left corner
#     img1[0:rows,int(width)-cols:int(width)]=img2 # logo for right corner
#     img3 =img1[0:rows,int(width)-cols:int(width)]
#     img3 = cv2.addWeighted(img3,0.7,img2,0.3,1)    # img for opacity(transparent)
#     img1[0:rows,int(width)-cols:int(width)]=img3   # logo for right corner after opacity
    img4=img1[0:rows,0:cols]
    img4 = cv2.addWeighted(img4,0.7,img2,0.3,1)
    img1[0:rows,0:cols]=img4
    vid_writr.write(frame)
    rect,frame=vid.read()
vid.release()
vid_writr.release()
    


# In[ ]:




