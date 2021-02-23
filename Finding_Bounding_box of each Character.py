#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[2]:


img = cv2.imread("OXFAM_EAWARD.jpg")


# In[3]:


h,w,c = img.shape


# In[7]:


boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b= b.split(" ")
    img = cv2.rectangle(img,(int(b[1]),h-int(b[2])),(int(b[3]),h-int(b[4])),(0,255,0),2)
plt.figure(figsize=(20,20))
plt.imshow(img)   


# In[ ]:




