#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


# In[2]:


img  =  cv2.imread('fruts.jpeg')


# In[3]:


cv2.imshow("fruts",img)
cv2.waitKey(0)


# In[4]:


box,label,conf = cv.detect_common_objects(img)


# In[5]:


Result = draw_bbox(img,box,label,conf)
Result


# In[6]:


cv2.imshow('Fruits_result',Result)
cv2.waitKey(0)


# In[7]:


plt.figure(figsize=(8,8))
plt.imshow(Result)
plt.show()


# In[ ]:




