#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox


# In[2]:


img = cv2.imread('traffic1.jpg')


# In[3]:


cv2.imshow("trafic1",img)
cv2.waitKey(0)


# In[4]:


plt.figure(figsize=(5,5))
plt.imshow(img)
plt.show()


# In[5]:


box,label,conf = cv.detect_common_objects(img)


# In[6]:


result = draw_bbox(img,box,label,conf)


# In[7]:


plt.figure(figsize=(20,8))
plt.imshow(result)
plt.show()


# In[ ]:




