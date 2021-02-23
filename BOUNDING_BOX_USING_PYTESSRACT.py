#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('invoice1.jpg')


# In[3]:


dic = pytesseract.image_to_data(img,output_type=Output.DICT)
dic.keys()


# In[4]:


n_boxes = len(dic["text"])
for i in range(n_boxes):
    if int(dic["conf"][i]) > 60:
        (x,y,w,h) = (dic["left"][i],dic['top'][i],dic['width'][i],dic['height'][i])
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
plt.figure(figsize=(20,20))        
plt.imshow(img)
plt.title("Bounding_box_with_Dictionary")


# In[5]:


n_boxes


# In[ ]:




