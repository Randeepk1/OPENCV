#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[18]:


img = cv2.imread("OXFAM_EAWARD.jpg")
plt.imshow(img)
plt.show
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[3]:


def gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[4]:


def remove_noise(img):
    return cv2.medianBlur(img,5)


# In[5]:


def thresholding(img):
    return cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# In[6]:


def dilate(img):
    kernal = np.ones((5,5),np.uint8)
    return cv2.dilate(img,kernal,iteration=1)


# In[7]:


def erode(img):
    kernal = np.ones((5,5),np.uint8)
    return cv2.erode(img,kernal,iteration = 1)


# In[8]:


def opening(img):
    kernal = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernal)


# In[9]:


def canny(img):
    return cv2.Canny(img,100,200)


# In[10]:


def match_template(img,templates):
    cv2.matchTemplate(img,templates,cv2.TM_CCOEFF_NORMED)


# In[11]:


gray = gray(img)
blur = remove_noise(gray)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)


# In[12]:


plt.figure(figsize=(15,15))
plt.subplot(331);plt.title("gray");plt.imshow(gray,cmap=cm.gray)
plt.subplot(332);plt.title("blur");plt.imshow(blur)
plt.subplot(333);plt.title("thresh");plt.imshow(thresh)
plt.subplot(334);plt.title("opening");plt.imshow(opening)
plt.subplot(335);plt.title("canny");plt.imshow(canny)


# In[13]:


img = gray
print(pytesseract.image_to_string(img))


# In[14]:


img = blur
print(pytesseract.image_to_string(img))


# In[15]:


img = thresh
print(pytesseract.image_to_string(img))


# In[16]:


img = opening
print(pytesseract.image_to_string(img))


# In[17]:


img = canny
print(pytesseract.image_to_string(img))


# In[ ]:




