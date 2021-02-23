#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[2]:


img = cv2.imread("py.webp")


# In[3]:


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"


# In[4]:


img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[5]:


print(pytesseract.image_to_string(img))

