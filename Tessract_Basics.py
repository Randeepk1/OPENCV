#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import pytesseract
import time


# In[8]:


# pip install --upgrade pip --user


# In[10]:


img = cv2.imread('you.jpg')


# In[11]:


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"


# In[12]:


print(pytesseract.image_to_string(img))


# In[16]:


start = time.time()
print(pytesseract.image_to_string(img))
end = time.time()
print("detection took",end-start,"(s)")


# In[ ]:




