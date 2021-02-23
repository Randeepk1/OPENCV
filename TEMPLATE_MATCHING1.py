#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[14]:


# img = cv2.imread("m2.jpg",0)
img2 = cv2.resize(img,(75,110),3)
template = img[40:150,100:175]
plt.imshow(img)
# template = cv2.imread("m1.jpg",0)
# plt.subplot(111);plt.imshow(img)
# plt.subplot(111);plt.imshow(template)


# In[15]:


template.shape,img2.shape


# In[16]:


w,h = template.shape[::-1]


# In[17]:


methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
for meth in methods:
    methed = eval(meth)
    res = cv2.matchTemplate(img2,template,methed)
    min_val,max_value,min_loc,max_loc = cv2.minMaxLoc(res)
    if methed in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        left_top = min_loc
    else:
        left_top = max_loc
    bottom_Right = (left_top[0]+w,left_top[1]+h)
    cv2.rectangle(img,left_top,bottom_Right,0,0,255,1)
    plt.figure(figsize=(10,10))
    plt.subplot(121);plt.imshow(res,cmap=cm.gray)
    plt.title("Matching Result");plt.xticks([]),plt.yticks([])
    plt.subplot(122);plt.imshow(img,cmap=cm.gray)
    plt.title("Detected Point");plt.xticks([]);plt.yticks([])
    plt.suptitle(meth)
    plt.show()
    


# In[ ]:




