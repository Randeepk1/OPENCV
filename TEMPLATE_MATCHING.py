#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[2]:


img = cv2.imread("mes1.jpg",0)
plt.imshow(img)
template = cv2.imread("mes1234.jpg",0)
plt.imshow(template)


# In[4]:


w,h = template.shape[::-1]
methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
for meth in methods:
    methed = eval(meth)
    res = cv2.matchTemplate(img,template,methed)
    min_val,max_value,min_loc,max_loc = cv2.minMaxLoc(res)
    if methed in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        left_top = min_loc
    else:
        left_top = max_loc
    bottom_Right = (left_top[0]+w,left_top[1]+h)
    cv2.rectangle(img,left_top,bottom_Right,255,1)
    plt.figure(figsize=(10,10))
    plt.subplot(121);plt.imshow(res,cmap=cm.gray)
    plt.title("Matching Result");plt.xticks([]),plt.yticks([])
    plt.subplot(122);plt.imshow(img,cmap=cm.gray)
    plt.title("Detected Point");plt.xticks([]);plt.yticks([])
    plt.suptitle(meth)
    plt.show()


# In[ ]:




