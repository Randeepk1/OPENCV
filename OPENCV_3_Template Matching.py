#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm


# In[2]:


# x =1 
# print(eval("x+1"))
# print(exec('x+=1'))


# In[3]:


imgage_40 = cv2.imread("messi1.jpg",0)
plt.imshow(imgage_40,cmap=cm.gray)
plt.show()
template = cv2.imread("messi2.jpg",0)
plt.imshow(template,cmap=cm.gray)
plt.show()
template.shape
template = template[0:90,80:130]
template
plt.imshow(template,cmap=cm.gray)
width,height = template.shape[::-1]
methods  = ["cv2.TM_CCOEFF","cv2.TM_CCOEFF_NORMED","cv2.TM_CCORR","cv2.TM_CCORR_NORMED","cv2.TM_SQDIFF","cv2.TM_SQDIFF_NORMED"]
for meth in methods:
    methods_new = eval(meth)
    res = cv2.matchTemplate(imgage_40,template,methods_new)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    if methods_new in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    botom_right = (top_left[0]+width,top_left[1]+height)
    cv2.rectangle(imgage_40,top_left,botom_right,100,1)


# In[4]:


plt.figure(figsize=(15,15))
plt.subplot(121);plt.imshow(res,cmap=cm.gray);plt.title("Matching Result");plt.xticks([]),plt.yticks([])
plt.subplot(122);plt.imshow(imgage_40,cmap=cm.gray);plt.title("Detected point");plt.xticks([]),plt.yticks([])
plt.suptitle(meth)
plt.show()


# In[ ]:




