#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


img = cv2.imread("RANDEEP.jpg")
h,w,c = img.shape


# In[3]:


font = cv2.FONT_HERSHEY_PLAIN                   #fonts
botom_left_corner_of_text =(10,h-355)             # where to fix text  (x,y)
font_scale =5                                   # scale
font_color = (0,0,255)                          # color
thickness =2                                    # thickness
line_type = 5                                   #line type (optional)


# In[4]:


img = cv2.putText(img,"RANDEEP",botom_left_corner_of_text,font,font_scale,font_color,thickness,line_type)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()


# In[ ]:





# In[ ]:




